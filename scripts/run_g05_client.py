"""Run a RoboColosseum G0.5 checkpoint on the real SO-101 arm (robot-side client).

Thin wrapper around GalaxeaVLA's official SO100 client
(vendor/GalaxeaVLA/experiments/so100/so100_policy_client.py), which talks to the
policy server started by scripts/run_g05_server.sh over WebSocket. The vendored
client is left untouched; three things are patched for our checkpoints:

1. Joint frame. The official client converts lerobot joints into Galaxea's own
   training frame (signs [1,-1,1,1,1,1], offsets [0,90,90,0,0,0]). Our g05-so101-*
   checkpoints were fine-tuned on our raw lerobot v3 datasets (their
   dataset_stats.json equals the dataset's meta/stats.json), so the conversion
   must be the identity -- otherwise shoulder_lift is inverted and shoulder_lift/
   elbow_flex are shifted by 90 degrees.
2. Home pose. Replaced by the training-state mean from the checkpoint's
   dataset_stats.json (the official one is Galaxea's data mean).
3. Cameras. The official client only opens OpenCV indices; our front camera is a
   RealSense D435. Cameras are opened with the same lerobot camera classes
   record_dataset.py uses. A --camera-index value of "rs:<serial>" opens a
   RealSense, anything else is an OpenCV index or device path.

Camera slots follow the training-time mapping in SO100CanonicalLerobotDatasetV3:
front -> exterior, wrist -> wrist_right, wrist_left zero-padded.

Run with envs/.venv (lerobot is already installed there):
    envs/.venv/bin/python scripts/run_g05_client.py --dry-run
It runs with --no-display (envs/.venv's OpenCV has no GUI): after homing, type the
task at the "Task>" prompt; typing a new line later updates the task.
Any extra flags are forwarded to the official client (see its --help).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
GALAXEA = REPO / "vendor" / "GalaxeaVLA"
CLIENT_PATH = GALAXEA / "experiments" / "so100" / "so100_policy_client.py"

DEFAULT_CHECKPOINT = "checkpoints/g05-so101-stack-white_bowls-100episodes"
DEFAULT_ARGS = {
    "--robot-port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114717-if00",
    "--robot-id": "follower_arm",
    "--camera-index": [
        "exterior:rs:262522073381",
        "wrist_right:/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
    ],
    "--camera-map": ["exterior:front", "wrist_right:wrist"],
    # Training chunks are sampled at the dataset fps (30), not the client's default 15.
    "--action-fps": "30",
    # envs/.venv has headless OpenCV (no GUI backend), so the client's cv2.imshow dashboard
    # crashes right after homing starts. Without it, the task is typed at a "Task>" prompt.
    "--no-display": None,
}
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30

sys.path[:0] = [str(GALAXEA / "src"), str(GALAXEA)]
spec = importlib.util.spec_from_file_location("so100_policy_client", CLIENT_PATH)
client = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = client
spec.loader.exec_module(client)


class LerobotCameraWorker:
    """Drop-in for the official CameraWorker, backed by lerobot cameras."""

    def __init__(self, index, width: int = CAM_WIDTH, height: int = CAM_HEIGHT, fps: int = CAM_FPS):
        index = str(index)
        if index.startswith("rs:"):
            from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

            self._cam = RealSenseCamera(RealSenseCameraConfig(
                serial_number_or_name=index[3:], width=width, height=height, fps=fps))
        else:
            from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

            self._cam = OpenCVCamera(OpenCVCameraConfig(
                index_or_path=int(index) if index.isdigit() else Path(index),
                width=width, height=height, fps=fps))
        self._cam.connect()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"Camera-{index}")
        self._thread.start()
        client.logger.info("[Camera %s] started (%dx%d @ %dfps)", index, width, height, fps)

    def _loop(self):
        while not self._stop.is_set():
            try:
                frame = self._cam.async_read(timeout_ms=500)  # RGB HWC uint8
            except Exception as e:  # noqa: BLE001 -- keep the thread alive across hiccups
                client.logger.warning("[Camera] read failed: %s", e)
                continue
            with self._lock:
                self._frame = frame

    def read_rgb_chw(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.transpose(2, 0, 1).copy()

    def read_bgr(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else cv2.cvtColor(self._frame, cv2.COLOR_RGB2BGR)

    def read_latest(self) -> np.ndarray:
        """Latest RGB HWC frame (for scripts/trial_recording.py's Recorder)."""
        with self._lock:
            if self._frame is None:
                raise RuntimeError("no frame yet")
            return self._frame.copy()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cam.disconnect()


def multi_run(argv: list[str]) -> None:
    """Connect once, run the policy many times (used by scripts/run_g05_checkpoint.sh).

    Same controls as the lerobot run scripts (scripts/multi_rollout.py):
        Enter            start a run (at the prompt) / stop the current run early
        q + Enter        quit (arm returns to the start pose, then disconnects)
        other text       at the prompt: set it as the task and start; during a run: update the task
    Each run lasts up to DURATION seconds (env, default 30; 0 = until Enter). Between runs the arm
    returns to the pose it had when the client connected. Built from the official client's pieces
    (FollowerArm, InferenceProducer, homing); each run opens a fresh InferenceProducer, whose
    first request sends a full observation, so the server re-infers instead of replaying a
    cached chunk from the previous run."""
    import argparse
    import logging
    import queue
    import time

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--robot-port", required=True)
    p.add_argument("--robot-id", required=True)
    p.add_argument("--action-fps", type=float, default=30.0)
    p.add_argument("--max-step-deg", type=float, default=10.0)
    p.add_argument("--camera-map", nargs="+", default=[])
    p.add_argument("--camera-index", nargs="+", default=[])
    p.add_argument("--dummy-camera-shape", type=client._parse_chw_shape, default=(3, 480, 640))
    p.add_argument("--dummy-camera-key", nargs="*", default=["wrist_left"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-display", action="store_true")  # always headless here
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    duration = float(os.environ.get("DURATION", "30"))
    task = os.environ.get("TASK", "").strip()
    camera_map = client._parse_key_value_pairs(args.camera_map)
    camera_index = client._parse_key_value_pairs(args.camera_index)
    expected_camera_keys = list(dict.fromkeys([*camera_map, *camera_index, *args.dummy_camera_key]))
    proprio_guard = client.build_proprio_guard(client.load_client_config(args.config))

    lines: queue.Queue[str] = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            lines.put(line.strip())
        lines.put("q")  # EOF

    follower = client._connect_follower(args)
    camera_workers = {k: client.CameraWorker(v) for k, v in camera_index.items()}
    time.sleep(0.5)

    # Start pose = where the arm is when the client connects (the rest pose), like the lerobot run
    # scripts. The official client instead homes to the training-state MEAN (e.g. shoulder_lift
    # ~-3 deg: arm raised mid-air). But every training episode starts from the rest pose
    # (shoulder_lift ~-102, elbow ~+90), so starting each trial from the mean pose -- with the
    # scene still in its initial state -- is a combination the model never saw. The mean pose also
    # sags under gravity, so homing to it always hit the 15 s timeout. G05_HOME=mean restores it.
    home_mode = os.environ.get("G05_HOME", "start")
    home = client._HOME_ARM if home_mode == "mean" else follower.get_state().copy()
    client.logger.info("[g05] start pose (%s): %s", home_mode, np.round(home, 1).tolist())

    if args.dry_run:
        class DryRunFollower:
            def get_state(self):
                return follower.get_state()

            def set_target(self, t):
                client.logger.info("[DryRun] would set_target %s", np.round(t, 1))

        active = DryRunFollower()
    else:
        active = follower

    def banner(msg: str) -> None:
        print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)

    producer = None
    runs = 0

    def run_trial() -> str:
        """One policy run until DURATION / Enter / q; typed text updates the task."""
        nonlocal task, producer
        producer = client.InferenceProducer(
            follower=active, camera_workers=camera_workers,
            ws_uri=f"ws://{args.host}:{args.port}", task=task,
            action_fps=args.action_fps, max_step_deg=args.max_step_deg,
            proprio_guard=proprio_guard, expected_camera_keys=expected_camera_keys,
            dummy_camera_shape=args.dummy_camera_shape,
        )
        producer.start()
        start = time.monotonic()
        try:
            while True:
                if duration > 0 and time.monotonic() - start >= duration:
                    return f"{duration:.0f}s limit reached"
                try:
                    line = lines.get(timeout=0.05)
                except queue.Empty:
                    continue
                if line.lower() in ("q", "quit", "exit"):
                    return "quit"
                if not line:
                    return "stopped with Enter"
                task = line
                producer.set_task(line)
        finally:
            producer.stop()
            producer.join(timeout=3.0)
            producer = None
            follower.set_target(follower.get_state())  # hold where it is

    def go_home() -> None:
        print(">>> Returning arm to start pose...", flush=True)
        follower.move_to_home(home=home, tol_deg=3.0)

    try:
        if home_mode == "mean":
            follower.move_to_home(home=home)
        if not task:
            task = input("Task> ").strip()
        threading.Thread(target=read_stdin, daemon=True, name="stdin").start()
        limit = f"{duration:.0f}s" if duration > 0 else "no time limit"
        if os.environ.get("RECORD") == "1":
            from trial_recording import LatestValue, model_name_from_path, run_recording_session, slug

            # For the LeRobot-format episode log: every target the policy hands to the arm.
            action_cache = LatestValue()
            set_target = active.set_target

            def set_target_logged(target):
                action_cache.set(target)
                set_target(target)

            active.set_target = set_target_logged

            # server camera key -> file name (exterior -> front, wrist_right -> wrist)
            camera_map_names = dict(camera_map)
            run_recording_session(
                cameras={camera_map_names.get(k, k): w for k, w in camera_workers.items()},
                fps=args.action_fps,
                model=os.environ.get("MODEL_NAME") or model_name_from_path(os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT)),
                task_name=os.environ.get("TASK_NAME") or slug(task),
                trials=int(os.environ.get("TRIALS", "25")),
                out_dir=Path(os.environ.get("RECORD_DIR", str(REPO / "outputs" / "recordings"))),
                limit=f"{limit} limit" if duration > 0 else limit,
                get_line=lines.get,
                run_trial=run_trial,
                go_home=go_home,
                is_shutdown=lambda: False,  # Ctrl-C raises KeyboardInterrupt here instead
                task=task,
                get_state=follower.get_state,  # refreshed continuously by the FollowerArm worker
                action_cache=action_cache,
            )
            return
        while True:
            banner(
                f"Ready for run #{runs + 1} ({limit}{', DRY RUN' if args.dry_run else ''}).\n"
                f"Task: {task}\n"
                "  Enter = start   |   q + Enter = quit   |   new task text + Enter = set task and start\n"
                "  (during a run) Enter = stop early, new text + Enter = update task, Ctrl-C = quit"
            )
            line = lines.get()
            if line.lower() in ("q", "quit", "exit"):
                break
            if line:
                task = line
            runs += 1
            print(f">>> Run #{runs} started: {task}", flush=True)
            reason = run_trial()
            print(f">>> Run #{runs} finished ({reason})", flush=True)
            if reason == "quit":
                break
            go_home()
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        if producer is not None:
            producer.stop()
            producer.join(timeout=3.0)
        print(f">>> Quitting{f' after {runs} run(s)' if runs else ''}: returning arm to start pose...", flush=True)
        try:
            follower.set_target(follower.get_state())
            follower.move_to_home(home=home, tol_deg=3.0)
        except Exception as e:  # noqa: BLE001 -- still release the hardware below
            client.logger.warning("[g05] final homing failed: %s", e)
        for cw in camera_workers.values():
            cw.release()
        follower.disconnect()


def main():
    checkpoint = REPO / os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT)
    stats = json.loads((checkpoint / "dataset_stats.json").read_text())
    home = np.array(stats["so100"]["state"]["right_arm"]["global_mean"], dtype=np.float32)

    client._SIGNS = np.ones(client.JOINT_COUNT, dtype=np.float32)
    client._OFFSETS = np.zeros(client.JOINT_COUNT, dtype=np.float32)
    client._HOME_ARM = home
    defaults = list(client.FollowerArm.move_to_home.__defaults__)
    defaults[0] = home
    client.FollowerArm.move_to_home.__defaults__ = tuple(defaults)
    client.CameraWorker = LerobotCameraWorker

    argv = sys.argv[1:]
    use_multi_run = "--multi-run" in argv
    if use_multi_run:
        argv.remove("--multi-run")
    for flag, value in DEFAULT_ARGS.items():
        if flag not in argv:
            if value is None:  # boolean flag
                argv += [flag]
            else:
                argv += [flag, *value] if isinstance(value, list) else [flag, value]
    sys.argv = [str(CLIENT_PATH), *argv]
    client.logger.info("[g05] checkpoint=%s home=%s", checkpoint, np.round(home, 1).tolist())
    if use_multi_run:
        multi_run(argv)
    else:
        client.main()


if __name__ == "__main__":
    main()
