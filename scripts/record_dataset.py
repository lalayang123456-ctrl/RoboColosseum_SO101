#!/usr/bin/env python3
"""Record a LeRobot teleop dataset on the SO-101, episode by episode, deciding
after each one whether to keep it or throw it away and redo it.

Why not plain `lerobot-record`: it saves every episode automatically (re-record
is only possible *during* an episode, via a pynput arrow-key listener that is
not installed in this venv). This script reuses lerobot's own record_loop and
LeRobotDataset writer -- so the on-disk format is exactly what lerobot-record
produces (v3.0, same as datasets/robocolosseum-so101-stack-white_bowls-100episodes)
-- and only changes the control flow. Keys are read straight from this
terminal, so no X/pynput is needed.

Per episode:
    [Enter] start recording   (between episodes the follower keeps copying the
                               leader, so you can reset the scene)
    [Enter] end the episode early (otherwise it stops after --episode-time)
    [s] save   [r] discard and re-record   [q] discard and quit
Ctrl-C at any point: already-saved episodes are kept, the current one is dropped.

Running it again with the same --root appends to the existing dataset.

Usage:
    envs/.venv/bin/python scripts/record_dataset.py --task "stack the white bowls"
    envs/.venv/bin/python scripts/record_dataset.py --root datasets/my_set --verify-only

Train on it with:
    lerobot-train --dataset.repo_id=<repo-id> --dataset.root=<root> ...
"""

import argparse
import contextlib
import json
import logging
import math
import os
import select
import shutil
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.utils import init_logging

BY_ID = "/dev/serial/by-id"
DEFAULT_FOLLOWER_PORT = f"{BY_ID}/usb-1a86_USB_Single_Serial_5B8E114717-if00"
DEFAULT_LEADER_PORT = f"{BY_ID}/usb-1a86_USB_Single_Serial_5B79016937-if00"
DEFAULT_FRONT_SERIAL = "262522073381"  # RealSense D435, table overview
# USB2.0_CAM1 mounted on the arm. Fixed by-id path: /dev/videoN numbers get
# reshuffled across reboots/replugs (index 4 once was this cam, later a RealSense node).
DEFAULT_WRIST_CAM = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
DEFAULT_TASK = (
    "Pick up the white plastic bowl on the right and stack it on top of the white plastic bowl on the left."
)
# Existing dataset recorded with the same arm/camera layout; new data is checked
# against its features so the two can be merged / trained on together.
REFERENCE_DATASET = Path("datasets/robocolosseum-so101-stack-white_bowls-100episodes")

ENTER = "\n"

# LeRobot only starts a new data/video/meta file when the current one would
# exceed these limits; making them tiny gives one file per episode
# (data/.../file-007.parquet == episode 7), the same layout as the reference
# dataset, instead of packing many episodes into one mp4.
ONE_FILE_PER_EPISODE_MB = 1e-9


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="local/so101-teleop", help="dataset name stored in meta; use <hf_user>/<name> if you'll push it")
    p.add_argument("--root", default="datasets/so101-teleop", help="dataset directory; appended to if it already exists")
    p.add_argument("--task", default=DEFAULT_TASK, help="language instruction stored with every frame")
    p.add_argument("--num-episodes", type=int, default=0, help="stop once the dataset holds this many episodes in total (counting earlier sessions); 0 = until you press q")
    p.add_argument("--episode-time", type=float, default=60.0, help="max seconds per episode")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    p.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    p.add_argument("--follower-id", default="follower_arm", help="calibration id (reuses teleoperate.sh's)")
    p.add_argument("--leader-id", default="leader_arm")
    p.add_argument("--front-serial", default=DEFAULT_FRONT_SERIAL)
    p.add_argument("--wrist-cam", default=DEFAULT_WRIST_CAM, help="device path or OpenCV index of the wrist camera")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--vcodec", default="h264", help="h264 matches the reference dataset; libsvtav1 is lerobot's default")
    p.add_argument("--packed", dest="one_file_per_episode", action="store_false",
                   help="pack many episodes per mp4/parquet (lerobot default) instead of one file per episode")
    p.add_argument("--no-streaming", action="store_true", help="write PNGs and encode on save instead of encoding live")
    p.add_argument("--display-data", action="store_true", help="live Rerun view of cameras/joints")
    p.add_argument("--verify-only", action="store_true", help="just check an existing dataset and exit")
    return p.parse_args()


class KeyReader:
    """Reads single keypresses from the terminal on a background thread.

    `arm(keys)` declares which keys the current phase reacts to; the first
    matching press is stored and ends lerobot's record_loop via
    events["exit_early"]. Ctrl-C still raises KeyboardInterrupt (cbreak mode
    keeps ISIG).
    """

    def __init__(self, events: dict):
        self.events = events
        self._valid: set[str] = set()
        self._pressed: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            ch = os.read(self._fd, 1).decode(errors="ignore")
            ch = ENTER if ch == "\r" else ch.lower()
            with self._lock:
                if self._pressed is None and ch in self._valid:
                    self._pressed = ch
                    self.events["exit_early"] = True

    def arm(self, keys: set[str]):
        with self._lock:
            self._valid = keys
            self._pressed = None
            self.events["exit_early"] = False

    def pressed(self) -> str | None:
        with self._lock:
            return self._pressed

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)


def build_robot(args) -> SOFollower:
    cameras = {
        "front": RealSenseCameraConfig(
            serial_number_or_name=args.front_serial, width=args.width, height=args.height, fps=args.fps
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=int(args.wrist_cam) if args.wrist_cam.isdigit() else Path(args.wrist_cam),
            width=args.width, height=args.height, fps=args.fps
        ),
    }
    return SOFollower(SO101FollowerConfig(port=args.follower_port, id=args.follower_id, cameras=cameras))


def build_teleop(args) -> SOLeader:
    return SOLeader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))


def dataset_features(robot, processors) -> dict:
    teleop_action_processor, _, robot_observation_processor = processors
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )


def open_dataset(args, robot, features) -> LeRobotDataset:
    root = Path(args.root)
    num_cameras = len(robot.cameras)
    common = dict(
        rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec),
        streaming_encoding=not args.no_streaming,
        encoder_threads=2,
        image_writer_threads=4 * num_cameras,
    )
    info = root / "meta" / "info.json"
    if info.exists() and json.loads(info.read_text()).get("total_episodes", 0) == 0:
        # Leftover from a run that died before saving anything: start it fresh.
        shutil.rmtree(root)
    if info.exists():
        dataset = LeRobotDataset.resume(args.repo_id, root=root, **common)
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, features)
        print(f"Appending to {root} ({dataset.num_episodes} episodes already saved)")
    else:
        dataset = LeRobotDataset.create(
            args.repo_id, args.fps, root=root, robot_type=robot.name, features=features, use_videos=True, **common
        )
        print(f"Created new dataset at {root}")
    if args.one_file_per_episode:
        dataset.meta.update_chunk_settings(
            data_files_size_in_mb=ONE_FILE_PER_EPISODE_MB, video_files_size_in_mb=ONE_FILE_PER_EPISODE_MB
        )
        # Episode metadata is buffered (10 episodes by default) before being
        # written; the rollover check needs each one on disk first.
        dataset.meta._metadata_buffer_size = 1
    return dataset


def shutdown_hardware(robot, teleop):
    """Disconnect whatever did come up. robot.is_connected is False as soon as
    one camera is down, so robot.disconnect() alone would leave the others'
    threads running (RealSense then aborts the process on exit)."""
    for dev in (robot.bus, *robot.cameras.values(), teleop.bus):
        if dev.is_connected:
            with contextlib.suppress(Exception):
                dev.disconnect()


def run_phase(keys: KeyReader, valid: set[str], dataset=None, duration=math.inf, **loop_kwargs) -> str | None:
    """Teleoperate (recording into `dataset` if given) until a valid key or `duration`."""
    keys.arm(valid)
    record_loop(dataset=dataset, control_time_s=duration, **loop_kwargs)
    return keys.pressed()


def record(args):
    processors = make_default_processors()
    robot = build_robot(args)
    teleop = build_teleop(args)
    features = dataset_features(robot, processors)

    # Hardware first, so a camera/arm that fails to come up leaves nothing on disk.
    try:
        teleop.connect()  # before the robot, like lerobot-record, so the follower isn't left idle
        robot.connect()
        dataset = open_dataset(args, robot, features)
    except BaseException:
        shutdown_hardware(robot, teleop)
        raise

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    keys = KeyReader(events)
    loop_kwargs = dict(
        robot=robot,
        events=events,
        fps=args.fps,
        teleop_action_processor=processors[0],
        robot_action_processor=processors[1],
        robot_observation_processor=processors[2],
        teleop=teleop,
        single_task=args.task,
        display_data=args.display_data,
    )
    saved_this_session = 0
    try:
        with VideoEncodingManager(dataset):
            if 0 < args.num_episodes <= dataset.num_episodes:
                print(f"Dataset already has {dataset.num_episodes} episodes (target {args.num_episodes}), nothing to record.")
            while args.num_episodes <= 0 or dataset.num_episodes < args.num_episodes:
                ep = dataset.num_episodes
                print(f"\n=== Episode {ep} === reset the scene, then [Enter] to start recording, [q] to quit")
                if run_phase(keys, {ENTER, "q"}, **loop_kwargs) == "q":
                    break

                print(f"● REC episode {ep} (max {args.episode_time:.0f}s) -- [Enter] to end early")
                t0 = time.perf_counter()
                run_phase(keys, {ENTER}, dataset=dataset, duration=args.episode_time, **loop_kwargs)
                elapsed = time.perf_counter() - t0
                n_frames = dataset.writer.episode_buffer["size"] if dataset.has_pending_frames() else 0

                # Frame timestamps are written as frame_index / fps, so a loop
                # that fell behind 30Hz makes the data play back too fast.
                real_fps = n_frames / elapsed if elapsed > 0 else 0.0
                print(f"■ Stopped: {n_frames} frames in {elapsed:.1f}s (effective {real_fps:.1f} Hz)")
                if n_frames and real_fps < 0.9 * args.fps:
                    print(f"  ⚠ loop ran well below {args.fps} Hz -- timing in this episode is off, consider [r]")

                print("  [s] save   [r] discard & re-record   [q] discard & quit")
                choice = run_phase(keys, {"s", "r", "q"}, **loop_kwargs)
                if choice == "s" and n_frames == 0:
                    print("  Episode is empty, nothing to save.")
                    choice = "r"
                if choice == "s":
                    dataset.save_episode()
                    saved_this_session += 1
                    print(f"  ✓ saved episode {ep}  (session: {saved_this_session}, total: {dataset.num_episodes})")
                else:
                    dataset.clear_episode_buffer()
                    print(f"  ✗ discarded episode {ep}")
                    if choice == "q":
                        break
    except KeyboardInterrupt:
        print("\nCtrl-C: dropping the unsaved episode, keeping everything already saved.")
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
    finally:
        keys.close()
        shutdown_hardware(robot, teleop)
        dataset.finalize()  # idempotent; VideoEncodingManager may already have done it

    print(f"\nSaved {saved_this_session} episode(s) this session.")
    return dataset.num_episodes


def verify(root: Path, repo_id: str) -> bool:
    """Reload the dataset from disk the way lerobot-train does and check it end to end."""
    print(f"\n=== Verifying {root} ===")
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print(f"  ✗ {msg}")

    ds = LeRobotDataset(repo_id, root=root)
    meta = ds.meta
    print(f"  {meta.total_episodes} episodes, {meta.total_frames} frames, fps={meta.fps}, robot_type={meta.robot_type}")
    if meta.total_episodes == 0:
        fail("dataset has no episodes")
        return False

    if REFERENCE_DATASET.exists():
        ref = LeRobotDataset(REFERENCE_DATASET.name, root=REFERENCE_DATASET).meta
        for key in ("robot_type", "fps"):
            if getattr(ref, key) != getattr(meta, key):
                fail(f"{key}={getattr(meta, key)} but reference has {getattr(ref, key)}")
        if set(ref.features) != set(meta.features):
            fail(f"feature keys differ from reference: {set(ref.features) ^ set(meta.features)}")
        for key, spec in ref.features.items():
            mine = meta.features.get(key)
            if mine and (tuple(mine["shape"]) != tuple(spec["shape"]) or mine.get("names") != spec.get("names")
                         or mine["dtype"] != spec["dtype"]):
                fail(f"{key}: {mine['dtype']}{mine['shape']} {mine.get('names')} != reference "
                     f"{spec['dtype']}{spec['shape']} {spec.get('names')}")
        if ok:
            print(f"  ✓ features/fps/robot_type match {REFERENCE_DATASET.name}")

    table = ds.hf_dataset.with_format(None)
    ep_col = np.array(table["episode_index"])
    fi_col = np.array(table["frame_index"])
    ts_col = np.array(table["timestamp"], dtype=np.float64)
    idx_col = np.array(table["index"])
    if not np.array_equal(idx_col, np.arange(len(idx_col))):
        fail("global 'index' column is not 0..N-1")
    for ep in range(meta.total_episodes):
        m = ep_col == ep
        n = int(m.sum())
        if n == 0:
            fail(f"episode {ep} has no frames")
            continue
        if not np.array_equal(fi_col[m], np.arange(n)):
            fail(f"episode {ep}: frame_index not contiguous")
        if not np.allclose(ts_col[m], np.arange(n) / meta.fps, atol=1e-4):
            fail(f"episode {ep}: timestamps not frame_index/fps")

    # Decode the first and last frame of every episode from the mp4s -- this
    # is what the training dataloader does, and it catches video/timestamp
    # desyncs that the parquet alone wouldn't.
    starts = [int(np.argmax(ep_col == ep)) for ep in range(meta.total_episodes)]
    ends = [int(len(ep_col) - 1 - np.argmax((ep_col == ep)[::-1])) for ep in range(meta.total_episodes)]
    for i in sorted(set(starts + ends)):
        try:
            item = ds[i]
        except Exception as e:  # noqa: BLE001
            fail(f"frame {i} (episode {int(ep_col[i])}) failed to load: {e}")
            continue
        for key in meta.video_keys:
            img = item[key]
            if tuple(img.shape) != (3, meta.features[key]["shape"][0], meta.features[key]["shape"][1]):
                fail(f"frame {i} {key}: decoded shape {tuple(img.shape)}")
    for key in ("action", "observation.state"):
        vals = np.array(table[key], dtype=np.float32)
        if not np.isfinite(vals).all():
            fail(f"{key} has NaN/inf values")
    if "action" in meta.stats and "observation.state" in meta.stats:
        print("  ✓ stats.json present for action / observation.state")
    else:
        fail("stats.json missing action/observation.state (normalization will fail)")

    print("  ✓ dataset loads and decodes -- ready for training" if ok else "  ✗ dataset has problems, see above")
    return ok


def main():
    init_logging()
    logging.getLogger().setLevel(logging.WARNING)  # keep lerobot's per-frame chatter out of the prompts
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)  # relative paths are from the repo root, like the other scripts
    # --display-data spawns the Rerun viewer from the venv's bin/, not on PATH unless the venv is activated.
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    if not args.verify_only:
        for dev in (args.follower_port, args.leader_port):
            if not Path(dev).exists():
                sys.exit(f"ERROR: {dev} does not exist (is the arm plugged in?)")
        if record(args) == 0:
            return
    sys.exit(0 if verify(Path(args.root), args.repo_id) else 1)


if __name__ == "__main__":
    main()
