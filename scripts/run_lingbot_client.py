#!/usr/bin/env python3
"""Robot client for the LingBot-VLA-v2 SO-101 checkpoint.

Reads the SO-101 arm + cameras (lerobot classes, main venv), sends observations
to the policy server started by scripts/run_lingbot_server.sh over websocket,
and executes the returned action chunk.

Usage (server must already be running):
    envs/.venv/bin/python scripts/run_lingbot_client.py --duration 30
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "lingbot-vla-v2"))
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: E402
from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower  # noqa: E402

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
WRIST_CAM = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
TASK = "Pick up the white plastic bowl on the right and stack it on top of the white plastic bowl on the left."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8006)
    p.add_argument("--arm-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114717-if00")
    p.add_argument("--robot-id", default="follower_arm")
    p.add_argument("--front-serial", default="262522073381")
    # By stable path, not a /dev/videoN index: those change across reboots (index 4 became a
    # RealSense IR sub-device, feeding the policy a grayscale top-down view as "wrist").
    p.add_argument("--wrist-cam", "--wrist-index", dest="wrist_cam", default=WRIST_CAM,
                   help="wrist camera device path or OpenCV index")
    p.add_argument("--task", default=TASK)
    p.add_argument("--duration", type=float, default=30.0, help="seconds; 0 = until Ctrl-C")
    p.add_argument("--fps", type=float, default=30.0, help="action execution rate")
    p.add_argument("--n-exec", type=int, default=25, help="steps of each returned chunk to execute")
    p.add_argument("--max-relative-target", type=float, default=10.0, help="safety clamp, deg per tick")
    p.add_argument("--multi-run", action="store_true",
                   help="connect once, then Enter = start/stop a run, q + Enter = quit")
    return p.parse_args()


def main():
    args = parse_args()

    cameras = {
        "front": RealSenseCameraConfig(serial_number_or_name=args.front_serial, width=640, height=480, fps=30),
        "wrist": OpenCVCameraConfig(
            index_or_path=int(args.wrist_cam) if str(args.wrist_cam).isdigit() else Path(args.wrist_cam),
            width=640, height=480, fps=30,
        ),
    }
    robot = SOFollower(
        SO101FollowerConfig(
            port=args.arm_port,
            id=args.robot_id,
            cameras=cameras,
            max_relative_target=args.max_relative_target,
        )
    )

    print(f"Connecting to policy server ws://{args.host}:{args.port} ...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    policy.reset("so101")

    robot.connect()
    print(f'Task: "{args.task}" | duration: {args.duration}s | fps: {args.fps} | n_exec: {args.n_exec}')

    if not args.multi_run:
        print("Press Ctrl-C to stop early.")
        try:
            run_once(robot, policy, args, stop_requested=lambda: False)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            robot.disconnect()
            print("Disconnected, done.")
        return

    multi_run(robot, policy, args)


def run_once(robot, policy, args, stop_requested) -> str:
    """One policy run: infer a chunk, execute n_exec steps of it, repeat until the duration
    limit or stop_requested() (checked every step)."""
    start = time.monotonic()
    period = 1.0 / args.fps
    while True:
        if args.duration > 0 and (time.monotonic() - start) >= args.duration:
            return f"{args.duration:.0f}s limit reached"
        if stop_requested():
            return "stopped with Enter"
        obs = robot.get_observation()
        request = {
            "observation.state": np.array([obs[f"{j}.pos"] for j in JOINT_ORDER], dtype=np.float32),
            "observation.images.front": np.ascontiguousarray(obs["front"]),
            "observation.images.wrist": np.ascontiguousarray(obs["wrist"]),
            "task": args.task,
        }
        t0 = time.monotonic()
        result = policy.infer(request)
        print(f"inference: {time.monotonic() - t0:.2f}s")

        chunk = np.asarray(result["action"], dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[None]
        chunk = chunk[: args.n_exec, : len(JOINT_ORDER)]

        for row in chunk:
            if args.duration > 0 and (time.monotonic() - start) >= args.duration:
                break
            if stop_requested():  # consumes the keypress, so return right here
                return "stopped with Enter"
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINT_ORDER, row)})
            time.sleep(period)


def move_to(robot, target: np.ndarray, duration_s: float = 3.0, hz: int = 50) -> None:
    """Smoothly interpolate the arm to `target` (joint degrees, JOINT_ORDER)."""
    obs = robot.get_observation()
    current = np.array([obs[f"{j}.pos"] for j in JOINT_ORDER], dtype=np.float32)
    steps = max(int(duration_s * hz), 1)
    for i in range(1, steps + 1):
        pos = current + (target - current) * (i / steps)
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINT_ORDER, pos)})
        time.sleep(1.0 / hz)


def multi_run(robot, policy, args) -> None:
    """Connected once, run many times -- same controls as scripts/multi_rollout.py:
    Enter = start a run / stop the current run early, q + Enter (or Ctrl-C) = quit.
    Each run starts from a fresh server-side policy state and ends with the arm back at the
    pose it had when the client connected."""
    import queue
    import threading

    lines: queue.Queue[str] = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            lines.put(line.strip().lower())
        lines.put("q")  # EOF

    threading.Thread(target=read_stdin, daemon=True, name="stdin").start()
    obs = robot.get_observation()
    start_pose = np.array([obs[f"{j}.pos"] for j in JOINT_ORDER], dtype=np.float32)
    limit = f"{args.duration:.0f}s" if args.duration > 0 else "no time limit"
    runs = 0

    quit_requested = False

    def stop_requested() -> bool:
        nonlocal quit_requested
        if lines.empty():
            return False
        quit_requested = lines.get_nowait() in ("q", "quit", "exit")
        return True

    def run_trial() -> str:
        policy.reset("so101")
        reason = run_once(robot, policy, args, stop_requested)
        return "quit" if quit_requested else reason

    def go_home() -> None:
        print(">>> Returning arm to start pose...", flush=True)
        move_to(robot, start_pose)

    try:
        if os.environ.get("RECORD") == "1":  # recorded + scored trials, see scripts/trial_recording.py
            from trial_recording import LatestValue, joints_from_dict, model_name_from_path, run_recording_session, slug

            # For the LeRobot-format episode log. This client reads the arm only once per chunk, so
            # the recorder thread reads the joint positions itself; a lock keeps its bus access from
            # interleaving with the control loop's.
            bus_lock = threading.Lock()
            state_cache, action_cache = LatestValue(), LatestValue()
            read_observation, send = robot.get_observation, robot.send_action

            def get_observation():
                with bus_lock:
                    obs = read_observation()
                state_cache.set(joints_from_dict(obs))
                return obs

            def send_action(action):
                with bus_lock:
                    sent = send(action)
                action_cache.set(joints_from_dict(sent if isinstance(sent, dict) else action))
                return sent

            robot.get_observation, robot.send_action = get_observation, send_action

            def get_state():
                # state_cache is only refreshed by the control loop (once per chunk); otherwise
                # read the arm here on every recorder tick.
                if state_cache.age() > 1.5 / args.fps:
                    with bus_lock:
                        return joints_from_dict(robot.bus.sync_read("Present_Position"))
                return state_cache.get()

            run_recording_session(
                cameras=robot.cameras,
                fps=args.fps,
                model=os.environ.get("MODEL_NAME") or model_name_from_path(os.environ.get("CHECKPOINT", "lingbot")),
                task_name=os.environ.get("TASK_NAME") or slug(args.task),
                trials=int(os.environ.get("TRIALS", "25")),
                out_dir=Path(os.environ.get("RECORD_DIR", str(Path(__file__).resolve().parent.parent / "outputs" / "recordings"))),
                limit=f"{limit} limit" if args.duration > 0 else limit,
                get_line=lines.get,
                run_trial=run_trial,
                go_home=go_home,
                is_shutdown=lambda: False,  # Ctrl-C raises KeyboardInterrupt here instead
                task=args.task,
                get_state=get_state,
                action_cache=action_cache,
            )
            return
        while True:
            print(f"\n{'=' * 70}\nReady for run #{runs + 1} ({limit}). Set up the scene, then:\n"
                  "  Enter = start   |   q + Enter = quit\n"
                  "  (during a run) Enter = stop early   |   Ctrl-C = quit\n" + "=" * 70, flush=True)
            if lines.get() in ("q", "quit", "exit"):
                break
            runs += 1
            policy.reset("so101")
            print(f">>> Run #{runs} started", flush=True)
            reason = run_once(robot, policy, args, stop_requested)
            if quit_requested:
                print(f">>> Run #{runs} finished (quit)", flush=True)
                break
            print(f">>> Run #{runs} finished ({reason})\n>>> Returning arm to start pose...", flush=True)
            move_to(robot, start_pose)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        print(f">>> Quitting{f' after {runs} run(s)' if runs else ''}: returning arm to start pose...", flush=True)
        try:
            move_to(robot, start_pose)
        finally:
            robot.disconnect()
            print("Disconnected, done.", flush=True)


if __name__ == "__main__":
    main()
