#!/usr/bin/env python3
"""Run the OFFICIAL allenai/MolmoAct2-SO100_101 checkpoint on the real SO-101 arm.

Unlike checkpoints/molmoact2-so101-stack-white_bowls-100episodes (a lerobot
policy, run via lerobot-rollout / run_checkpoint.sh), this checkpoint is a raw
HF `transformers` custom-code model (AutoModelForImageTextToText +
model.predict_action(...)) -- it's a language-conditioned generalist policy
for SO-100/101, so any task instruction can be tried, not just what it was
explicitly fine-tuned on. This script wires it up to the real arm + cameras
directly (reusing lerobot's robot/camera classes for hardware I/O only).

Usage:
    envs/.venv/bin/python scripts/run_official_so101_checkpoint.py \
        --task "stack the white bowls" --duration 30
"""

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

DEFAULT_CHECKPOINT = "checkpoints/MolmoAct2-SO100_101"
DEFAULT_ARM_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114717-if00"
DEFAULT_FRONT_SERIAL = "262522073381"
DEFAULT_WRIST_INDEX = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--arm-port", default=DEFAULT_ARM_PORT)
    p.add_argument("--robot-id", default="follower_arm")
    p.add_argument("--front-serial", default=DEFAULT_FRONT_SERIAL)
    p.add_argument("--wrist-index", type=int, default=DEFAULT_WRIST_INDEX)
    p.add_argument(
        "--task",
        default="Pick up the white plastic bowl on the right and stack it on top of the white plastic bowl on the left.",
    )
    p.add_argument("--duration", type=float, default=30.0, help="seconds; 0 = run until Ctrl-C")
    p.add_argument("--fps", type=float, default=10.0, help="action execution rate")
    p.add_argument("--num-steps", type=int, default=10, help="flow-matching solver steps")
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=10.0,
        help="safety clamp: max degrees/step the arm may move per control tick "
        "(this checkpoint is untested on this hardware, so start conservative)",
    )
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--no-cuda-graph", action="store_true")
    return p.parse_args()


def build_robot(args) -> SOFollower:
    cameras = {
        "front": RealSenseCameraConfig(
            serial_number_or_name=args.front_serial, width=640, height=480, fps=30
        ),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_index, width=640, height=480, fps=30),
    }
    cfg = SO101FollowerConfig(
        port=args.arm_port,
        id=args.robot_id,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
    )
    return SOFollower(cfg)


def build_model(args):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(args.checkpoint, trust_remote_code=True, dtype=dtype)
        .to("cuda")
        .eval()
    )
    return processor, model


def state_vector(obs: dict) -> np.ndarray:
    return np.array([obs[f"{j}.pos"] for j in JOINT_ORDER], dtype=np.float32)


def main():
    args = parse_args()

    print(f"Loading model from {args.checkpoint} ({args.dtype})...")
    processor, model = build_model(args)

    print(f"Connecting robot @ {args.arm_port} ...")
    robot = build_robot(args)
    robot.connect()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    autocast_ctx = torch.autocast("cuda", dtype=dtype) if dtype == torch.bfloat16 else torch.no_grad()

    print(f'Task: "{args.task}" | duration: {args.duration}s | fps: {args.fps}')
    print("Press Ctrl-C to stop early.")

    start = time.monotonic()
    step_period = 1.0 / args.fps
    try:
        while args.duration <= 0 or (time.monotonic() - start) < args.duration:
            obs = robot.get_observation()
            state = state_vector(obs)
            images = [obs["front"], obs["wrist"]]

            with torch.inference_mode(), autocast_ctx:
                out = model.predict_action(
                    processor=processor,
                    images=images,
                    task=args.task,
                    state=state,
                    norm_tag="so100_so101_molmoact2",
                    inference_action_mode="continuous",
                    enable_depth_reasoning=False,
                    num_steps=args.num_steps,
                    normalize_language=True,
                    enable_cuda_graph=not args.no_cuda_graph,
                )

            actions = out.actions.detach().float().cpu().numpy()
            if actions.ndim == 3:  # (batch, horizon, action_dim) -> drop batch dim
                actions = actions[0]
            actions = actions[:, : len(JOINT_ORDER)]

            for row in actions:
                if args.duration > 0 and (time.monotonic() - start) >= args.duration:
                    break
                action_dict = {f"{j}.pos": float(v) for j, v in zip(JOINT_ORDER, row)}
                robot.send_action(action_dict)
                time.sleep(step_period)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        robot.disconnect()
        print("Disconnected, done.")


if __name__ == "__main__":
    main()
