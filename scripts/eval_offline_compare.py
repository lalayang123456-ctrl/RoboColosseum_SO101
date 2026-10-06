#!/usr/bin/env python3
"""Offline sanity check: for one episode of the training dataset, feed each
sampled (image, state, task) into the fine-tuned checkpoint and compare its
predicted action against the recorded ground-truth action at that same
timestep. This is a teacher-forced, single-step comparison (each query is
independent -- it does not roll the policy forward using its own outputs),
which is the standard way to check whether a policy reproduces the
demonstrations it was trained on.

Usage:
    envs/.venv/bin/python scripts/eval_offline_compare.py --episode 0 --num-samples 8
"""

import argparse

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

DEFAULT_CHECKPOINT = "checkpoints/molmoact2-so101-stack-white_bowls-100episodes"
DEFAULT_DATASET_ROOT = "datasets/robocolosseum-so101-stack-white_bowls-100episodes"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument(
        "--frame-range",
        type=str,
        default=None,
        help="e.g. '130:150' to densely sample within-episode frame indices [start, end) "
        "instead of evenly spanning the whole episode",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from {args.dataset_root} ...")
    dataset = LeRobotDataset(
        repo_id="robocolosseum-so101-stack-white_bowls-100episodes", root=args.dataset_root
    )

    ep = dataset.meta.episodes[args.episode]
    from_idx, to_idx = ep["dataset_from_index"], ep["dataset_to_index"]
    length = to_idx - from_idx
    if args.frame_range:
        lo, hi = (int(x) for x in args.frame_range.split(":"))
        sample_indices = [from_idx + o for o in range(lo, hi)]
    else:
        sample_offsets = np.linspace(0, length - 1, args.num_samples, dtype=int)
        sample_indices = [from_idx + o for o in sample_offsets]

    task = dataset[from_idx]["task"]
    print(f"Episode {args.episode}: {length} frames, task = {task!r}")
    print(f"Sampling {args.num_samples} frames at dataset indices: {sample_indices}")

    print(f"Loading policy from {args.checkpoint} ...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.pretrained_path = args.checkpoint
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    device = torch.device(policy.config.device)
    results = []

    for idx in sample_indices:
        sample = dataset[idx]
        gt_action = sample["action"].numpy()

        obs = {
            "observation.state": sample["observation.state"].numpy(),
            "observation.images.front": sample["observation.images.front"].permute(1, 2, 0).numpy(),
            "observation.images.wrist": sample["observation.images.wrist"].permute(1, 2, 0).numpy(),
        }
        # images are already float32 [0,1] HWC-after-permute; prepare_observation_for_inference
        # expects uint8 HWC (it does the /255 conversion itself), so convert back for it.
        obs["observation.images.front"] = (obs["observation.images.front"] * 255).astype(np.uint8)
        obs["observation.images.wrist"] = (obs["observation.images.wrist"] * 255).astype(np.uint8)

        policy.reset()
        with torch.inference_mode():
            observation = prepare_observation_for_inference(obs, device, task, dataset.meta.robot_type)
            observation = preprocessor(observation)
            action = policy.select_action(observation)
            action = postprocessor(action)
        pred_action = action.squeeze(0).cpu().numpy()

        results.append(
            {
                "frame_index": int(sample["frame_index"]),
                "gt": gt_action,
                "pred": pred_action,
            }
        )

    # --- report ---
    print()
    header = f"{'frame':>6} | " + " | ".join(f"{j:>13}" for j in JOINT_NAMES)
    print(header)
    print("-" * len(header))
    abs_errors = []
    for r in results:
        diff = np.abs(r["gt"] - r["pred"])
        abs_errors.append(diff)
        print(f"{r['frame_index']:>6} | GT:  " + " | ".join(f"{v:>10.2f}" for v in r["gt"]))
        print(f"{'':>6} | Pred:" + " | ".join(f"{v:>10.2f}" for v in r["pred"]))
        print(f"{'':>6} | |Err|" + " | ".join(f"{v:>10.2f}" for v in diff))
        print("-" * len(header))

    abs_errors = np.stack(abs_errors)
    print("\nMean absolute error per joint (degrees, gripper in 0-100 range):")
    for j, e in zip(JOINT_NAMES, abs_errors.mean(axis=0)):
        print(f"  {j:>15}: {e:.3f}")
    print(f"\nOverall mean absolute error: {abs_errors.mean():.3f}")


if __name__ == "__main__":
    main()
