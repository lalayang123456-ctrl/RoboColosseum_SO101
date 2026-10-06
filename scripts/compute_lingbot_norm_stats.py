#!/usr/bin/env python3
"""Reconstruct norm_stats.json for the lingbot-vla-v2 SO-101 robot_config.

The original training run's robot_config.yaml + norm_stats.json (LingBot-VLA's
external, checkpoint-independent config files -- see scripts/run_official_so101_checkpoint.py's
counterpart note and the conversation for why these aren't bundled with the HF
checkpoint) aren't published, so this recomputes an equivalent norm_stats.json
directly from the training dataset using the same algorithm as LingBot-VLA's
own lingbotvla.utils.normalize.RunningStats (mean/std/quantiles), matching its
JSON schema exactly -- without needing their full training environment.

Usage: envs/.venv/bin/python scripts/compute_lingbot_norm_stats.py            (bowls, defaults)
       envs/.venv/bin/python scripts/compute_lingbot_norm_stats.py \
           --dataset-root datasets/<dataset> --out configs/lingbot/norm_stats/<name>.json
"""

import glob
import json

import numpy as np
import pandas as pd

DATASET_ROOT = "datasets/robocolosseum-so101-stack-white_bowls-100episodes"
OUT_PATH = "configs/lingbot/norm_stats/so101_stack_white_bowls.json"

# SO-101: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
ARM_SLICE = slice(0, 5)
EFFECTOR_SLICE = slice(5, 6)

# The arm action is trained as a delta from the current state (subtract_state=True),
# over a 50-step chunk (chunk_size in the checkpoint's lingbotvla_cli.yaml), so its
# norm stats are over (action[t+k] - state[t]) for k in [0, CHUNK), merged over k.
CHUNK = 50


def stats_for(arr: np.ndarray) -> dict:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    q01, q02, q98, q99 = np.quantile(arr, [0.01, 0.02, 0.98, 0.99], axis=0)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "q02": q02.tolist(),
        "q98": q98.tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=DATASET_ROOT)
    p.add_argument("--out", default=OUT_PATH)
    args = p.parse_args()
    dataset_root, out_path = args.dataset_root, args.out

    files = sorted(glob.glob(f"{dataset_root}/data/chunk-*/*.parquet"))
    print(f"Reading {len(files)} parquet files...")
    dfs = [pd.read_parquet(f, columns=["observation.state", "action", "episode_index"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total frames: {len(df)}")

    state = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy())
    assert state.shape[1] == 6 and action.shape[1] == 6, (state.shape, action.shape)

    ep = df["episode_index"].to_numpy() if "episode_index" in df else None
    deltas = []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        st, ac = state[idx][:, ARM_SLICE], action[idx][:, ARM_SLICE]
        for k in range(CHUNK):
            n = len(idx) - k
            if n > 0:
                deltas.append(ac[k:k + n] - st[:n])
    arm_delta = np.concatenate(deltas, axis=0)

    norm_stats = {
        "action.arm.position": stats_for(arm_delta),
        "action.effector.position": stats_for(action[:, EFFECTOR_SLICE]),
        "observation.state.arm.position": stats_for(state[:, ARM_SLICE]),
        "observation.state.effector.position": stats_for(state[:, EFFECTOR_SLICE]),
    }

    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"norm_stats": norm_stats}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
