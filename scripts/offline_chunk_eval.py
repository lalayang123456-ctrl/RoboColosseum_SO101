#!/usr/bin/env python3
"""Offline sanity check for a lerobot policy checkpoint (pi05 / GR00T / MolmoAct2 / ...).

Feeds frames from the checkpoint's training dataset through the SAME preprocessor -> policy ->
postprocessor the robot run uses (including the fixes in scripts/multi_rollout.py), predicts a full
action chunk from each frame, and compares step k with the action recorded at frame idx+k.
A correct pipeline gives ~0.5-3 deg mean error; a wrong joint order / units / normalization /
relative-action decode gives tens of degrees or a clear per-joint bias. It is teacher-forced: it
checks that inputs and outputs are handled correctly, not that the task succeeds on the arm.

No arm or cameras needed; uses ~10-15 GB of GPU memory.

Usage:
    envs/.venv/bin/python scripts/offline_chunk_eval.py <checkpoint dir> <dataset root> <dataset repo id> [episodes=0,50] [frames per episode=10]
e.g.
    envs/.venv/bin/python scripts/offline_chunk_eval.py \\
        checkpoints/pi05-so101-stack-cubes-40k/checkpoints/040000/pretrained_model \\
        datasets/so101-stack-cubes Jiamo0912/so101-stack-cubes
References (overall mean error): pi05 0.6-0.8 deg, GR00T 0.4 deg, MolmoAct2 1.0-1.7 deg.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ckpt, root, repo = sys.argv[1], sys.argv[2], sys.argv[3]
episodes = [int(e) for e in (sys.argv[4] if len(sys.argv) > 4 else "0,50").split(",")]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 10

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0], f"--policy.path={ckpt}"]  # multi_rollout reads the checkpoint's train crop from this
import multi_rollout  # noqa: E402, F401

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402

ds = LeRobotDataset(repo, root=root)
cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.pretrained_path = ckpt
policy = make_policy(cfg, ds_meta=ds.meta).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt)
horizon = cfg.chunk_size
errs = []
for ep in episodes:
    lo, hi = ds.meta.episodes["dataset_from_index"][ep], ds.meta.episodes["dataset_to_index"][ep]
    for idx in np.linspace(lo, hi - horizon - 1, n).astype(int):
        s = ds[int(idx)]
        obs = {"observation.state": s["observation.state"].numpy()}
        for cam in ("front", "wrist"):
            obs[f"observation.images.{cam}"] = (s[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        policy.reset()
        with torch.inference_mode():
            o = prepare_observation_for_inference(obs, torch.device("cuda"), s["task"], ds.meta.robot_type)
            chunk = post(policy.predict_action_chunk(pre(o)))  # whole chunk, decoded against this frame's state
        pred = chunk.float().cpu().numpy()[0, :, :6]
        gt = np.stack([ds.hf_dataset[int(idx) + k]["action"].numpy()[:6] for k in range(horizon)])
        errs.append(pred - gt)

e = np.stack(errs)
joints = ["pan", "lift", "elbow", "w_flex", "w_roll", "grip"]
print(f"\n{ckpt}\n{len(e)} chunks x {horizon} steps (task: {s['task']})")
print("step | mean |error| deg per joint                      || signed bias (pred - gt)")
print("     | " + " ".join(f"{j:>6}" for j in joints) + " || " + " ".join(f"{j:>6}" for j in joints))
for k in [0, 8, 16, 24, horizon - 1]:
    print(f"{k:4d} | " + " ".join(f"{v:6.2f}" for v in np.abs(e[:, k]).mean(0)) + " || " + " ".join(f"{v:+6.2f}" for v in e[:, k].mean(0)))
print("all  | " + " ".join(f"{v:6.2f}" for v in np.abs(e).mean((0, 1))) + " || " + " ".join(f"{v:+6.2f}" for v in e.mean((0, 1)))
      + f"   overall {np.abs(e).mean():.2f}")
