#!/usr/bin/env python3
"""Upload the recorded evaluation episodes (LeRobot v3.0 datasets written by
scripts/trial_recording.py) to the Hugging Face Hub, one dataset repo per task + policy.

    envs/.venv/bin/python scripts/upload_eval_datasets.py            # private repos
    envs/.venv/bin/python scripts/upload_eval_datasets.py --public
    envs/.venv/bin/python scripts/upload_eval_datasets.py --dry-run  # only list what would go up

Uses the cached `hf auth login` token and refuses to run if it is not the --owner account.
"""

import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent / "so101_finetuning_eval"
POLICIES = {"pi05": "pi0.5", "groot": "GR00T N1.7", "molmoact2": "MolmoAct2", "g05": "G0.5", "lingbot": "LingBot-VLA-v2"}

CARD = """---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- so101
- evaluation
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# SO-101 evaluation rollouts: {task} / {policy_name}

Real-robot evaluation episodes of a fine-tuned **{policy_name}** policy on the SO-101 arm, recorded in
[LeRobot](https://github.com/huggingface/lerobot) {version} format (same layout and features as the training data).

- **Task prompt:** {prompt}
- **Policy checkpoint:** `{model}`
- **Episodes:** {episodes} ({frames} frames, {fps} fps)
- **Features:** `observation.state` and `action` (6 joint positions, degrees), `observation.images.front`, `observation.images.wrist`
- **Average score:** {avg:.2f} / 4

`action` is the command sent to the arm by the policy, `observation.state` is the measured joint position.
An episode starts at the first action of the policy and ends when the trial was stopped or hit its time limit.

## Scores

Each trial was scored by hand from 0 to 4 (4 = task completed). They are also in `eval_trials.csv`.

| episode_index | trial | score | frames |
|---|---|---|---|
{rows}
"""


def find_datasets():
    for info_path in sorted(ROOT.glob("*/*/dataset/*/meta/info.json")):
        ds = info_path.parent.parent
        task, policy = ds.parts[-4], ds.parts[-3]
        yield task, policy, ds


def card(task: str, policy: str, ds: Path) -> str:
    import pandas as pd

    info = json.loads((ds / "meta/info.json").read_text())
    trials = list(csv.DictReader(open(ds / "eval_trials.csv")))
    return CARD.format(
        task=task, policy_name=POLICIES.get(policy, policy), version=info["codebase_version"],
        prompt=pd.read_parquet(ds / "meta/tasks.parquet").index[0], model=ds.name,
        episodes=info["total_episodes"], frames=info["total_frames"], fps=info["fps"],
        avg=sum(float(t["score"]) for t in trials) / len(trials),
        rows="\n".join(f"| {t['episode_index']} | {t['run']} | {t['score']} | {t['frames']} |" for t in trials),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--owner", default="Jiamo0912")
    p.add_argument("--prefix", default="so101-eval")
    p.add_argument("--public", action="store_true", help="create public repos (default: private)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    datasets = list(find_datasets())
    api = HfApi()
    if not args.dry_run:
        user = api.whoami()["name"]
        if user.lower() != args.owner.lower():
            raise SystemExit(f"Logged in as '{user}', not '{args.owner}'. Run: envs/.venv/bin/hf auth login")

    links = []
    for task, policy, ds in datasets:
        repo_id = f"{args.owner}/{args.prefix}-{task.replace('_', '-')}-{policy}"
        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"{task:26s} {policy:10s} -> {url}", flush=True)
        links.append((task, policy, url))
        if args.dry_run:
            continue
        api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=ds,
                          commit_message=f"Evaluation rollouts: {ds.name} on {task}")
        api.upload_file(repo_id=repo_id, repo_type="dataset", path_or_fileobj=card(task, policy, ds).encode(),
                        path_in_repo="README.md", commit_message="Add dataset card")
        remote = {f for f in api.list_repo_files(repo_id, repo_type="dataset")}
        local = {str(f.relative_to(ds)) for f in ds.rglob("*") if f.is_file()}
        missing = sorted(local - remote)
        print(f"    uploaded {len(local) - len(missing)}/{len(local)} files" + (f" -- MISSING {missing[:3]}" if missing else ""), flush=True)

    print("\nLinks:")
    for task, policy, url in links:
        print(f"{task}\t{policy}\t{url}")


if __name__ == "__main__":
    main()
