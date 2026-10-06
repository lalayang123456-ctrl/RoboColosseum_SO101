#!/usr/bin/env bash
# Offline sanity check for a LingBot-VLA-v2 checkpoint + its norm stats: starts the policy server
# on a spare port, sends frames from the training dataset through the same websocket client the
# robot uses, and compares each returned 25-step chunk with the recorded actions. No arm, no cameras.
# Needs ~14 GB of free GPU memory.
#
# usage: bash scripts/lingbot_offline_probe.sh <hf_ckpt dir> <norm.json> <dataset root> <dataset repo id> [episodes=0,50] [frames per episode=6]
# e.g.   bash scripts/lingbot_offline_probe.sh \
#          checkpoints/lingbot-vla-v2-6b-so101-cube-drawer-10ep/checkpoints/global_step_24170/hf_ckpt \
#          configs/lingbot/norm_stats/so101_cube_drawer.json datasets/so101-cube-drawer Jiamo0912/so101-cube-drawer
# Reference: bowls model 2.36 deg overall, screwdriver model 2.63 deg overall.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CKPT="$1"; NORM="$2"; ROOT="$3"; REPO="$4"; EPISODES="${5:-0,50}"; N="${6:-6}"
PORT="${PORT:-8799}"
LOG="$(mktemp)"

CKPT="$CKPT" NORM="$NORM" PORT="$PORT" bash scripts/run_lingbot_server.sh > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; for _ in $(seq 1 15); do kill -0 $PID 2>/dev/null || break; sleep 1; done; kill -9 $PID 2>/dev/null; rm -f "$LOG"' EXIT
start=$(date +%s)
until ss -ltn "sport = :$PORT" | grep -q LISTEN; do
  kill -0 $PID 2>/dev/null || { echo "server exited while loading:"; tail -n 20 "$LOG"; exit 1; }
  [ $(( $(date +%s) - start )) -gt 600 ] && { echo "server not listening after 600 s"; exit 1; }
  sleep 2
done
echo "server up after $(( $(date +%s) - start ))s | norm: $(basename "$NORM")"

envs/.venv/bin/python - "$PORT" "$ROOT" "$REPO" "$EPISODES" "$N" <<'EOF' 2>&1 | grep -E "chunks x|step |^  +[0-9a-z]"
import sys

import numpy as np

sys.path.insert(0, "vendor/lingbot-vla-v2")
from deploy.websocket_client_policy import WebsocketClientPolicy

from lerobot.datasets.lerobot_dataset import LeRobotDataset

port, root, repo = int(sys.argv[1]), sys.argv[2], sys.argv[3]
episodes, n, n_exec = [int(e) for e in sys.argv[4].split(",")], int(sys.argv[5]), 25
ds = LeRobotDataset(repo, root=root)
policy = WebsocketClientPolicy(host="127.0.0.1", port=port)
errs = []
for ep in episodes:
    lo, hi = ds.meta.episodes["dataset_from_index"][ep], ds.meta.episodes["dataset_to_index"][ep]
    for idx in np.linspace(lo, hi - n_exec - 1, n).astype(int):
        s = ds[int(idx)]
        policy.reset("so101")
        img = lambda k: np.ascontiguousarray((s[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8))  # noqa: E731
        req = {"observation.state": s["observation.state"].numpy().astype(np.float32),
               "observation.images.front": img("observation.images.front"),
               "observation.images.wrist": img("observation.images.wrist"), "task": s["task"]}
        chunk = np.asarray(policy.infer(req)["action"], dtype=np.float32)[:n_exec, :6]
        gt = np.stack([ds.hf_dataset[int(idx) + k]["action"].numpy()[:6] for k in range(len(chunk))])
        errs.append(np.abs(chunk - gt))
e = np.stack(errs)
print(f"{len(e)} chunks x {e.shape[1]} steps | mean |error| deg per joint:")
print("  step      pan   lift  elbow w_flex w_roll   grip")
for k in [0, 8, 16, e.shape[1] - 1]:
    print(f"  {k:4d}   " + " ".join(f"{v:6.2f}" for v in e[:, k].mean(0)))
print("  all    " + " ".join(f"{v:6.2f}" for v in e.mean((0, 1))) + f"   | overall {e.mean():.2f}")
EOF
