#!/usr/bin/env bash
# Start the LingBot-VLA-v2 policy server (websocket) for the SO-101 fine-tuned
# checkpoint. Uses the separate envs/.venv-lingbot (torch 2.8.0 + lingbot pins);
# run scripts/run_lingbot_client.py (main venv) in another terminal to drive the arm.
#
# The robot_config.yaml + norm_stats.json used here are RECONSTRUCTIONS (see
# configs/lingbot/robot_configs/so101.yaml) -- the originals from training were
# never published with the checkpoint.
#
# Usage: bash scripts/run_lingbot_server.sh            (port 8006, bowls checkpoint)
#        PORT=8010 bash scripts/run_lingbot_server.sh
#        CKPT=<.../hf_ckpt> NORM=configs/lingbot/norm_stats/<dataset>.json bash scripts/run_lingbot_server.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/vendor/lingbot-vla-v2"

PORT="${PORT:-8006}"
CKPT="${CKPT:-$ROOT/checkpoints/lingbot-vla-v2-6b-so101-stack-white_bowls-100episodes/checkpoints/global_step_2340/hf_ckpt}"
# Norm stats must come from the checkpoint's own training dataset
# (scripts/compute_lingbot_norm_stats.py --dataset-root ... --out ...).
NORM="${NORM:-$ROOT/configs/lingbot/norm_stats/so101_stack_white_bowls.json}"
# Relative paths are relative to the repo root (this script cd's into vendor/ above).
case "$CKPT" in /*) ;; *) CKPT="$ROOT/$CKPT" ;; esac
case "$NORM" in /*) ;; *) NORM="$ROOT/$NORM" ;; esac
[ -f "$NORM" ] || { echo "ERROR: norm stats $NORM not found" >&2; exit 1; }
[ -f "$CKPT/config.json" ] || { echo "ERROR: checkpoint $CKPT/config.json not found" >&2; exit 1; }

# Qwen3-VL-4B-Instruct base (downloaded into the HF cache)
QWEN_SNAP="$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/* | head -1)"
export QWEN3VL_PATH="$QWEN_SNAP"

# The server resolves configs/robot_configs/<robo_name>.yaml relative to cwd.
mkdir -p configs/robot_configs
cp "$ROOT/configs/lingbot/robot_configs/so101.yaml" configs/robot_configs/so101.yaml

export PYTHONNOUSERSITE=1
export LINGBOT_ATTN_IMPL=sdpa
# Triton compiles a C helper at runtime and needs Python.h; the headers were
# extracted from the libpython3.12-dev .deb into envs/pyheaders (no sudo needed).
export C_INCLUDE_PATH="$ROOT/envs/pyheaders/usr/include:$ROOT/envs/pyheaders/usr/include/python3.12:$ROOT/envs/pyheaders/usr/include/x86_64-linux-gnu/python3.12:${C_INCLUDE_PATH:-}"
echo "checkpoint: $CKPT"
echo "qwen3-vl:   $QWEN3VL_PATH"
echo "norm stats: $NORM"
exec "$ROOT/envs/.venv-lingbot/bin/python" -m deploy.lingbot_vla_v2_policy \
  --model_path "$CKPT" \
  --robot_norm_path "$NORM" \
  --use_length 25 \
  --use_bf16 true \
  --use_compile false \
  --port "$PORT"
