#!/usr/bin/env bash
# Serve a G0.5 (GalaxeaVLA) SO-101 checkpoint over WebSocket for
# scripts/run_g05_client.py. Runs in GalaxeaVLA's own Python 3.10 venv
# (vendor/GalaxeaVLA/.venv, built with `uv sync`); the checkpoint's
# .hydra/config.yaml resolves checkpoints/qwen3_5_2b_base_processor,
# checkpoints/action_tokenizer.pt and configs/data/parts_meta/so101.yaml
# relative to vendor/GalaxeaVLA.
#
# Usage:
#   bash scripts/run_g05_server.sh
#   CHECKPOINT=checkpoints/g05-... ACTION_STEPS=16 bash scripts/run_g05_server.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"

CHECKPOINT="${CHECKPOINT:-checkpoints/g05-so101-stack-white_bowls-100episodes}"
PORT="${PORT:-8765}"
# Steps executed per inference (model horizon is 32 steps @ 30 fps).
ACTION_STEPS="${ACTION_STEPS:-32}"

GALAXEA="$REPO/vendor/GalaxeaVLA"
export PYTHONPATH="$GALAXEA/src:$GALAXEA:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
# flash-attn-4 (CuTe) kernels fail to compile on the RTX 5090 (sm_120); use SDPA.
export G05_DISABLE_FLASH_ATTN=1

cd "$GALAXEA"
exec ./.venv/bin/python scripts/serve_policy.py \
  --ckpt_path "$REPO/$CHECKPOINT/checkpoints/model_state_dict.pt" \
  --host 127.0.0.1 --port "$PORT" --device cuda --action_steps "$ACTION_STEPS" \
  eval_embodiment=so100 \
  model.model_weights_to_bf16=true \
  model.use_torch_compile=false \
  model.model_arch.attn_implementation=sdpa
