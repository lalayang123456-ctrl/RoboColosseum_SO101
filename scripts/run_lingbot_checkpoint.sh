#!/usr/bin/env bash
# One-command LingBot-VLA-v2 run on the real SO-101 arm, same interface as the other
# run_*_checkpoint.sh scripts:
#   1. starts the policy server (scripts/run_lingbot_server.sh) in the background -- the 6B
#      model loads once; its log goes to outputs/lingbot_server_logs/,
#   2. waits until the server listens,
#   3. runs the robot client (scripts/run_lingbot_client.py --multi-run): Enter = start a run,
#      Enter during a run = stop early, q + Enter = quit; each run lasts up to DURATION s and
#      ends with the arm back at the pose it had when the client connected,
#   4. stops the server when the client exits (also on Ctrl-C or an error).
#
# CHECKPOINT is the hf_ckpt dir; NORM must be the norm stats of the checkpoint's own training
# dataset (scripts/compute_lingbot_norm_stats.py --dataset-root ... --out ...).
#
# Usage:
#   CHECKPOINT=checkpoints/lingbot-vla-v2-6b-so101-black-screwdriver-box-to-table-10ep/checkpoints/global_step_14840/hf_ckpt \
#   NORM=configs/lingbot/norm_stats/so101_black_screwdriver_box_to_table.json \
#   TASK="Take the black screwdriver out of the box and place it on the table." \
#   bash scripts/run_lingbot_checkpoint.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT="${CHECKPOINT:-checkpoints/lingbot-vla-v2-6b-so101-stack-white_bowls-100episodes/checkpoints/global_step_2340/hf_ckpt}"
NORM="${NORM:-configs/lingbot/norm_stats/so101_stack_white_bowls.json}"
TASK="${TASK:-Pick up the white plastic bowl on the right and stack it on top of the white plastic bowl on the left.}"
DURATION="${DURATION:-30}"            # seconds per run; 0 = run until you press Enter
PORT="${PORT:-8006}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-600}"  # seconds to wait for the model to load
VENV_PY="envs/.venv/bin/python"

[ -f "$CHECKPOINT/config.json" ] || { echo "ERROR: $CHECKPOINT/config.json not found." >&2; exit 1; }
[ -f "$NORM" ] || { echo "ERROR: norm stats $NORM not found." >&2; exit 1; }
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "ERROR: something is already listening on port $PORT (another LingBot server?)." \
       "Stop it first, or set PORT=... to use another port." >&2
  exit 1
fi

mkdir -p outputs/lingbot_server_logs
LOG="outputs/lingbot_server_logs/$(basename "$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")")_$(date +%Y%m%d_%H%M%S).log"

echo "Running $CHECKPOINT (LingBot-VLA-v2) on SO-101"
echo "Norm stats: $NORM"
echo "Task: \"$TASK\" | duration: ${DURATION}s per run"
echo "Starting policy server on port $PORT (log: $LOG) ..."

CKPT="$CHECKPOINT" NORM="$NORM" PORT="$PORT" bash scripts/run_lingbot_server.sh > "$LOG" 2>&1 &
SERVER_PID=$!  # run_lingbot_server.sh execs the python server, so this is the server itself

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping policy server..."
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

start=$(date +%s)
until ss -ltn "sport = :$PORT" | grep -q LISTEN; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: policy server exited while loading. Last lines of $LOG:" >&2
    tail -n 30 "$LOG" >&2
    exit 1
  fi
  if [ $(( $(date +%s) - start )) -ge "$SERVER_TIMEOUT" ]; then
    echo "ERROR: policy server not listening after ${SERVER_TIMEOUT}s (see $LOG)." >&2
    exit 1
  fi
  sleep 2
done
echo "Policy server ready after $(( $(date +%s) - start ))s."
echo

"$VENV_PY" scripts/run_lingbot_client.py --multi-run --host 127.0.0.1 --port "$PORT" \
  --task "$TASK" --duration "$DURATION"
