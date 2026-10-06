#!/usr/bin/env bash
# One-command G0.5 (GalaxeaVLA) run on the real SO-101 arm, same interface as the other
# run_*_checkpoint.sh scripts:
#   1. starts the policy server (scripts/run_g05_server.sh) in the background -- the model
#      loads once, ~20 s, ~12 GB GPU; its log goes to outputs/g05_server_logs/,
#   2. waits until the server listens,
#   3. runs the robot client (scripts/run_g05_client.py --multi-run): the arm homes to the
#      checkpoint's start pose, then Enter = start a run, Enter during a run = stop early,
#      q + Enter = quit; each run lasts up to DURATION s and ends with the arm back at the
#      start pose,
#   4. stops the server when the client exits (also on Ctrl-C or an error).
#
# Usage:
#   CHECKPOINT=checkpoints/g05-so101-black-screwdriver-box-to-table-10ep \
#   TASK="Take the black screwdriver out of the box and place it on the table." \
#   bash scripts/run_g05_checkpoint.sh
#   DRY_RUN=1 ...   infer and log targets without moving the arm (homing still moves it)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT="${CHECKPOINT:-checkpoints/g05-so101-stack-white_bowls-100episodes}"
TASK="${TASK:-Pick up the white plastic bowl on the right and stack it on top of the white plastic bowl on the left.}"
DURATION="${DURATION:-30}"            # seconds per run; 0 = run until you press Enter
DRY_RUN="${DRY_RUN:-0}"
PORT="${PORT:-8765}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-300}"  # seconds to wait for the model to load
VENV_PY="envs/.venv/bin/python"
export CHECKPOINT TASK DURATION PORT

if [ ! -f "$CHECKPOINT/checkpoints/model_state_dict.pt" ]; then
  echo "ERROR: $CHECKPOINT/checkpoints/model_state_dict.pt not found." >&2
  exit 1
fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "ERROR: something is already listening on port $PORT (another g05 server?)." \
       "Stop it first, or set PORT=... to use another port." >&2
  exit 1
fi

mkdir -p outputs/g05_server_logs
LOG="outputs/g05_server_logs/$(basename "$CHECKPOINT")_$(date +%Y%m%d_%H%M%S).log"

echo "Running $CHECKPOINT (G0.5) on SO-101"
echo "Task: \"$TASK\" | duration: ${DURATION}s per run$([ "$DRY_RUN" = 1 ] && echo " | DRY RUN")"
echo "Starting policy server on port $PORT (log: $LOG) ..."

bash scripts/run_g05_server.sh > "$LOG" 2>&1 &
SERVER_PID=$!  # run_g05_server.sh execs the python server, so this is the server itself

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

CLIENT_ARGS=(--multi-run --host 127.0.0.1 --port "$PORT")
[ "$DRY_RUN" = 1 ] && CLIENT_ARGS+=(--dry-run)
"$VENV_PY" scripts/run_g05_client.py "${CLIENT_ARGS[@]}"
