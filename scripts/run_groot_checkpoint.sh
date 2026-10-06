#!/usr/bin/env bash
# Run the groot-n17-so101-stack-white_bowls checkpoint on the real SO-101 arm.
# Same lerobot-native format as run_checkpoint.sh (config.json declares
# type=groot, same observation.state/images.front/images.wrist/action schema)
# -- this is just that script pointed at a different checkpoint, PLUS one
# real difference: this checkpoint's config.json has use_relative_actions=true
# (it predicts action deltas, not absolute joint positions). GrootPolicy's
# select_action() explicitly refuses to run under the default sync inference
# engine for relative-action policies (cached chunk actions could get decoded
# against a newer, stale observation state) -- confirmed via a direct load
# test. The fix is --inference.type=rtc below, which the lerobot-rollout
# context builder ok's for relative-action policies (RTC postprocesses the
# whole chunk instead of decoding step-by-step).
#
# Also needs an HF_TOKEN with access granted to the gated base model
# nvidia/Cosmos-Reason2-2B (visit that model's HF page to request access,
# then `export HF_TOKEN=hf_...` before running this script).
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/run_groot_checkpoint.sh
#   HF_TOKEN=hf_... TASK="pick up the cube" DURATION=60 bash scripts/run_groot_checkpoint.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARM_PORT="${ARM_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114717-if00}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

FRONT_CAM_TYPE="intelrealsense"
FRONT_CAM_SERIAL="${FRONT_CAM_SERIAL:-262522073381}"
FRONT_CAM_WIDTH="${FRONT_CAM_WIDTH:-640}"
FRONT_CAM_HEIGHT="${FRONT_CAM_HEIGHT:-480}"
FRONT_CAM_FPS="${FRONT_CAM_FPS:-30}"

source scripts/_wrist_cam.sh  # wrist camera found by name (its /dev/videoN changes across reboots)
WRIST_CAM_WIDTH="${WRIST_CAM_WIDTH:-640}"
WRIST_CAM_HEIGHT="${WRIST_CAM_HEIGHT:-480}"
WRIST_CAM_FPS="${WRIST_CAM_FPS:-30}"

TASK="${TASK:-Pick up the white plastic bowl on the right and stack it on top of the white plastic bowl on the left.}"
DURATION="${DURATION:-30}"
FPS="${FPS:-30}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"

CHECKPOINT="${CHECKPOINT:-checkpoints/groot-n17-so101-stack-white_bowls-100episodes}"
VENV_PY="envs/.venv/bin/python"

# How many actions of each predicted chunk to execute before switching to the next one.
# RTC's default queue_threshold (30) re-plans as soon as <=30 actions are left, i.e.
# immediately for a 32- (or 16-) step chunk: each chunk ran only for the inference delay
# (~3-9 steps) before being replaced. Relative-action GR00T also runs with RTC's smoothing
# prefix disabled, so every chunk is an independent sample -> the policy switched plans every
# ~0.1-0.3 s (jerky, never finishing a motion). Default: execute half of each chunk.
CHUNK_SIZE="$("$VENV_PY" -c "import json,sys; print(json.load(open(sys.argv[1] + '/config.json'))['chunk_size'])" "$CHECKPOINT")"
EXEC_STEPS="${EXEC_STEPS:-$((CHUNK_SIZE / 2))}"
QUEUE_THRESHOLD=$((CHUNK_SIZE - EXEC_STEPS))

if [ ! -e "$ARM_PORT" ]; then
  echo "ERROR: $ARM_PORT does not exist." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. This checkpoint needs a token with access to" \
       "the gated nvidia/Cosmos-Reason2-2B base model." >&2
  exit 1
fi

echo "Running $CHECKPOINT on SO-101 @ $ARM_PORT (front=RealSense $FRONT_CAM_SERIAL, wrist=cam$WRIST_CAM)"
echo "Task: \"$TASK\" | duration: ${DURATION}s | fps: $FPS | chunk $CHUNK_SIZE, execute $EXEC_STEPS per chunk (queue_threshold=$QUEUE_THRESHOLD)"
echo "Model loads once, then runs repeatedly: Enter = start/stop a run, q + Enter or Ctrl-C = quit."
echo

"$VENV_PY" scripts/multi_rollout.py \
  --strategy.type=base \
  --inference.type=rtc \
  --inference.queue_threshold="$QUEUE_THRESHOLD" \
  --policy.path="$CHECKPOINT" \
  --robot.type=so101_follower \
  --robot.port="$ARM_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="{ front: {type: $FRONT_CAM_TYPE, serial_number_or_name: $FRONT_CAM_SERIAL, width: $FRONT_CAM_WIDTH, height: $FRONT_CAM_HEIGHT, fps: $FRONT_CAM_FPS}, wrist: {type: opencv, index_or_path: $WRIST_CAM, width: $WRIST_CAM_WIDTH, height: $WRIST_CAM_HEIGHT, fps: $WRIST_CAM_FPS} }" \
  --task="$TASK" \
  --duration="$DURATION" \
  --fps="$FPS" \
  --display_data="$DISPLAY_DATA"
