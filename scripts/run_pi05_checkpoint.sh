#!/usr/bin/env bash
# Run the pi05-so101-stack-white_bowls checkpoint on the real SO-101 arm.
# Same lerobot-native format as run_checkpoint.sh (config.json declares
# type=pi05). This checkpoint's config also declares a 3rd, "empty_camera_0"
# image input (a pi05/pi0-family architecture quirk: the base model expects up
# to 3 camera slots) -- lerobot's pi05 modeling code auto-fills any declared
# camera the robot doesn't provide with a zero-padded image, so we only need
# to wire up the 2 real cameras below, same as every other checkpoint.
#
# Also needs an HF_TOKEN with access granted to the gated base model
# google/paligemma-3b-pt-224 (visit that model's HF page to request access,
# then `export HF_TOKEN=hf_...` before running this script).
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/run_pi05_checkpoint.sh
#   HF_TOKEN=hf_... TASK="pick up the cube" DURATION=60 bash scripts/run_pi05_checkpoint.sh

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

CHECKPOINT="${CHECKPOINT:-checkpoints/pi05-so101-stack-white_bowls-100episodes}"
VENV_PY="envs/.venv/bin/python"

if [ ! -e "$ARM_PORT" ]; then
  echo "ERROR: $ARM_PORT does not exist." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. This checkpoint needs a token with access to" \
       "the gated google/paligemma-3b-pt-224 base model." >&2
  exit 1
fi

echo "Running $CHECKPOINT on SO-101 @ $ARM_PORT (front=RealSense $FRONT_CAM_SERIAL, wrist=cam$WRIST_CAM)"
echo "Task: \"$TASK\" | duration: ${DURATION}s | fps: $FPS"
echo "Model loads once, then runs repeatedly: Enter = start/stop a run, q + Enter or Ctrl-C = quit."
echo

"$VENV_PY" scripts/multi_rollout.py \
  --strategy.type=base \
  --policy.path="$CHECKPOINT" \
  --robot.type=so101_follower \
  --robot.port="$ARM_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="{ front: {type: $FRONT_CAM_TYPE, serial_number_or_name: $FRONT_CAM_SERIAL, width: $FRONT_CAM_WIDTH, height: $FRONT_CAM_HEIGHT, fps: $FRONT_CAM_FPS}, wrist: {type: opencv, index_or_path: $WRIST_CAM, width: $WRIST_CAM_WIDTH, height: $WRIST_CAM_HEIGHT, fps: $WRIST_CAM_FPS} }" \
  --task="$TASK" \
  --duration="$DURATION" \
  --fps="$FPS" \
  --display_data="$DISPLAY_DATA"
