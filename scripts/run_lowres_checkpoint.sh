#!/usr/bin/env bash
# Run a 240x320-trained checkpoint (ACT / Diffusion Policy) on the real SO-101 arm.
#
# Same as run_checkpoint.sh, but goes through scripts/lowres_rollout.py: cameras
# still capture at 640x480 and each frame is area-downscaled to IMG_WIDTH x
# IMG_HEIGHT (default 320x240) before reaching the policy -- the wrist webcam
# has no native 320x240 mode (it snaps to 424x240).
#
# Defaults below are already set for this machine's confirmed setup: follower
# arm (by-id serial path), front = RealSense D435 (by serial number, via the
# proper pyrealsense2 SDK backend), wrist = the small USB webcam physically
# mounted on the arm. Override any of it via env vars if your setup differs.
#
# Usage:
#   bash scripts/run_lowres_checkpoint.sh                 # ACT (default)
#   CHECKPOINT=checkpoints/diffusion_so101_stack_white_bowls_240x320 bash scripts/run_lowres_checkpoint.sh
#   DURATION=60 bash scripts/run_lowres_checkpoint.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- Fill these in for your setup (or export as env vars) ---
# NOTE: /dev/ttyACM0 vs ttyACM1 is NOT stable on this machine (observed to
# flip across reconnects). Default uses the fixed by-serial-number symlink
# under /dev/serial/by-id/, which always points at the follower arm
# (serial 5B8E114717) regardless of which ttyACMx number the kernel assigns.
ARM_PORT="${ARM_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114717-if00}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"   # reuse the calibration already done via teleoperate.sh

# front = Intel RealSense D435, fixed overview of the whole table. Accessed
# via the proper pyrealsense2 SDK backend (type=intelrealsense), NOT plain
# OpenCV/V4L2 -- the RealSense exposes 6 raw /dev/videoN sub-devices (depth,
# IR x2, color, metadata) and guessing the wrong one, or driving the right
# one through generic V4L2, hits hard-to-diagnose failures (stuck at a bogus
# 360p GREY sub-stream, or V4L2 .set() calls that report failure even when
# the value already matches -- confirmed both while wiring this up).
FRONT_CAM_TYPE="intelrealsense"
FRONT_CAM_SERIAL="${FRONT_CAM_SERIAL:-262522073381}"
FRONT_CAM_WIDTH="${FRONT_CAM_WIDTH:-640}"
FRONT_CAM_HEIGHT="${FRONT_CAM_HEIGHT:-480}"
FRONT_CAM_FPS="${FRONT_CAM_FPS:-30}"

# wrist = the small USB2.0_CAM1 module mounted on the arm itself (confirmed
# physically). Plain OpenCV/V4L2 works fine for this one. Found by name, since
# its /dev/videoN number changes across reboots (see scripts/_wrist_cam.sh).
source scripts/_wrist_cam.sh
WRIST_CAM_WIDTH="${WRIST_CAM_WIDTH:-640}"
WRIST_CAM_HEIGHT="${WRIST_CAM_HEIGHT:-480}"
WRIST_CAM_FPS="${WRIST_CAM_FPS:-30}"

TASK="${TASK:-Pick up the white plastic bowl on the right and stack it on top of the white plastic bowl on the left.}"
DURATION="${DURATION:-30}"   # seconds; 0 = run until Ctrl-C
FPS="${FPS:-30}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"   # true = live camera/action viz via rerun (pip install rerun-sdk first)
# ---------------------------------------------------------------

CHECKPOINT="${CHECKPOINT:-checkpoints/act_so101_stack_white_bowls_240x320}"
IMG_WIDTH="${IMG_WIDTH:-320}"     # policy input size; frames are downscaled to this
IMG_HEIGHT="${IMG_HEIGHT:-240}"
export IMG_WIDTH IMG_HEIGHT
VENV_PY="envs/.venv/bin/python"

if [ ! -e "$ARM_PORT" ]; then
  echo "ERROR: $ARM_PORT does not exist. Run 'bash scripts/find_hardware.sh' first" \
       "and set ARM_PORT to the right device." >&2
  exit 1
fi

echo "Running $CHECKPOINT on SO-101 @ $ARM_PORT (front=RealSense $FRONT_CAM_SERIAL, wrist=cam$WRIST_CAM, policy input ${IMG_WIDTH}x${IMG_HEIGHT})"
echo "Task: \"$TASK\" | duration: ${DURATION}s | fps: $FPS"
echo "Press Ctrl-C to stop early."
echo

"$VENV_PY" scripts/lowres_rollout.py \
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
