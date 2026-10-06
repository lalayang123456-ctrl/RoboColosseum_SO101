# Sourced by the run_*_checkpoint.sh scripts: resolve the wrist camera by its V4L2 NAME, not by a
# fixed /dev/videoN index. The index is not stable across reboots: after one, the wrist camera
# (USB2.0_CAM1) moved from /dev/video4 to /dev/video0 and /dev/video4 became a RealSense
# sub-device (its IR stream), so every rollout silently fed the policy a grayscale IR top-down view
# as the "wrist" image instead of the arm-mounted camera.
#
# Override with WRIST_CAM=/dev/videoN (or an index) if needed.

WRIST_CAM_NAME="${WRIST_CAM_NAME:-USB2.0_CAM1}"
# Same stable path scripts/view_cameras.py uses.
WRIST_CAM_BY_ID="/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
if [ -z "${WRIST_CAM:-}" ] && [ -e "$WRIST_CAM_BY_ID" ]; then
  WRIST_CAM="$WRIST_CAM_BY_ID"
fi
if [ -z "${WRIST_CAM:-}" ]; then
  for d in /sys/class/video4linux/video*; do
    # index 0 = the capture node; the same device also exposes a metadata node (index 1).
    if grep -q "$WRIST_CAM_NAME" "$d/name" 2>/dev/null && [ "$(cat "$d/index" 2>/dev/null)" = "0" ]; then
      WRIST_CAM="/dev/$(basename "$d")"
      break
    fi
  done
fi
if [ -z "${WRIST_CAM:-}" ]; then
  echo "ERROR: wrist camera '$WRIST_CAM_NAME' not found under /sys/class/video4linux." \
       "Check its USB cable, or set WRIST_CAM=/dev/videoN." >&2
  exit 1
fi
