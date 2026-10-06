#!/usr/bin/env bash
# Teleoperate one SO-101 (the "follower") by hand-moving the other (the "leader").
#
# You have two identical, fully-motorized SO-101 arms connected — either can act
# as leader or follower, it's just a role assigned by --robot.type/--teleop.type.
# Pick which one you'll hold and move (leader) and which one should copy it
# (follower), then set the ports below. If you guess wrong, the follower just
# won't move — swap FOLLOWER_PORT/LEADER_PORT and rerun.
#
# First time for a given --robot.id/--teleop.id: lerobot will walk you through
# an interactive calibration (move each joint through its full range, press
# Enter) before teleop starts. Clear space around both arms first.
#
# Usage:
#   FOLLOWER_PORT=/dev/ttyACM0 LEADER_PORT=/dev/ttyACM1 bash scripts/teleoperate.sh
#
# NOTE: /dev/ttyACM0 vs ttyACM1 is NOT stable on this machine -- it has been
# observed to flip between the two arms across reconnects/USB re-enumeration.
# The defaults below instead use the fixed by-serial-number symlinks under
# /dev/serial/by-id/, which always point to the same physical arm no matter
# what ttyACMx number the kernel assigns it this time.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BY_ID=/dev/serial/by-id
FOLLOWER_PORT="${FOLLOWER_PORT:-$BY_ID/usb-1a86_USB_Single_Serial_5B8E114717-if00}"
LEADER_PORT="${LEADER_PORT:-$BY_ID/usb-1a86_USB_Single_Serial_5B79016937-if00}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"
LEADER_ID="${LEADER_ID:-leader_arm}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"   # true = live viz via rerun

VENV_PY="envs/.venv/bin/python"

for dev in "$FOLLOWER_PORT" "$LEADER_PORT"; do
  if [ ! -e "$dev" ]; then
    echo "ERROR: $dev does not exist." >&2
    exit 1
  fi
done

echo "Follower: $FOLLOWER_PORT (id=$FOLLOWER_ID) <- copies -- Leader: $LEADER_PORT (id=$LEADER_ID)"
echo "Press Ctrl-C to stop."
echo

"$VENV_PY" -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --display_data="$DISPLAY_DATA"
