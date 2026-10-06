#!/usr/bin/env bash
# Fix: /dev/ttyACM0 and /dev/ttyACM1 (the SO-101 arms) are root:dialout mode 660,
# and the current user isn't in the `dialout` group, so opening them fails with
# "Permission denied" (confirmed via pyserial).
#
# This does two things:
#   1. A temporary ACL grant so the ports work RIGHT NOW, this session, without
#      logging out (resets on unplug/replug/reboot).
#   2. Adds the user to `dialout` permanently — takes effect after you log out
#      and back in (or reboot).
#
# Needs sudo — run it yourself.

set -euo pipefail

for dev in /dev/ttyACM0 /dev/ttyACM1; do
  if [ -e "$dev" ]; then
    sudo setfacl -m "u:${USER}:rw" "$dev"
    echo "Granted temporary access: $dev"
  fi
done

sudo usermod -aG dialout "$USER"
echo
echo "Done. Ports are usable right now. For this to survive reboots/replugs"
echo "without needing setfacl again, log out and back in (dialout group)."
