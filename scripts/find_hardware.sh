#!/usr/bin/env bash
# One-time hardware discovery: find the SO-101 arm's serial port and the
# camera device indices for the wrist + front cameras.
#
# Usage: bash scripts/find_hardware.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VENV_PY=envs/.venv/bin/python

echo "=== Finding the SO-101 arm's serial port ==="
echo "Follow the prompts: it will ask you to unplug then replug the arm's USB cable."
"$VENV_PY" -m lerobot.scripts.lerobot_find_port

echo
echo "=== Finding camera devices ==="
echo "This opens each detected camera briefly and saves a sample frame under"
echo "outputs/captured_images/ so you can tell which index is the wrist cam"
echo "vs. the front/third cam."
"$VENV_PY" -m lerobot.scripts.lerobot_find_cameras opencv
