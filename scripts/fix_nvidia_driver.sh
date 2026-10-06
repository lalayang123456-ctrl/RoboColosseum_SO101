#!/usr/bin/env bash
# Fix: RTX 5090 Laptop GPU (Blackwell) is not detected by nvidia-smi.
#
# Root cause (confirmed via `journalctl -k | grep nvidia`):
#   "NVRM: The NVIDIA GPU 0000:02:00.0 (PCI ID: 10de:2c58) installed in this
#    system requires use of the NVIDIA open kernel modules."
# Blackwell (RTX 50-series) GPUs only work with the "open" NVIDIA kernel
# modules. This machine currently has the closed-source `nvidia-driver-580`
# installed, which loads but cannot talk to the GPU (GPU Firmware: N/A in
# /proc/driver/nvidia/gpus/*/information).
#
# Fix: swap to the -open driver variant (same version, already available
# in apt) and reboot. This needs sudo, which Claude Code's auto mode will
# not run on your behalf even given a password from an .env file (blocked
# as "credential materialization") - run it yourself:
#
#   bash scripts/fix_nvidia_driver.sh
#
# Then reboot, and verify with: nvidia-smi

set -euo pipefail

sudo apt update
sudo apt install -y nvidia-driver-580-open
sudo apt remove -y nvidia-driver-580 nvidia-dkms-580 2>/dev/null || true
sudo apt autoremove -y

echo
echo "Driver package swapped. Reboot now, then run 'nvidia-smi' to confirm"
echo "the RTX 5090 shows up."
