#!/usr/bin/env python3
"""Smoke test: verify the venv can see the GPU and parse the molmoact2 checkpoint config.
Run with: envs/.venv/bin/python scripts/check_env.py
"""

import sys
from pathlib import Path

import torch

print(f"torch {torch.__version__} (cuda build {torch.version.cuda})")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU visible to torch — see scripts/fix_nvidia_driver.sh and reboot.")

repo_root = Path(__file__).resolve().parent.parent
ckpt_dir = repo_root / "checkpoints" / "molmoact2-so101-stack-white_bowls-100episodes"

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

cfg = MolmoAct2Config.from_pretrained(str(ckpt_dir))
print(f"Loaded checkpoint config OK: chunk_size={cfg.chunk_size}, base={cfg.checkpoint_path}")
print(
    "Note: first real inference call will auto-download the base VLM "
    f"'{cfg.checkpoint_path}' from Hugging Face (~29GB) into the HF cache."
)

sys.exit(0)
