# RoboColosseum

Workspace for running policy checkpoints and real-robot evaluation.

## Layout

- `checkpoints/` — downloaded or locally-trained policy checkpoints, one subfolder per checkpoint (matches its Hugging Face repo name). Each contains `config.json`, `model.safetensors`, pre/postprocessor files, and `train_config.json`.
- `datasets/` — local copies of episode datasets used for training or eval.
- `configs/` — eval/task configs (robot setup, camera layout, task params).
- `scripts/` — run/eval/download helper scripts.
- `outputs/eval/` — eval run outputs: logs, rollout videos, metrics. One subfolder per run, e.g. `outputs/eval/<checkpoint-name>_<date>/`.
- `envs/` — environment definitions (`requirements.txt` / `environment.yml`) for reproducing the Python setup.

## Checkpoints

| Checkpoint | Source | Task |
|---|---|---|
| `molmoact2-so101-stack-white_bowls-100episodes` | [tsangb34/molmoact2-so101-stack-white_bowls-100episodes](https://huggingface.co/tsangb34/molmoact2-so101-stack-white_bowls-100episodes) | SO-101 arm: stack white bowls (trained on 100 episodes) |

## Environment

GPU: **RTX 5090 Laptop GPU** (Blackwell). This machine has no system `git`, `pip`, or `huggingface-cli`, and `sudo` needs a password, so:

- Checkpoints are fetched with `scripts/download_hf_checkpoint.sh` (raw `wget` against HF `resolve/main` URLs — no `git`/`huggingface-cli` needed).
- The Python env lives in `envs/.venv`, built without any apt packages: `python3 -m venv --without-pip` + bootstrapping `pip` via `get-pip.py`.

**GPU driver: fixed and verified.** `nvidia-driver-580-open` + MOK enrollment done; `nvidia-smi` lists the RTX 5090 and `torch` runs CUDA kernels on it (bf16 matmul tested at sm_120). See `scripts/fix_nvidia_driver.sh` for how, if this is ever needed on another machine — Blackwell GPUs only work with NVIDIA's *open* kernel modules, and enrolling the new module's Machine-Owner Key (MOK) at the blue firmware screen on the next boot is required, not optional.

**Python env** (already set up; `torch`/`torchvision` must come from the `cu128` wheel index for Blackwell support — see comment in `envs/requirements.txt`):

```bash
source envs/.venv/bin/activate
python scripts/check_env.py   # verifies torch, GPU visibility, and checkpoint config load
```

**Note:** this checkpoint's `config.json` references a base VLM at `allenai/MolmoAct2` (`checkpoint_path`). The `lerobot` MolmoAct2 policy code auto-downloads that repo (~29GB) from Hugging Face into the HF cache (`~/.cache/huggingface`) the first time you actually instantiate/run the policy — budget time and disk for that on first run.

## Running on the real SO-101

1. Connect the SO-101 follower arm (USB) plus the wrist camera and a third/front camera.
2. Find the arm's serial port and camera indices:
   ```bash
   bash scripts/find_hardware.sh
   ```
3. Run the checkpoint (edit the defaults at the top of the script, or pass as env vars):
   ```bash
   ARM_PORT=/dev/ttyACM0 FRONT_CAM=0 WRIST_CAM=1 TASK="stack the white bowls" \
     bash scripts/run_checkpoint.sh
   ```
   This calls `lerobot-rollout --strategy.type=base` — autonomous policy rollout, no dataset recording, with a `duration`-second time limit (default 30s; set `DURATION=0` to run until Ctrl-C). Camera dict keys are `front`/`wrist` to match the checkpoint's trained `observation.images.front` / `observation.images.wrist` keys. Set `DISPLAY_DATA=true` for a live Rerun view of the camera feed and actions.
