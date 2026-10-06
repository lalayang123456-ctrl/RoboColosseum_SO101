# SO101 calibration

Copies of the LeRobot calibration files for this rig (calibrated 2026-09-22).
LeRobot reads them from `~/.cache/huggingface/lerobot/calibration/`, not from here.

To restore on a new machine:

```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration
cp -r configs/calibration/robots configs/calibration/teleoperators ~/.cache/huggingface/lerobot/calibration/
```

- `robots/so_follower/follower_arm.json` - follower arm (`--robot.id=follower_arm`)
- `teleoperators/so_leader/leader_arm.json` - leader arm (`--teleop.id=leader_arm`)

These values are specific to these two arms; other hardware must be recalibrated.
