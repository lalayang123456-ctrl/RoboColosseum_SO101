# SO101 calibration

Copies of the LeRobot calibration files for this rig. LeRobot reads them from
`~/.cache/huggingface/lerobot/calibration/`, not from here.

To restore on a new machine (only when the same physical arms move with it):

```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration
cp -r configs/calibration/robots configs/calibration/teleoperators ~/.cache/huggingface/lerobot/calibration/
```

- `robots/so_follower/follower_arm_open.json` - follower arm used by the Colosseum Client
  (`robot_id: follower_arm_open`); recalibrated 2026-10-19 after the arm swap, serial port
  `usb-1a86_USB_Single_Serial_5B79017804`.
- `robots/so_follower/follower_arm.json` - identical copy under the id the `scripts/` tools use
  (`--robot.id=follower_arm`). Keep the two in sync: the driver refuses to start when the
  calibration stored in the motors differs from the file.
- `teleoperators/so_leader/leader_arm.json` - leader arm (`--teleop.id=leader_arm`, 2026-09-22).

These values are specific to these arms; other hardware must be recalibrated.
