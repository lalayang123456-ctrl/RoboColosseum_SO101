#!/usr/bin/env python3
"""Print adapter_config joint_low/joint_high (degrees) implied by a LeRobot SO101 follower calibration.

LeRobot's degree mode centres each joint on the middle of its calibrated range:
degrees = (ticks - mid) * 360 / 4095.

    python3 colosseum/tools/joint_limits.py ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json
"""
import json
import sys

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
calibration = json.load(open(sys.argv[1]))
half = [round((calibration[j]['range_max'] - calibration[j]['range_min']) / 2 * 360 / 4095, 1) for j in JOINTS]
print('  joint_low:', [-value for value in half])
print('  joint_high:', half)
