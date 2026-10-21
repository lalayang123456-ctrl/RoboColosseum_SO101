#!/usr/bin/env python3
"""Live read-out of the SO101 follower joints. Read-only: it never enables torque or writes
configuration, so the arm can be moved by hand while it runs. Ctrl-C to stop.

Shows degrees in the Client's frame (LeRobot current calibration; gripper 0-100) and in the
earlier SO100 frame the generalist checkpoints were trained in.

    colosseum-client/.venv/bin/python colosseum/tools/read_joints.py /dev/serial/by-id/YOUR_PORT [robot_id] [seconds]

Do not run it while an evaluation or another script is driving the arm.
"""
import sys
import time

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MOTORS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
SIGNS, OFFSETS = np.array([1, -1, 1, 1, 1, 1]), np.array([0, 90, 90, 0, 0, 0])
port = sys.argv[1]
robot = SO101Follower(SO101FollowerConfig(port=port, id=sys.argv[2] if len(sys.argv) > 2 else 'follower_arm', use_degrees=True))
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else float('inf')
robot.bus.connect()
try:
    print('calibration file and motors agree:', robot.bus.is_calibrated)
    print('             ' + ''.join(f'{name:>15s}' for name in MOTORS))
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        values = robot.bus.sync_read('Present_Position')
        current = np.array([values[name] for name in MOTORS])
        legacy = SIGNS * current + OFFSETS
        print('\rcurrent      ' + ''.join(f'{v:15.1f}' for v in current)
              + '   |  legacy ' + ' '.join(f'{v:6.1f}' for v in legacy), end='', flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    print()
    robot.bus.disconnect(False)  # leave the torque state untouched
