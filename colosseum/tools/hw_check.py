#!/usr/bin/env python3
"""Hardware check of the SO101 driver, no model and no Router.

Enables torque at the current pose, reads state and both cameras, holds, nudges wrist_roll by
+2 deg and the gripper by +5, comes back, then returns to the start pose and releases torque.
Saves head_image.png/.npy, left_image.png/.npy and state.npy into OUT for the model probes.
PLACE THE ARM IN ITS REST POSE FIRST.

    colosseum-client/.venv/bin/python colosseum/tools/hw_check.py colosseum-client/configs/robot.so101.yaml outputs/colosseum/probe
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from colosseum_client import RobotClientConfig, make_robot

config = RobotClientConfig.from_yaml(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
started = time.monotonic()
robot = make_robot(config)
print(f'opened in {time.monotonic() - started:.1f}s; start pose {robot.start}')
try:
    times = []
    for _ in range(30):
        tick = time.monotonic()
        observation = robot.get_observation()
        times.append(time.monotonic() - tick)
        time.sleep(max(0, 1 / 30 - times[-1]))
    state = np.r_[observation.joints, observation.gripper].astype(np.float32)
    print(f'joints {observation.joints} gripper {observation.gripper} cartesian {observation.cartesian_position.shape}')
    print(f'get_observation: mean {1e3 * np.mean(times):.1f} ms, max {1e3 * np.max(times):.1f} ms')
    np.save(out / 'state.npy', state)
    for role, image in observation.images.items():
        print(f'{role}: {image.shape} {image.dtype}, channel means {image.mean((0, 1)).round(1)}')
        Image.fromarray(image).save(out / f'{role}.png')
        np.save(out / f'{role}.npy', image)

    def run(target, steps, label):
        for _ in range(steps):
            tick = time.monotonic()
            robot.execute(target)
            time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
        now = robot.get_observation()
        print(f'{label}: target {np.asarray(target)} measured {np.r_[now.joints, now.gripper]}')

    base = state.astype(np.float64)
    run(base, 15, 'hold')
    nudge = base.copy()
    nudge[4] += 2
    nudge[5] = min(100, nudge[5] + 5)
    run(nudge, 30, 'nudge')
    run(base, 30, 'back')
finally:
    started = time.monotonic()
    robot.close()
    print(f'closed in {time.monotonic() - started:.1f}s; look at the PNGs in {out}: correct cameras, colours not swapped?')
