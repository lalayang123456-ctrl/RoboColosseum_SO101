#!/usr/bin/env python3
"""lerobot-rollout, but feeding the policy camera frames downscaled to a lower resolution.

For checkpoints trained on 240x320 frames (e.g. act_/diffusion_so101_stack_white_bowls_240x320).
Their training data was recorded at 640x480 and then area-downscaled, and the wrist webcam has
no 320x240 mode anyway (it snaps to 424x240, a different aspect/FOV). So the cameras keep
capturing at the configured 640x480, and every frame is cv2.INTER_AREA-resized to
IMG_WIDTH x IMG_HEIGHT (default 320x240) before it reaches the policy. The robot's
observation_features report the downscaled shape so the rollout pipeline matches the policy.

Takes exactly the same CLI args as lerobot-rollout (see scripts/run_lowres_checkpoint.sh).
"""

import os

import cv2

import multi_rollout  # noqa: F401  -- load the model once, run it many times (Enter to start/stop)
from lerobot.robots.so_follower.so_follower import SOFollower
from multi_rollout import main  # lerobot_rollout.main + exit hard on crash

IMG_WIDTH = int(os.environ.get("IMG_WIDTH", 320))
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", 240))

_orig_cameras_ft = SOFollower._cameras_ft.fget
_orig_get_observation = SOFollower.get_observation


def _cameras_ft(self):
    return {k: (IMG_HEIGHT, IMG_WIDTH, shape[2]) for k, shape in _orig_cameras_ft(self).items()}


def get_observation(self):
    obs = _orig_get_observation(self)
    for cam_key in self.cameras:
        img = obs.get(cam_key)
        if img is not None and img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            obs[cam_key] = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return obs


SOFollower._cameras_ft = property(_cameras_ft)
SOFollower.get_observation = get_observation

if __name__ == "__main__":
    main()
