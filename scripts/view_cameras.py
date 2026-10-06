#!/usr/bin/env python3
"""Live side-by-side viewer for the front and wrist cameras, so you can
physically aim them while watching the feed. Ctrl-C to quit.

front = Intel RealSense D435 (accessed via the pyrealsense2 SDK, same as the
        real inference scripts use -- NOT a raw /dev/videoN index, which for
        this camera can silently pick the wrong sub-stream).
wrist = the small USB webcam mounted on the arm, via plain OpenCV/V4L2, opened
        by its fixed /dev/v4l/by-id path (/dev/videoN numbers get reshuffled
        across reboots/replugs).

Frames are shown in a Rerun viewer window: the cv2 in this venv is the
headless build (opencv-python-headless overrides opencv-python), so
cv2.imshow has no GUI backend here.

Only one process can hold a camera at a time -- close this before recording.
To watch the cameras *while* recording, use record_dataset.py --display-data.

Usage: envs/.venv/bin/python scripts/view_cameras.py
       envs/.venv/bin/python scripts/view_cameras.py --front-serial 262522073381 --wrist /dev/video0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

DEFAULT_WRIST_CAM = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"

parser = argparse.ArgumentParser()
parser.add_argument("--front-serial", default="262522073381")
parser.add_argument("--wrist", default=DEFAULT_WRIST_CAM, help="device path or OpenCV index")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

front_cam = RealSenseCamera(
    RealSenseCameraConfig(
        serial_number_or_name=args.front_serial, width=args.width, height=args.height, fps=30
    )
)
wrist_cam = OpenCVCamera(
    OpenCVCameraConfig(
        index_or_path=int(args.wrist) if args.wrist.isdigit() else Path(args.wrist),
        width=args.width,
        height=args.height,
        fps=30,
    )
)

# The viewer binary lives in the venv's bin/, which isn't on PATH unless the venv is activated.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
rr.init("so101_cameras", spawn=True)
rr.send_blueprint(
    rrb.Horizontal(rrb.Spatial2DView(origin="front", name="front"), rrb.Spatial2DView(origin="wrist", name="wrist"))
)

front_cam.connect()
try:
    wrist_cam.connect()
except BaseException:
    front_cam.disconnect()
    raise

print("Streaming to the Rerun window. Ctrl-C here to quit.")
try:
    while True:
        t0 = time.perf_counter()
        # Both lerobot cameras return RGB, which is what Rerun expects.
        rr.log("front", rr.Image(front_cam.read()).compress(jpeg_quality=85))
        rr.log("wrist", rr.Image(wrist_cam.read()).compress(jpeg_quality=85))
        time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t0)))
except KeyboardInterrupt:
    pass
finally:
    front_cam.disconnect()
    wrist_cam.disconnect()
