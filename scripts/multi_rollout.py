#!/usr/bin/env python3
"""lerobot-rollout that runs the same policy many times without reloading it.

The model is loaded and the robot connected ONCE; then it loops:

    wait for Enter  ->  run the policy for up to DURATION seconds (Enter ends it early)
                    ->  arm smoothly returns to its start pose  ->  wait for Enter ...

Type `q` + Enter at the prompt (or Ctrl-C at any time) to quit; the arm returns to
its start pose and disconnects as usual. Policy state (action queue / RTC chunk queue /
interpolator) is reset before every run, same as lerobot's episodic strategy.

Takes exactly the same CLI args as `lerobot-rollout --strategy.type=base ...`
(`--duration` is the per-run limit; 0 = run until Enter). If stdin is not a
terminal there is nobody to press Enter, so it falls back to a single run.
"""

import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import lerobot.scripts.lerobot_rollout as lerobot_rollout
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

_lines: "queue.Queue[str]" = queue.Queue()


def _stdin_reader():
    for line in sys.stdin:
        _lines.put(line.strip().lower())
    _lines.put("q")  # EOF


def _banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def _model_name() -> str:
    """<checkpoint repo dir>[_<step>] from --policy.path, e.g. pi05-so101-cube-drawer-40k_020000."""
    from trial_recording import model_name_from_path

    return model_name_from_path(_policy_path() or "model")


class MultiRunStrategy(BaseStrategy):
    def run(self, ctx) -> None:
        if not sys.stdin.isatty():
            logger.info("stdin is not a terminal -> single run")
            return super().run(ctx)
        threading.Thread(target=_stdin_reader, daemon=True, name="stdin").start()
        if os.environ.get("RECORD") == "1":
            return self._record_session(ctx)

        cfg = ctx.runtime.cfg
        shutdown = ctx.runtime.shutdown_event
        limit = f"{cfg.duration:.0f}s" if cfg.duration > 0 else "no time limit"
        run_idx = 0
        while not shutdown.is_set():
            _banner(
                f"Ready for run #{run_idx + 1} ({limit}). Set up the scene, then:\n"
                "  Enter      = start      |  q + Enter = quit\n"
                "  (during a run) Enter = stop this run early   |  Ctrl-C = quit"
            )
            if not self._wait_for_line(shutdown) or shutdown.is_set():
                break
            run_idx += 1
            print(f">>> Run #{run_idx} started", flush=True)
            reason = self._run_once(ctx)
            print(f">>> Run #{run_idx} finished ({reason})", flush=True)
            if shutdown.is_set():
                break
            if cfg.return_to_initial_position and ctx.hardware.initial_position:
                print(">>> Returning arm to start pose...", flush=True)
                self._return_to_initial_position(ctx.hardware)
        print(f">>> Quitting after {run_idx} run(s)", flush=True)

    def _record_session(self, ctx) -> None:
        """RECORD=1: TRIALS (default 25) recorded + scored runs -- see scripts/trial_recording.py."""
        from trial_recording import LatestValue, joints_from_dict, run_recording_session, slug

        cfg = ctx.runtime.cfg
        shutdown = ctx.runtime.shutdown_event

        # For the LeRobot-format episode log: remember the joint state the control loop reads and
        # every command it sends (send_action returns what was actually sent, after any clipping).
        robot = ctx.hardware.robot_wrapper
        state_cache, action_cache = LatestValue(), LatestValue()
        read_observation, send = robot.get_observation, robot.send_action

        def get_observation():
            obs = read_observation()
            try:
                state_cache.set(joints_from_dict(obs))
            except (KeyError, TypeError):
                pass
            return obs

        def send_action(action):
            sent = send(action)
            try:
                action_cache.set(joints_from_dict(sent if isinstance(sent, dict) else action))
            except (KeyError, TypeError):
                pass
            return sent

        robot.get_observation, robot.send_action = get_observation, send_action

        def get_state():
            # The control loop refreshes the cache every tick. While it is blocked in a synchronous
            # inference call, read the arm from here on every recorder tick instead
            # (ThreadSafeRobot serialises bus access), so the logged state stays at the full rate.
            if state_cache.age() > 1.5 / cfg.fps:
                try:
                    return joints_from_dict(read_observation())
                except (KeyError, TypeError):
                    return None
            return state_cache.get()

        def get_line() -> str | None:
            while not shutdown.is_set():
                try:
                    return _lines.get(timeout=0.2)
                except queue.Empty:
                    continue
            return None

        def go_home() -> None:
            if cfg.return_to_initial_position and ctx.hardware.initial_position:
                print(">>> Returning arm to start pose...", flush=True)
                self._return_to_initial_position(ctx.hardware)

        run_recording_session(
            cameras=ctx.hardware.robot_wrapper.inner.cameras,
            fps=cfg.fps,
            model=os.environ.get("MODEL_NAME") or _model_name(),
            task_name=os.environ.get("TASK_NAME") or slug(cfg.task or "task"),
            trials=int(os.environ.get("TRIALS", "25")),
            out_dir=Path(os.environ.get("RECORD_DIR", "outputs/recordings")),
            limit=f"{cfg.duration:.0f}s limit" if cfg.duration > 0 else "no time limit",
            get_line=get_line,
            run_trial=lambda: self._run_once(ctx),
            go_home=go_home,
            is_shutdown=shutdown.is_set,
            task=cfg.task or "",
            get_state=get_state,
            action_cache=action_cache,
        )

    @staticmethod
    def _wait_for_line(shutdown) -> bool:
        """Block until a line is entered; False means quit."""
        while not shutdown.is_set():
            try:
                line = _lines.get(timeout=0.2)
            except queue.Empty:
                continue
            return line not in ("q", "quit", "exit")
        return False

    def _run_once(self, ctx) -> str:
        """One policy run; same control loop as BaseStrategy.run, plus Enter-to-stop."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(cfg.fps)

        # Drop leftover state from the previous run (queued action chunks etc.).
        engine.reset()
        interpolator.reset()
        self._cached_obs_processed = None
        while not _lines.empty():
            _lines.get_nowait()

        start_time = time.perf_counter()
        engine.resume()
        try:
            while not ctx.runtime.shutdown_event.is_set():
                loop_start = time.perf_counter()
                if cfg.duration > 0 and (loop_start - start_time) >= cfg.duration:
                    return f"{cfg.duration:.0f}s limit reached"
                if not _lines.empty():
                    _lines.get_nowait()
                    return "stopped with Enter"

                obs = robot.get_observation()
                obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue
                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                dt = time.perf_counter() - loop_start
                if (sleep_t := control_interval - dt) > 0:
                    precise_sleep(sleep_t)
            return "Ctrl-C"
        finally:
            engine.pause()


def _fix_molmoact2_action_horizon() -> None:
    """lerobot 0.6.1 bug: MolmoAct2Policy raises max_action_horizon to chunk_size on the HF model
    and backbone configs, but not on the action expert's own config, so a checkpoint trained with
    chunk_size > the base model's limit (e.g. 32 vs allenai/MolmoAct2-SO100_101's 30) crashes with
    "Action sequence length 32 exceeds configured max_action_horizon=30". The action expert uses
    RoPE (no horizon-sized weights), so raising the limit everywhere is safe."""
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    orig = MolmoAct2Policy._override_loaded_max_action_horizon

    def override(self, action_horizon: int) -> None:
        orig(self, action_horizon)
        hf_model = self._hf_model()
        for module in hf_model.modules():
            cfg = getattr(module, "config", None)
            if cfg is not None and hasattr(cfg, "max_action_horizon"):
                cfg.max_action_horizon = int(action_horizon)
        expert_cfg = getattr(getattr(hf_model, "config", None), "action_expert_config", None)
        if expert_cfg is not None:
            expert_cfg.max_action_horizon = int(action_horizon)

    MolmoAct2Policy._override_loaded_max_action_horizon = override


def _fix_molmoact2_action_frame_transform() -> None:
    """lerobot 0.6.1 bug: fine-tuned MolmoAct2 checkpoints get saved with joint_signs/joint_offsets
    set (a frame conversion meant only for ZERO-SHOT MolmoAct2-SO100_101 on new-calibration arms).
    In training that only transforms the state input; action targets are normalized in the arm's own
    frame. So the model already outputs arm-frame actions, and the inference-only
    molmoact2_action_frame_transform converts them a second time: shoulder_lift comes out as
    ~90 - x and elbow_flex as ~x - 90 (offline check vs. dataset: ~180 / ~88 deg mean error on
    those joints, ~2 deg on the others). Skip that step; keep the state transform (it matches
    training). Set MOLMOACT2_ACTION_FRAME_TRANSFORM=1 to restore it for true zero-shot use."""
    import os

    if os.environ.get("MOLMOACT2_ACTION_FRAME_TRANSFORM") == "1":
        return
    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2ActionFrameTransformStep

    MolmoAct2ActionFrameTransformStep.__call__ = lambda self, transition: transition


def _policy_path() -> str | None:
    """--policy.path from the command line (or MOLMOACT2_POLICY_PATH for offline scripts)."""
    import os

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a.startswith("--policy.path="):
            return a.split("=", 1)[1]
        if a == "--policy.path" and i + 1 < len(args):
            return args[i + 1]
    return os.environ.get("MOLMOACT2_POLICY_PATH")


def _center_crop_box(h: int, w: int, scale: float, ratio: float) -> tuple[int, int, int, int]:
    """Deterministic (center) version of RandomResizedCrop.get_params: a crop of area scale*h*w
    and aspect `ratio`, or torchvision's fallback when that does not fit. Returns (top, left, h, w)."""
    import math

    cw, ch = round(math.sqrt(h * w * scale * ratio)), round(math.sqrt(h * w * scale / ratio))
    if not (0 < cw <= w and 0 < ch <= h):
        if w / h > ratio:
            ch, cw = h, round(h * ratio)
        elif w / h < ratio:
            cw, ch = w, round(w / ratio)
        else:
            ch, cw = h, w
    return (h - ch) // 2, (w - cw) // 2, ch, cw


def _molmoact2_training_crop() -> None:
    """Match a MolmoAct2 checkpoint's training-time image geometry at inference.

    MolmoAct2 checkpoints here train with image_transforms incl. a RandomResizedCrop applied to
    every sample, which fixes what the model has seen. Read that crop from the checkpoint's own
    train_config.json and apply its deterministic center version + resize (the random rotation /
    color jitter are pure augmentation and are skipped):
      - bowls MolmoAct2 (size 256x256, scale 0.9025, ratio 1.0): a square crop of that area does
        not fit 640x480, so torchvision always fell back to the center 480x480 -> 256x256, i.e.
        the model only ever saw that view. Offline, teacher-forced on dataset episodes 0 and 50
        (24 frames): overall error 1.96 -> 1.55 deg vs. feeding the full frame.
      - screwdriver MolmoAct2 (size 480x640, scale 0.9025, ratio 4:3): random 95%-side crops
        resized back to 640x480 -> center 608x456 crop -> 640x480 (training object scale).
    No-op if the checkpoint has no crop. Set MOLMOACT2_TRAIN_CROP=0 to always feed full frames."""
    import json
    import os
    from pathlib import Path

    import torch
    from torchvision.transforms.v2 import functional as F

    if os.environ.get("MOLMOACT2_TRAIN_CROP") == "0":
        return
    path = _policy_path()
    cfg_file = Path(path) / "train_config.json" if path else None
    if cfg_file is None or not cfg_file.is_file():
        return
    tfs = json.loads(cfg_file.read_text()).get("dataset", {}).get("image_transforms", {})
    crop = tfs.get("tfs", {}).get("crop", {})
    if not tfs.get("enable") or crop.get("type") != "RandomResizedCrop":
        return
    kw = crop.get("kwargs", {})
    size = [int(s) for s in kw["size"]] if isinstance(kw["size"], list) else [int(kw["size"])] * 2
    scale = sum(kw.get("scale", [1.0, 1.0])) / 2
    ratio = sum(kw.get("ratio", [4 / 3, 4 / 3])) / 2

    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep

    orig = MolmoAct2PackInputsProcessorStep._extract_images

    def extract_images(self, observation, batch_size):
        observation = dict(observation)
        for key in self._resolve_image_keys(observation):
            img = observation[key]
            if torch.is_tensor(img) and img.ndim >= 3:
                top, left, ch, cw = _center_crop_box(*img.shape[-2:], scale, ratio)
                observation[key] = F.resize(F.crop(img, top, left, ch, cw), size, antialias=True)
        return orig(self, observation, batch_size)

    MolmoAct2PackInputsProcessorStep._extract_images = extract_images
    logger.info("MolmoAct2 train-crop: center (%.4f area, ratio %.3f) -> %s", scale, ratio, size)


def _retry_arm_configure(attempts: int = 3, wait_s: float = 0.5) -> None:
    """Retry SOFollower.configure() on a dropped bus packet.

    On connect, lerobot writes ~10 config registers per motor with a single try each, so one
    lost status packet ("Failed to write 'Lock' on id_=3 ... There is no status packet!")
    aborts the whole run. It is intermittent (afterwards 3500 reads in a row succeed), and
    configure() only (re)writes settings, so retrying it is safe."""
    from lerobot.robots.so_follower.so_follower import SOFollower

    orig = SOFollower.configure

    def configure(self) -> None:
        for attempt in range(1, attempts + 1):
            try:
                return orig(self)
            except ConnectionError as e:
                if attempt == attempts:
                    raise
                logger.warning("Arm configure failed (attempt %d/%d): %s -- retrying", attempt, attempts, e)
                time.sleep(wait_s)

    SOFollower.configure = configure


def main() -> None:
    """Run lerobot_rollout; on a crash, exit hard instead of hanging.

    After an unhandled exception the process otherwise hangs ("FATAL: exception not rethrown")
    and keeps the arm's serial port, the cameras and GPU memory until it is killed."""
    import os
    import traceback

    try:
        lerobot_rollout.main()
    except BaseException as e:
        if isinstance(e, SystemExit) and not e.code:
            raise
        traceback.print_exc()
        print("\n>>> Crashed -- exiting so the arm port, cameras and GPU are released.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def install() -> None:
    """Make lerobot_rollout use MultiRunStrategy for --strategy.type=base."""
    create = lerobot_rollout.create_strategy
    lerobot_rollout.create_strategy = lambda c: MultiRunStrategy(c) if c.type == "base" else create(c)
    _fix_molmoact2_action_horizon()
    _fix_molmoact2_action_frame_transform()
    _molmoact2_training_crop()
    _retry_arm_configure()


install()

if __name__ == "__main__":
    main()
