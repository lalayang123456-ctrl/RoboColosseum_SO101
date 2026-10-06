"""Recorded + scored evaluation trials, shared by scripts/multi_rollout.py (lerobot policies:
pi05 / GR00T / MolmoAct2 / ...) and scripts/run_g05_client.py (G0.5). Enabled with RECORD=1.

Per trial: Enter = run the policy AND record the front + wrist cameras; Enter again = stop.
Then  s + Enter = save  -> type a numeric score + Enter -> both videos are saved as
      <model>-<task>-<run>-<score>-front.mp4 / -wrist.mp4 in RECORD_DIR/<model>/, or, with
      EVAL_DIR set, as EVAL_DIR/front/<model>-<task>-<run>-<score>.mp4 and EVAL_DIR/wrist/...;
      r + Enter = re-record the same trial (discards the videos).
The arm returns to its start pose after every trial; the session ends after TRIALS (default 25)
saved trials. q + Enter quits. Numbering continues after the highest trial already saved for this
model + task, so a restart resumes.

Every saved trial is also written as an episode of a LeRobot v3.0 dataset (same features, fps,
h264 video and one-file-per-episode layout as the training datasets): measured joint positions as
observation.state, the command sent to the arm as action, both camera streams and the task prompt,
sampled at the same fixed rate as the videos. It lives in <output dir>/dataset/<model>/, with
eval_trials.csv mapping episode_index -> trial number and score. RECORD_DATASET=0 turns it off.
"""

import csv
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMAGE_SHAPE = (480, 640, 3)
# Tiny file-size limits = one data/video file per episode, like the training datasets
# (see scripts/record_dataset.py).
ONE_FILE_PER_EPISODE_MB = 1e-9


def joints_from_dict(d: dict) -> np.ndarray:
    """{"shoulder_pan.pos": ..} or {"shoulder_pan": ..} -> float32[6] in dataset joint order."""
    return np.array([d[f"{j}.pos"] if f"{j}.pos" in d else d[j] for j in JOINTS], dtype=np.float32)


class LatestValue:
    """Thread-safe holder for the most recent sample (e.g. the last action sent to the arm)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: np.ndarray | None = None
        self._time = 0.0

    def set(self, value) -> None:
        with self._lock:
            self._value = np.asarray(value, dtype=np.float32).copy()
            self._time = time.perf_counter()

    def get(self) -> np.ndarray | None:
        with self._lock:
            return None if self._value is None else self._value.copy()

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._value is None else time.perf_counter() - self._time

    def clear(self) -> None:
        with self._lock:
            self._value = None


class EpisodeLogger:
    """Writes evaluated trials as episodes of a LeRobot v3.0 dataset, with the same writer settings
    as scripts/record_dataset.py so the result matches the training datasets (features, fps,
    robot_type, h264 video, one file per episode).

    add() is called from the Recorder thread once per tick; frames before the policy has sent its
    first action are skipped (there is no action to store yet), so an episode starts at the first
    command. save()/discard() are called from the main thread after the Recorder has stopped."""

    def __init__(self, root: Path, repo_id: str, fps: int, task: str,
                 get_state: Callable[[], np.ndarray | None], get_action: Callable[[], np.ndarray | None]) -> None:
        import av
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset

        av.logging.set_level(av.logging.ERROR)  # libx264 prints a page of stats per episode otherwise
        self.root, self.task = Path(root), task
        self._get_state, self._get_action = get_state, get_action
        self.frames = 0
        names = [f"{j}.pos" for j in JOINTS]
        features = {
            "action": {"dtype": "float32", "shape": (len(JOINTS),), "names": names},
            "observation.state": {"dtype": "float32", "shape": (len(JOINTS),), "names": names},
            **{f"observation.images.{cam}": {"dtype": "video", "shape": IMAGE_SHAPE, "names": ["height", "width", "channels"]}
               for cam in ("front", "wrist")},
        }
        common = dict(rgb_encoder=RGBEncoderConfig(vcodec="h264"), streaming_encoding=True, encoder_threads=2,
                      image_writer_threads=8)
        info = self.root / "meta" / "info.json"
        if info.exists():
            import json

            if json.loads(info.read_text()).get("total_episodes", 0) == 0:
                shutil.rmtree(self.root)  # leftover from a session that saved nothing
        if info.exists():
            self.dataset = LeRobotDataset.resume(repo_id, root=self.root, **common)
        else:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.dataset = LeRobotDataset.create(repo_id, int(fps), root=self.root, robot_type="so_follower",
                                                 features=features, use_videos=True, **common)
        self.dataset.meta.update_chunk_settings(data_files_size_in_mb=ONE_FILE_PER_EPISODE_MB,
                                                video_files_size_in_mb=ONE_FILE_PER_EPISODE_MB)
        self.dataset.meta._metadata_buffer_size = 1

    def add(self, images: dict[str, np.ndarray]) -> None:
        state, action = self._get_state(), self._get_action()
        if state is None or action is None or any(images.get(c) is None or images[c].shape != IMAGE_SHAPE for c in ("front", "wrist")):
            return
        self.dataset.add_frame({
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "observation.images.front": images["front"],
            "observation.images.wrist": images["wrist"],
            "task": self.task,
        })
        self.frames += 1

    def save(self, run: int, score: str) -> int | None:
        """Write the buffered frames as one episode; returns its episode_index (None if empty)."""
        frames, self.frames = self.frames, 0
        if frames == 0:
            return None
        episode = self.dataset.num_episodes
        self.dataset.save_episode()
        # lerobot keeps the newest episode's data + metadata parquet files open (no footer, so
        # unreadable) until the next save or finalize(). Close them now so a crash or power loss
        # cannot lose the episode just saved; with one file per episode the next save opens new ones.
        try:
            self.dataset.writer.close_writer()
            self.dataset.meta._close_writer()
        except Exception as e:  # noqa: BLE001 -- finalize() at session end closes them anyway
            logger.warning("could not close the episode files early: %s", e)
        trials = self.root / "eval_trials.csv"
        new = not trials.exists()
        with open(trials, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["episode_index", "run", "score", "frames", "saved_at"])
            w.writerow([episode, run, score, frames, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        return episode

    def discard(self) -> None:
        self.frames = 0
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

    def close(self) -> None:
        self.discard()
        self.dataset.finalize()


class Recorder:
    """Records cameras to one .mp4 each from a background thread at a fixed rate. It reads each
    camera's latest frame itself (anything with read_latest() -> RGB HWC uint8) instead of taking
    frames from the control loop, so the video stays real-time even while the policy pauses for
    inference."""

    def __init__(self, cameras: dict, paths: dict[str, Path], fps: float, episode_logger: "EpisodeLogger | None" = None) -> None:
        import cv2

        self._cv2 = cv2
        self._cameras, self._paths, self._fps = cameras, paths, fps
        self._episode_logger = episode_logger
        self._writers: dict = {}
        self._stop = threading.Event()
        self.frames = 0
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Recorder")
        self._thread.start()

    def _loop(self) -> None:
        cv2, period = self._cv2, 1.0 / self._fps
        next_t = time.perf_counter()
        while not self._stop.is_set():
            images = {}
            for name, cam in self._cameras.items():
                try:
                    frame = cam.read_latest()
                except Exception as e:  # noqa: BLE001 -- a late frame must not kill the recording
                    logger.debug("recorder: %s read failed: %s", name, e)
                    continue
                images[name] = frame
                writer = self._writers.get(name)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(self._paths[name]), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
                    self._writers[name] = writer
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if self._episode_logger is not None:
                try:
                    self._episode_logger.add(images)
                except Exception as e:  # noqa: BLE001 -- the dataset must never break the video recording
                    print(f">>> WARNING: dataset logging stopped for this trial: {e}", flush=True)
                    self._episode_logger = None
            self.frames += 1
            next_t += period
            if (sleep_t := next_t - time.perf_counter()) > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        for writer in self._writers.values():
            writer.release()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def model_name_from_path(path: str | Path) -> str:
    """.../<repo>/checkpoints/<step>[/...] -> <repo>_<step> (step like 020000 or step_023050);
    otherwise the model dir name (a trailing pretrained_model / hf_ckpt is skipped)."""
    path = Path(path)
    parts = path.parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == "checkpoints" and re.fullmatch(r"(step_|global_step_)?\d+", parts[i + 1]):
            return f"{parts[i - 1]}_{parts[i + 1]}"
    return path.parent.name if path.name in ("pretrained_model", "hf_ckpt") else path.name


def _banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def run_recording_session(
    *,
    cameras: dict,
    fps: float,
    model: str,
    task_name: str,
    trials: int,
    out_dir: Path,
    limit: str,
    get_line: Callable[[], str | None],
    run_trial: Callable[[], str],
    go_home: Callable[[], None],
    is_shutdown: Callable[[], bool],
    task: str = "",
    get_state: Callable[[], np.ndarray | None] | None = None,
    action_cache: LatestValue | None = None,
) -> None:
    """get_line() blocks for the next input line (None = shutting down); run_trial() runs the
    policy until Enter / time limit and returns why it stopped.

    With get_state (measured joint positions, float32[6]) and action_cache (the caller stores every
    command it sends to the arm in it), each saved trial is also written as a LeRobot v3.0 episode
    in <output dir>/dataset/<model>/ -- see EpisodeLogger. `task` is the full language prompt.

    Output layout:
      EVAL_DIR set (env):  EVAL_DIR/<camera>/<model>-<task>-<run>-<score>.mp4   (one folder per camera)
      otherwise:           <out_dir>/<model>/<model>-<task>-<run>-<score>-<camera>.mp4"""
    eval_dir = os.environ.get("EVAL_DIR")
    camera_dirs = bool(eval_dir)
    out_dir = Path(eval_dir) if camera_dirs else Path(out_dir) / model
    out_dir.mkdir(parents=True, exist_ok=True)
    if camera_dirs:
        for cam in cameras:
            (out_dir / cam).mkdir(exist_ok=True)
    prefix = f"{model}-{task_name}-"
    ref_cam = "front" if "front" in cameras else next(iter(cameras))

    def update_results() -> None:
        """EVAL_DIR = <root>/<task>/<policy>: refresh <root>/results.md + results.csv."""
        if not camera_dirs:
            return
        try:
            from eval_summary import build

            build(out_dir.parent.parent)
        except Exception as e:  # noqa: BLE001 -- the results list must never break recording
            print(f">>> WARNING: could not update results list: {e}", flush=True)

    def final_path(name: str, cam: str) -> Path:
        return out_dir / cam / f"{name}.mp4" if camera_dirs else out_dir / f"{name}-{cam}.mp4"

    def saved_runs() -> set[int]:
        pattern = f"{ref_cam}/{prefix}*.mp4" if camera_dirs else f"{prefix}*-{ref_cam}.mp4"
        runs = set()
        for f in out_dir.glob(pattern):
            head = f.name[len(prefix):].split("-", 1)[0]
            if head.isdigit():
                runs.add(int(head))
        return runs

    def line() -> str | None:
        text = get_line()
        return None if text is None else text.strip().lower()

    run = max(saved_runs(), default=0) + 1
    layout = f"<camera>/{prefix}<run>-<score>.mp4" if camera_dirs else f"{prefix}<run>-<score>-<camera>.mp4"
    print(f">>> Recording to {out_dir}/  as  {layout}", flush=True)
    if run > 1:
        print(f">>> Found trials up to {run - 1} already saved; continuing at trial {run}.", flush=True)
    tmp = {name: out_dir / f".recording-{name}.mp4" for name in cameras}

    episode_logger = None
    if get_state is not None and action_cache is not None and os.environ.get("RECORD_DATASET", "1") != "0":
        dataset_root = out_dir / "dataset" / model
        try:
            episode_logger = EpisodeLogger(dataset_root, f"eval/{model}-{task_name}", int(fps), task, get_state, action_cache.get)
            print(f">>> LeRobot v3.0 dataset: {dataset_root}/  ({episode_logger.dataset.num_episodes} episodes already saved)", flush=True)
        except Exception as e:  # noqa: BLE001 -- keep recording videos even if the dataset can't be opened
            print(f">>> WARNING: could not open the LeRobot dataset ({e}); recording videos only.", flush=True)
    try:
        _run_trials(run, trials, limit, fps, prefix, tmp, cameras, out_dir, line, run_trial, go_home, is_shutdown,
                    final_path, update_results, episode_logger, action_cache)
    finally:
        if episode_logger is not None:
            try:
                episode_logger.close()
            except Exception as e:  # noqa: BLE001
                print(f">>> WARNING: closing the LeRobot dataset failed: {e}", flush=True)
    print(f">>> Recording session over: {len(saved_runs())}/{trials} trials saved in {out_dir}/", flush=True)


def _run_trials(run, trials, limit, fps, prefix, tmp, cameras, out_dir, line, run_trial, go_home, is_shutdown,
                final_path, update_results, episode_logger, action_cache) -> None:
    while run <= trials and not is_shutdown():
        _banner(
            f"Trial {run}/{trials} ({limit}). Set up the scene, then:\n"
            "  Enter = start the policy + recording (front + wrist)   |   q + Enter = quit\n"
            "  (during the run) Enter = stop recording"
        )
        text = line()
        if text is None or text in ("q", "quit", "exit"):
            break
        if action_cache is not None:
            action_cache.clear()  # no stale command from the previous trial
        if episode_logger is not None:
            episode_logger.discard()
        recorder = Recorder(cameras, tmp, fps=fps, episode_logger=episode_logger)
        print(f">>> Trial {run}: running + recording...", flush=True)
        try:
            reason = run_trial()
        finally:
            recorder.stop()
        logged = f", {episode_logger.frames} frames logged with joint states + actions" if episode_logger is not None else ""
        print(f">>> Trial {run}: recording stopped ({reason}), {recorder.frames / fps:.1f}s{logged}", flush=True)
        if is_shutdown() or reason == "quit":
            for p in tmp.values():
                p.unlink(missing_ok=True)
            if episode_logger is not None:
                episode_logger.discard()
            break

        choice = ""
        while choice not in ("s", "r", "q") and not is_shutdown():
            _banner(f"Trial {run}: save or re-record?\n"
                    "  s + Enter = save   |   r + Enter = re-record   |   q + Enter = discard and quit")
            choice = (line() or "q")[:1]
        if choice != "s":
            for p in tmp.values():
                p.unlink(missing_ok=True)
            if episode_logger is not None:
                episode_logger.discard()
            if choice == "q":
                break
            print(f">>> Trial {run}: discarded, re-recording.", flush=True)
            go_home()
            continue

        score = None
        while score is None and not is_shutdown():
            print(f"Score for trial {run} (a number) + Enter: ", end="", flush=True)
            text = line() or ""
            if re.fullmatch(r"-?\d+(\.\d+)?", text):
                score = text
            else:
                print(f"  '{text}' is not a number, try again.", flush=True)
        if score is None:
            if episode_logger is not None:
                episode_logger.discard()
            break
        name = f"{prefix}{run}-{score}"
        for cam, p in tmp.items():
            if p.exists():
                p.rename(final_path(name, cam))
                print(f">>> Saved {final_path(name, cam)}", flush=True)
            else:
                print(f">>> WARNING: no frames were recorded from the {cam} camera.", flush=True)
        if episode_logger is not None:
            try:
                episode = episode_logger.save(run, score)
                print(f">>> Saved LeRobot episode {episode} (trial {run})" if episode is not None
                      else ">>> WARNING: no frames with joint states + actions were logged for this trial.", flush=True)
            except Exception as e:  # noqa: BLE001 -- the videos and the score are already saved
                print(f">>> WARNING: saving the LeRobot episode failed: {e}", flush=True)
        update_results()
        run += 1
        go_home()
