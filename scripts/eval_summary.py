#!/usr/bin/env python3
"""Build the evaluation results list from the recorded videos.

Scans <root>/<task>/<policy>/{front,wrist}/<model>-<task_name>-<run>-<score>.mp4 (written by the
RECORD=1 + EVAL_DIR runs, see scripts/trial_recording.py) and writes:
  <root>/results_cn.md  overview (trials, mean/min/max score per model) + per-trial table with video links
  <root>/results.md     the same in English
  <root>/results.csv    one row per saved trial
Models listed in <root>/eval_commands.md that have no trials yet show up as 0/<TRIALS>.

The files are regenerated automatically after every saved trial; to rebuild by hand:
    envs/.venv/bin/python scripts/eval_summary.py [root]      (default: so101_finetuning_eval)
"""

import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

VIDEO_RE = re.compile(r"^(?P<model>.+)-(?P<task>[a-z0-9_]+)-(?P<run>\d+)-(?P<score>-?\d+(?:\.\d+)?)\.mp4$")
CAMERAS = ("front", "wrist")
OUT_FILES = {"en": "results.md", "cn": "results_cn.md"}
TASK_ORDER = ["stack_white_bowls", "screwdriver_box_to_table", "cube_drawer", "stack_cubes", "upright_bottle"]
LANG = {
    "cn": {
        "tasks": {"stack_white_bowls": "叠碗 stack white bowls", "screwdriver_box_to_table": "螺丝刀 screwdriver box → table",
                  "cube_drawer": "抽屉 cube drawer", "stack_cubes": "叠方块 stack cubes", "upright_bottle": "扶正瓶子 upright bottle"},
        "title": "# SO-101 fine-tuning 测评结果",
        "note": "> 根据本目录下已保存的视频**自动生成**，每保存一轮都会更新，请不要手动编辑。 手动刷新：`envs/.venv/bin/python scripts/eval_summary.py`",
        "updated": "> 更新时间：{time} ｜ 每一轮的明细：[results.csv](results.csv) ｜ [English](results.md)",
        "overview": "## 总览",
        "overview_head": "| 任务 | policy | 模型 | 进度 | 平均分 | 最低 | 最高 | 视频位置 |",
        "details": "## 每一轮明细",
        "none": "**{policy}** · `{model}` · 0/{target}：还没有录制",
        "model_line": "**{policy}** · `{model}` · {n}/{target} · 平均分 {mean}",
        "trial_head": "| 轮次 | 分数 | 顶部视频 | 夹爪视频 | 录制时间 |",
        "missing": "缺失",
    },
    "en": {
        "tasks": {"stack_white_bowls": "Stack white bowls", "screwdriver_box_to_table": "Screwdriver: box → table",
                  "cube_drawer": "Cube drawer", "stack_cubes": "Stack cubes", "upright_bottle": "Upright bottle"},
        "title": "# SO-101 fine-tuning evaluation results",
        "note": "> **Generated automatically** from the videos saved in this folder and refreshed after every saved trial; do not edit by hand. To refresh manually: `envs/.venv/bin/python scripts/eval_summary.py`",
        "updated": "> Updated: {time} | Per-trial data: [results.csv](results.csv) | [中文](results_cn.md)",
        "overview": "## Overview",
        "overview_head": "| Task | Policy | Model | Progress | Mean score | Min | Max | Videos |",
        "details": "## Per-trial details",
        "none": "**{policy}** · `{model}` · 0/{target}: not recorded yet",
        "model_line": "**{policy}** · `{model}` · {n}/{target} · mean score {mean}",
        "trial_head": "| Trial | Score | Front (top) video | Wrist video | Recorded |",
        "missing": "missing",
    },
}


def planned_models(root: Path) -> dict[tuple[str, str, str], int]:
    """(task_dir, policy_dir, model) -> target trials, from the commands in eval_commands.md."""
    md = root / "eval_commands.md"
    if not md.is_file():
        return {}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trial_recording import model_name_from_path

    planned = {}
    for block in re.findall(r"```bash\n(.*?)```", md.read_text(), re.S):
        env = dict(re.findall(r"\b([A-Z_]+)=(\S+)", block))
        if "EVAL_DIR" not in env or "CHECKPOINT" not in env:
            continue
        parts = Path(env["EVAL_DIR"]).parts
        if len(parts) < 2:
            continue
        model = env.get("MODEL_NAME") or model_name_from_path(env["CHECKPOINT"])
        planned[(parts[-2], parts[-1], model)] = int(env.get("TRIALS", "25"))
    return planned


def collect(root: Path):
    """{(task_dir, policy_dir, model): {run: {"score": str, "front": Path, "wrist": Path, "time": float}}}"""
    trials = defaultdict(dict)
    for video in sorted(root.glob("*/*/*/*.mp4")):
        cam, policy, task = video.parent.name, video.parent.parent.name, video.parent.parent.parent.name
        m = VIDEO_RE.match(video.name)
        if cam not in CAMERAS or not m:
            continue
        run = int(m["run"])
        entry = trials[(task, policy, m["model"])].setdefault(run, {"score": m["score"], "task_name": m["task"]})
        entry[cam] = video
        entry["time"] = max(entry.get("time", 0.0), video.stat().st_mtime)
    return trials


def fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def build(root: Path) -> None:
    root = Path(root)
    trials = collect(root)
    planned = planned_models(root)
    keys = sorted(set(trials) | set(planned), key=lambda k: (TASK_ORDER.index(k[0]) if k[0] in TASK_ORDER else 99, k[1], k[2]))

    rows = []
    for task, policy, model in keys:
        for run, t in sorted(trials.get((task, policy, model), {}).items()):
            rows.append({
                "task": task, "policy": policy, "model": model, "run": run, "score": t["score"],
                "front_video": str(t["front"].relative_to(root)) if "front" in t else "",
                "wrist_video": str(t["wrist"].relative_to(root)) if "wrist" in t else "",
                "recorded_at": datetime.fromtimestamp(t["time"]).strftime("%Y-%m-%d %H:%M"),
            })
    with open(root / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "policy", "model", "run", "score", "front_video", "wrist_video", "recorded_at"])
        w.writeheader()
        w.writerows(rows)

    now = datetime.now()
    for lang, t in LANG.items():
        (root / OUT_FILES[lang]).write_text(render(root, keys, trials, planned, t, now))


def render(root: Path, keys, trials, planned, t: dict, now: datetime) -> str:
    out = [t["title"], "", t["note"], t["updated"].format(time=f"{now:%Y-%m-%d %H:%M}"), "",
           t["overview"], "", t["overview_head"], "|---|---|---|---|---|---|---|---|"]
    for task, policy, model in keys:
        ts = trials.get((task, policy, model), {})
        target = planned.get((task, policy, model), 25)
        scores = [float(x["score"]) for x in ts.values()]
        stats = [fmt(sum(scores) / len(scores)), fmt(min(scores)), fmt(max(scores))] if scores else ["—", "—", "—"]
        folder = f"{task}/{policy}"
        out.append(f"| {task} | {policy} | `{model}` | {len(ts)}/{target} | {' | '.join(stats)} | [{folder}/]({folder}) |")

    out += ["", t["details"], ""]
    for task in dict.fromkeys(k[0] for k in keys):
        out += [f"### {t['tasks'].get(task, task)}", ""]
        for _, policy, model in [k for k in keys if k[0] == task]:
            ts = trials.get((task, policy, model), {})
            target = planned.get((task, policy, model), 25)
            if not ts:
                out += [t["none"].format(policy=policy, model=model, target=target), ""]
                continue
            scores = [float(x["score"]) for x in ts.values()]
            out += [t["model_line"].format(policy=policy, model=model, n=len(ts), target=target, mean=fmt(sum(scores) / len(scores))),
                    "", t["trial_head"], "|---|---|---|---|---|"]
            for run, x in sorted(ts.items()):
                links = [f"[{c}]({x[c].relative_to(root)})" if c in x else t["missing"] for c in CAMERAS]
                out.append(f"| {run} | {x['score']} | {links[0]} | {links[1]} | {datetime.fromtimestamp(x['time']):%m-%d %H:%M} |")
            out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("so101_finetuning_eval"))
    print("updated results.md / results_cn.md / results.csv")
