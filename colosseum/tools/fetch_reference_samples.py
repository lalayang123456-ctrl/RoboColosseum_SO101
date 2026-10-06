#!/usr/bin/env python3
"""Download 3 episodes of a public SO100 dataset from the models' training mixture and extract
12 reference samples (two views, state, ground-truth action chunk) for openloop_check.py.

Needs huggingface_hub, pyarrow and PyAV (av): run it in the MolmoAct2 environment.

    envs/.venv/bin/python colosseum/tools/fetch_reference_samples.py outputs/colosseum/ref
"""
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = 'Beegbrain/pick_lemon_and_drop_in_bowl'  # LeRobot v2.1, 30 fps, legacy SO100 joint frame
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
fetch = lambda name: hf_hub_download(REPO, name, repo_type='dataset')  # noqa: E731
tasks = [json.loads(line) for line in open(fetch('meta/tasks.jsonl'))]
samples = []
for episode in (0, 7, 15):
    table = pq.read_table(fetch(f'data/chunk-000/episode_{episode:06d}.parquet')).to_pydict()
    states = np.array(table['observation.state'], np.float32)
    actions = np.array(table['action'], np.float32)
    indices = [0, len(states) // 4, len(states) // 2, 3 * len(states) // 4]
    frames = {}
    for camera in ('realsense_top', 'realsense_side'):
        container = av.open(fetch(f'videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4'))
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                frames[camera, index] = frame.to_ndarray(format='rgb24')
        container.close()
    for index in indices:
        key = f'ep{episode}_t{index}'
        np.save(out / f'{key}_head.npy', frames['realsense_top', index])
        np.save(out / f'{key}_wrist.npy', frames['realsense_side', index])
        np.save(out / f'{key}_state.npy', states[index])
        np.save(out / f'{key}_actions.npy', actions[index:index + 50])
        samples.append({'key': key, 'task': tasks[table['task_index'][index]]['task']})
json.dump(samples, open(out / 'samples.json', 'w'))
print(f'{len(samples)} samples in {out}; rest state of episode 0: {np.load(out / "ep0_t0_state.npy").round(1)}')
