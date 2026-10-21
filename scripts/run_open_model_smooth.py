#!/usr/bin/env python3
"""Like run_open_model.py, but without the pause at every action chunk.

run_open_model.py (and the Colosseum Client) stop sending commands while the model computes
the next chunk. Here the next chunk is requested in the background *before* the current one
is used up, so the arm keeps moving at the control rate:

  - a chunk predicted from the observation taken at step s holds the actions for steps
    s, s+1, s+2, ...; by the time it arrives, some of those steps are already in the past,
    so execution switches to it at the entry for the current step (latency compensation);
  - the request is sent early enough for the measured inference latency, so the switch
    happens after about --replan-steps steps of each chunk.

This only works when inference takes less than half a chunk: the entries that are still in
the future when a chunk arrives must last until the next one arrives. MolmoAct2 (~0.4 s for
30 steps) and pi0.5 (~0.6 s for 50 steps) qualify at 30 Hz. G05 (~0.8 s) qualifies at its
upstream 15 Hz (12 of 32 steps past on arrival) but not at 30 Hz (24 of 32); when it does not
fit, the script waits at the end of each chunk for a chunk computed from a fresh observation,
like run_open_model.py, and says so.

G05 defaults follow its upstream SO100 deployment (15 Hz, bfloat16, at most 10 deg per tick,
start from the upstream home pose); see run_open_model.py for the options that change this.

Standalone: no Router, no recording, no scoring.

    colosseum-client/.venv/bin/python scripts/run_open_model_smooth.py                 # pi0.5 (default)
    colosseum-client/.venv/bin/python scripts/run_open_model_smooth.py --model molmoact2
    colosseum-client/.venv/bin/python scripts/run_open_model_smooth.py --model g05
    colosseum-client/.venv/bin/python scripts/run_open_model_smooth.py --dry-run       # print, do not move

Per trial: type the task + Enter to start; Enter again to stop early; the arm then returns to
the pose it started from and releases torque. Empty task or q quits.
"""
import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_open_model import (MODELS, REPO, add_execution_arguments, enter_pressed, infer, infer_g05,  # noqa: E402
                            limit_step, measured, prepare_start, resolve_execution, start_worker)

# Starting guess for inference latency in seconds; replaced by measurements during the trial.
LATENCY = {'pi05': 0.7, 'molmoact2': 0.4, 'g05': 0.9, 'groot': 0.2}
MARGIN_STEPS = 2


def run_trial(robot, name, port, task, *, duration, hz, replan_steps, execute, pool, infer_chunk,
              max_step_deg=0, g05_frequency=15, max_steps=800):
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    period = 1 / hz

    def request(step, overlapping=True):
        """Read the robot on the control thread; run the model on the worker thread."""
        observation = robot.get_observation()
        state = np.r_[observation.joints, observation.gripper]
        images = observation.images['head_image'], observation.images['left_image']
        sent = time.monotonic()

        def work():
            return infer_chunk(name, port, *images, state, task, g05_frequency), time.monotonic() - sent
        return step, state, pool.submit(work), overlapping

    print('Computing the first chunk (the arm holds still)...', flush=True)
    _, first, future, _ = request(0)
    plan, took = future.result()
    base, latency = 0, max(LATENCY[name], 0.0)  # the first call can include warm-up; do not learn from it
    limit = f'{max_steps} steps' + (f' or {duration:.0f} s' if duration else '')
    print(f'First chunk in {took:.2f}s. Running "{task}" for up to {limit}. Press Enter to stop.', flush=True)

    step, pending, last, warned = 0, None, None, False
    held = switches = dropped = 0
    moved, latencies = np.zeros(6), []
    start = time.monotonic()
    try:
        while step < max_steps and not (duration and time.monotonic() - start >= duration) and not enter_pressed():
            tick = time.monotonic()
            if pending is not None and pending[2].done():
                new_base, state, done, overlapping = pending
                pending = None
                new_plan, took = done.result()
                latencies.append(took)
                latency = 0.7 * latency + 0.3 * took
                moved = np.maximum(moved, np.abs(state - first))
                if not overlapping:
                    new_base = step  # the arm waited at the observed pose: start at the first entry
                if step - new_base < len(new_plan):
                    plan, base = new_plan, new_base
                    switches += 1
                    print(f'step {step:4d}  state {state}  ->  chunk end {plan[-1]}  '
                          f'(inference {took:.2f}s, resumed at entry {step - base}/{len(plan)})', flush=True)
                else:
                    dropped += 1  # arrived after every step it covers had passed
            index = step - base
            lead = math.ceil(latency * hz) + MARGIN_STEPS
            if 2 * math.ceil(latency * hz) + MARGIN_STEPS <= len(plan):
                if pending is None and index >= min(replan_steps, len(plan)) - lead:
                    pending = request(step)
            else:
                if not warned:
                    warned = True
                    print(f'Inference (~{latency:.2f}s = {lead} steps) is more than half of a {len(plan)}-step chunk: '
                          'cannot overlap, waiting at the end of each chunk instead.', flush=True)
                if pending is None and index >= min(replan_steps, len(plan)):
                    pending = request(step, overlapping=False)
            waiting = pending is not None and not pending[3]  # keep the pose the request observed
            if index < len(plan) and not waiting:
                last = plan[index]
            else:
                held += 1  # hold the last target until the next chunk arrives
            if execute and last is not None:
                robot.execute(limit_step(last, measured(robot), max_step_deg) if max_step_deg else last)
            step += 1
            time.sleep(max(0, period - (time.monotonic() - tick)))
    finally:
        if pending is not None:
            pending[2].cancel()
    elapsed = time.monotonic() - start
    return dict(steps=step, seconds=elapsed, hz=step / elapsed if elapsed else 0, switches=switches, held=held,
                dropped=dropped, latency=float(np.mean(latencies)) if latencies else took, moved=moved)


def infer_chunk(name, port, external, wrist, state, task, g05_frequency):
    if name == 'g05':
        return infer_g05(port, external, wrist, state, task, g05_frequency, MODELS[name]['chunk'])
    return infer(port, external, wrist, state, task)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(MODELS), default='pi05')
    parser.add_argument('--config', default=str(REPO / 'colosseum-client/configs/robot.so101.yaml'),
                        help='Colosseum robot config; only its arm and camera settings are used')
    parser.add_argument('--max-steps', type=int, default=800,
                        help='stop a trial after this many executed steps (default 800, like the Colosseum Open track)')
    parser.add_argument('--duration', type=float, default=0,
                        help='optional wall-clock limit per trial in seconds; 0 = no time limit (default)')
    parser.add_argument('--replan-steps', type=int,
                        help='switch to a fresh chunk after about this many steps of the current one '
                             '(default: 32 for g05, 30 otherwise)')
    add_execution_arguments(parser)
    parser.add_argument('--dry-run', action='store_true', help='read the robot and print chunks without executing them')
    args = parser.parse_args()
    model = MODELS[args.model]
    run = resolve_execution(args, args.model, 'replan_steps')
    if not 1 <= run['chunk_steps'] <= model['chunk']:
        parser.error(f'--replan-steps must be in [1, {model["chunk"]}] for {args.model}')
    print(f'{args.model}: {run["hz"]:g} Hz, replan after {run["chunk_steps"]} steps, '
          f'max step {run["max_step_deg"] or "off"}, start pose {run["home"]}'
          + (f', {"float32" if args.g05_fp32 else "bfloat16"}, frequency input {args.g05_frequency:g}'
             if args.model == 'g05' else ''))

    from colosseum_client import RobotClientConfig, make_robot
    config = RobotClientConfig.from_yaml(args.config)
    if config.robot_type != 'so101' or config.test:
        sys.exit('The config must be a real robot_type: so101 configuration')
    log = REPO / 'outputs/colosseum' / f'{args.model}-standalone.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    # G05's server must serve its whole chunk so later entries are available after the latency.
    worker = start_worker(args.model, log, model['chunk'], args.g05_fp32)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                print('\nPlace the arm in its rest pose and set up the scene.')
                task = input('Task> ').strip()
                if task.lower() in {'', 'q', 'quit', 'exit'}:
                    break
                robot = make_robot(config)  # enables torque at the current pose
                try:
                    prepare_start(robot, run['home'], not args.dry_run)
                    r = run_trial(robot, args.model, model['port'], task, duration=args.duration, hz=run['hz'],
                                  replan_steps=run['chunk_steps'], execute=not args.dry_run, pool=pool,
                                  infer_chunk=infer_chunk, max_step_deg=run['max_step_deg'],
                                  g05_frequency=args.g05_frequency, max_steps=args.max_steps)
                    print(f'Stopped after {r["steps"]} steps in {r["seconds"]:.1f}s ({r["hz"]:.1f} Hz). '
                          f'{r["switches"]} chunks, mean inference {r["latency"]:.2f}s, '
                          f'{r["held"]} steps held waiting for a chunk, {r["dropped"]} late chunks dropped.')
                    print(f'Largest joint excursion from the start pose: {r["moved"]}')
                except KeyboardInterrupt:
                    print('\nInterrupted.')
                finally:
                    print('Returning to the start pose and releasing torque...', flush=True)
                    robot.close()
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        if worker is not None:
            worker.terminate()
            worker.wait()


if __name__ == '__main__':
    main()
