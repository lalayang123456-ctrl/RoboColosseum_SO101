#!/usr/bin/env python3
"""Run a generalist SO100/101 checkpoint on the real SO-101, standalone (no Router, no recording).

Loads the model once, then repeatedly asks for a task instruction and runs it on the arm.
It reuses the two pieces already validated for Colosseum: the SO101 hardware driver from
colosseum-client (same arm/camera settings as configs/robot.so101.yaml) and the model worker
from colosseum-policy-server, but talks to the worker directly.

    colosseum-client/.venv/bin/python scripts/run_open_model.py                 # pi0.5 (default)
    colosseum-client/.venv/bin/python scripts/run_open_model.py --model molmoact2
    colosseum-client/.venv/bin/python scripts/run_open_model.py --model g05
    colosseum-client/.venv/bin/python scripts/run_open_model.py --model groot
    colosseum-client/.venv/bin/python scripts/run_open_model.py --dry-run       # print actions, do not move
    colosseum-client/.venv/bin/python scripts/run_open_model.py --model g05 --colosseum   # same inference and
                                                           # execution as the Colosseum evaluation, any model

Per trial: type the task + Enter to start; Enter again to stop early; the arm then returns to
the pose it started from and releases torque. Empty task or q quits.

G05 defaults follow its upstream SO100 deployment (experiments/so100 in GalaxeaVLA) instead of
the Colosseum settings: bfloat16 weights, 15 Hz, 32 steps per chunk, at most 10 deg per tick,
and the arm first moves to the upstream home pose (the training-data mean, arm raised) before
the model takes over. Override any of them with --hz, --chunk-steps, --max-step-deg, --home
and --g05-fp32; `--hz 30 --chunk-steps 30 --max-step-deg 0 --home rest --g05-fp32` reproduces
what the Colosseum evaluation does (add --g05-frequency 30 for that too). The other models
keep 30 Hz from the rest pose; --home mean is available for them too.

G05 takes a "frequency" value with each observation. Measured on 11 training-distribution
samples (first 1.07 s of each chunk): with frequency 15 the predicted steps match the 30 fps
demonstration frame by frame (2.8 deg mean error; 5.7 deg against every second frame), and
with frequency 30 the error is 4.1 deg. So 15 is the better input, and one predicted step
corresponds to 1/30 s of demonstration regardless: --hz 15 (upstream) plays the motion at
half the demonstrated speed, --hz 30 at the demonstrated speed.
"""
import argparse
import base64
import json
import select
import socket
import subprocess
import sys
import termios
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODELS = {
    'pi05': dict(port=9212, python=REPO / 'envs/pi05_so101/.venv/bin/python', checkpoint=REPO / 'checkpoints/pi05-so100_101',
                 arguments=['pi05_lerobot', '--robot-type', 'so101', '--device', 'cuda'], chunk=50),
    'molmoact2': dict(port=9201, python=REPO / 'envs/.venv/bin/python', checkpoint=REPO / 'checkpoints/MolmoAct2-SO100_101',
                      arguments=['molmoact2', '--robot-type', 'so101', '--device', 'cuda:0', '--dtype', 'bfloat16'], chunk=30),
    'groot': dict(port=9217, python=REPO / 'envs/.venv/bin/python', checkpoint=REPO / 'checkpoints/groot-so100_101',
                  arguments=['groot_n17', '--robot-type', 'so101', '--base-model', str(REPO / 'checkpoints/GR00T-N1.7-3B'),
                             '--device', 'cuda'], chunk=40),
    # Upstream G05 server: it caches a chunk and serves --action-steps steps of it, one per request.
    'g05': dict(port=9204, python=REPO / 'vendor/GalaxeaVLA/.venv/bin/python',
                checkpoint=REPO / 'checkpoints/G05/g05-so101/checkpoints/model_state_dict.pt',
                arguments=['g05', '--robot-type', 'so101', '--source-root', str(REPO / 'vendor/GalaxeaVLA'), '--device', 'cuda'],
                chunk=32, defaults=dict(hz=15, chunk_steps=32, max_step_deg=10, home='mean')),
}
DEFAULTS = dict(hz=30, chunk_steps=30, max_step_deg=0, home='rest')
# --colosseum: what the Colosseum evaluation does for each model. The Client executes a whole chunk of
# the Router's max_horizon (30/50/40/30) at 30 Hz from wherever the arm is, with no per-tick cap; its
# Policy Server loads G05 with the checkpoint's own float32 settings and sends frequency = 30.
COLOSSEUM = {
    'molmoact2': dict(hz=30, chunk_steps=30, max_step_deg=0, home='rest'),
    'pi05': dict(hz=30, chunk_steps=50, max_step_deg=0, home='rest'),
    'groot': dict(hz=30, chunk_steps=40, max_step_deg=0, home='rest'),
    'g05': dict(hz=30, chunk_steps=30, max_step_deg=0, home='rest', g05_fp32=True, g05_frequency=30),
}
# Upstream G05 server settings (experiments/so100/start_server.sh). torch.compile is left off: it
# only affects speed and is slow to warm up.
G05_BF16 = ['--g05-override', 'model.model_weights_to_bf16=true',
            '--g05-override', 'model.model_arch.attn_implementation=sdpa']
# Upstream G05 home pose: the training-data mean, in the arm's own frame (degrees, gripper 0-100).
HOME_MEAN = np.array([3.1, -34.3, 31.5, 55.9, -12.3, 13.4], dtype=np.float32)
# These checkpoints use LeRobot's earlier SO100 joint frame: model = sign * arm + offset (degrees).
SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float32)
OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=np.float32)


def listening(port):
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(('127.0.0.1', port)) == 0


def die_with_parent():
    """Run in the child before exec: have the kernel stop the worker if this script dies for any
    reason (closed terminal, kill), so it never lingers holding the GPU and its port."""
    import ctypes
    import signal
    ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def start_worker(name, log, chunk_steps, g05_fp32=False):
    """Return the worker process, or None when one is already serving this model's port."""
    model = MODELS[name]
    extra = ['--action-steps', str(chunk_steps), *([] if g05_fp32 else G05_BF16)] if name == 'g05' else []
    if listening(model['port']):
        print(f'Reusing the {name} worker already listening on port {model["port"]}.')
        return None
    if subprocess.run(['pgrep', '-f', '[c]olosseum-policy-local'], capture_output=True).returncode == 0:
        sys.exit('The Colosseum Policy Server is running and may hold the GPU. Stop it (Ctrl-C in its terminal) first.')
    command = [str(model['python']), '-m', 'colosseum_policy_server.model_service', *model['arguments'],
               '--checkpoint', str(model['checkpoint']), '--port', str(model['port']), *extra]
    print(f'Loading {name} (log: {log}) ...', flush=True)
    process = subprocess.Popen(command, stdout=open(log, 'wb'), stderr=subprocess.STDOUT, start_new_session=True,
                               preexec_fn=die_with_parent)
    started = time.monotonic()
    while not listening(model['port']):
        if process.poll() is not None:
            sys.exit(f'Model worker exited with status {process.returncode}; see {log}')
        if time.monotonic() - started > 600:
            process.terminate()
            sys.exit(f'Model worker did not start within 600 s; see {log}')
        time.sleep(1)
    print(f'Model ready in {time.monotonic() - started:.0f} s.')
    return process


def encode(array):
    array = np.ascontiguousarray(array)
    return {'__numpy__': base64.b64encode(array.tobytes()).decode('ascii'), 'dtype': array.dtype.str, 'shape': list(array.shape)}


def infer(port, external, wrist, state, task, num_steps=10):
    """state/actions are in the arm's own frame; returns an (N, 6) float32 chunk."""
    payload = dict(external_cam=encode(external), wrist_cam=encode(wrist),
                   state=encode((SIGNS * state + OFFSETS).astype(np.float32)), instruction=task, num_steps=num_steps)
    request = Request(f'http://127.0.0.1:{port}/act', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=120) as response:
        raw = json.loads(response.read())
    if 'actions' not in raw:
        raise RuntimeError(f'Model worker returned no actions: {raw}')
    value = raw['actions']
    actions = np.frombuffer(base64.b64decode(value['__numpy__']), dtype=np.dtype(value['dtype'])).reshape(value['shape'])
    return ((actions.astype(np.float32) - OFFSETS) * SIGNS).astype(np.float32)


def infer_g05(port, external, wrist, state, task, hz, max_steps):
    """One connection per chunk: infer once, then drain the steps the server cached for it."""
    import msgpack
    from websockets.sync.client import connect

    def pack(value):
        return msgpack.packb(value, default=lambda a: {b'__ndarray__': True, b'data': a.tobytes(),
                                                      b'dtype': a.dtype.str, b'shape': a.shape})

    def unpack(raw):
        def hook(value):
            value = {k.decode() if isinstance(k, bytes) else k: v for k, v in value.items()}
            if '__ndarray__' in value:
                return np.frombuffer(value['data'], dtype=np.dtype(value['dtype'])).reshape(tuple(value['shape'])).copy()
            return value
        return msgpack.unpackb(raw, object_hook=hook, raw=False, strict_map_key=False)

    exterior = np.ascontiguousarray(external.transpose(2, 0, 1))
    request = {'images': {'exterior': exterior, 'wrist_right': np.ascontiguousarray(wrist.transpose(2, 0, 1)),
                          'wrist_left': np.zeros_like(exterior)},
               'state': {'right_arm': (SIGNS * state + OFFSETS).astype(np.float32)},
               'task': task, 'embodiment_type': 'so100', 'frequency': float(hz)}
    steps = []
    with connect(f'ws://127.0.0.1:{port}', compression=None, max_size=None, open_timeout=30) as socket:
        socket.recv(timeout=30)  # greeting: {"action_steps": N}
        while True:
            socket.send(pack(request))
            response = unpack(socket.recv(timeout=240))
            if 'error' in response or 'right_arm' not in response.get('action', {}):
                raise RuntimeError(f'G05 server returned no action: {response.get("error", response)}')
            steps.append((np.asarray(response['action']['right_arm'], np.float32).reshape(6) - OFFSETS) * SIGNS)
            if response.get('need_obs', True) or len(steps) >= max_steps:
                return np.stack(steps).astype(np.float32)
            request = {}


def add_execution_arguments(parser):
    parser.add_argument('--hz', type=float, help='action execution rate (default: 15 for g05, 30 otherwise)')
    parser.add_argument('--max-step-deg', type=float,
                        help='scale each command so no joint moves more than this from the measured '
                             'position per tick; 0 disables (default: 10 for g05, 0 otherwise)')
    parser.add_argument('--home', choices=['rest', 'mean'],
                        help='rest: start where the arm is; mean: first move to the upstream G05 home pose '
                             f'{[round(float(v), 1) for v in HOME_MEAN]} (default: mean for g05, rest otherwise)')
    parser.add_argument('--g05-fp32', action='store_true', help='load G05 in float32 instead of upstream bfloat16')
    parser.add_argument('--g05-frequency', type=float, default=15,
                        help='frequency value sent to G05 with each observation (upstream client: 15). It is a '
                             'model input, separate from --hz; see the note in the module docstring')


def resolve_execution(args, name, steps_option):
    """Fill unset options from the model's defaults and return them as a dict."""
    defaults = {**DEFAULTS, **MODELS[name].get('defaults', {})}
    chosen = {key: getattr(args, key if key != 'chunk_steps' else steps_option) for key in defaults}
    settings = {key: defaults[key] if chosen[key] is None else chosen[key] for key in defaults}
    if settings['hz'] <= 0 or settings['max_step_deg'] < 0:
        raise SystemExit('--hz must be positive and --max-step-deg must not be negative')
    return settings


def limit_step(target, current, max_deg):
    """Upstream G05 clip_action: scale the whole delta so no joint exceeds max_deg this tick."""
    delta = np.asarray(target, dtype=np.float64) - current
    biggest = float(np.max(np.abs(delta)))
    if not max_deg or biggest <= max_deg:
        return np.asarray(target, dtype=np.float32)
    return (current + delta * (max_deg / biggest)).astype(np.float32)


def measured(robot):
    observation = robot.get_observation()
    return np.r_[observation.joints, observation.gripper].astype(np.float64)


def move_to(robot, target, *, step_deg=1.5, hz=30, timeout=15):
    """Interpolate from the measured pose to target at no more than step_deg per tick."""
    start = measured(robot)
    steps = max(1, int(np.ceil(np.max(np.abs(target - start)) / step_deg)))
    for index in range(1, steps + 1):
        tick = time.monotonic()
        robot.execute(start + (target - start) * index / steps)
        time.sleep(max(0, 1 / hz - (time.monotonic() - tick)))
    deadline = time.monotonic() + timeout  # let the arm settle; gravity can leave a small error
    while time.monotonic() < deadline and np.max(np.abs(measured(robot)[:5] - target[:5])) > 3:
        robot.execute(target)
        time.sleep(1 / hz)
    return measured(robot)


def enter_pressed():
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def run_trial(robot, name, port, task, *, duration, hz, chunk_steps, execute, max_step_deg=0, g05_frequency=15,
              max_steps=800):
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    limit = f'{max_steps} steps' + (f' or {duration:.0f} s' if duration else '')
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)  # drop keys typed before the trial, e.g. a second Enter
    print(f'Running "{task}" for up to {limit}. Press Enter to stop.', flush=True)
    start, step, moved = time.monotonic(), 0, np.zeros(6)
    first = None
    timed_out = lambda: bool(duration) and time.monotonic() - start >= duration  # noqa: E731
    while step < max_steps and not timed_out():
        observation = robot.get_observation()
        state = np.r_[observation.joints, observation.gripper]
        first = state if first is None else first
        began = time.monotonic()
        images = observation.images['head_image'], observation.images['left_image']
        actions = (infer_g05(port, *images, state, task, g05_frequency, chunk_steps) if name == 'g05'
                   else infer(port, *images, state, task)[:chunk_steps])
        print(f'step {step:4d}  state {state}  ->  chunk end {actions[-1]}  (inference {time.monotonic() - began:.2f}s)', flush=True)
        for action in actions:
            tick = time.monotonic()
            if enter_pressed() or step >= max_steps or timed_out():
                moved = np.abs(state - first)
                return step, moved
            if execute:
                robot.execute(limit_step(action, measured(robot), max_step_deg) if max_step_deg else action)
            step += 1
            time.sleep(max(0, 1 / hz - (time.monotonic() - tick)))
        moved = np.maximum(moved, np.abs(state - first))
    return step, moved


def prepare_start(robot, home, execute):
    """Optionally move from the rest pose to the upstream home pose before the model takes over."""
    if home != 'mean':
        return
    if not execute:
        print('Dry run: not moving to the home pose, so the model sees the rest pose instead.', flush=True)
        return
    print(f'Moving to the home pose {[round(float(v), 1) for v in HOME_MEAN]} ...', flush=True)
    reached = move_to(robot, HOME_MEAN.astype(np.float64))
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    print(f'Home pose reached: {reached}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(MODELS), default='pi05')
    parser.add_argument('--config', default=str(REPO / 'colosseum-client/configs/robot.so101.yaml'),
                        help='Colosseum robot config; only its arm and camera settings are used')
    # For g05 this is also how many steps of each chunk its server serves; it is fixed at load time.
    parser.add_argument('--max-steps', type=int, default=800,
                        help='stop a trial after this many executed steps (default 800, like the Colosseum Open track)')
    parser.add_argument('--duration', type=float, default=0,
                        help='optional wall-clock limit per trial in seconds; 0 = no time limit (default)')
    parser.add_argument('--chunk-steps', type=int,
                        help='steps executed per inference before replanning (default: 32 for g05, 30 otherwise)')
    add_execution_arguments(parser)
    parser.add_argument('--colosseum', action='store_true',
                        help='use exactly the settings of the Colosseum evaluation for this model (same worker '
                             'command, chunk length = Router max_horizon, 30 Hz, rest pose, no step cap); '
                             'overrides --hz, --chunk-steps, --max-step-deg and the --g05-* options; '
                             'an explicit --home is kept, so --colosseum --home mean changes only the start pose')
    parser.add_argument('--dry-run', action='store_true', help='read the robot and print actions without executing them')
    args = parser.parse_args()
    model = MODELS[args.model]
    if args.colosseum:
        preset = COLOSSEUM[args.model]
        args.hz, args.chunk_steps, args.max_step_deg = (preset[k] for k in ('hz', 'chunk_steps', 'max_step_deg'))
        args.home = args.home or preset['home']  # an explicit --home is kept: only the start pose differs
        args.g05_fp32 = preset.get('g05_fp32', args.g05_fp32)
        args.g05_frequency = preset.get('g05_frequency', args.g05_frequency)
    run = resolve_execution(args, args.model, 'chunk_steps')
    if not 1 <= run['chunk_steps'] <= model['chunk']:
        parser.error(f'--chunk-steps must be in [1, {model["chunk"]}] for {args.model}')
    print(('Colosseum settings. ' if args.colosseum else '') + f'{args.model}: {run["hz"]:g} Hz, {run["chunk_steps"]} steps per chunk, '
          f'max step {run["max_step_deg"] or "off"}, start pose {run["home"]}'
          + (f', {"float32" if args.g05_fp32 else "bfloat16"}, frequency input {args.g05_frequency:g}'
             if args.model == 'g05' else ''))

    from colosseum_client import RobotClientConfig, make_robot
    config = RobotClientConfig.from_yaml(args.config)
    if config.robot_type != 'so101' or config.test:
        sys.exit('The config must be a real robot_type: so101 configuration')
    log = REPO / 'outputs/colosseum' / f'{args.model}-standalone.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    worker = start_worker(args.model, log, run['chunk_steps'], args.g05_fp32)
    try:
        while True:
            print('\nPlace the arm in its rest pose and set up the scene.')
            task = input('Task> ').strip()
            if task.lower() in {'', 'q', 'quit', 'exit'}:
                break
            robot = make_robot(config)  # enables torque at the current pose
            try:
                prepare_start(robot, run['home'], not args.dry_run)
                steps, moved = run_trial(robot, args.model, model['port'], task, duration=args.duration,
                                         hz=run['hz'], chunk_steps=run['chunk_steps'], execute=not args.dry_run,
                                         max_step_deg=run['max_step_deg'], g05_frequency=args.g05_frequency,
                                         max_steps=args.max_steps)
                print(f'Stopped after {steps} steps. Largest joint excursion from the start pose: {moved}')
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
