#!/usr/bin/env python3
"""Run a generalist SO100/101 checkpoint on the real SO-101, standalone (no Router, no recording).

Loads the model once, then repeatedly asks for a task instruction and runs it on the arm.
It reuses the two pieces already validated for Colosseum: the SO101 hardware driver from
colosseum-client (same arm/camera settings as configs/robot.so101.yaml) and the model worker
from colosseum-policy-server, but talks to the worker directly.

    colosseum-client/.venv/bin/python scripts/run_open_model.py                 # pi0.5 (default)
    colosseum-client/.venv/bin/python scripts/run_open_model.py --model molmoact2
    colosseum-client/.venv/bin/python scripts/run_open_model.py --model g05
    colosseum-client/.venv/bin/python scripts/run_open_model.py --dry-run       # print actions, do not move

Per trial: type the task + Enter to start; Enter again to stop early; the arm then returns to
the pose it started from and releases torque. Empty task or q quits.
"""
import argparse
import base64
import json
import select
import socket
import subprocess
import sys
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
    # Upstream G05 server: it caches a chunk and serves --action-steps steps of it, one per request.
    'g05': dict(port=9204, python=REPO / 'vendor/GalaxeaVLA/.venv/bin/python',
                checkpoint=REPO / 'checkpoints/G05/g05-so101/checkpoints/model_state_dict.pt',
                arguments=['g05', '--robot-type', 'so101', '--source-root', str(REPO / 'vendor/GalaxeaVLA'), '--device', 'cuda'],
                chunk=32),
}
# These checkpoints use LeRobot's earlier SO100 joint frame: model = sign * arm + offset (degrees).
SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float32)
OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=np.float32)


def listening(port):
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(('127.0.0.1', port)) == 0


def start_worker(name, log, chunk_steps):
    """Return the worker process, or None when one is already serving this model's port."""
    model = MODELS[name]
    extra = ['--action-steps', str(chunk_steps)] if name == 'g05' else []
    if listening(model['port']):
        print(f'Reusing the {name} worker already listening on port {model["port"]}.')
        return None
    if subprocess.run(['pgrep', '-f', '[c]olosseum-policy-local'], capture_output=True).returncode == 0:
        sys.exit('The Colosseum Policy Server is running and may hold the GPU. Stop it (Ctrl-C in its terminal) first.')
    command = [str(model['python']), '-m', 'colosseum_policy_server.model_service', *model['arguments'],
               '--checkpoint', str(model['checkpoint']), '--port', str(model['port']), *extra]
    print(f'Loading {name} (log: {log}) ...', flush=True)
    process = subprocess.Popen(command, stdout=open(log, 'wb'), stderr=subprocess.STDOUT, start_new_session=True)
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


def enter_pressed():
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def run_trial(robot, name, port, task, *, duration, hz, chunk_steps, execute):
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    print(f'Running "{task}" for up to {duration:.0f} s. Press Enter to stop.', flush=True)
    start, step, moved = time.monotonic(), 0, np.zeros(6)
    first = None
    while time.monotonic() - start < duration:
        observation = robot.get_observation()
        state = np.r_[observation.joints, observation.gripper]
        first = state if first is None else first
        began = time.monotonic()
        images = observation.images['head_image'], observation.images['left_image']
        actions = (infer_g05(port, *images, state, task, hz, chunk_steps) if name == 'g05'
                   else infer(port, *images, state, task)[:chunk_steps])
        print(f't={began - start:5.1f}s  state {state}  ->  chunk end {actions[-1]}  (inference {time.monotonic() - began:.2f}s)', flush=True)
        for action in actions:
            tick = time.monotonic()
            if enter_pressed() or tick - start >= duration:
                moved = np.abs(state - first)
                return step, moved
            if execute:
                robot.execute(action)
            step += 1
            time.sleep(max(0, 1 / hz - (time.monotonic() - tick)))
        moved = np.maximum(moved, np.abs(state - first))
    return step, moved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(MODELS), default='pi05')
    parser.add_argument('--config', default=str(REPO / 'colosseum-client/configs/robot.so101.yaml'),
                        help='Colosseum robot config; only its arm and camera settings are used')
    # For g05 this is also how many steps of each chunk its server serves; it is fixed at load time.
    parser.add_argument('--duration', type=float, default=30, help='seconds per trial')
    parser.add_argument('--hz', type=float, default=30, help='action execution rate')
    parser.add_argument('--chunk-steps', type=int, default=30, help='steps executed per inference before replanning')
    parser.add_argument('--dry-run', action='store_true', help='read the robot and print actions without executing them')
    args = parser.parse_args()
    model = MODELS[args.model]
    if not 1 <= args.chunk_steps <= model['chunk']:
        parser.error(f'--chunk-steps must be in [1, {model["chunk"]}] for {args.model}')

    from colosseum_client import RobotClientConfig, make_robot
    config = RobotClientConfig.from_yaml(args.config)
    if config.robot_type != 'so101' or config.test:
        sys.exit('The config must be a real robot_type: so101 configuration')
    log = REPO / 'outputs/colosseum' / f'{args.model}-standalone.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    worker = start_worker(args.model, log, args.chunk_steps)
    try:
        while True:
            print('\nPlace the arm in its rest pose and set up the scene.')
            task = input('Task> ').strip()
            if task.lower() in {'', 'q', 'quit', 'exit'}:
                break
            robot = make_robot(config)  # enables torque at the current pose
            try:
                steps, moved = run_trial(robot, args.model, model['port'], task, duration=args.duration, hz=args.hz,
                                         chunk_steps=args.chunk_steps, execute=not args.dry_run)
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
