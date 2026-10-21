#!/usr/bin/env python3
"""Run a generalist SO100/101 checkpoint on the real SO-101 with its authors' own inference settings.

Standalone: no Router, no recording, no scoring. Unlike run_open_model.py there is nothing to
tune here; every model runs the way its model card or upstream deployment code runs it:

  molmoact2  model card of allenai/MolmoAct2-SO100_101
             predict_action(num_steps=10, continuous, no depth reasoning, normalize_language=True,
             enable_cuda_graph=True, norm_tag=so100_so101_molmoact2), 3 warm-up calls.
             Precision: the card reports results in float32 (24-26 GB) and gives bfloat16 weights
             under torch.autocast for smaller GPUs. float32 runs out of memory on a 24 GB card, so
             bfloat16 is the default; pass --molmo-float32 on a larger GPU.
             Execution: the whole 30-step chunk at 30 Hz (norm_stats: action_horizon = n_action_steps = 30).

  pi05       model card of hqfang/pi05-so100_101
             float32 weights, saved pre/post processors, bundled tokenizer, compile off,
             10 denoising steps, external image in camera_0, wrist image in camera_1, the other
             slots zero images flagged as padding.
             Execution: the whole 50-step chunk at 30 Hz (n_action_steps = 50, as select_action does).

  g05        GalaxeaVLA experiments/so100 (start_server.sh + so100_policy_client.py)
             server: --action_steps 32, eval_embodiment=so100, bfloat16 weights, torch.compile,
             sdpa attention; client: 15 Hz, frequency 15, at most 10 deg per tick, exterior +
             wrist_right images with a zero wrist_left, and the arm first moves to the upstream
             home pose (training-data mean) before the model takes over.

Two things are ours because the arm requires them, not the models:
  - state/actions are converted between this arm's LeRobot calibration frame and the earlier
    SO100 frame all three checkpoints were trained in (model = sign * arm + offset);
  - commands pass through the SO101 driver (joint-limit clamp, return to the start pose on exit).

    colosseum-client/.venv/bin/python scripts/run_official.py --model pi05
    colosseum-client/.venv/bin/python scripts/run_official.py --model molmoact2
    colosseum-client/.venv/bin/python scripts/run_official.py --model g05

Per trial: place the arm in its rest pose, type the task + Enter to start, Enter again to stop
early. The arm then returns to the pose it started from and releases torque. Empty task or q quits.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_open_model import REPO, die_with_parent, listening, prepare_start, run_trial  # noqa: E402

SERVICE = ['-m', 'colosseum_policy_server.model_service']
OFFICIAL = {
    'molmoact2': dict(
        port=9201, python=REPO / 'envs/.venv/bin/python',
        command=[str(REPO / 'scripts/official_molmoact2_worker.py'),
                 '--checkpoint', str(REPO / 'checkpoints/MolmoAct2-SO100_101'), '--dtype', 'bfloat16'],
        hz=30, chunk_steps=30, max_step_deg=0, home='rest',
        summary='bfloat16 under autocast, CUDA graph on, 10 denoising steps; 30 Hz, 30 steps per chunk'),
    'pi05': dict(
        port=9212, python=REPO / 'envs/pi05_so101/.venv/bin/python',
        command=[*SERVICE, 'pi05_lerobot', '--robot-type', 'so101', '--device', 'cuda', '--num-steps', '10',
                 '--checkpoint', str(REPO / 'checkpoints/pi05-so100_101')],
        hz=30, chunk_steps=50, max_step_deg=0, home='rest',
        summary='float32, saved processors, 10 denoising steps; 30 Hz, 50 steps per chunk'),
    'g05': dict(
        port=9204, python=REPO / 'vendor/GalaxeaVLA/.venv/bin/python',
        command=[*SERVICE, 'g05', '--robot-type', 'so101', '--source-root', str(REPO / 'vendor/GalaxeaVLA'),
                 '--device', 'cuda', '--action-steps', '32',
                 '--checkpoint', str(REPO / 'checkpoints/G05/g05-so101/checkpoints/model_state_dict.pt'),
                 '--g05-override', 'model.model_weights_to_bf16=true',
                 '--g05-override', 'model.use_torch_compile=true',
                 '--g05-override', 'model.model_arch.attn_implementation=sdpa'],
        hz=15, chunk_steps=32, max_step_deg=10, home='mean', g05_frequency=15,
        summary='bfloat16, torch.compile, sdpa; 15 Hz, frequency 15, 32 steps per chunk, '
                'max 10 deg per tick, starts from the upstream home pose'),
}


def start_worker(name, command, log):
    model = OFFICIAL[name]
    if listening(model['port']):
        sys.exit(f'Port {model["port"]} is already in use by another model worker. Stop it first (the Colosseum '
                 'Policy Server or another run_* script), so this run is guaranteed to use the official settings.')
    if subprocess.run(['pgrep', '-f', '[c]olosseum-policy-local'], capture_output=True).returncode == 0:
        sys.exit('The Colosseum Policy Server is running and may hold the GPU. Stop it (Ctrl-C in its terminal) first.')
    print(f'Loading {name} (log: {log}) ...', flush=True)
    process = subprocess.Popen([str(model['python']), *command, '--port', str(model['port'])],
                               stdout=open(log, 'wb'), stderr=subprocess.STDOUT, start_new_session=True,
                               preexec_fn=die_with_parent)
    started = time.monotonic()
    while not listening(model['port']):
        if process.poll() is not None:
            sys.exit(f'Model worker exited with status {process.returncode}; see {log}')
        if time.monotonic() - started > 900:
            process.terminate()
            sys.exit(f'Model worker did not start within 900 s; see {log}')
        time.sleep(1)
    print(f'Model ready in {time.monotonic() - started:.0f} s.')
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(OFFICIAL), required=True)
    parser.add_argument('--config', default=str(REPO / 'colosseum-client/configs/robot.so101.yaml'),
                        help='Colosseum robot config; only its arm and camera settings are used')
    parser.add_argument('--max-steps', type=int, default=800,
                        help='stop a trial after this many executed steps (default 800, like the Colosseum Open track)')
    parser.add_argument('--duration', type=float, default=0,
                        help='optional wall-clock limit per trial in seconds; 0 = no time limit (default)')
    parser.add_argument('--molmo-float32', action='store_true',
                        help="MolmoAct2 in float32, the precision of the authors' reported results; needs more than 24 GB")
    parser.add_argument('--dry-run', action='store_true',
                        help='read the robot and print chunks without moving (G05 then starts from the rest pose)')
    args = parser.parse_args()
    model = OFFICIAL[args.model]
    command, summary = list(model['command']), model['summary']
    if args.model == 'molmoact2' and args.molmo_float32:
        command[command.index('--dtype') + 1] = 'float32'
        summary = summary.replace('bfloat16 under autocast', 'float32')
    print(f'{args.model}, official settings: {summary}')

    from colosseum_client import RobotClientConfig, make_robot
    config = RobotClientConfig.from_yaml(args.config)
    if config.robot_type != 'so101' or config.test:
        sys.exit('The config must be a real robot_type: so101 configuration')
    log = REPO / 'outputs/colosseum' / f'{args.model}-official.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    worker = start_worker(args.model, command, log)
    try:
        while True:
            print('\nPlace the arm in its rest pose and set up the scene.')
            task = input('Task> ').strip()
            if task.lower() in {'', 'q', 'quit', 'exit'}:
                break
            robot = make_robot(config)  # enables torque at the current pose
            try:
                prepare_start(robot, model['home'], not args.dry_run)
                steps, moved = run_trial(robot, args.model, model['port'], task, duration=args.duration,
                                         hz=model['hz'], chunk_steps=model['chunk_steps'], execute=not args.dry_run,
                                         max_step_deg=model['max_step_deg'],
                                         g05_frequency=model.get('g05_frequency', 15), max_steps=args.max_steps)
                np.set_printoptions(precision=1, suppress=True, linewidth=200)
                print(f'Stopped after {steps} steps. Largest joint excursion from the first observed pose: {moved}')
            except KeyboardInterrupt:
                print('\nInterrupted.')
            finally:
                print('Returning to the start pose and releasing torque...', flush=True)
                robot.close()
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        worker.terminate()
        worker.wait()


if __name__ == '__main__':
    main()
