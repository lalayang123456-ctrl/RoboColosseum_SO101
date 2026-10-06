#!/usr/bin/env python3
"""Open-loop check through the running Policy Server: predicted chunks vs dataset ground truth.

Dataset samples are in the legacy SO100 frame. They are converted to the Client's
current frame before sending, exactly what the Client would report for that pose,
and returned actions are converted back for comparison.

REF is the directory written by fetch_reference_samples.py; OUT receives one .npz per model.
Expected on a correct setup: model MAE about 2.5-4 deg, clearly below the hold-pose MAE (9-13).

    colosseum-policy-server/.venv/bin/python colosseum/tools/openloop_check.py \\
        colosseum-policy-server/configs/local-runtime-so101.yaml outputs/colosseum/ref outputs/colosseum \\
        molmoact2-so101 pi05-so101 g05-so101
"""
import asyncio, json, sys, time
import numpy as np
from websockets.asyncio.client import connect
from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.local_protocol import decode_control, encode_control
from colosseum_policy_server.local_runtime import RuntimeConfig
from colosseum_policy_server.model_adapters.so101_vla import LEGACY_OFFSETS as OFF, LEGACY_SIGNS as SGN
from colosseum_policy_server.tensors import tensor_from_numpy, tensor_to_numpy
config_path, ref, out = sys.argv[1:4]
names = sys.argv[4:]
models = RuntimeConfig.from_yaml(config_path).models
samples = json.load(open(f'{ref}/samples.json'))
np.set_printoptions(precision=1, suppress=True, linewidth=220)
def image(sensor, array):
    return pb.Image(sensor_id=sensor, encoding=pb.RAW_RGB, height=array.shape[0], width=array.shape[1], data=array.tobytes())
async def run(name):
    model = models[name]
    async with connect('ws://127.0.0.1:8000', compression=None, max_size=32*1024*1024, open_timeout=10, ping_timeout=900) as ws:
        t = time.monotonic()
        await ws.send(encode_control({'type': 'prepare', 'protocol_version': 1, 'run_id': 'check', 'preparation_id': 'p',
                                      'model': model.public_spec(), 'robot_type': 'so101'}))
        raw = await asyncio.wait_for(ws.recv(), 900)
        if pb.RelayFrame.FromString(raw).type == pb.ERROR:
            print(name, 'PREPARE ERROR', pb.Error.FromString(pb.RelayFrame.FromString(raw).payload).message); return
        print(f'\n===== {name}: ready in {time.monotonic()-t:.0f}s (max_horizon {model.max_horizon})')
        rows, results = [], {}
        for seq, sample in enumerate(samples, 1):
            k = sample['key']; legacy = np.load(f'{ref}/{k}_state.npy'); truth = np.load(f'{ref}/{k}_actions.npy')
            current = ((legacy - OFF) * SGN).astype(np.float32)
            obs = pb.Observation(instruction=sample['task'], control_step=0, state={
                'joint_position': tensor_from_numpy(current[:5]), 'gripper_position': tensor_from_numpy(current[5:]),
                'cartesian_position': tensor_from_numpy(np.empty(0, np.float32))},
                sensors=[image('head_image', np.load(f'{ref}/{k}_head.npy')), image('left_image', np.load(f'{ref}/{k}_wrist.npy'))])
            await ws.send(pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION, session_id='check', sequence=seq, deadline_ms=120000,
                                        payload=obs.SerializeToString()).SerializeToString())
            reply = pb.RelayFrame.FromString(await asyncio.wait_for(ws.recv(), 300))
            if reply.type != pb.ACTION_PLAN:
                print(k, 'INFERENCE ERROR', pb.Error.FromString(reply.payload).message); return
            predicted = SGN * tensor_to_numpy(pb.ActionPlan.FromString(reply.payload).actions) + OFF
            n = min(len(predicted), len(truth)); predicted, truth = predicted[:n], truth[:n]
            model_error = np.abs(predicted - truth).mean(0); hold_error = np.abs(legacy - truth).mean(0)
            rows.append((k, n, model_error, hold_error, np.abs(truth[-1] - legacy).max()))
            results[k] = predicted
        np.savez(f'{out}/openloop_{name}.npz', **results)
        print('sample        steps | model MAE per joint (deg; gripper 0-100)     mean | hold-pose MAE mean | GT motion over chunk')
        for k, n, m, h, motion in rows:
            print(f'{k:13s} {n:5d} | {m}  {m.mean():5.1f} | {h.mean():18.1f} | {motion:6.1f}')
        M = np.array([r[2] for r in rows]); H = np.array([r[3] for r in rows])
        moving = np.array([r[4] for r in rows]) > 5
        print(f'ALL     model MAE {M.mean():.1f}  vs hold-pose {H.mean():.1f}   per joint {M.mean(0)}')
        print(f'MOVING  model MAE {M[moving].mean():.1f}  vs hold-pose {H[moving].mean():.1f}   ({moving.sum()} samples where the arm moves > 5 deg)')
        await ws.send(pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id='check').SerializeToString())
async def main():
    for name in names:
        await run(name)
        await asyncio.sleep(1)
asyncio.run(main())
