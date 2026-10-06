#!/usr/bin/env python3
"""Act as the Client towards a running colosseum-policy-local: prepare one model, send a real observation.

OBS is the directory written by hw_check.py (head_image.npy, left_image.npy, state.npy).
The Policy Server must already be running; it launches the model worker itself.
Expect: "ready", then three (N, 6) plans whose first step is close to the state.

    colosseum-policy-server/.venv/bin/python colosseum/tools/runtime_probe.py \\
        colosseum-policy-server/configs/local-runtime-so101.yaml molmoact2-so101 outputs/colosseum/probe "Pick up the red cube."
"""
import asyncio, sys, time
import numpy as np
from websockets.asyncio.client import connect
from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.local_protocol import decode_control, encode_control
from colosseum_policy_server.local_runtime import RuntimeConfig
from colosseum_policy_server.tensors import tensor_from_numpy, tensor_to_numpy
config_path, name, scratch, task = sys.argv[1:5]
model = RuntimeConfig.from_yaml(config_path).models[name]
state = np.load(f'{scratch}/state.npy').astype(np.float32)
def image(sensor):
    a = np.load(f'{scratch}/{sensor}.npy')
    return pb.Image(sensor_id=sensor, encoding=pb.RAW_RGB, height=a.shape[0], width=a.shape[1], data=a.tobytes())
async def main():
    async with connect('ws://127.0.0.1:8000', compression=None, max_size=32*1024*1024, open_timeout=10, ping_timeout=600) as ws:
        t = time.monotonic()
        await ws.send(encode_control({'type': 'prepare', 'protocol_version': 1, 'run_id': 'probe', 'preparation_id': 'p1',
                                      'model': model.public_spec(), 'robot_type': 'so101'}))
        raw = await asyncio.wait_for(ws.recv(), 900)
        frame = pb.RelayFrame.FromString(raw)
        if frame.type == pb.ERROR:
            e = pb.Error.FromString(frame.payload); print('ERROR', e.code, e.message); return
        print('ready:', decode_control(raw)['type'], f'after {time.monotonic()-t:.1f}s')
        np.set_printoptions(precision=1, suppress=True, linewidth=200)
        for seq in (1, 2, 3):
            obs = pb.Observation(instruction=task, control_step=(seq-1)*30, state={'joint_position': tensor_from_numpy(state[:5]),
                  'gripper_position': tensor_from_numpy(state[5:]), 'cartesian_position': tensor_from_numpy(np.empty(0, np.float32))},
                  sensors=[image('head_image'), image('left_image')])
            t = time.monotonic()
            await ws.send(pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION, session_id='probe', sequence=seq, deadline_ms=120000,
                                        payload=obs.SerializeToString()).SerializeToString())
            reply = pb.RelayFrame.FromString(await asyncio.wait_for(ws.recv(), 300))
            if reply.type != pb.ACTION_PLAN:
                e = pb.Error.FromString(reply.payload); print('ERROR', e.code, e.message); return
            plan = pb.ActionPlan.FromString(reply.payload); actions = tensor_to_numpy(plan.actions)
            print(f'plan {seq}: {actions.shape} hz {plan.control_hz} in {time.monotonic()-t:.2f}s')
        print('state      ', state); print('action[0]  ', actions[0]); print('action[-1] ', actions[-1])
        print('max |first step - state|', np.abs(actions[0]-state).max().round(1), '| max step-to-step', np.abs(np.diff(actions, axis=0)).max().round(2))
        await ws.send(pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id='probe').SerializeToString())
asyncio.run(main())
