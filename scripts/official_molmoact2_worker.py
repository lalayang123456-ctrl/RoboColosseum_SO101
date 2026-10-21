#!/usr/bin/env python3
"""MolmoAct2-SO100_101 served exactly as its model card calls it.

Same HTTP /act protocol as colosseum-policy-model, but the predict_action call is the one from
https://huggingface.co/allenai/MolmoAct2-SO100_101: num_steps=10, continuous actions, no depth
reasoning, normalize_language=True, enable_cuda_graph=True, images [external, wrist].
Precision follows the card too: float32 (what the authors report results with, ~24-26 GB), or
bfloat16 weights under torch.autocast when that does not fit.

Run in the MolmoAct2 environment:
    envs/.venv/bin/python scripts/official_molmoact2_worker.py --checkpoint checkpoints/MolmoAct2-SO100_101 --port 9201
"""
import argparse
import logging
import os
from http.server import HTTPServer
from pathlib import Path

import numpy as np

log = logging.getLogger('official_molmoact2')


class Runtime:
    action_dim = 6
    image_keys = ('external_cam', 'wrist_cam')

    def __init__(self, checkpoint, dtype, cuda_graph, warmup):
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.Image, self.dtype, self.cuda_graph = torch, Image, dtype, cuda_graph
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(checkpoint), trust_remote_code=True, dtype=getattr(torch, dtype)).to('cuda').eval()
        blank = np.zeros((480, 640, 3), np.uint8)
        for index in range(warmup):  # the card: the first calls are slow while CUDA graphs are captured
            self.infer({'external_cam': blank, 'wrist_cam': blank, 'instruction': 'warm up',
                        'state': np.array([0, 189, 181, 60, 0, 1], np.float32)})
            log.info('warm-up call %d/%d done', index + 1, warmup)

    def infer(self, request):
        torch = self.torch
        state = np.asarray(request['state'], dtype=np.float32).reshape(-1)
        images = [self.Image.fromarray(np.asarray(request[key], dtype=np.uint8)) for key in self.image_keys]
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if self.dtype == 'bfloat16' else torch.no_grad()
        with torch.inference_mode(), precision:
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=request['instruction'],
                state=state,
                norm_tag='so100_so101_molmoact2',
                inference_action_mode='continuous',
                enable_depth_reasoning=False,
                num_steps=10,
                normalize_language=True,
                enable_cuda_graph=self.cuda_graph,
            )
        actions = out.actions.detach().float().cpu().numpy()
        if actions.ndim == 3:
            actions = actions[0]
        return {'actions': actions[:, :self.action_dim].astype(np.float32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--no-cuda-graph', action='store_true')
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    logging.basicConfig(level=logging.INFO)
    from colosseum_policy_server.model_service import http_handler
    runtime = Runtime(args.checkpoint.resolve(), args.dtype, not args.no_cuda_graph, args.warmup)
    import torch
    log.info('ready: dtype=%s cuda_graph=%s peak GPU memory %.1f GB', args.dtype, not args.no_cuda_graph,
             torch.cuda.max_memory_allocated() / 2**30)
    with HTTPServer(('127.0.0.1', args.port), http_handler(runtime)) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
