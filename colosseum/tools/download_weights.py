#!/usr/bin/env python3
"""Download the three SO100/101 Open-track checkpoints at the revisions registered on the Router.

About 50 GB. OpenGalaxea/G05 is gated: the HF account behind HF_TOKEN must have accepted its
terms at https://huggingface.co/OpenGalaxea/G05 first. Re-running resumes.

    HF_TOKEN=hf_... colosseum-client/.venv/bin/python colosseum/tools/download_weights.py checkpoints
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
MODELS = [
    ('allenai/MolmoAct2-SO100_101', '152569fe57914d97be91055800035f54e250d009', 'MolmoAct2-SO100_101', None),
    ('hqfang/pi05-so100_101', 'd3204e03ae84d232e2493a933fd460515904eb58', 'pi05-so100_101', None),
    ('OpenGalaxea/G05', 'e312be81e90c56a55bcb26b57429bd39a335b449', 'G05',
     ['g05-so101/*', 'g05-so101/.hydra/*', 'g05-so101/checkpoints/*', 'action_tokenizer.pt',
      'qwen3_5_2b_base_processor/*', 'README.md', 'licenses/*']),
]
for repo, revision, folder, patterns in MODELS:
    print(f'== {repo}@{revision[:8]} -> {root / folder}', flush=True)
    snapshot_download(repo, revision=revision, local_dir=str(root / folder), allow_patterns=patterns)
print('done')
