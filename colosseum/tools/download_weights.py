#!/usr/bin/env python3
"""Download the SO100/101 Open-track VLA checkpoints at the revisions registered on the Router.

About 95 GB. OpenGalaxea/G05 is gated: the HF account behind HF_TOKEN must have accepted its
terms at https://huggingface.co/OpenGalaxea/G05 first. Re-running resumes.

    HF_TOKEN=hf_... colosseum-client/.venv/bin/python colosseum/tools/download_weights.py checkpoints

RoboColosseum/LingBot-MolmoAct2-So101 is no longer publicly listed on the Hub (checked
2026-10-21). If it fails here, copy checkpoints/LingBot-MolmoAct2-So101 from a machine that has it
(colosseum/README.md, section 4); the other models are still downloaded.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
# (repo, revision, folder, allow_patterns, optional)
MODELS = [
    ('allenai/MolmoAct2-SO100_101', '152569fe57914d97be91055800035f54e250d009', 'MolmoAct2-SO100_101', None, False),
    ('hqfang/pi05-so100_101', 'd3204e03ae84d232e2493a933fd460515904eb58', 'pi05-so100_101', None, False),
    ('OpenGalaxea/G05', 'e312be81e90c56a55bcb26b57429bd39a335b449', 'G05',
     ['g05-so101/*', 'g05-so101/.hydra/*', 'g05-so101/checkpoints/*', 'action_tokenizer.pt',
      'qwen3_5_2b_base_processor/*', 'README.md', 'licenses/*'], False),
    ('hqfang/groot-so100_101', 'f5e17af8948a647e4c23186a27927bf73d6d784e', 'groot-so100_101', None, False),
    # Base snapshot the GR00T policy loader initializes from before loading the fine-tuned weights.
    ('nvidia/GR00T-N1.7-3B', '2fc962b973bccdd5d8ce4f67cc63b264d6886495', 'GR00T-N1.7-3B', None, False),
    ('RoboColosseum/LingBot-MolmoAct2-So101', 'd360710b21beec2804e42ac2d6035022c2a4c1a8',
     'LingBot-MolmoAct2-So101', None, True),
    # LingBot reads only the Qwen3-VL config, tokenizer and processor; the 8 GB of weights are not needed.
    ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17', 'Qwen3-VL-4B-Instruct',
     ['*.json', '*.txt', '*.jinja', 'README.md'], False),
]
missing = []
for repo, revision, folder, patterns, optional in MODELS:
    print(f'== {repo}@{revision[:8]} -> {root / folder}', flush=True)
    try:
        snapshot_download(repo, revision=revision, local_dir=str(root / folder), allow_patterns=patterns)
    except Exception as exc:
        if not optional:
            raise
        print(f'   FAILED ({type(exc).__name__}); copy {root / folder} from another machine', flush=True)
        missing.append(folder)
print('done' if not missing else f'done, missing: {", ".join(missing)} (see colosseum/README.md section 4)')
