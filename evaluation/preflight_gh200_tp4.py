"""Single-GPU preflight of every TP4 validation check that needs no peer rank.

One process builds vLLM's TP group, then plays each of the four ranks in turn:
original factors bit for bit and DiffKV attention at TP4-local geometry.
Cross-rank gathers stay in `evaluation.validate_gh200_tp4`.
"""

import argparse
import json
from pathlib import Path
import socket

import torch
import torch.distributed as dist

from basisserve.core.qwen3_8b_vllm_c1 import file_sha256
from basisserve.core.tp4_c1_factors import TP_SIZE, load_manifest
from evaluation.run_gh200_tp4_smoke import FACTORS
from evaluation.validate_gh200_tp4 import CASES, LAYERS, local_attention, original_factors, tp_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert tp_group(1, 0).world_size == 1
    print('TP GROUP READY', flush=True)
    records = []
    for width, root in FACTORS.items():
        root = root.resolve()
        manifest = load_manifest(root, file_sha256(root / 'results.json'))
        assert manifest['fit_config']['cache_rank_per_head'] == width
        for layer in LAYERS:
            for rank in range(TP_SIZE):
                encoder, _ = original_factors(root, manifest, layer, rank, device)
                for mode, batch, query_length in CASES:
                    _, _, error = local_attention(encoder, rank, mode, batch, query_length, device)
                    records.append(dict(value_rank=width, layer=layer, tp_rank=rank, mode=mode, batch=batch,
                                        attention_relative_l2=error, factors_identical=True, keys_unchanged=True))
            print('PASS', f'v{width}', 'layer', layer, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='complete', hostname=socket.gethostname(),
                                           checks=records), indent=2) + '\n')
    print('PREFLIGHT COMPLETE', args.output, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
