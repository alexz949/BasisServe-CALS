"""Compare preconditioners on the same captured, gated linear subproblem."""
import json
import time
from pathlib import Path

import torch

from basisserve.core.qwen35_gated_v_als import GatedVCapture, GatedVBlock, initialize_gated_v, gated_v_update
from evaluation.qwen35_hybrid_common import atomic_save


def main():
    torch.set_num_threads(2)
    root = Path('results/q35_hybrid/capture')
    saved = torch.load(root / 'fit_000.pt', map_location='cpu', weights_only=True, mmap=True)['layers'][3]
    weight = torch.load(root / 'weights.pt', weights_only=True)[3]
    capture = GatedVCapture(saved['z'].cuda(), saved['gate'].cuda(), weight.cuda(), saved['target'].cuda(), torch.arange(16) // 4)
    e, r = initialize_gated_v(capture, 64, device='cuda')
    r, rd = gated_v_update(GatedVBlock(capture, e, block='decoder', chunk_rows=2048), r)
    print(json.dumps({'decoder': rd}), flush=True)
    records = {}
    for mode in ('jacobi', 'separable'):
        op = GatedVBlock(capture, r, block='encoder', chunk_rows=2048)
        started = time.monotonic()
        _, record = gated_v_update(op, e, encoder_preconditioner=mode)
        record['seconds'] = time.monotonic() - started
        records[mode] = record
        print(json.dumps({mode: record}), flush=True)
    atomic_save('results/q35_hybrid/encoder_pcg_verified.json', records)


if __name__ == '__main__':
    main()
