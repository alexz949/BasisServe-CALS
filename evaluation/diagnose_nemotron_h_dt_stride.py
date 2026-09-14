"""Isolate native dt scan addressing at Nemotron's actual projection stride."""
import argparse
import torch
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=('contiguous', 'strided'), required=True)
    args = parser.parse_args()
    stride = 16384 + 20480 + 256
    for length in (32768, 65389):
        dt = torch.empty_strided((1, length, 256),
            (length * stride, stride, 1), device='cuda', dtype=torch.bfloat16)
        dt.fill_(0.01)
        if args.layout == 'contiguous':
            dt = dt.contiguous()
        A = -torch.ones(256, device='cuda', dtype=torch.float32)
        print('SCAN', args.layout, length, 'stride', dt.stride(),
            'maximum_offset', (length - 1) * dt.stride(1) + 255, flush=True)
        cumulative, transformed = _chunk_cumsum_fwd(dt, A, 128, dt_softplus=True)
        torch.cuda.synchronize()
        assert torch.isfinite(cumulative).all() and torch.isfinite(transformed).all()
        print('PASS', args.layout, length, flush=True)
        del dt, A, cumulative, transformed


if __name__ == '__main__':
    main()
