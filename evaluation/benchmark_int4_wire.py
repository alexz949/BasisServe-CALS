"""Synthetic, real multi-GPU communication smoke; not a model PPL test."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from basisserve.kernels.int4_wire import (
    pack_int4, unpack_int4, all_gather_int4, communication_bytes,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', default='results/q3-int4-wire/smoke.json')
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    from basisserve.core.qwen3_8b_tp4_decode import Qwen3TP4C1DecodeAttention
    results = []
    for vrank in (64, 96):
        width = 32 * vrank // world
        for tokens in (1, 16, 128):
            torch.manual_seed(123 + rank)
            x = torch.randn(tokens, width, device='cuda', dtype=torch.bfloat16)
            all_x = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(all_x, x)
            reference = torch.cat([unpack_int4(pack_int4(a), width) for a in all_x], -1)
            output = all_gather_int4(x)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            torch.manual_seed(456)
            decoder = torch.randn(world * width, 4096, device='cuda', dtype=torch.bfloat16) / (world * width)**0.5
            expected = reference @ decoder
            stub = SimpleNamespace(wire_dtype='int4', local_wire_width=width,
                                   global_decoder=decoder, _update_wire_amax=lambda _: None)
            for local in (x.T.contiguous(), x.reshape(tokens, 8, 1, vrank)):
                actual = Qwen3TP4C1DecodeAttention._project_output(stub, local)
                torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)
            prefill = x.reshape(1, tokens, 8, vrank).transpose(1, 2).contiguous()
            actual = Qwen3TP4C1DecodeAttention._project_output(stub, prefill)
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
            dense = torch.cat(all_x, -1) @ decoder
            rel_mse = ((expected.float()-dense.float()).square().sum()/dense.float().square().sum()).item()
            baseline = x @ decoder[rank * width:(rank + 1) * width]
            buffer = torch.empty_like(baseline)
            packet = pack_int4(x)
            received = torch.empty((world*tokens, packet.shape[1]), device='cuda', dtype=torch.uint8)
            bf16_received = torch.empty((world*tokens, width), device='cuda', dtype=x.dtype)
            def dense_ar():
                buffer.copy_(baseline)
                dist.all_reduce(buffer)
            def timed(fn):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    fn()
                end.record()
                end.synchronize()
                ms = torch.tensor(start.elapsed_time(end)/20, device='cuda')
                dist.all_reduce(ms, op=dist.ReduceOp.MAX)
                return ms.item()
            record = dict(rank=vrank, tokens=tokens, tp=world,
                          synthetic_decoder_rel_mse=rel_mse,
                          int4_pack_gather_unpack_ms=timed(lambda: all_gather_int4(x)),
                          int4_collective_only_ms=timed(lambda: dist.all_gather_into_tensor(received, packet)),
                          bf16_latent_collective_only_ms=timed(lambda: dist.all_gather_into_tensor(bf16_received, x)),
                          dense_copy_allreduce_ms=timed(dense_ar),
                          **communication_bytes(tokens, width, world))
            results.append(record)
            if rank == 0:
                print(json.dumps(record), flush=True)
    if rank == 0:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(synthetic=True, quantizer='dynamic symmetric INT4; FP32 row scale; no outlier channel', results=results), indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
