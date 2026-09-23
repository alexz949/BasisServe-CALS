"""Compare TP8 construction with unchanged upstream ShadowKV get_svd."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basisserve.core.shadowkv_tp8 import build_global_factors
from evaluation import official_shadowkv_cpu


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    assert dist.get_world_size() == 8
    official_shadowkv_cpu.ROOT = args.upstream.resolve()
    cache_class = official_shadowkv_cpu.load_cache_class()
    results = []
    cases = [(layer, 1, 512) for layer in range(8)] + [(3, 2, 2048)]
    for layer, batch, length in cases:
        torch.manual_seed(1700 + rank)
        local = torch.randn(batch, 1, length, 128, device=device, dtype=torch.bfloat16)
        shared_u = torch.empty(batch, length, 160, device=device, dtype=torch.bfloat16)
        local_sv = torch.empty(batch, 1, 128, 160, device=device, dtype=torch.bfloat16)
        record = build_global_factors(local, layer, shared_u, local_sv)
        reference_u = shared_u.clone()
        dist.broadcast(reference_u, src=0)
        assert torch.equal(shared_u, reference_u)
        # The all-gather below belongs only to this small reference test.
        all_keys = [torch.empty_like(local) for _ in range(8)]
        all_sv = [torch.empty_like(local_sv) for _ in range(8)]
        dist.all_gather(all_keys, local)
        dist.all_gather(all_sv, local_sv)
        if rank == layer % 8:
            global_key = torch.cat(all_keys, dim=1)
            global_sv = torch.cat(all_sv, dim=1)
            for request in range(batch):
                reference = cache_class.__new__(cache_class)
                reference.batch_size = 1
                reference.num_key_value_heads = 8
                reference.head_dim = 128
                reference.rank = 160
                reference.num_layers = 1
                reference.prefilled_batch = 0
                reference.dtype = torch.bfloat16
                reference.get_svd(global_key[request:request + 1], 0)
                expected_u = reference.U[0, 0].to(device)
                expected_sv = reference.SV[0, 0].to(device)
                # Allow the arbitrary SVD sign, but verify every SV shard.
                signs = (shared_u[request].float() * expected_u.float()).sum(0).sign()
                signs[signs == 0] = 1
                torch.testing.assert_close(shared_u[request].float() * signs,
                                           expected_u.float(), rtol=0.03, atol=0.003)
                torch.testing.assert_close(global_sv[request].float() * signs,
                                           expected_sv.float(), rtol=0.03, atol=0.03)
                actual = shared_u[request].float() @ global_sv[request].permute(2, 0, 1).reshape(160, 1024).float()
                expected = expected_u.float() @ expected_sv.permute(2, 0, 1).reshape(160, 1024).float()
                relative_error = float((actual - expected).norm() / expected.norm())
                assert relative_error < 0.01
                print(json.dumps({"layer": layer, "request": request,
                                  "reconstruction_relative_error": relative_error}), flush=True)
        dist.barrier()
        results.append({"layer": layer, "batch": batch, "length": length, **record})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"factors_rank{rank}.json").write_text(json.dumps({
        "status": "complete", "rank": rank, "cases": results,
        "upstream": str(args.upstream.resolve()), "pytorch": torch.__version__,
        "command": sys.argv, "tf32": torch.backends.cuda.matmul.allow_tf32,
    }, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
