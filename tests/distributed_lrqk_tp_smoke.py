"""Tensor-parallel LRQK factory (basisserve/core/lrqk_tp.py: Keys on host, Values on GPU) against the unchanged
upstream factory (Keys and Values on host). The GPU hit cache must hold the same Keys and Values either way.

Every rank builds the same synthetic full-head prefill and decode tensors. The reference is upstream
LightAttentionIndicesFactory over all 32 query heads with upstream's own cast_lrqk_prefill/decode (rank 0).
Each rank runs the TP factory on its local heads; per decode step the local hit indices, GPU hit cache and B
factors are gathered and compared with the reference. Run with torchrun --nproc-per-node 1, 4 or 8.
"""

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

from basisserve.core import lrqk_tp


def gather(tensor, dim):
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor.contiguous())
    return torch.cat(parts, dim=dim)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=72)  # crosses one lite-token flush (64)
    args = parser.parse_args()
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    world = dist.get_world_size()
    module = lrqk_tp.upstream()
    factory_class = lrqk_tp.make_factory_class()
    generator = torch.Generator(device=device).manual_seed(4242)
    draw = lambda *shape: (torch.randn(*shape, device=device, generator=generator) * 0.5).to(torch.bfloat16)
    length, steps = args.length, args.steps
    query, key, value = draw(1, 32, length, 128), draw(1, 8, length, 128), draw(1, 8, length, 128)
    decode = [(draw(1, 32, 1, 128), draw(1, 8, 1, 128), draw(1, 8, 1, 128)) for _ in range(steps)]
    kv, heads = 8 // world, 32 // world
    own_q, own_kv = slice(rank * heads, (rank + 1) * heads), slice(rank * kv, (rank + 1) * kv)

    reference = None
    if rank == 0:
        module.cast_lrqk_decode = module.upstream_cast_lrqk_decode
        reference = module.LightAttentionIndicesFactory(
            num_lite_tokens=lrqk_tp.CONFIG["lite"], attn_topk=lrqk_tp.CONFIG["topk"], num_key_value_groups=4,
            r=lrqk_tp.CONFIG["rank"], max_iter=lrqk_tp.CONFIG["iterations"], tol=lrqk_tp.CONFIG["tolerance"],
            capacity=length + steps + 2 * lrqk_tp.CONFIG["lite"], init_aq_ak_method=module.InitAQAK.randn)
        torch.manual_seed(lrqk_tp.SEED)
        reference.prefill(query, key, value)
        reference_steps = []
        for q, k, v in decode:
            key_gpu, value_gpu = reference.decode(q, k, v)
            reference_steps.append((reference.hit_indices.clone(), key_gpu.clone(), value_gpu.clone(),
                                    reference.B_Q.clone(), reference.B_K.clone()))
        module.cast_lrqk_decode = lrqk_tp.cast_decode
    dist.barrier()

    tp = factory_class(batch=1, query_heads=heads, kv_heads=kv, value_dim=128, capacity=length + steps,
                       head_start=rank * heads, device=device)
    tp._first_layer = True
    tp.prefill_row(0, query[:, own_q].contiguous(), key[:, own_kv].contiguous(), value[:, own_kv].contiguous())
    rows = []
    for step, (q, k, v) in enumerate(decode):
        key_gpu, value_gpu = tp.decode(q[:, own_q].contiguous(), k[:, own_kv].contiguous(), v[:, own_kv].contiguous())
        hits, keys, values = gather(tp.hit_indices, 1), gather(key_gpu, 1), gather(value_gpu, 1)
        b_q, b_k = gather(tp.B_Q, 1), gather(tp.B_K, 1)
        if rank == 0:
            expected = reference_steps[step]
            left = expected[0].reshape(32, -1).sort(-1).values
            right = hits.reshape(32, -1).sort(-1).values
            overlap = float(min(len(set(a.tolist()) & set(b.tolist())) / a.numel() for a, b in zip(left, right)))
            # Slot order may differ only where the hit sets differ; compare caches through their hit order.
            rows.append(dict(step=step, identical_hits=bool(torch.equal(left, right)), minimum_hit_overlap=overlap,
                             key_cache_equal=bool(torch.equal(expected[1], keys)),
                             value_cache_equal=bool(torch.equal(expected[2], values)),
                             b_q_max_abs=float((expected[3].float() - b_q.float()).abs().max()),
                             b_k_max_abs=float((expected[4].float() - b_k.float()).abs().max())))
    if rank == 0:
        result = dict(world_size=world, length=length, steps=steps, config=lrqk_tp.CONFIG,
                      all_hits_identical=all(r["identical_hits"] for r in rows),
                      minimum_hit_overlap=min(r["minimum_hit_overlap"] for r in rows),
                      all_caches_equal=all(r["key_cache_equal"] and r["value_cache_equal"] for r in rows),
                      maximum_b_difference=max(max(r["b_q_max_abs"], r["b_k_max_abs"]) for r in rows),
                      steps_detail=rows, reference="upstream LightAttentionIndicesFactory + cast_lrqk_* (eager)",
                      status="complete")
        result["status"] = "complete" if result["minimum_hit_overlap"] >= 0.99 else "mismatch"
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "steps_detail"}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
