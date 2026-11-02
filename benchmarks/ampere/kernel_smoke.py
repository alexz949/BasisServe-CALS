"""Single GPU correctness smoke for the Ampere joint routing pipeline."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from basisserve.kernels.mapped_host_paged_attention import (
    _load_extension,
    append_mapped_host_key,
    mapped_host_bf16_empty,
    mapped_host_device_pointer,
)
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from benchmarks.ampere.two_stage import compile_fine
from benchmarks.system.fused_candidates import candidates
from benchmarks.system.two_stage_router import Metadata


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "basisserve/kernels/csrc/conditional_router_page32.cu",
    "basisserve/kernels/csrc/fused_decode_postprocess.cu",
    "basisserve/kernels/csrc/persistent_key_slots.cu",
    "basisserve/kernels/slot_indexed_attention.py",
    "benchmarks/system/two_stage_router.py",
    "benchmarks/system/fused_candidates.py",
    "benchmarks/ampere/two_stage.py",
    "benchmarks/ampere/kernel_smoke.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extension(name, source):
    path = ROOT / source
    major, minor = torch.cuda.get_device_capability()
    return load(
        name=f"ampere_sm{major}{minor}_{name}_{sha256(path)[:12]}",
        verbose=os.environ.get("BASISSERVE_VERBOSE_BUILD", "0") == "1",
        sources=[str(path)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
    )


@torch.inference_mode()
def check(length, full, fine, postprocess, slots):
    device = "cuda"
    dtype = torch.bfloat16
    historical = length - 64
    q = torch.randn(1, 4, 1, 128, device=device, dtype=dtype)
    key = torch.randn(1, 1, length, 128, device=device, dtype=dtype)
    value = torch.randn(1, 1, length, 96, device=device, dtype=dtype)
    base = torch.randn(1, 1, historical, 16, device=device, dtype=dtype) * 0.1
    residual = torch.randn_like(base) * 0.1
    right = torch.randn(1, 16, 128, device=device, dtype=dtype) * 0.1
    bias = torch.randn(1, 128, device=device, dtype=dtype) * 0.1
    residual_query = torch.randn(4, 128, 16, device=device, dtype=dtype) * 0.1
    angle = torch.randn(historical, 64, device=device)
    cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
    meta = Metadata(key, length + 1)
    candidates_ids = candidates(meta.scores(q), historical)
    n = candidates_ids.shape[-1]
    query_code = torch.empty(1, 1, 4, 16, device=device, dtype=dtype)
    candidate_scores = torch.empty(1, 1, 4, n, device=device)
    full_scores = torch.empty(1, 1, 4, math.ceil(historical / 32), device=device)
    inputs = (q, base, residual, right, bias, residual_query, cos, sin, query_code)
    fine.conditional_router_page_lse(
        *inputs, candidate_scores, 128**-0.5, False, candidates_ids
    )
    full.conditional_router_page_lse(
        *inputs, full_scores, 128**-0.5, False
    )
    expected = full_scores.gather(
        -1, candidates_ids[:, :, None].expand(-1, -1, 4, -1).clamp_max(full_scores.shape[-1] - 1)
    )
    valid = candidates_ids < full_scores.shape[-1]
    torch.testing.assert_close(
        candidate_scores.masked_select(valid[:, :, None]),
        expected.masked_select(valid[:, :, None]),
        rtol=0.005,
        atol=0.02,
    )
    assert bool(torch.isneginf(candidate_scores.masked_select(~valid[:, :, None].expand_as(candidate_scores))).all())

    selected = torch.empty(1, 1, 62, device=device, dtype=torch.long)
    support = torch.empty(1, 1, 2048, device=device, dtype=torch.long)
    postprocess.select_and_pack(
        candidate_scores, candidates_ids, selected, support, historical, length
    )
    assert selected[0, 0, 0].item() == 0
    assert bool((support[..., -64:] == torch.arange(historical, length, device=device)).all())
    assert bool(((support < length) | (support == -1)).all())

    host = mapped_host_bf16_empty(batch=1, kv_heads=1, capacity=length)
    append_mapped_host_key(host, key, start=0)
    pointer = mapped_host_device_pointer(host)
    cache = torch.empty(1, 1, 2048, 128, device=device, dtype=dtype)
    resident = torch.full((1, 1, 2048), -1, device=device, dtype=torch.long)
    lookup = torch.full((1, 1, length), -1, device=device, dtype=torch.int32)
    selected_slots = torch.empty_like(support)
    missing = torch.empty_like(support, dtype=torch.int32)
    counts = torch.empty(1, 1, 2, device=device, dtype=torch.int32)
    slots.refresh(pointer, cache, support, resident, lookup, selected_slots, missing, counts, True)
    valid_support = support >= 0
    actual_key = cache.gather(2, selected_slots.clamp_min(0)[..., None].expand(-1, -1, -1, 128))
    expected_key = key.gather(2, support.clamp_min(0)[..., None].expand(-1, -1, -1, 128))
    torch.testing.assert_close(
        actual_key.masked_select(valid_support[..., None]),
        expected_key.masked_select(valid_support[..., None]),
        rtol=0,
        atol=0,
    )
    first_counts = counts.cpu().flatten().tolist()
    slots.refresh(pointer, cache, support, resident, lookup, selected_slots, missing, counts, True)
    second_counts = counts.cpu().flatten().tolist()
    assert second_counts[0] == second_counts[1] == int(valid_support.sum())

    workspace = (
        torch.empty(1, 4, 16, 96, device=device),
        torch.empty(1, 4, 16, device=device),
        torch.empty(1, 4, 1, 96, device=device, dtype=dtype),
    )
    actual = slot_indexed_attention(
        q, cache, value, support, selected_slots, workspace, scale=128**-0.5
    )
    gathered_value = value.gather(2, support.clamp_min(0)[..., None].expand(-1, -1, -1, 96))
    scores = q.float() @ expected_key.repeat_interleave(4, 1).float().transpose(-1, -2)
    scores = (scores * (128**-0.5)).masked_fill(
        ~valid_support.repeat_interleave(4, 1)[:, :, None], -torch.inf
    )
    expected_attention = scores.softmax(-1) @ gathered_value.repeat_interleave(4, 1).float()
    torch.testing.assert_close(actual.float(), expected_attention, rtol=0.02, atol=0.02)
    return {
        "length": length,
        "candidate_pages": n,
        "coarse_pruned": n < math.ceil(length / 32),
        "first_slot_counts": first_counts,
        "second_slot_counts": second_counts,
        "attention_max_abs_error": float((actual.float() - expected_attention).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    capability = torch.cuda.get_device_capability()
    assert capability in ((8, 0), (8, 6))
    torch.manual_seed(20260922)
    torch.set_num_threads(2)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
    args.output.mkdir(parents=True, exist_ok=True)
    full = _load_extension(value_dim=96, queries_per_kv=4, page_size=32,
                           base_rank=16, residual_rank=16)
    fine = compile_fine(args.output / "build")
    postprocess = extension("postprocess", "basisserve/kernels/csrc/fused_decode_postprocess.cu")
    slots = extension("slots", "basisserve/kernels/csrc/persistent_key_slots.cu")
    checks = [check(length, full, fine, postprocess, slots) for length in (4096, 32768)]
    result = {
        "status": "complete",
        "gpu": torch.cuda.get_device_name(),
        "capability": capability,
        "checks": checks,
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
    }
    (args.output / "kernel_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
