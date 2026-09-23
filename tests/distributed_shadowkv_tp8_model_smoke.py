"""Compare short real-text TP8 generation with the existing official adapter."""

import argparse
import gc
import json
import os
from datetime import timedelta
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basisserve.core.shadowkv_tp8_attention import install_tp8_shadowkv
from benchmarks.system.bench_llama31_8b_tp8_combined import DEFAULT_MODEL, _bind_cpu, _choose
from benchmarks.system.numa_memory import bind_host_allocations
from evaluation import official_shadowkv_cpu


def full_logits(logits, vocab_size):
    if logits.shape[-1] == vocab_size:
        return logits.detach().cpu().reshape(-1)
    assert logits.shape[-1] * 8 == vocab_size
    shards = [torch.empty_like(logits) for _ in range(8)]
    dist.all_gather(shards, logits)
    return torch.cat(shards, dim=-1).cpu().reshape(-1)


def positions_snapshot(cache):
    shards = [torch.empty_like(cache.position_ids) for _ in range(8)]
    dist.all_gather(shards, cache.position_ids)
    return torch.cat(shards, dim=2).cpu()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=Path,
                        default=Path("/workspace/runs/l31-cal128/tp8-benchmark-prompts/p4096_c0.safetensors"))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=8)
    args = parser.parse_args()
    assert args.decode_steps > 0
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    bind_host_allocations(_bind_cpu(local_rank))
    dist.init_process_group("nccl", timeout=timedelta(minutes=20), device_id=device)
    assert dist.get_world_size() == 8
    official_shadowkv_cpu.ROOT = args.upstream.resolve()
    cache_class = official_shadowkv_cpu.load_cache_class()
    tokens = load_file(str(args.tokens))["input_ids"][:1, :4096].long().to(device)
    assert tokens.shape == (1, 4096)
    args.output.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=8), local_files_only=True).eval()
    modules, cache = install_tp8_shadowkv(model, cache_class, batch=1, length=4096,
                                         decode_steps=args.decode_steps)
    logits_history, selection_history, generated = [], [], []
    with patch("torch.svd", wraps=torch.svd) as svd:
        output = model(input_ids=tokens, use_cache=False, logits_to_keep=1)
        assert bool(torch.isfinite(output.logits).all())
        logits_history.append(full_logits(output.logits, model.config.vocab_size))
        token = _choose(output.logits, model.config.vocab_size)
        generated.append(token.clone())
        del output
        cache.H2D()
        selection_history.append(positions_snapshot(cache))
        prefill_svd_calls = svd.call_count
        assert prefill_svd_calls == 4
        for step in range(args.decode_steps):
            output = model(input_ids=token, position_ids=torch.tensor([[4096 + step]], device=device),
                           use_cache=False, logits_to_keep=1)
            assert bool(torch.isfinite(output.logits).all())
            logits_history.append(full_logits(output.logits, model.config.vocab_size))
            token = _choose(output.logits, model.config.vocab_size)
            generated.append(token.clone())
            selection_history.append(positions_snapshot(cache))
            del output
        assert svd.call_count == prefill_svd_calls
    assert all(len(module.factor_records) == 1 for module in modules)
    ids = torch.cat(generated, dim=-1)
    rank_ids = [torch.empty_like(ids) for _ in range(8)]
    dist.all_gather(rank_ids, ids)
    assert all(torch.equal(ids, other) for other in rank_ids)
    ids = ids.cpu().reshape(-1).tolist()
    tp_logits = torch.stack(logits_history)
    tp_selections = torch.stack(selection_history)
    del model, modules, cache, logits_history, selection_history, generated, rank_ids
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    success = torch.ones((), device=device, dtype=torch.int32)
    if rank == 0:
        reference = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": str(device)}, local_files_only=True).eval()
        official_shadowkv_cpu.install(reference)
        reference_cache = official_shadowkv_cpu.make_cache(
            cache_class, reference.config, 4096, args.decode_steps, device)
        positions = torch.arange(4096 + args.decode_steps, device=device)[None]
        cos, sin = reference.model.rotary_emb(torch.empty(1, device=device, dtype=torch.bfloat16), positions)
        cos_sin = torch.cat((cos[0, :, :64], sin[0, :, :64]), -1).contiguous()
        for layer in reference.model.layers:
            layer.self_attn._official_shadow_cache = reference_cache
            layer.self_attn._shadow_cos_sin = cos_sin
        reference_logits, reference_selections, reference_ids = [], [], []
        for step in range(args.decode_steps + 1):
            output = reference(
                input_ids=tokens if step == 0 else reference_token,
                position_ids=positions[:, :4096] if step == 0 else positions[:, 4095 + step:4096 + step],
                use_cache=False, logits_to_keep=1)
            assert bool(torch.isfinite(output.logits).all())
            reference_logits.append(output.logits.cpu().reshape(-1))
            reference_token = output.logits.argmax(-1)
            reference_ids.append(int(reference_token.item()))
            if step == 0:
                reference_cache.H2D()
            reference_selections.append(reference_cache.position_ids.cpu().clone())
            del output
        ref_logits = torch.stack(reference_logits)
        ref_selections = torch.stack(reference_selections)
        relative_error = ((tp_logits.float() - ref_logits.float()).norm(dim=-1)
                          / ref_logits.float().norm(dim=-1)).tolist()
        overlaps = [len(set(left) & set(right)) / 256
                    for left, right in zip(tp_selections.reshape(-1, 256).tolist(),
                                           ref_selections.reshape(-1, 256).tolist())]
        checks = {"identical_generated_tokens": ids == reference_ids,
                  "logits_relative_error_below_0_02": max(relative_error) < 0.02,
                  "minimum_selection_overlap_at_least_0_95": min(overlaps) >= 0.95}
        passed = all(checks.values())
        success.fill_(int(passed))
        result = {"status": "complete" if passed else "mismatch", "checks": checks,
                  "tp8_generated_ids": ids, "reference_generated_ids": reference_ids,
                  "logits_relative_error_per_output": relative_error,
                  "minimum_selection_overlap": min(overlaps),
                  "mean_selection_overlap": sum(overlaps) / len(overlaps),
                  "prefill_svd_calls_per_tp_rank": prefill_svd_calls,
                  "decode_svd_calls": 0, "model": str(args.model), "tokens": str(args.tokens),
                  "reference": "Existing unchanged official_shadowkv_cpu adapter; single GPU",
                  "tp8_decode_backend": "upstream flash_attn_with_kvcache",
                  "scope": "Short real-text correctness smoke, not a quality benchmark or speed trial",
                  "command": sys.argv, "pytorch": torch.__version__}
        (args.output / "model_reference.json").write_text(json.dumps(result, indent=2) + "\n")
        save_file({"tp8_logits": tp_logits, "reference_logits": ref_logits,
                   "tp8_selections": tp_selections, "reference_selections": ref_selections},
                  str(args.output / "model_reference.safetensors"))
        print(json.dumps(result), flush=True)
    dist.broadcast(success, src=0)
    assert bool(success.item()), "Model-level ShadowKV smoke mismatch; inspect saved comparison."
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
