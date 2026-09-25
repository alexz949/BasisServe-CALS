"""Fixed active-batch decode with sequential prompt admission and independent KV."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from benchmarks.system import bench_tp1_dense_full_v6 as dense
from benchmarks.system import bench_tp1_sparse_full_v6 as sparse


@contextmanager
def prefill_row(layers, row):
    """Use views into the final batch allocation, without duplicating historical KV."""
    saved = []
    for layer in layers:
        cache = {name: getattr(layer, name) for name in (
            "key_cache", "value_cache", "host_key", "host_value", "base_cache", "residual_cache"
        ) if isinstance(getattr(layer, name, None), torch.Tensor)}
        saved.append(cache)
        for name, tensor in cache.items():
            setattr(layer, name, tensor[row:row + 1])
        layer.length = 0
    yield
    for layer, cache in zip(layers, saved, strict=True):
        for name, tensor in cache.items():
            setattr(layer, name, tensor)


def memory():
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
                  if ":" in line)
    return dict(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                gpu_reserved_bytes=torch.cuda.memory_reserved(),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                host_rss_bytes=int(status["VmRSS"].split()[0]) * 1024,
                host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dense_local", "dense_k_offload", "basis_k_offload"), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.context >= 4096 and args.batch > 0 and args.steps > 0
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "result.json").exists()

    def phase(name, **extra):
        record = dict(phase=name, time=time.time(), **extra)
        (args.output / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    torch.set_num_threads(2)
    torch.manual_seed(20260925)
    torch.backends.cuda.matmul.allow_tf32 = False
    capacity = args.context + max(args.warmup, args.steps)
    phase("extensions")
    extension = None
    if args.method != "dense_local":
        extension = sparse._load_extension(value_dim=128, queries_per_kv=4,
            page_size=32 if args.method == "basis_k_offload" else 1, base_rank=16, residual_rank=16)
    slots = sparse._load_slot_extension() if args.method == "basis_k_offload" else None
    phase("model_load")
    model = AutoModelForCausalLM.from_pretrained(dense.MODEL_PATH, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").to("cuda").eval()
    phase("cache_allocation")
    common = dict(capacity=capacity, batch_size=args.batch, profile_components=False, validate=args.validate)
    if args.method == "basis_k_offload":
        layers = sparse._install_sparse_attention(model, storage="offload", mode="optimized",
            router_extension=extension, fine_router_extension=None, attention_extension=None,
            routing="full", key_reuse=True, slot_extension=slots, postprocess_extension=None, **common)
    else:
        layers = dense._install_dense_attention(model,
            storage="local" if args.method == "dense_local" else "k_offload",
            host_extension=extension, **common)
    windows = load_file(str(dense.TOKENS_PATH))["input_ids"]
    assert windows.shape[1] >= args.context and windows.shape[0] >= args.batch
    next_tokens = []
    phase("prefill", **memory())
    for row in range(args.batch):
        phase("prefill", request=row, **memory())
        with prefill_row(layers, row):
            output = model.model(input_ids=windows[row:row + 1, :args.context].to("cuda"), use_cache=False)
            token = model.lm_head(output.last_hidden_state[:, -1:]).argmax(-1)
            next_tokens.append(token)
            del output
        torch.cuda.synchronize()
    initial = torch.cat(next_tokens, dim=0)
    del next_tokens, windows
    torch.cuda.empty_cache()
    resident = memory()

    def reset():
        for layer in layers:
            layer.length = args.context
            if getattr(layer, "key_reuse", False):
                layer.slot_resident.fill_(-1)
                layer.slot_lookup.fill_(-1)

    def step(tokens, position):
        assert tokens.shape == (args.batch, 1)
        positions = torch.full((args.batch, 1), position, device="cuda", dtype=torch.long)
        hidden = model.model(input_ids=tokens, position_ids=positions, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden[:, -1:])
        return logits.argmax(-1), torch.isfinite(logits).all()

    phase("warmup", **resident)
    tokens = initial
    for index in range(args.warmup):
        tokens, finite = step(tokens, args.context + index)
        assert bool(finite)
    if args.validate:
        assert all(layer.validated for layer in layers)
        if args.method == "basis_k_offload":
            assert all(layer.router_validated for layer in layers)
    # Numerical validation runs only in warmup, never in the timed decode window.
    for layer in layers:
        layer.validate = False
    reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    phase("decode", **memory())
    tokens = initial
    samples, generated, finite_flags = [], [], []
    start = time.perf_counter()
    for index in range(args.steps):
        begin = time.perf_counter()
        tokens, finite = step(tokens, args.context + index)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1000)
        generated.append(tokens)
        finite_flags.append(finite)
    elapsed = time.perf_counter() - start
    assert bool(torch.stack(finite_flags).all())
    assert all(layer.length == args.context + args.steps for layer in layers)
    result = dict(method=args.method, context=args.context, batch=args.batch,
        active_batch=[args.batch] * args.steps, decode_steps=args.steps, warmup_steps=args.warmup,
        decode_seconds=elapsed, aggregate_tokens_per_second=args.batch * args.steps / elapsed,
        mean_step_ms=statistics.fmean(samples), median_step_ms=statistics.median(samples),
        step_ms=samples, decode_resident=resident, final_memory=memory(),
        host_key_bytes=sum(layer.host_key.numel() * layer.host_key.element_size()
                           for layer in layers if layer.host_key is not None),
        tokens=torch.cat(generated, dim=1).cpu().tolist(), all_logits_finite=True,
        validation=args.validate, environment="basis", torch=torch.__version__,
        prefill_rope="inplace_chunks_2048_existing_operator",
        gpu=torch.cuda.get_device_name(), prompt_rows=list(range(args.batch)),
        protocol="Sequential prefill admission; simultaneous fixed-batch greedy decode. "
                 "128 measured decode forwards by default, excludes first token from prefill. "
                 "Warmup cache lengths and K slots reset before measurement; no EOS stopping. "
                 "Timing includes logits, argmax, finite reduction and per-step synchronization; not E2E.")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    phase("complete", aggregate_tokens_per_second=result["aggregate_tokens_per_second"],
          mean_step_ms=result["mean_step_ms"])


if __name__ == "__main__":
    main()
