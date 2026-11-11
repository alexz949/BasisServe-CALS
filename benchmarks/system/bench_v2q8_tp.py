"""V2Q8 benchmark, TP1 or tensor parallel (torchrun --nproc-per-node=W): one arm, one cohort, one mode.

  torchrun --nproc-per-node=4 benchmarks/system/bench_v2q8_tp.py --arm lrqk_koff_v128 --mode request \
      --length 65536 --batch 16 --cohort 0 --inputs .../tp4_64k/inputs/prompts.safetensors --output DIR ...

Arms (experiment design: BasisKV and LRQK offload exact Keys to host, ShadowKV offloads Values):
  basis_koff_v128[_reuse]  BasisKV, original V128, B16R16 Page32 (32 sink + 1952 routed + 64 recent), exact K in
                           mapped pinned host memory with GPU Key slots; _reuse reads Base from the rotated V cache
  basis_koff_v96[_reuse]   BasisKV, compressed V96, B16R16 Page4 (2048 routed + 64 recent, no sink), same placement
  shadowkv_upstream_v128   round 2: the upstream ShadowKV runtime (W=1) or its minimal TP port; V128 on pinned host
  lrqk_upstream_koff_v128  round 2: the upstream LRQK runtime; exact K on pinned host, V on GPU (requested placement)
  shadowkv_voff_v128       round 1: official ShadowKV cache + CUDA kernels through the earlier TP adapter
  lrqk_koff_v128 / _v96    round 1: official LRQK factory through the earlier TP adapter
  dense_v128               original model, GPU-resident K/V, FlashAttention-2 decode (reference only)
Prefill schedule: BasisKV, dense and the round-1 adapters prefill every batch row as its own request; the upstream
baselines use their own schedule (ShadowKV batch_prefill: groups of 4, or 8 for 120K-200K prompts; LRQK: the whole
batch in one forward). A row's TTFT is the end of the prefill call that contains it; batch_ready (all rows
prefilled and placed) has the same meaning for every arm. All rows then decode together for a fixed output count.

Modes (after a same-shape warmup request and fresh state):
  request    synchronized phase boundaries only: per-row prefill end (TTFT), batch ready, decode, total
  steady     prefill, then 16 conditioning + 128 measured decode steps, CUDA events and synchronized wall per rank,
             phase memory and a tensor inventory (GPU and host)
  breakdown  instrumented separately: every component timed with synchronized wall clock (the same method for
             every arm), including ShadowKV SVD and LRQK ALS fitting; not additive to the request/steady timings
Each rank writes rank_<r>.json; rank 0 writes result.json with every rank's numbers.
"""

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/system"))

ARMS = {
    "basis_koff_v128": dict(method="basis", value=128, reuse=False),
    "basis_koff_v128_reuse": dict(method="basis", value=128, reuse=True),
    "basis_koff_v96": dict(method="basis", value=96, reuse=False),
    "basis_koff_v96_reuse": dict(method="basis", value=96, reuse=True),
    "shadowkv_voff_v128": dict(method="shadowkv", value=128),
    # The upstream ShadowKV runtime itself (W=1) or its minimal TP port (basisserve/core/shadowkv_upstream_tp.py).
    "shadowkv_upstream_v128": dict(method="shadowkv_upstream", value=128),
    # The upstream LRQK runtime with Values kept on GPU (basisserve/core/lrqk_upstream.py).
    "lrqk_upstream_koff_v128": dict(method="lrqk_upstream", value=128),
    "lrqk_koff_v128": dict(method="lrqk", value=128),
    "lrqk_koff_v96": dict(method="lrqk", value=96),
    "dense_v128": dict(method="dense", value=128),
}
# BasisKV page geometry per Value width: the banks that exist (V128: Table 3 runtime bank; V96: Table 2 Page4 bank).
BASIS_GEOMETRY = {128: dict(page_size=32, routed_tokens=1984, sink_tokens=32),
                  96: dict(page_size=4, routed_tokens=2048, sink_tokens=0)}


class Timers:
    """Synchronized wall-clock component timers for the breakdown mode (one method for every arm)."""

    def __init__(self):
        self.enabled, self.phase, self.seconds, self.calls, self.open = False, "setup", {}, {}, {}

    def start(self, label):
        if self.enabled:
            torch.cuda.synchronize()
            self.open[label] = time.perf_counter()

    def stop(self, label):
        if self.enabled and label in self.open:
            torch.cuda.synchronize()
            key = (self.phase, label)
            self.seconds[key] = self.seconds.get(key, 0.0) + time.perf_counter() - self.open.pop(label)
            self.calls[key] = self.calls.get(key, 0) + 1

    @contextmanager
    def __call__(self, label):
        self.start(label)
        yield
        self.stop(label)


class CollectiveLog:
    """Logical collective traffic of one forward: every torch.distributed call (attention outputs, HF row-parallel
    MLP AllReduce, ShadowKV's Key gather / U broadcast / SV scatter), plus the vocab-gathering lm_head, whose
    DTensor redistribute bypasses torch.distributed and is recorded from its output shape."""

    def __init__(self, world):
        self.active, self.calls, self.world = False, [], world
        original_batch = dist.batch_isend_irecv

        def batch_isend_irecv(operations):
            if self.active:
                for operation in operations:
                    self.record("p2p " + operation.op.__name__, operation.tensor)
            return original_batch(operations)

        dist.batch_isend_irecv = batch_isend_irecv
        for name in ("all_reduce", "all_gather_into_tensor", "broadcast", "all_gather"):
            original = getattr(dist, name)

            def wrapped(*positional, _name=name, _original=original, **keywords):
                if self.active:
                    tensor = positional[1] if _name == "all_gather_into_tensor" else positional[0]
                    if torch.is_tensor(tensor):
                        self.record(_name, tensor)
                return _original(*positional, **keywords)

            setattr(dist, name, wrapped)

    def record(self, op, tensor):
        self.calls.append(dict(op=op, shape=list(tensor.shape), dtype=str(tensor.dtype),
                               bytes=tensor.numel() * tensor.element_size()))

    def capture(self, model, function):
        def observer(op):
            def hook(module, inputs, output):  # must return None: a forward hook's return value replaces the output
                if self.active:
                    self.record(op, output)
            return hook

        hooks = []
        if self.world > 1:
            for name, module in model.named_modules():
                if name == "lm_head":
                    hooks.append(module.register_forward_hook(observer("all_gather (HF DTensor, lm_head vocab)")))
        self.calls, self.active = [], True
        result = function()
        torch.cuda.synchronize()
        self.active = False
        for hook in hooks:
            hook.remove()
        summary = {}
        for call in self.calls:
            key = f"{call['op']} {call['shape']} {call['dtype']}"
            row = summary.setdefault(key, dict(op=call["op"], shape=call["shape"], dtype=call["dtype"],
                                               calls=0, bytes_per_call=call["bytes"], logical_bytes=0))
            row["calls"] += 1
            row["logical_bytes"] += call["bytes"]
        return result, list(summary.values())


def nccl_kernels(function):
    """Measured NCCL kernel count and CUDA time of one call (torch.profiler; whatever API issued them)."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        function()
        torch.cuda.synchronize()
    rows = {}
    for event in profiler.events():
        if "nccl" in event.name.lower() and event.device_type.name == "CUDA":
            row = rows.setdefault(event.name, dict(kernel=event.name, calls=0, cuda_us=0.0))
            row["calls"] += 1
            row["cuda_us"] += event.device_time
    return sorted(rows.values(), key=lambda r: -r["cuda_us"])


def sync():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return time.perf_counter()


def memory():
    import psutil
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                device_used=total - free, process_rss=psutil.Process().memory_info().rss)


def build_shadowkv_upstream(args, world, timers):
    """The upstream ShadowKV runtime: its own model, batch_prefill (request groups) and cache, timed from outside.
    Breakdown timers wrap upstream functions and cache methods; no upstream code is edited."""
    from benchmarks.system import audit_tp1_request as frozen
    from basisserve.core import shadowkv_upstream_tp as upstream_tp

    device = torch.device("cuda", torch.cuda.current_device())
    llama = upstream_tp.build(frozen.MODEL, batch=args.batch, max_length=args.length + args.output_tokens + 8,
                              device=device)
    instrument = args.mode == "breakdown"
    if instrument:
        base, _, _ = upstream_tp.import_upstream()

        def timed(function, label):
            def measured(*positional, **keywords):
                with timers(label):
                    return function(*positional, **keywords)
            return measured

        base.flash_attn_with_kvcache = timed(base.flash_attn_with_kvcache, "flash_attention")
        torch.svd = timed(torch.svd, "svd")
        upstream_tp.PHASE = timers
    cache_labels = dict(get_svd="get_svd_total", prefill_kv_cache="landmarks_outliers_and_value_offload",
                        update_kv_cache="append_new_token", get_retrieval_position_ids="selection_landmark_scores_topk",
                        get_value_cache="value_miss_fetch_h2d", get_key_cache="key_miss_reconstruct")

    def reset():
        cache = upstream_tp.fresh_cache(llama, length=args.length, output_tokens=args.output_tokens)
        if instrument:
            for name, label in cache_labels.items():
                setattr(cache, name, timed(getattr(cache, name), label))

    reset()

    def forward(tokens, start):
        return llama.inference(input_ids=tokens, position_ids=llama.get_ctx(tokens))[:, -1].float()

    def prefill_batch(prompts):
        """Upstream batch_prefill; each request's first-token time is the end of its prefill group."""
        ends = []

        def inference(input_ids, position_ids):
            output = type(llama).inference(llama, input_ids=input_ids, position_ids=position_ids)
            ends.extend([sync()] * input_ids.shape[0])
            return output

        llama.inference = inference
        logits = llama.batch_prefill(prompts)
        del llama.inference
        return logits[:, -1].argmax(-1)[:, None], ends

    def place():
        with timers("post_prefill_h2d"):
            llama.kv_cache.H2D()

    holder = torch.nn.Module()  # no parameters: the inventory walks the upstream model object instead
    holder.model = SimpleNamespace(layers=[])
    configuration = dict(value="original V128 and Wo",
        placement="Values offloaded: full prompt V on pinned host (upstream ShadowKVCache_CPU)",
        implementation="upstream models.llama.Llama + ShadowKVCache_CPU + CUDA kernels "
                       "(ByteDance-Seed/ShadowKV@e51904cd), source unmodified" + (
                           "" if world == 1 else "; minimal TP port basisserve/core/shadowkv_upstream_tp.py"),
        shadowkv=dict(rank=upstream_tp.RANK, chunk=upstream_tp.CHUNK, sparse_budget=upstream_tp.SPARSE_BUDGET),
        prefill="upstream batch_prefill (request groups of 4; 8 for 120K-200K prompts)",
        tp=None if world == 1 else "local heads and MLP slice per rank, AllReduce after wo and down_proj, global "
           "SVD on the layer owner (layer % W) with Key gather and U/SV scatter; embedding, norms, lm_head replicated",
        generation_buffer_tokens=llama.kv_cache.k_cache_buffer.shape[-2],
        generation_buffer_policy="at least actual initial support plus output-token count; capacity only")
    return SimpleNamespace(model=holder, layers=[], forward=forward, reset=reset, place=place,
                           prefill_batch=prefill_batch, llama=llama, configuration=configuration,
                           state=lambda: llama)


def build_lrqk_upstream(args, world, timers):
    """The upstream LRQK runtime (HF model + load_model_hack + DynamicLRQKCache), Values on GPU, timed from
    outside. The whole batch is prefilled in one forward, as upstream does. Breakdown timers wrap upstream
    functions and methods; no upstream code is edited."""
    from transformers import AutoModelForCausalLM, DistributedConfig
    from benchmarks.system import audit_tp1_request as frozen
    from evaluation.chunked_prefill_mlp import ChunkedTokenwise
    from basisserve.core import lrqk_upstream

    module, split_kv, k_only, factory_class = lrqk_upstream.make_classes()
    options = dict(dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True)
    model = (AutoModelForCausalLM.from_pretrained(frozen.MODEL, **options).cuda() if world == 1 else
             AutoModelForCausalLM.from_pretrained(frozen.MODEL, distributed_config=DistributedConfig(tp_size=world),
                                                  **options)).eval()
    for layer in model.model.layers:  # as system_benchmarks/tp1_offline: exact tokenwise prefill MLP chunks
        layer.mlp = ChunkedTokenwise(layer.mlp, chunk_size=1024)
    model = lrqk_upstream.install(model, world=world)
    if args.mode == "breakdown":
        def timed(function, label):
            def measured(*positional, **keywords):
                with timers(label):
                    return function(*positional, **keywords)
            return measured

        for name, label in (("cast_lrqk_prefill", "als_factor_fit"), ("cast_lrqk_decode", "factor_update"),
                            ("flash_attn_with_kvcache", "flash_attention")):
            setattr(module, name, timed(getattr(module, name), label))
        for owner, name, label in ((k_only, "compute_hit_indices", "proxy_scores_and_topk"),
                                   (k_only, "only_indices_in_cpu", "hit_miss_bookkeeping"),
                                   (split_kv, "append", "kv_store_k_host_v_gpu"),
                                   (factory_class, "prefill", "lrqk_prefill_total"),
                                   (factory_class, "decode", "lrqk_decode_total")):
            setattr(owner, name, timed(getattr(owner, name), label))
        torch.cuda.empty_cache = timed(torch.cuda.empty_cache, "empty_cache")
        lrqk_upstream.PHASE = timers
    holder, vocab = {}, model.config.vocab_size

    def reset():
        holder["cache"] = None
        gc.collect()
        holder["cache"] = lrqk_upstream.new_cache(factory_class,
                                                  max_sequence_length=args.length + args.output_tokens + 64)

    reset()

    def forward(tokens, start):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=tokens, past_key_values=holder["cache"], use_cache=True, return_dict=True,
                           logits_to_keep=1).logits
        logits = logits.full_tensor() if hasattr(logits, "full_tensor") else logits
        if logits.shape[-1] != vocab:
            parts = [torch.empty_like(logits) for _ in range(world)]
            dist.all_gather(parts, logits.contiguous())
            logits = torch.cat(parts, dim=-1)
        return logits[:, -1].float()

    def prefill_batch(prompts):
        token = forward(prompts, 0).argmax(-1)[:, None]
        return token, [sync()] * prompts.shape[0]

    factories = lambda: [holder["cache"].lwattn[i] for i in sorted(holder["cache"].lwattn)]
    configuration = dict(value="original V128 and Wo",
        placement="Keys offloaded: exact K in an upstream pinned host pool; Values in an upstream GPU pool; GPU "
                  "hit cache of top-k + lite per query head; hit cache and A_K parked in host memory during prefill",
        implementation="upstream lrqk_attention (tenghuilee/LRQK@caf16293): load_model_hack, DynamicLRQKCache, "
                       "LightAttentionIndicesOffloadPrefill, torch.compile as upstream; source unmodified; "
                       "basisserve/core/lrqk_upstream.py splits the K/V pool (Values on GPU)",
        lrqk=dict(lrqk_upstream.CONFIG, lambdas=1.0, init="randn"),
        prefill="whole batch in one forward (upstream)", mlp="ChunkedTokenwise(1024) as tp1_offline",
        tp=None if world == 1 else "HF tensor parallel; local head counts on each attention module")
    return SimpleNamespace(model=model, layers=[], forward=forward, reset=reset, place=lambda: None,
                           prefill_batch=prefill_batch, factories=factories, configuration=configuration,
                           state=lambda: holder["cache"])


def build(args, rank, world, timers):
    from transformers import AutoModelForCausalLM, DistributedConfig
    from benchmarks.system import audit_tp1_request as frozen
    from evaluation.chunked_prefill_mlp import ChunkedTokenwise

    spec = ARMS[args.arm]
    if spec["method"] == "shadowkv_upstream":
        return build_shadowkv_upstream(args, world, timers)
    if spec["method"] == "lrqk_upstream":
        return build_lrqk_upstream(args, world, timers)
    if spec["method"] == "basis":
        # BasisKV uses the archived Table 3 kernels (slot refresh, slot-indexed attention) from the frozen tree.
        # Baselines must not: their code (c1_*, compressed_v_decode_attention) is the repository's.
        sys.path.insert(0, str(frozen.SNAPSHOT))
        harness = frozen.module_from_file(
            "frozen_request_harness", frozen.SNAPSHOT / "benchmarks/system/bench_tp1_sparse_full_v6.py")
    options = dict(dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    model = (AutoModelForCausalLM.from_pretrained(frozen.MODEL, **options).cuda() if world == 1 else
             AutoModelForCausalLM.from_pretrained(frozen.MODEL, distributed_config=DistributedConfig(tp_size=world),
                                                  **options)).eval()
    capacity = args.length + args.output_tokens
    value_checkpoint = args.value_checkpoint if spec["value"] == 96 else None
    value = "compressed V96 (paper Table 2 shared checkpoint)" if spec["value"] == 96 else "original V128 and Wo"
    output = ("AllGather of local V96 coordinates + replicated GQA-tied decoder" if spec["value"] == 96 and world > 1
              else "GQA-tied V96 decoder" if spec["value"] == 96 else "o_proj" + (" (row-parallel AllReduce)" if world > 1 else ""))
    place = lambda: None
    if spec["method"] == "shadowkv":
        from basisserve.core import shadowkv_tp8_attention as adapter
        from basisserve.core.shadowkv_tp8 import make_tp8_cache
        from evaluation import official_shadowkv_cpu
        official_shadowkv_cpu.ROOT = ROOT / "external/ShadowKV"
        cache_class = official_shadowkv_cpu.load_cache_class()
        phase = (lambda name, layer, request: timers(name)) if args.mode == "breakdown" else None
        layers, cache = adapter.install_tp8_shadowkv(model, cache_class, batch=args.batch, length=args.length,
                                                     decode_steps=args.output_tokens, phase=phase)
        holder = {"cache": cache}

        def place():
            with timers("post_prefill_h2d"):
                holder["cache"].H2D()

        def reset_shadowkv():
            # A fresh official cache per request (allocated before the timed request, like every arm's caches).
            holder["cache"] = make_tp8_cache(cache_class, layers=32, batch=args.batch, length=args.length,
                                             decode_steps=args.output_tokens, device=torch.device("cuda"))
            for layer in layers:
                layer.cache, layer.length = holder["cache"], 0

        for layer in layers:
            layer.reset = reset_shadowkv
        configuration = dict(value=value, placement="Values offloaded: full prompt V on pinned host, GPU buffer of "
                             "local + outlier + selected + generated tokens",
            implementation="official ShadowKVCache_CPU + CUDA kernels (ByteDance-Seed/ShadowKV@e51904cd)",
            shadowkv=dict(rank=160, chunk=8, routed=2048, outlier_chunks=48, local_chunks=4),
            keys="global rank-160 pre-RoPE SVD on the layer owner; U replicated, SV per local KV head; selected "
                 "misses reconstructed by batch_gather_gemm + RoPE",
            values="selected misses fetched by gather_copy_with_offsets on a copy stream",
            decode="FlashAttention-2 (vLLM build) over the selected-token buffer", output=output)
    elif spec["method"] == "lrqk":
        from basisserve.core import lrqk_tp
        layers = lrqk_tp.install(model, batch=args.batch, capacity=capacity, rank=rank, world_size=world,
                                 chunked=ChunkedTokenwise, value_checkpoint=value_checkpoint)
        for layer in layers:
            layer.timers = timers if args.mode == "breakdown" else None
        configuration = dict(value=value, placement="Keys offloaded: exact K in a pinned host pool; Values in a "
                             "GPU-resident cache; GPU hit cache of top-k + lite per query head",
            implementation="official LightAttentionIndicesFactory (tenghuilee/LRQK@caf16293) on local heads",
            lrqk=dict(lrqk_tp.CONFIG, lambdas=1.0, init="randn (all heads, local slice)"),
            misses="Keys gathered on CPU (cpp_kernel.d0_index_select) and copied to GPU; Values gathered on GPU",
            convergence="deltas averaged over ranks, tested per request",
            decode="FlashAttention-2 (vLLM build) over the hit cache" if spec["value"] == 128 else
                   "Triton compressed-V attention over the hit cache", output=output,
            upstream_float32_matmul_precision=torch.get_float32_matmul_precision(),
            upstream_torch_compile=os.environ.get("LRQK_TORCH_COMPILE") == "1")
    elif spec["method"] == "basis":
        from table3_page_router import build as build_router
        from benchmarks.system import v2q8_basis
        geometry = BASIS_GEOMETRY[spec["value"]]
        folder = args.build_root / (f"p{geometry['page_size']}_v{spec['value']}_t{geometry['routed_tokens']}"
                                    f"_s{geometry['sink_tokens']}")
        router = anchors = None
        for turn in range(2):  # rank 0 compiles first; the others load the cached build
            if (turn == 0) == (rank == 0):
                router, anchors = build_router(frozen.SNAPSHOT, folder, geometry["page_size"],
                                               value_dim=spec["value"], routed_tokens=geometry["routed_tokens"],
                                               sink_tokens=geometry["sink_tokens"])
            if world > 1:
                dist.barrier()
        layers = v2q8_basis.install(
            model, harness=harness, router=router, bank=args.bank_v96 if spec["value"] == 96 else args.bank_v128,
            value_checkpoint=value_checkpoint, batch=args.batch, capacity=capacity, offload=True,
            value_reuse=spec["reuse"], rank=rank, world=world, **geometry)
        for layer in layers:
            layer.timers = timers if args.mode == "breakdown" else None
        configuration = dict(value=value, placement="Keys offloaded: exact K in mapped pinned host memory with "
                             "persistent GPU Key slots; Values and routing codes on GPU",
            router="B16R16 full-scan page router", **geometry, recent_tokens=64,
            base_code="first 16 coordinates of the rotated Value cache" if spec["reuse"] else "separate Base cache",
            output=output, router_build=anchors)
    else:
        from benchmarks.system import v2q8_tp_baselines
        layers = v2q8_tp_baselines.install(model, "dense", value_checkpoint=None, batch=args.batch,
                                           capacity=capacity, rank=rank, world=world, chunked=ChunkedTokenwise)
        configuration = dict(value=value, placement="GPU-resident K/V", decode="FlashAttention-2 (vLLM build)",
                             output=output)
    vocab = model.config.vocab_size

    def forward(tokens, start):
        positions = torch.arange(start, start + tokens.shape[1], device="cuda")[None].expand(tokens.shape[0], -1)
        hidden = model.model(input_ids=tokens, position_ids=positions, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden[:, -1:])
        logits = logits.full_tensor() if hasattr(logits, "full_tensor") else logits
        if logits.shape[-1] != vocab:
            parts = [torch.empty_like(logits) for _ in range(world)]
            dist.all_gather(parts, logits.contiguous())
            logits = torch.cat(parts, dim=-1)
        return logits[:, -1].float()

    def reset():
        if spec["method"] == "shadowkv":
            layers[0].reset()
            return
        for layer in layers:
            layer.reset()

    return SimpleNamespace(model=model, layers=layers, forward=forward, reset=reset, place=place,
                           configuration=configuration, state=lambda: layers)


def fresh(runtime):
    torch.cuda.synchronize()
    runtime.reset()
    gc.collect()
    torch.manual_seed(20260924)
    sync()


def prefill_rows(runtime, prompts):
    """Prefill each row as its own request, then place the cache for decode (ShadowKV H2D; a no-op elsewhere).
    Returns first tokens, each row's synchronized first-token time, and the batch-ready time."""
    if hasattr(runtime, "prefill_batch"):  # upstream ShadowKV: its own batch_prefill
        token, ends = runtime.prefill_batch(prompts)
        runtime.place()
        return token, ends, sync()
    first, ends = [], []
    for row in range(prompts.shape[0]):
        for layer in runtime.layers:
            layer.prefill_row = row
        first.append(runtime.forward(prompts[row:row + 1], 0).argmax(-1))
        ends.append(sync())
    runtime.place()
    return torch.cat(first)[:, None], ends, sync()


def request(runtime, prompts, count, forced=None):
    """forced: [B, count] token ids fed instead of the model's own argmax (teacher forcing; consistency checks).
    The log-probability of every fed token is recorded either way; its cost is inside the decode time."""
    torch.cuda.reset_peak_memory_stats()
    begin = sync()
    token, ends, ready = prefill_rows(runtime, prompts)
    prefill_memory = memory()
    torch.cuda.reset_peak_memory_stats()
    resume = sync()
    argmax, fed, logprobs = [token], [], []
    if forced is not None:
        token = forced[:, :1]
    fed.append(token)
    length = prompts.shape[1]
    for step in range(count - 1):
        logits = runtime.forward(token, length + step)
        choice = logits.argmax(-1)[:, None]
        token = choice if forced is None else forced[:, step + 1:step + 2]
        logprobs.append(torch.log_softmax(logits, -1).gather(-1, token))
        argmax.append(choice)
        fed.append(token)
    stop = sync()
    decode_seconds = stop - resume
    # Prefill calls may cover several rows (upstream baselines): a call's duration is split evenly over its rows.
    groups, start = [], begin
    for index, end in enumerate(ends):
        if index == len(ends) - 1 or ends[index + 1] != end:
            size = index + 1 - sum(g[0] for g in groups)
            groups.append((size, end - start))
            start = end
    return dict(request_seconds=(ready - begin) + decode_seconds, batch_ready_seconds=ready - begin,
        # A row's TTFT is the end of the prefill call containing it (rows or groups run one after another).
        ttft_seconds=[end - begin for end in ends],
        prefill_call_rows_and_seconds=groups,
        prefill_seconds_per_request=[seconds / size for size, seconds in groups for _ in range(size)],
        placement_seconds=ready - ends[-1], decode_seconds=decode_seconds,
        decode_tokens_per_second=prompts.shape[0] * (count - 1) / decode_seconds,
        decode_mean_ms_per_step=1000 * decode_seconds / (count - 1),
        memory_after_prefill=prefill_memory, memory_after_decode=memory(),
        generated_tokens=torch.cat(fed if forced is None else argmax, 1).tolist(),
        teacher_forced=forced is not None,
        fed_token_logprobs=torch.cat(logprobs, 1).tolist() if logprobs else [])


def steady(runtime, prompts, warmup, steps, log):
    torch.cuda.reset_peak_memory_stats()
    begin = sync()
    token, _, ready = prefill_rows(runtime, prompts)
    prefill_seconds = ready - begin
    phases = dict(after_prefill=memory())
    torch.cuda.reset_peak_memory_stats()
    length, cuda_ms, wall_ms, tokens, communication = prompts.shape[1], [], [], [token], None
    for step in range(warmup + steps):
        if step == warmup:
            phases["decode_ready"] = memory()
            torch.cuda.reset_peak_memory_stats()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # Local synchronization only: a barrier inside the timed step would add its own latency. Ranks are coupled
        # by the forward's collectives; the summary takes the slowest rank per step.
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_event.record()
        if step == warmup - 1:
            logits, communication = log.capture(runtime.model, lambda: runtime.forward(token, length + step))
        else:
            logits = runtime.forward(token, length + step)
        end_event.record()
        torch.cuda.synchronize()
        stop = time.perf_counter()
        token = logits.argmax(-1)[:, None]
        tokens.append(token)
        if step >= warmup:
            cuda_ms.append(start_event.elapsed_time(end_event))
            wall_ms.append(1000 * (stop - start))
    phases["decode_end"] = memory()
    profiled = nccl_kernels(lambda: runtime.forward(token, length + warmup + steps))
    return dict(cuda_ms=cuda_ms, wall_ms=wall_ms, prefill_seconds=prefill_seconds, memory=phases,
                decode_step_communication=communication, decode_step_nccl_kernels=profiled,
                generated_tokens=torch.cat(tokens, 1).tolist())


PARENTS = {"attention_block": ("cache_append", "router_scan", "page_selection", "key_refresh", "sparse_attention"),
           "get_svd_total": ("svd", "k_gather", "factor_redistribution_incl_owner_svd_wait"),
           "lrqk_prefill_total": ("als_factor_fit", "proxy_scores_and_topk", "kv_store_k_host_v_gpu"),
           "lrqk_decode_total": ("factor_update", "proxy_scores_and_topk", "hit_miss_bookkeeping", "miss_indices",
                                 "key_miss_fetch_cpu_gather_h2d", "value_miss_gather_gpu", "kv_store_k_host_v_gpu")}


def breakdown(runtime, prompts, warmup, steps, timers):
    """Instrumented run: synchronized timers around every component (prefill per request, decode per step) plus
    whole-forward totals. Components exclude nothing on purpose; their sum differs from the uninstrumented total
    because synchronization serializes overlapped work (e.g. ShadowKV's copy stream)."""
    fresh(runtime)
    timers.seconds, timers.calls, timers.enabled = {}, {}, True
    timers.phase = "prefill"
    first = []
    if hasattr(runtime, "prefill_batch"):  # upstream ShadowKV: its own batch_prefill (request groups)
        with timers("model_forward_total"):
            first.append(runtime.prefill_batch(prompts)[0][:, 0])
    else:
        for row in range(prompts.shape[0]):
            for layer in runtime.layers:
                layer.prefill_row = row
            with timers("model_forward_total"):
                first.append(runtime.forward(prompts[row:row + 1], 0).argmax(-1))
    runtime.place()
    token, length = torch.cat(first)[:, None], prompts.shape[1]
    for step in range(warmup + steps):
        timers.phase = "decode_conditioning" if step < warmup else "decode"
        with timers("model_forward_total"):
            token = runtime.forward(token, length + step).argmax(-1)[:, None]
    timers.enabled = False
    batch = prompts.shape[0]
    prefill = {label: dict(seconds_per_request=seconds / batch, calls=timers.calls[(phase, label)])
               for (phase, label), seconds in timers.seconds.items() if phase == "prefill"}
    decode = {label: dict(ms_per_step=1000 * seconds / steps, calls_per_step=timers.calls[(phase, label)] / steps)
              for (phase, label), seconds in timers.seconds.items() if phase == "decode"}
    for name, rows, unit in (("prefill", prefill, "seconds_per_request"), ("decode", decode, "ms_per_step")):
        total = rows.get("model_forward_total", {}).get(unit, 0.0)
        # Enclosing timers of the upstream runtimes and their named children; the remainder is reported as
        # <parent>_rest (ShadowKV get_svd: sync + gc.collect + empty_cache + U/SV copies to host; LRQK factory:
        # cache writes, repeat_kv, sync_cache, and the prefill parking / first-decode restore).
        for parent, children in PARENTS.items():
            if parent in rows:
                inner = sum(rows[label][unit] for label in children if label in rows)
                rows[f"{parent}_rest"] = {unit: rows[parent][unit] - inner}
        # Parents (BasisKV attention_block, upstream ShadowKV get_svd, LRQK factory) enclose other components;
        # count leaves and the <parent>_rest remainders only.
        # post_prefill_h2d (ShadowKV placement) runs outside the forward and is reported on its own.
        named = sum(row[unit] for label, row in rows.items()
                    if label not in ("model_forward_total", "attention_block", "post_prefill_h2d", *PARENTS))
        rows["other_model_work"] = {unit: total - named, "note": "forward total minus the named components "
                                    "(projections, MLP, norms, embedding, lm_head, collectives)"}
    return dict(prefill=prefill, decode=decode, batch=batch, measured_decode_steps=steps,
                scope="Instrumented separately with synchronized wall-clock timers; components serialized; prefill "
                      "components summed over layers and averaged per request, decode summed over layers per step.")


def support(runtime, arm):
    """Actual physical support per (request, KV head) at the end of the run, on this rank."""
    layers, method = runtime.layers, ARMS[arm]["method"]
    if method == "basis":
        counts = [float((layer.support_ids >= 0).sum(-1).float().mean()) for layer in layers]
    elif method == "lrqk":
        counts = []
        for layer in layers:
            factory = layer.factory
            lite = torch.tensor(factory.lite_indices, device="cuda").view(1, 1, -1).expand(
                factory.batch, factory.hit_indices.shape[1], -1)
            ids = torch.cat((factory.hit_indices[..., 0], lite), dim=-1)
            grouped = ids.reshape(factory.batch, layer.base.kv_heads, -1).sort(-1).values
            counts.append(float((1 + (grouped[..., 1:] != grouped[..., :-1]).sum(-1)).float().mean()))
    elif method == "shadowkv":
        cache = layers[0].cache  # local + outlier + routed + generated tokens per (request, KV head)
        counts = [float(cache.sparse_end + cache.gen_offset)] * len(layers)
    elif method == "shadowkv_upstream":
        cache = runtime.llama.kv_cache
        counts = [float(cache.sparse_end + cache.gen_offset)] * cache.num_layers
    elif method == "lrqk_upstream":
        counts = []
        for factory in runtime.factories():
            batch, heads = factory.hit_indices.shape[:2]
            lite = torch.tensor(factory.lite_indices, device="cuda").view(1, 1, -1).expand(batch, heads, -1)
            ids = torch.cat((factory.hit_indices[..., 0], lite), dim=-1)
            grouped = ids.reshape(batch, heads // 4, -1).sort(-1).values
            counts.append(float((1 + (grouped[..., 1:] != grouped[..., :-1]).sum(-1)).float().mean()))
    else:
        counts = [float(layer.length) for layer in layers]
    return dict(mean_tokens_per_request_kv_head=statistics.fmean(counts), per_layer=counts)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--mode", choices=("request", "steady", "breakdown"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--cohort", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--bank-v96", type=Path, required=True)
    parser.add_argument("--bank-v128", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-tokens", type=Path, default=None,
                        help="request mode: JSON [B][count] token ids to feed instead of argmax (teacher forcing)")
    args = parser.parse_args()
    warmup, steps = (2, 4) if args.smoke else (16, 128)
    args.output_tokens = ((8 if args.smoke else 128) if args.mode == "request" else
                          warmup + steps + (1 if args.mode == "steady" else 0))
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    from v2q8_affinity import bind_to_gpu_node
    # Slot = physical GPU index: jobs on GPU 0 and GPU 1 (same NUMA node) get disjoint CPU halves.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = int(visible.split(",")[local_rank]) if visible else local_rank
    placement = bind_to_gpu_node(local_rank, physical)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260924)
    args.output.mkdir(parents=True, exist_ok=True)
    log, timers = CollectiveLog(world), Timers()
    progress = lambda phase: rank == 0 and print(json.dumps(dict(arm=args.arm, mode=args.mode, batch=args.batch,
        phase=phase, time_utc=time.strftime("%H:%M:%S", time.gmtime()))), flush=True)
    progress("model_load_and_build")
    started = time.perf_counter()
    runtime = build(args, rank, world, timers)
    setup_seconds = time.perf_counter() - started
    cohort = load_file(str(args.inputs))[f"cohort_{args.cohort}"]
    cohort = cohort[None] if cohort.ndim == 1 else cohort  # TP1 cohorts hold one prompt, TP4 cohorts sixteen
    prompts = cohort[:args.batch, :args.length].long().cuda()
    assert prompts.shape == (args.batch, args.length)
    record = dict(rank=rank, world=world, cpu_placement=placement,
                  excluded_setup_seconds=setup_seconds, memory_model_loaded=memory())
    progress("same_shape_warmup_request")
    fresh(runtime)
    request(runtime, prompts, args.output_tokens)
    fresh(runtime)
    progress(f"measured_{args.mode}")
    if args.mode == "request":
        forced = (torch.tensor(json.loads(args.force_tokens.read_text()), device="cuda")[:args.batch,
                  :args.output_tokens] if args.force_tokens else None)
        record["measurement"] = request(runtime, prompts, args.output_tokens, forced)
    elif args.mode == "steady":
        record["measurement"] = steady(runtime, prompts, warmup, steps, log)
    else:
        record["measurement"] = breakdown(runtime, prompts, warmup, steps, timers)
    record["support"] = support(runtime, args.arm)
    if args.mode == "steady":
        from bench_v2q8_tp1 import inventory
        record["inventory"] = inventory(runtime)
        fresh(runtime)
        for layer in runtime.layers:
            layer.prefill_row = 0
        _, record["prefill_one_request_communication"] = log.capture(
            runtime.model, lambda: runtime.forward(prompts[:1], 0))
    (args.output / f"rank_{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    agree = True
    if "generated_tokens" in record["measurement"]:
        tokens = torch.tensor(record["measurement"]["generated_tokens"], device="cuda")
        gathered = [torch.empty_like(tokens) for _ in range(world)]
        dist.all_gather(gathered, tokens)
        agree = all(torch.equal(gathered[0], other) for other in gathered)
    dist.barrier()
    if rank == 0:
        ranks = [json.loads((args.output / f"rank_{r}.json").read_text()) for r in range(world)]
        result = dict(status="complete" if agree else "cross_rank_token_mismatch", arm=args.arm, mode=args.mode,
            tp=world, batch=args.batch, context_tokens=args.length, cohort=args.cohort,
            output_tokens=args.output_tokens, configuration=runtime.configuration,
            cross_rank_tokens_agree=agree, environment=dict(
                command=sys.argv, python=platform.python_version(), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, nccl=".".join(map(str, torch.cuda.nccl.version())),
                conda_environment=Path(sys.prefix).name, host=platform.node(),
                nccl_env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                gpus=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
                                              "--format=csv,noheader"], text=True).strip(),
                git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()),
            ranks=ranks)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        progress(result["status"])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
