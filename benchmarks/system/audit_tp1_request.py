"""Instrumented request smoke; timings are diagnostic, not formal E2E results."""

import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "results/system_benchmarks/tp1_sparse_local/frozen_source"
sys.path.insert(0, str(ROOT))
MODEL = Path("/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
TOKENS = Path("/workspace/runs/l31-cal128/calibration/windows.safetensors")


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Audit:
    def __init__(self):
        self.records = []
        self.phase = "setup"
        self.request_start = None
        self.restore_starts = {}

    def now(self):
        torch.cuda.synchronize()
        return time.perf_counter()

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            start = self.now()
            output = original(*args, **kwargs)
            stop = self.now()
            self.records.append(dict(label=label, phase=self.phase,
                                     start=start, stop=stop, seconds=stop-start))
            return output

        setattr(owner, name, measured)


def local_model(args, audit):
    # Import the preceding tested harness from its frozen source tree.
    config = module_from_file("router_configuration", ROOT / "benchmarks/system/bench_tp1_router_config.py")
    sys.path.insert(0, str(SNAPSHOT))
    filename = "bench_tp1_sparse_full_v6.py" if args.method == "basis" else "bench_tp1_dense_full_v6.py"
    harness = module_from_file("frozen_request_harness", SNAPSHOT / "benchmarks/system" / filename)
    from transformers import AutoModelForCausalLM

    router = config.build("w16u1") if args.method == "basis" else None
    attention = harness._load_extension(value_dim=128, queries_per_kv=4,
                                        page_size=1, base_rank=16, residual_rank=16) if router else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to("cuda").eval()
    common = dict(capacity=args.length + args.output_tokens, storage="local",
                  profile_components=args.profile_components, validate=args.validate)
    if args.method == "basis":
        layers = harness._install_sparse_attention(
            model, **common, mode="optimized", router_extension=router,
            fine_router_extension=None, attention_extension=attention,
            routing="full", key_reuse=False, slot_extension=None,
            postprocess_extension=None)
    else:
        layers = harness._install_dense_attention(model, **common, host_extension=None)

    def forward(tokens):
        start = layers[0].length
        positions = torch.arange(start, start + tokens.shape[1], device=tokens.device)[None]
        output = model.model(input_ids=tokens, position_ids=positions, use_cache=False)
        return model.lm_head(output.last_hidden_state[:, -1:])

    def reset():
        for layer in layers:
            layer.length = 0

    return SimpleNamespace(prefill=forward, decode=forward, place=lambda: None,
        reset=reset, state=lambda: layers, configuration=dict(storage="GPU-local K/V",
        prompt_specific_fit=False, routing="full_scan" if router else "dense",
        validation_enabled=args.validate))


def shadow_model(args, audit):
    from benchmarks.system.paper_faithful_compat import install_shadowkv_import_adapters
    upstream_root = ROOT / "external/ShadowKV"
    install_shadowkv_import_adapters(upstream_root)
    from transformers import LlamaConfig, LlamaForCausalLM

    if not hasattr(LlamaConfig, "rope_theta"):
        LlamaConfig.rope_theta = property(lambda c: c.rope_parameters.get("rope_theta", c.default_theta))
    original = LlamaForCausalLM.from_pretrained

    def load(*positional, **keywords):
        loaded = original(*positional, **keywords)
        for layer in loaded.model.layers:
            layer.self_attn.rotary_emb = loaded.model.rotary_emb
        return loaded

    LlamaForCausalLM.from_pretrained = load
    package = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(
        "models", loader=None, is_package=True))
    package.__path__ = [str(upstream_root / "models")]
    sys.modules["models"] = package
    from models.llama import Llama

    model = Llama(model_name=str(MODEL), device="cuda:0", batch_size=1,
                  max_length=args.length + args.output_tokens + 8,
                  attn_mode="shadowkv_cpu", sparse_budget=2048, rank=160,
                  chunk_size=8, minference=False)
    def reset():
        # Recreate request state, including the original CPU placement of scratch.
        model.kv_cache = None
        model.init_kv_cache(2048, 160, 8, model.config)
        cache = model.kv_cache
        chunks = args.length // cache.chunk_size - cache.local_chunk
        chunks -= chunks % 8
        local_tokens = args.length - chunks * cache.chunk_size
        required = cache.sparse_budget + cache.outlier_chunk * cache.chunk_size + local_tokens + args.output_tokens
        for name in ("k_cache_buffer", "v_cache_buffer"):
            buffer = getattr(cache, name)
            shape = list(buffer.shape)
            shape[-2] = max(shape[-2], required)
            if shape != list(buffer.shape):
                setattr(cache, name, torch.zeros(shape, device=buffer.device, dtype=buffer.dtype))
        if audit is not None:
            audit.wrap(cache, "get_svd", "svd_and_factor_storage")
            audit.wrap(cache, "prefill_kv_cache", "landmarks_and_cache_preparation")
            audit.wrap(cache, "H2D", "post_prefill_placement")

    reset()

    def decode(token):
        return model.inference(input_ids=token, position_ids=model.get_ctx(token))

    return SimpleNamespace(prefill=model.batch_prefill, decode=decode,
        place=lambda: model.kv_cache.H2D(), reset=reset, state=lambda: model.kv_cache,
        configuration=dict(
        storage="upstream CPU V / low-rank reconstructed K", rank=160,
        chunk_size=8, sparse_budget=2048, prompt_specific_fit=True,
        generation_buffer_tokens=model.kv_cache.k_cache_buffer.shape[-2],
        generation_buffer_policy="at least actual initial support plus output-token count; capacity only"))


def lrqk_model(args, audit):
    from benchmarks.system.paper_faithful_compat import install_flash_attn_adapter
    install_flash_attn_adapter()
    sys.path.insert(0, str(ROOT / "external/LRQK"))
    sys.path.insert(0, str(ROOT / "external/LRQK/cpp_kernel"))
    import lrqk_attention as upstream
    from transformers import AutoModelForCausalLM

    if audit is not None:
        install_lrqk_audit(upstream, audit)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda:0").eval()
    model.config._attn_implementation = "flash_attention_2"
    model = upstream.load_model_hack(str(MODEL), device="cuda:0", base_model=model)
    if args.length >= 130048:
        from evaluation.chunked_prefill_mlp import ChunkedTokenwise
        for layer in model.model.layers:
            layer.mlp = ChunkedTokenwise(layer.mlp, chunk_size=1024)
    cache = None

    def reset():
        nonlocal cache
        cache = None
        cache = upstream.DynamicLRQKCache(
            num_key_value_groups=4, r=32, num_active_tokens=2048, lite_tokens=64,
            max_iter=(2, 2), tol=1e-8,
            max_sequence_length=args.length + args.output_tokens + 64,
            lwattn_factory=upstream.LightAttentionIndicesOffloadPrefill,
            init_aq_ak_method=upstream.InitAQAK.randn)

    reset()

    def forward(token):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(input_ids=token, past_key_values=cache, use_cache=True,
                         return_dict=True, logits_to_keep=1).logits

    return SimpleNamespace(prefill=forward, decode=forward, place=lambda: None,
        reset=reset, state=lambda: cache, configuration=dict(storage="upstream CPU exact K/V",
        rank=32, active_tokens=2048, lite_tokens=64, prompt_specific_fit=True,
        prefill_mlp_chunk_size=1024 if args.length >= 130048 else None))


def install_lrqk_audit(upstream, audit):
    audit.wrap(upstream, "cast_lrqk_prefill", "prompt_factor_fit")
    audit.wrap(upstream.LightAttentionIndicesOffloadPrefill, "prefill", "construction_and_cache_preparation")
    outer_decode = upstream.LightAttentionIndicesOffloadPrefill.decode
    inner_decode = upstream.LightAttentionIndicesFactory.decode

    def enter_decode(layer, *positional, **keywords):
        if layer.Kgpu_temp is not None:
            audit.restore_starts[id(layer)] = audit.now()
        return outer_decode(layer, *positional, **keywords)

    def restored_decode(layer, *positional, **keywords):
        start = audit.restore_starts.pop(id(layer), None)
        if start is not None:
            stop = audit.now()
            audit.records.append(dict(label="lazy_first_decode_restore", phase=audit.phase,
                                      start=start, stop=stop, seconds=stop-start))
        return inner_decode(layer, *positional, **keywords)

    upstream.LightAttentionIndicesOffloadPrefill.decode = enter_decode
    upstream.LightAttentionIndicesFactory.decode = restored_decode
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dense", "basis", "shadowkv", "lrqk"), required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--cohort", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.validate = True
    args.profile_components = False
    # This diagnostic deliberately cannot be used as a long-context formal runner.
    assert 8192 <= args.length <= 16384 and 2 <= args.output_tokens <= 8
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "command.json").write_text(json.dumps(sys.argv, indent=2) + "\n")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    audit = Audit()
    factory = shadow_model if args.method == "shadowkv" else lrqk_model if args.method == "lrqk" else local_model
    runtime = factory(args, audit)
    windows = load_file(str(TOKENS))["input_ids"]
    assert args.cohort < windows.shape[0] and args.length <= windows.shape[1]
    prompt = windows[args.cohort:args.cohort+1, :args.length].long().cuda()
    torch.cuda.reset_peak_memory_stats()
    audit.phase = "prefill"
    start = audit.now()
    logits = runtime.prefill(prompt)
    prefill_end = audit.now()
    finite_checks = [torch.isfinite(logits).all()]
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [token]
    audit.phase = "placement"
    runtime.place()
    placement_end = audit.now()
    audit.phase = "decode"
    steps = []
    for _ in range(args.output_tokens - 1):
        begin = audit.now()
        logits = runtime.decode(token)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        end = audit.now()
        steps.append(end-begin)
        finite_checks.append(torch.isfinite(logits).all())
    stop = audit.now()
    finite = bool(torch.stack(finite_checks).all())
    restores = [r for r in audit.records if r["label"] == "lazy_first_decode_restore"]
    ready = max(r["stop"] for r in restores) if restores else (
        placement_end if args.method == "shadowkv" else prefill_end)
    totals = {}
    for record in audit.records:
        totals[record["label"]] = totals.get(record["label"], 0.0) + record["seconds"]
        record["start_from_request_seconds"] = record.pop("start") - start
        record["stop_from_request_seconds"] = record.pop("stop") - start
    counts = {label: sum(r["label"] == label for r in audit.records) for label in totals}
    expected = {"shadowkv": {"svd_and_factor_storage": 32,
                            "landmarks_and_cache_preparation": 32, "post_prefill_placement": 1},
                "lrqk": {"prompt_factor_fit": 32, "construction_and_cache_preparation": 32,
                         "lazy_first_decode_restore": 32}}.get(args.method, {})
    counts_ok = counts == expected
    result = dict(status="complete" if finite and counts_ok else "validation_failed",
        method=args.method, environment="basis", gpu=torch.cuda.get_device_name(),
        context_tokens=args.length, output_tokens=args.output_tokens, cohort=args.cohort,
        configuration=runtime.configuration, all_logits_finite=finite,
        generated_tokens=torch.cat(generated, dim=1).tolist(), actual_decode_calls=len(steps),
        diagnostic_prefill_seconds=prefill_end-start,
        diagnostic_post_prefill_ready_seconds=ready-prefill_end,
        ready_overlaps_first_decode=bool(restores),
        diagnostic_request_seconds=stop-start, diagnostic_decode_seconds=steps,
        component_seconds=totals, component_calls=counts, expected_calls=expected,
        component_counts_match=counts_ok, records=audit.records,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        scope="Synchronized, instrumented, cold request smoke with validation. Not formal latency. Nested component times must not be added. LRQK readiness includes preceding first-decode work, not just restoration. Model load and upfront allocations excluded.")
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
