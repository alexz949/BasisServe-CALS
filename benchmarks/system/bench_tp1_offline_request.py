"""Fresh-state TP1 requests and steady decode, with separate diagnostic profiles."""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.system.audit_tp1_request import (
    Audit, install_lrqk_audit, local_model, lrqk_model, shadow_model,
)


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def progress(args, phase):
    data = dict(method=args.method, phase=phase, context=args.length, cohort=args.cohort)
    (args.output / "progress.json").write_text(json.dumps(data) + "\n")
    print(json.dumps(data), flush=True)


def fresh(runtime):
    torch.cuda.synchronize()
    runtime.reset()
    gc.collect()
    torch.manual_seed(20260924)
    torch.cuda.synchronize()


def generate(runtime, prompt, count, ready_marker=False):
    # Only phase boundaries synchronize. No per-step events or finite checks.
    tokens = []
    ready_times = []
    start = sync_time()
    logits = runtime.prefill(prompt)
    prefill_end = sync_time()
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    tokens.append(token)
    runtime.place()
    placed = sync_time()
    if ready_marker:
        import lrqk_attention as upstream
        parent = upstream.LightAttentionIndicesFactory.decode
        last = runtime.state().lwattn[31]

        def mark(layer, *positional, **keywords):
            if layer is last:
                # The upstream restoration stream has already synchronized here.
                ready_times.append(time.perf_counter())
                upstream.LightAttentionIndicesFactory.decode = parent
            return parent(layer, *positional, **keywords)

        upstream.LightAttentionIndicesFactory.decode = mark
    steady_begin = None
    for index in range(count - 1):
        if index == 16:
            steady_begin = sync_time()
        logits = runtime.decode(token)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(token)
    stop = sync_time()
    return dict(request_seconds=stop-start, prefill_inclusive_seconds=prefill_end-start,
        post_prefill_phase_seconds=placed-prefill_end, decode_phase_seconds=stop-placed,
        post_prefill_ready_seconds=(ready_times[0]-prefill_end) if ready_times else (
            placed-prefill_end if runtime.configuration["storage"].startswith("upstream CPU V") else 0.0),
        ready_overlaps_first_decode=ready_marker,
        steady_tail_wall_mean_ms=1000*(stop-steady_begin)/(count-17) if steady_begin is not None else None,
        steady_tail_decode_steps=max(0, count-17),
        final_logits_finite=bool(torch.isfinite(logits).all()),
        generated_tokens=torch.cat(tokens, dim=1).tolist(), actual_decode_calls=count-1)


def profile_build(runtime, prompt, method):
    fresh(runtime)
    audit = Audit()
    if method == "shadowkv":
        for name, label in (("get_svd", "svd_and_factor_storage"),
                            ("prefill_kv_cache", "landmarks_and_cache_preparation"),
                            ("H2D", "post_prefill_placement")):
            audit.wrap(runtime.state(), name, label)
    else:
        import lrqk_attention as upstream
        install_lrqk_audit(upstream, audit)
    audit.phase = "prefill"
    begin = sync_time()
    logits = runtime.prefill(prompt)
    prefill_end = sync_time()
    first_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    audit.phase = "placement"
    runtime.place()
    placed = sync_time()
    audit.phase = "decode"
    logits = runtime.decode(first_token)
    sync_time()
    second_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    totals, counts = {}, {}
    restores = [r for r in audit.records if r["label"] == "lazy_first_decode_restore"]
    ready = max(r["stop"] for r in restores) if restores else placed
    for row in audit.records:
        label = row["label"]
        totals[label] = totals.get(label, 0.0) + row["seconds"]
        counts[label] = counts.get(label, 0) + 1
        row["start"] -= begin
        row["stop"] -= begin
    construction = (totals["svd_and_factor_storage"] + totals["landmarks_and_cache_preparation"]
                    + totals["post_prefill_placement"]) if method == "shadowkv" else (
                        totals["construction_and_cache_preparation"] + totals["lazy_first_decode_restore"])
    expected = ({"svd_and_factor_storage": 32, "landmarks_and_cache_preparation": 32,
                 "post_prefill_placement": 1} if method == "shadowkv" else
                {"prompt_factor_fit": 32, "construction_and_cache_preparation": 32,
                 "lazy_first_decode_restore": 32})
    return dict(component_seconds=totals, component_calls=counts,
        counts_match=counts == expected, construction_including_preparation_seconds=construction,
        diagnostic_post_prefill_ready_seconds=ready-prefill_end,
        first_two_tokens=torch.cat((first_token, second_token), dim=1).tolist(),
        records=audit.records,
        scope="Separate synchronized profile; nested fitting time is not added twice. Not subtractable from uninstrumented E2E.")


def steady(runtime, prompt, warmup, steps):
    logits = runtime.prefill(prompt)
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    runtime.place()
    cuda_ms, wall_ms, components, generated = [], [], [], []
    finite = True
    for index in range(warmup + steps):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = sync_time()
        begin.record()
        logits = runtime.decode(token)
        end.record()
        stop = sync_time()
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        finite = finite and bool(torch.isfinite(logits).all())
        if index >= warmup:
            cuda_ms.append(begin.elapsed_time(end))
            wall_ms.append(1000*(stop-start))
            if runtime.state()[0].profile_components:
                layer_times = [layer.component_times() for layer in runtime.state()]
                components.append({key: sum(row[key] for row in layer_times) for key in layer_times[0]})
    return dict(cuda_ms=cuda_ms, wall_ms=wall_ms,
        cuda_median_ms=statistics.median(cuda_ms), cuda_mean_ms=statistics.fmean(cuda_ms),
        wall_median_ms=statistics.median(wall_ms), all_logits_finite=finite,
        generated_tokens=torch.cat(generated, dim=1).tolist(), components=components)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dense", "basis", "shadowkv", "lrqk"), required=True)
    parser.add_argument("--mode", choices=("request", "steady"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--cohort", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.validate = False
    args.profile_components = False
    warmup, steps = (2, 4) if args.smoke else (16, 128)
    requested_tokens = args.output_tokens
    if args.mode == "steady":
        assert args.method in ("dense", "basis")
        args.output_tokens = warmup + steps + 1
    assert args.length + args.output_tokens <= 131072
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "command.json").write_text(json.dumps(sys.argv, indent=2) + "\n")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260924)
    progress(args, "model_load")
    factory = shadow_model if args.method == "shadowkv" else lrqk_model if args.method == "lrqk" else local_model
    runtime = factory(args, None)
    prompt = load_file(str(args.inputs))[f"cohort_{args.cohort}"][:args.length][None].long().cuda()
    assert prompt.shape[1] == args.length
    progress(args, "same_shape_runtime_warmup")
    fresh(runtime)
    warm = generate(runtime, prompt, args.output_tokens)
    assert warm["final_logits_finite"]
    progress(args, "fresh_request_state")
    fresh(runtime)
    del warm
    torch.cuda.reset_peak_memory_stats()
    progress(args, "measured_" + args.mode)
    if args.mode == "request":
        measured = generate(runtime, prompt, args.output_tokens, ready_marker=args.method == "lrqk")
        valid = measured["final_logits_finite"]
    else:
        measured = steady(runtime, prompt, warmup, steps)
        valid = measured["all_logits_finite"]
    peak = torch.cuda.max_memory_allocated()/2**30
    result = dict(status="measured", method=args.method, mode=args.mode,
        context_tokens=args.length, cohort=args.cohort, output_tokens=requested_tokens,
        configuration=runtime.configuration, environment="basis", gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, peak_allocated_gib=peak, measurement=measured,
        scope="Same-shape runtime warmup; fresh request state and representation. Model loading, upfront cache allocation and input transfer excluded. Greedy fixed-length output ignores EOS. No build profiling in primary timing. Request steady tail is wall mean after 16 decode calls, not CUDA median.")
    (args.output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.mode == "request" and args.method in ("shadowkv", "lrqk") and (requested_tokens == 128 or args.smoke):
        progress(args, "separate_build_profile")
        profile = profile_build(runtime, prompt, args.method)
        profile["first_two_tokens_match"] = profile["first_two_tokens"] == [measured["generated_tokens"][0][:2]]
        valid = valid and profile["counts_match"] and profile["first_two_tokens_match"]
        result["build_profile"] = profile
    elif args.mode == "request" and args.method in ("dense", "basis"):
        result["build_profile"] = dict(prompt_specific_fit_seconds=0.0,
            ordinary_encoding_and_cache_writes="included in prefill")
    if args.mode == "steady":
        progress(args, "separate_attention_profile")
        for layer in runtime.state():
            names = ("attention_block", "cache_append", "dense_attention") if args.method == "dense" else (
                "attention_block", "cache_append", "router_scan", "page_selection", "sparse_attention")
            layer.profile_events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for name in names}
            layer.profile_components = True
        fresh(runtime)
        profile_steps = 4 if args.smoke else 32
        profile = steady(runtime, prompt, warmup, profile_steps)
        profile["prefix_tokens_match"] = profile["generated_tokens"] == [measured["generated_tokens"][0][:warmup+profile_steps]]
        valid = valid and profile["all_logits_finite"] and profile["prefix_tokens_match"]
        result["attention_profile"] = profile
    result["status"] = "complete" if valid else "validation_failed"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    progress(args, result["status"])


if __name__ == "__main__":
    main()
