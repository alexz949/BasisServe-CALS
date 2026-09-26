"""Single-GPU compact global decoder timings using real C1 weights and latents."""

import argparse
import gc
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from basisserve.kernels.fp8_wire import quantize_e4m3_static, quantize_e4m3_tensorwise_col_major, scaled_mm_e4m3_static
from evaluation.benchmark_fp8_feature_decoder import _time_cuda, _error
from evaluation.eval_qwen3_latent_a8 import SOURCES, load_fixed_model
from evaluation.eval_qwen3_kv4_fp8_ppl import write
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 96], choices=(64, 96))
    parser.add_argument("--prior", type=Path, default=ROOT / "results/q3-kv4-fp8/formal")
    parser.add_argument("--output", type=Path, default=ROOT / "results/q3-a8-dec")
    args = parser.parse_args()
    layers = [18] if args.phase == "smoke" else [0, 18, 35]
    batches = [1, 16] if args.phase == "smoke" else [1, 4, 16, 64, 128]
    warmup, repeats = (3, 10) if args.phase == "smoke" else (20, 100)
    directory = args.output / f"decoder_{args.phase}"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source"
    source.mkdir(exist_ok=True)
    for name in (*SOURCES, "evaluation/benchmark_qwen3_latent_decoder.py",
                 "evaluation/benchmark_fp8_feature_decoder.py"):
        original = ROOT / name
        destination = source / original.name
        if destination.exists():
            assert destination.read_bytes() == original.read_bytes()
        else:
            shutil.copy2(original, destination)
    records = []
    for rank in args.ranks:
        model, tokenizer, projections, decoders, handles, schedule, protocol = load_fixed_model(args.prior, rank)
        captured = {}

        def capture(name, value):
            captured[name] = value[0].reshape(-1, 4096).detach().clone()

        observers = [decoders[f"{i}.decoder"].register_forward_pre_hook(
            lambda m, x, name=i: capture(name, x)) for i in layers]
        train = _token_ids(tokenizer, "wikitext2", "train", None).reshape(-1)
        start = protocol["calibration_starts"][0]
        model.model(input_ids=train[start:start + 128][None].cuda(), use_cache=False)
        for observer in observers:
            observer.remove()
        for layer in layers:
            decoder = decoders[f"{layer}.decoder"]
            active = torch.tensor([q * 128 + j for q in range(32)
                for j in range(schedule[layer][q // 4])], device="cuda")
            weight = decoder.weight.index_select(1, active).contiguous()
            latent = captured[layer].index_select(1, active).contiguous()
            codes_w, scale_w = quantize_e4m3_tensorwise_col_major(weight.T)
            scale_a = decoder.input_scale
            for batch in batches:
                value = latent[:batch].contiguous()
                codes = quantize_e4m3_static(value, scale_a)

                def bf16():
                    return F.linear(value, weight)

                def a8_bf16():
                    return F.linear((codes.float() * scale_a).to(torch.bfloat16), weight)

                def w8a8():
                    return scaled_mm_e4m3_static(codes, codes_w,
                        left_scale=scale_a, right_scale=scale_w, out_dtype=torch.bfloat16)

                def quantize_a8_bf16():
                    encoded = quantize_e4m3_static(value, scale_a)
                    return F.linear((encoded.float() * scale_a).to(torch.bfloat16), weight)

                def quantize_w8a8():
                    encoded = quantize_e4m3_static(value, scale_a)
                    return scaled_mm_e4m3_static(encoded, codes_w,
                        left_scale=scale_a, right_scale=scale_w, out_dtype=torch.bfloat16)

                operations = dict(bf16=bf16, a8_bf16=a8_bf16, w8a8=w8a8,
                    quantize=lambda: quantize_e4m3_static(value, scale_a),
                    quantize_a8_bf16=quantize_a8_bf16, quantize_w8a8=quantize_w8a8)
                times, errors = {}, {}
                reference = bf16()
                for name, operation in operations.items():
                    times[name], result = _time_cuda(operation, warmup=warmup, repeats=repeats)
                    if name != "quantize":
                        assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
                        errors[name] = _error(reference, result)
                record = dict(rank=rank, layer=layer, rows=batch, latent_width=value.shape[1],
                    output_width=4096, latency_ms=times, error_vs_bf16=errors,
                    decoder_speedup=times["bf16"] / times["w8a8"],
                    quantize_decoder_speedup=times["bf16"] / times["quantize_w8a8"],
                    latent_bytes=dict(bf16=value.numel() * 2, fp8=codes.numel(), static_scale_bytes=4))
                records.append(record)
                print("CASE", json.dumps(record), flush=True)
                write(directory / "progress.json", records)
        for handle in handles:
            handle.remove()
        del model, tokenizer, projections, decoders, handles, captured, decoder
        gc.collect()
        torch.cuda.empty_cache()
    write(directory / "results.json", dict(status="complete", command=shlex.join(sys.argv),
        environment="basis", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        phase=args.phase, warmup=warmup, repeats=repeats, records=records,
        protocol="CUDA event medians, eager kernels; offline weight quantization excluded; actual compact checkpoint weights",
        scope="global decoder on one GPU; no collectives, no TP8 speedup, no full-model E2E",
        inputs="128 train tokens under BF16 encoder/decoder + fixed KV4; row counts emulate GEMM batch shape, not concurrent requests"))
    lines = ["# Compact Decoder Microbenchmark", "", f"Phase: {args.phase}. Environment: basis.",
        "Real adaptive C1 decoder weights; fixed KV4; BF16 encoder. Single GPU, no communication.",
        "CUDA event median timings. W8A8 decoder excludes activation quantization; combined timing includes it.", "",
        "| Rank | Layer | Rows | K | BF16 ms | A8/BF16 ms | W8A8 ms | Quantize + W8A8 ms | Combined speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        t = r["latency_ms"]
        lines.append(f"| {r['rank']} | {r['layer']} | {r['rows']} | {r['latent_width']} | {t['bf16']:.6f} | {t['a8_bf16']:.6f} | {t['w8a8']:.6f} | {t['quantize_w8a8']:.6f} | {r['quantize_decoder_speedup']:.3f} |")
    lines += ["", "Not a serving benchmark: repeated train-sample latents, warm weights, no TP8 collectives or scheduling.",
        "Wire payload counts are analytical and exclude protocol overhead; no measured communication speedup is claimed.",
        "PPL uses padded HF projections for consistency with the earlier quality baseline; these timings remove inactive padding.", ""]
    (directory / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
