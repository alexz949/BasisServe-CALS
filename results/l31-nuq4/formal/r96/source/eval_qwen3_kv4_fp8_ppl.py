"""WT2 PPL: same C1 checkpoint, NUQ4 K/V, and actual E4M3 W8A8 GEMMs."""

import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from basisserve.core.qwen3_kv4_fp8_quality import install_factors, install_nuq4_hooks
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids, WIKITEXT_REVISION

HF_REV = "3df8ff71c718ebcb096f479209baf95cd46abc60"
CACHE = Path("/workspace/.cache/huggingface/hub")
MODEL = CACHE / "models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
BANKS = CACHE / f"models--alexz949--BasisServe-CALS/snapshots/{HF_REV}/ICLR-results/qwen3-8b/checkpoints"
ARMS = ("bf16", "kv4", "fp8", "kv4_fp8")


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def fit_quantizers(model, upstream, modules, indices, train, starts, length, output):
    records = {name: [] for name in modules}

    def capture(name, value):
        record = [value.reshape(-1, 1024).index_select(-1, indices[name]).detach().float().cpu(), None]
        records[name].append(record)

        def gradient(grad):
            record[1] = grad.reshape(-1, 1024).index_select(-1, indices[name]).detach().float().square().cpu()

        value.register_hook(gradient)

    handles = [module.register_forward_hook(lambda m, x, y, name=name: capture(name, y))
               for name, module in modules.items()]
    for i, start in enumerate(starts):
        tokens = train[start:start + length][None].cuda()
        embedded = model.get_input_embeddings()(tokens).detach().requires_grad_(True)
        logits = model(inputs_embeds=embedded, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                               tokens[:, 1:].reshape(-1), reduction="sum")
        assert torch.isfinite(loss)
        loss.backward()
        print(f"FISHER {i + 1}/{len(starts)} loss={float(loss.detach()):.6f}", flush=True)
        del tokens, embedded, logits, loss
    for handle in handles:
        handle.remove()
    quantizers = {}
    for name in modules:
        data = torch.cat([r[0] for r in records[name]])
        fisher = torch.cat([r[1] for r in records[name]])
        assert torch.isfinite(data).all() and torch.isfinite(fisher).all() and fisher.sum() > 0
        fitter = SimpleNamespace(out=data, perchannel=True, qchannel=0 if name.endswith(".k") else -1,
                                 bits=4, nsamples=len(starts))
        quantizers[name] = upstream.SimQuant.quantize(
            fitter, include_sparse=True, sparsity_threshold=0.99, nuq=True,
            fisher=fisher / fisher.mean(), first_few_fp16=-1,
        )
        del records[name], data, fisher, fitter
        print(f"CODEBOOK {name}", flush=True)
    torch.save(quantizers, output / "quantizers.pt")
    return quantizers


@torch.inference_mode()
def calibrate_fp8(model, projections, train, starts, length):
    for module in projections.values():
        module.begin_calibration()
    for i, start in enumerate(starts):
        model.model(input_ids=train[start:start + length][None].cuda(), use_cache=False)
        print(f"FP8_CALIBRATION {i + 1}/{len(starts)}", flush=True)
    scales = {}
    for name, module in projections.items():
        module.end_calibration()
        scales[name] = dict(input_scale=float(module.input_scale), weight_scale=float(module.weight_scale),
                            observed_amax=float(module.observed_amax))
    return scales


@torch.inference_mode()
def evaluate(model, tokens, length, windows, progress):
    nll, count = 0.0, 0
    for i in range(windows):
        batch = tokens[i * length:(i + 1) * length][None].cuda()
        hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state[0]
        loss = torch.zeros((), device="cuda", dtype=torch.float32)
        for start in range(0, length - 1, 128):
            stop = min(start + 128, length - 1)
            logits = model.lm_head(hidden[start:stop]).float()
            loss += F.cross_entropy(logits, batch[0, start + 1:stop + 1], reduction="sum")
        assert torch.isfinite(loss)
        nll += float(loss)
        count += length - 1
        metrics = dict(ppl=math.exp(nll / count), nll_sum=nll, tokens=count,
                       windows=i + 1, requested_windows=windows, seqlen=length)
        write(progress, metrics)
        print(f"PPL {i + 1}/{windows} {metrics['ppl']:.6f}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--ranks", nargs="+", type=int, choices=(64, 96), default=[64, 96])
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--checkpoints", type=Path, default=BANKS)
    parser.add_argument("--upstream", type=Path, default=ROOT / "external/KVQuant")
    parser.add_argument("--output", type=Path, default=ROOT / "results/q3-kv4-fp8")
    args = parser.parse_args()
    assert len(set(args.ranks)) == len(args.ranks)
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = args.upstream / "quant/kvquant/simquant_module_quantizer.py"
    spec = importlib.util.spec_from_file_location("kvquant_official", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    output = args.output / args.phase
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "source"
    snapshots.mkdir(exist_ok=True)
    for path in (Path(__file__), ROOT / "basisserve/core/qwen3_kv4_fp8_quality.py",
                 ROOT / "basisserve/kernels/fp8_wire.py", source):
        destination = snapshots / path.name
        if destination.exists():
            assert destination.read_bytes() == path.read_bytes(), str(path)
        else:
            shutil.copy2(path, destination)
    length = 256 if args.phase == "smoke" else 2048
    calibration_length = 128 if args.phase == "smoke" else 2048
    calibration_windows = 1 if args.phase == "smoke" else 16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train = _token_ids(tokenizer, "wikitext2", "train", None).reshape(-1)
    test = _token_ids(tokenizer, "wikitext2", "test", None).reshape(-1)
    starts = torch.randint(0, train.numel() - calibration_length, (calibration_windows,),
                           generator=torch.Generator().manual_seed(0)).tolist()
    windows = 2 if args.phase == "smoke" else test.numel() // length
    protocol = dict(model=str(args.model), checkpoint_root=str(args.checkpoints), hf_revision=HF_REV,
        tp=1, batch_size=1, environment="basis", gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, dataset="WikiText2", dataset_revision=WIKITEXT_REVISION,
        evaluation_split="test", seqlen=length, windows=windows,
        calibration_split="train", calibration_length=calibration_length, calibration_starts=starts,
        arms=ARMS, factor_validation="structure only; no SHA256", ranks=args.ranks,
        kv="official NUQ4; 0.99 percentile outlier rule; no first-token exclusion; no rotation",
        key_location="after k_norm, before RoPE; static per-channel",
        value_location="active C1 latent coordinates; dynamic per-token over all KV heads",
        fisher="BF16 C1 arm-specific squared activation gradients; frozen across quantized arms",
        projections="folded V encoder and output decoder only; padded HF V/O slots",
        fp8="actual E4M3 W8A8 scaled GEMM; BF16 output; use_fast_accum=False",
        fp8_scales="tensorwise weights; static per-projection training absmax activation scales; matching KV setting",
        other_modules="BF16; full attention; no sparse routing; no MLP quantization",
        packed_cache=False, performance_benchmark=False,
        upstream_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip())
    manifest = output / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["protocol"] == json.loads(json.dumps(protocol))
    else:
        write(manifest, dict(protocol=protocol, command=shlex.join(sys.argv),
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    summary = {}
    for rank in args.ranks:
        directory = output / f"r{rank}"
        directory.mkdir(exist_ok=True)
        result_path = directory / "results.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            assert existing["status"] == "complete" and existing["protocol"] == json.loads(json.dumps(protocol))
            summary[str(rank)] = existing["results"]
            continue
        print(f"LOAD R{rank}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
            local_files_only=True, attn_implementation="sdpa").eval().cuda()
        model.config.use_cache = False
        for param in model.parameters():
            param.requires_grad_(False)
        projections, modules, indices, schedule = install_factors(model, args.checkpoints / f"Q3-8B-C1-R{rank}", rank)
        results = {}
        quantizers = None
        if (directory / "quantizers.pt").exists():
            quantizers = torch.load(directory / "quantizers.pt", map_location="cpu", weights_only=False)
        for arm in ARMS:
            print(f"ARM R{rank} {arm}", flush=True)
            write(directory / "progress.json", dict(arm=arm, phase="preparation"))
            for module in projections.values():
                module.fp8 = False
                module.reset_stats()
            kv4 = arm in ("kv4", "kv4_fp8")
            fp8 = arm in ("fp8", "kv4_fp8")
            if kv4 and quantizers is None:
                quantizers = fit_quantizers(model, upstream, modules, indices, train,
                                           starts, calibration_length, directory)
            handles = install_nuq4_hooks(upstream, quantizers, modules, indices) if kv4 else []
            scales = calibrate_fp8(model, projections, train, starts, calibration_length) if fp8 else None
            if fp8:
                write(directory / f"{arm}_scales.json", scales)
            for module in projections.values():
                module.fp8 = fp8
                module.reset_stats()
            started = time.perf_counter()
            metrics = evaluate(model, test, length, windows, directory / "progress.json")
            metrics.update(wall_seconds=time.perf_counter() - started, quant_kv=kv4, fp8_gemm=fp8,
                fp8_projection_stats={name: dict(calls=m.calls, input_elements=m.elements,
                    clipped_elements=int(m.clipped)) for name, m in projections.items()} if fp8 else {})
            if fp8:
                assert all(m.calls == windows for m in projections.values())
            results[arm] = metrics
            for handle in handles:
                handle.remove()
            write(directory / f"{arm}.json", metrics)
        baseline = results["bf16"]["ppl"]
        for metrics in results.values():
            metrics.update(delta_ppl=metrics["ppl"] - baseline,
                           relative_ppl_percent=100 * (metrics["ppl"] / baseline - 1))
        write(result_path, dict(status="complete", nominal_rank=rank, layer_ranks=schedule,
                                 protocol=protocol, results=results))
        summary[str(rank)] = results
        del model, projections, modules, indices, quantizers
        gc.collect()
        torch.cuda.empty_cache()
    write(output / "summary.json", summary)
    lines = ["# Qwen3-8B-Base KV4 + FP8 PPL", "", f"Phase: **{args.phase}**. Environment: `basis`; single L40S.",
        "", "Smoke numbers are diagnostics, not full-test PPL." if args.phase == "smoke" else
        "Full WikiText2 test, non-overlapping 2048-token windows, B1; FP32 summed cross entropy.",
        "", "| Nominal rank | Arm | PPL | Delta PPL | Change |", "|---:|---|---:|---:|---:|"]
    for rank, results in summary.items():
        for arm, metrics in results.items():
            lines.append(f"| {rank} | {arm} | {metrics['ppl']:.6f} | {metrics['delta_ppl']:+.6f} | {metrics['relative_ppl_percent']:+.3f}% |")
    lines += ["", "R64/R96 are equivalent average ranks of the original adaptive checkpoints, not uniform per-layer ranks.",
        "K/V NUQ4 uses the official quantize/dequantize quality simulation with outliers; this is not a packed-cache speed test.",
        "FP8 uses actual E4M3 GEMMs for folded V encoder and output decoder only; other components remain BF16.",
        "All fitted quantities use WT2 train, never test. Activation scales are frozen for evaluation, not inferred from future test tokens.",
        "Structure-only factor validation; no SHA256. No TP8 verification.", "", "Command:", "```bash", shlex.join(sys.argv), "```", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines))
    print("COMPLETE", args.phase, flush=True)


if __name__ == "__main__":
    main()
