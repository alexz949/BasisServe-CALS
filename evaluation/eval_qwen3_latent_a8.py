"""Fixed NUQ4, BF16 encoder: BF16 latent, A8 latent, or A8 + W8 decoder."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from basisserve.core.qwen3_kv4_fp8_quality import install_factors, install_nuq4_hooks
from basisserve.core.qwen3_latent_a8_quality import install_latent_decoders
from evaluation.eval_qwen3_kv4_fp8_ppl import evaluate, write
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids

ARMS = ("bf16", "a8", "w8a8")
SOURCES = ("evaluation/eval_qwen3_latent_a8.py", "basisserve/core/qwen3_latent_a8_quality.py",
           "evaluation/eval_qwen3_kv4_fp8_ppl.py", "basisserve/core/qwen3_kv4_fp8_quality.py",
           "basisserve/kernels/fp8_wire.py", "external/KVQuant/quant/kvquant/simquant_module_quantizer.py")


def load_fixed_model(prior, rank):
    protocol = json.loads((prior / "manifest.json").read_text())["protocol"]
    assert protocol["environment"] == "basis" and protocol["seqlen"] == 2048
    assert torch.__version__ == protocol["torch"] and torch.cuda.get_device_name() == protocol["gpu"]
    for name in SOURCES[2:]:
        path = ROOT / name
        assert path.read_bytes() == (prior / "source" / path.name).read_bytes(), name
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = ROOT / SOURCES[-1]
    spec = importlib.util.spec_from_file_location("fixed_kvquant", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parents[2], text=True).strip() == protocol["upstream_commit"]
    model = AutoModelForCausalLM.from_pretrained(protocol["model"], dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").eval().cuda()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    projections, modules, indices, schedule = install_factors(model,
        Path(protocol["checkpoint_root"]) / f"Q3-8B-C1-R{rank}", rank)
    codes = torch.load(prior / f"r{rank}/quantizers.pt", map_location="cpu", weights_only=False)
    handles = install_nuq4_hooks(upstream, codes, modules, indices)
    scales = json.loads((prior / f"r{rank}/kv4_fp8_scales.json").read_text())
    decoders = install_latent_decoders(model, projections, scales)
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True)
    return model, tokenizer, projections, decoders, handles, schedule, protocol


def worker(args):
    output = args.output / args.phase / f"r{args.rank}"
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer, projections, decoders, handles, schedule, protocol = load_fixed_model(args.prior, args.rank)
    tokens = _token_ids(tokenizer, "wikitext2", "test", None).reshape(-1)
    windows = 2 if args.phase == "smoke" else tokens.numel() // 2048
    assert tokens.numel() // 2048 == protocol["windows"] == 146
    write(output / "manifest.json", dict(command=shlex.join(sys.argv), phase=args.phase,
        prior=str(args.prior), nominal_rank=args.rank, schedule=schedule, windows=windows,
        protocol=protocol, overrides=dict(encoder="BF16 in every arm", kv="fixed previous formal NUQ4",
        latent="decoder input after attention; scalar per-layer E4M3 scale",
        scales="reused train-only kv4_fp8 decoder scales; shared unchanged across new arms",
        arms=ARMS, tp=1, collectives=False, performance_benchmark=False)))
    write(output / "scales.json", {n: dict(input_scale=float(m.input_scale),
        weight_scale=float(m.weight_scale)) for n, m in decoders.items()})
    # Calibration was collected with BF16 projections and matching KV4 in the prior run.
    if args.phase == "smoke":
        train = _token_ids(tokenizer, "wikitext2", "train", None).reshape(-1)
        start = protocol["calibration_starts"][0]
        batch = train[start:start + 128][None].cuda()
        with torch.inference_mode():
            first = model.model(input_ids=batch, use_cache=False).last_hidden_state.clone()
            for mode in ("a8", "w8a8"):
                for decoder in decoders.values():
                    decoder.mode = mode
                assert torch.isfinite(model.model(input_ids=batch, use_cache=False).last_hidden_state).all()
            for decoder in decoders.values():
                decoder.mode = "bf16"
            repeated = model.model(input_ids=batch, use_cache=False).last_hidden_state
            torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        write(output / "order_check.json", dict(status="passed", bf16_after_modes_exact=True))
    results = {}
    for arm in ARMS:
        assert not (output / f"{arm}.json").exists()
        for module in projections.values():
            module.reset_stats()
        for decoder in decoders.values():
            decoder.mode = arm
        print("ARM", args.rank, arm, flush=True)
        start = time.perf_counter()
        metrics = evaluate(model, tokens, 2048, windows, output / f"{arm}_progress.json")
        encoder_calls = sum(m.calls for n, m in projections.items() if n.endswith("encoder"))
        assert encoder_calls == 0 and all(not m.fp8 for n, m in projections.items() if n.endswith("encoder"))
        calls = sum(m.calls for m in decoders.values())
        assert calls == (0 if arm == "bf16" else 36 * windows)
        metrics.update(wall_seconds=time.perf_counter() - start, encoder_fp8_calls=encoder_calls,
            latent_quantize_calls=calls, decoder_fp8_gemm_calls=calls if arm == "w8a8" else 0,
            projection_stats={n: dict(calls=m.calls, input_elements=m.elements,
                clipped_elements=int(m.clipped)) for n, m in decoders.items()})
        if arm == "bf16" and args.phase == "formal":
            old = json.loads((args.prior / f"r{args.rank}/kv4.json").read_text())
            assert math.isclose(metrics["ppl"], old["ppl"], rel_tol=1e-6)
        write(output / f"{arm}.json", metrics)
        results[arm] = metrics
        print("ARM_COMPLETE", args.rank, arm, metrics["ppl"], flush=True)
    for handle in handles:
        handle.remove()
    write(output / "results.json", dict(status="complete", results=results))


def summarize(args):
    directory = args.output / args.phase
    rows = {}
    lines = ["# BF16 Encoder, A8 Latent and W8 Decoder", "",
        f"Phase: {args.phase}. Environment: basis. Qwen3-8B-Base, adaptive R64/R96.",
        "KV4 NUQ4 is fixed in all arms. This is single-GPU PPL, not a TP8 communication benchmark.", "",
        "| Rank | KV4 + BF16 | KV4 + A8 | KV4 + A8/W8 | A8 delta | W8 increment over A8 |",
        "|---:|---:|---:|---:|---:|---:|"]
    for rank in args.ranks:
        result = json.loads((directory / f"r{rank}/results.json").read_text())["results"]
        rows[str(rank)] = result
        b, a, w = [result[arm]["ppl"] for arm in ARMS]
        lines.append(f"| {rank} | {b:.6f} | {a:.6f} | {w:.6f} | {a-b:+.6f} | {w-a:+.6f} |")
    lines += ["", "Encoder remains BF16. A8: E4M3 quantize/dequantize then BF16 decoder GEMM.",
        "A8/W8: the same latent scale, E4M3 decoder weights, actual FP8 GEMM with BF16 output.",
        "Static scales come from the previous full train-only calibration under BF16 projections + KV4.",
        "PPL includes quantized upstream layers. W8 increment is an end-to-end difference, not independent additive error.",
        "No SHA256. NUQ4 remains quantize/dequantize quality simulation, not packed INT4 storage.",
        "Smoke covers only two 2048-token windows; formal covers all 146 windows / 298862 targets.",
        "Per-worker commands and settings: r64/manifest.json and r96/manifest.json.", ""]
    write(directory / "summary.json", rows)
    (directory / "SUMMARY.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--prior", type=Path, default=ROOT / "results/q3-kv4-fp8/formal")
    parser.add_argument("--output", type=Path, default=ROOT / "results/q3-a8-dec")
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 96], choices=(64, 96))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--rank", type=int, choices=(64, 96))
    args = parser.parse_args()
    if args.rank is not None:
        worker(args)
        return
    assert len(args.ranks) == len(args.gpus) and len(set(args.gpus)) == len(args.gpus)
    directory = args.output / args.phase
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source"
    source.mkdir(exist_ok=True)
    for name in SOURCES:
        original = ROOT / name
        destination = source / original.name
        if destination.exists():
            assert original.read_bytes() == destination.read_bytes()
        else:
            shutil.copy2(original, destination)
    write(directory / "command.json", dict(command=shlex.join(sys.argv),
        created_utc=datetime.now(timezone.utc).isoformat(), environment="basis", gpus=args.gpus))
    running = []
    for rank, gpu in zip(args.ranks, args.gpus):
        log = (directory / f"r{rank}.log").open("a")
        command = [sys.executable, str(Path(__file__).resolve()), "--phase", args.phase,
            "--prior", str(args.prior), "--output", str(args.output), "--rank", str(rank)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        running.append((rank, process, log))
        print("START", rank, gpu, process.pid, flush=True)
    outcomes = []
    for rank, process, log in running:
        code = process.wait()
        log.close()
        outcomes.append(dict(rank=rank, returncode=code))
    write(directory / "outcomes.json", outcomes)
    assert all(row["returncode"] == 0 for row in outcomes), outcomes
    summarize(args)
    print("COMPLETE", args.phase, flush=True)


if __name__ == "__main__":
    main()
