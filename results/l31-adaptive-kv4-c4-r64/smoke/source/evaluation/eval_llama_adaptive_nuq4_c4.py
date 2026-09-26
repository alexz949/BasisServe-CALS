"""Paired BF16/KV4 C4 PPL for archived Llama Two-Sided KL V64/V96 allocations."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from basisserve.core.llama_adaptive_nuq4_quality import install_adaptive_factors
from basisserve.core.qwen3_kv4_fp8_quality import install_nuq4_hooks
from evaluation.calibrate_llama_nuq4 import MODEL
from evaluation.eval_qwen3_kv4_fp8_ppl import fit_quantizers, evaluate, write
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids, WIKITEXT_REVISION

DATA = ROOT / "ICLR-results/llama31-8b-instruct"
WINDOWS = DATA / "calibration/c4_validation_128x2048/windows.safetensors"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--rank", type=int, choices=(64, 96), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    factors = DATA / f"c1/two-sided-kl/R{args.rank}-D0-S8"
    reference_path = DATA / f"quality/L31-8B-I-C1-R{args.rank}/c4.json"
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output.resolve() / args.phase
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "result.json").exists()
    reference = json.loads(reference_path.read_text())
    assert reference["model"]["path"] == str(MODEL)
    assert reference["checkpoint"]["directory"] == str(factors)
    assert reference["compression"]["equivalent_rank_target"] == args.rank
    assert reference["compression"]["allocation"] == "two_sided_factorized_terminal_kl_alpha1"
    schedule = reference["compression"]["layer_ranks"]
    data = load_file(str(WINDOWS))
    assert set(data) == {"input_ids"} and data["input_ids"].shape == (128, 2048)
    windows = 2 if args.phase == "smoke" else 128
    test = data["input_ids"][:windows].long().reshape(-1)
    assert test.min() >= 0 and test.max() < 128256
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    train = _token_ids(tokenizer, "wikitext2", "train", None).reshape(-1)
    length, samples = (128, 1) if args.phase == "smoke" else (2048, 16)
    starts = torch.randint(0, train.numel() - length, (samples,),
                           generator=torch.Generator().manual_seed(0)).tolist()
    sources = [ROOT / p for p in (
        "evaluation/eval_llama_adaptive_nuq4_c4.py", "basisserve/core/llama_adaptive_nuq4_quality.py",
        "basisserve/core/qwen3_kv4_fp8_quality.py", "evaluation/eval_qwen3_kv4_fp8_ppl.py",
        "evaluation/calibrate_llama_nuq4.py", "scripts/eval_svdllm_safetensors_ppl_accelerate.py",
        "external/KVQuant/quant/kvquant/simquant_module_quantizer.py")]
    upstream_source = sources[-1]
    sources.append(ROOT / "tests/test_llama_adaptive_nuq4_quality.py")
    if args.phase == "formal":
        smoke = json.loads((args.output / "smoke/result.json").read_text())
        assert smoke["status"] == "complete" and smoke["protocol"]["layer_ranks"] == schedule
        for source in sources:
            assert source.read_bytes() == (args.output / "smoke/source" / source.relative_to(ROOT)).read_bytes()
    for source in sources:
        target = output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, target)
    protocol = dict(model=str(MODEL), checkpoint=str(factors), historical_reference=str(reference_path),
        historical_c4_bf16_ppl=reference["metrics"]["ppl"], layer_ranks=schedule,
        equivalent_rank=args.rank, allocation=f"Two-Sided KL, equivalent V{args.rank} (not uniform V{args.rank})",
        phase=args.phase, environment="basis", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        calibration_split="WT2 train", calibration_dataset_revision=WIKITEXT_REVISION,
        calibration_length=length, calibration_starts=starts,
        evaluation_split="C4 validation", evaluation_windows=str(WINDOWS),
        windows=windows, seqlen=2048, scored_tokens=windows * 2047, batch_size=1, tp=1,
        kv="official Fisher-weighted NUQ4 + 0.99 outlier rule; no rotation or first-token exclusion",
        k="static per-channel pre-RoPE", v="dynamic per-token over all active adaptive latent coordinates",
        other_modules="BF16; encoder/decoder BF16; no A8, FP8 GEMM, sparse routing or MLP change",
        factor_validation="structure and rank schedule only; no SHA256",
        packed_cache=False, performance_benchmark=False)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["protocol"] == protocol
    else:
        write(manifest_path, dict(protocol=protocol, command=shlex.join(sys.argv)))
    spec = importlib.util.spec_from_file_location("kvquant_adaptive_c4", upstream_source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    print(f"LOAD adaptive V{args.rank}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    model.requires_grad_(False)
    modules, indices = install_adaptive_factors(model, factors, schedule, args.rank)
    path = output / "quantizers.pt"
    if path.exists():
        codes = torch.load(path, map_location="cpu", weights_only=False)
    else:
        print("CALIBRATION WT2 train", samples, length, flush=True)
        codes = fit_quantizers(model, upstream, modules, indices, train, starts, length, output)
    assert set(codes) == set(modules)
    for name, (hi, lo, lut) in codes.items():
        assert hi.shape == lo.shape and torch.isfinite(hi).all() and torch.isfinite(lo).all()
        assert (hi >= lo).all()
        poles = torch.as_tensor(lut[0]).flatten()
        assert poles.numel() == 16 and torch.isfinite(poles).all()
        if name.endswith(".k"):
            assert hi.numel() == 1024
    results = {}
    for arm in ("bf16", "kv4"):
        print("C4 ARM", arm, flush=True)
        handles = install_nuq4_hooks(upstream, codes, modules, indices) if arm == "kv4" else []
        started = time.perf_counter()
        metrics = evaluate(model, test, 2048, windows, output / f"{arm}_progress.json")
        assert metrics["tokens"] == windows * 2047
        metrics["evaluation_wall_seconds"] = time.perf_counter() - started
        results[arm] = metrics
        for handle in handles:
            handle.remove()
        write(output / f"{arm}.json", metrics)
    for source in sources:
        assert source.read_bytes() == (output / "source" / source.relative_to(ROOT)).read_bytes()
    delta = results["kv4"]["ppl"] - results["bf16"]["ppl"]
    relative = 100 * delta / results["bf16"]["ppl"]
    write(output / "result.json", dict(status="complete", protocol=protocol, results=results,
        delta_ppl=delta, relative_ppl_percent=relative, source_bytes_match=True))
    lines = [f"# Llama-3.1-8B-Instruct Two-Sided KL V{args.rank}: C4 KV4 PPL", "",
        f"Phase: **{args.phase}**; environment: `basis`, one L40S, B1/TP1.", "",
        "Smoke calibration and evaluation only; not a formal quality result." if args.phase == "smoke" else
        "Frozen C4 validation 128 x 2048 windows; 262,016 scored tokens, no cross-document transitions.", "",
        "| Arm | C4 PPL |", "|---|---:|",
        f"| Same-checkpoint BF16 | {results['bf16']['ppl']:.6f} |",
        f"| Same-checkpoint KV4 | {results['kv4']['ppl']:.6f} |", "",
        f"KV4 increment: **{delta:+.6f} PPL ({relative:+.3f}%)**.",
        f"Historical BF16 full-C4 reference: {reference['metrics']['ppl']:.6f}; not used to calculate the increment.", "",
        f"Checkpoint: `{factors.relative_to(ROOT)}`.",
        f"Ranks are adaptive, averaging {args.rank}; this is not a uniform-rank experiment.",
        f"Fresh calibration: WT2 train, {samples} x {length} tokens, seed 0; no C4 validation used for fitting.",
        "K: pre-RoPE per-channel NUQ4; V: active latent coordinates, per-token over all KV heads.",
        "Official Fisher-weighted NUQ4 plus 0.99 outlier rule, no rotation or first-token exclusion.",
        "Encoder/decoder and all other modules remain BF16; no A8 or FP8 GEMM.",
        "Quantize/dequantize quality simulation, not packed-cache serving performance.",
        "No SHA256 checks. Codebook checks and final source-byte comparisons passed.",
        "No GitHub/HF upload or old result overwrite.", "", "Command (basis):", "```bash",
        shlex.join(sys.argv), "```", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines))
    print("COMPLETE", args.phase, "BF16", results["bf16"]["ppl"], "KV4", results["kv4"]["ppl"], flush=True)


if __name__ == "__main__":
    main()
