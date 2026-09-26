"""Llama-3.1-8B-Instruct V96 KV4 PPL with frozen train-only NUQ4 codebooks."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from basisserve.core.llama_nuq4_quality import install_llama_factors
from basisserve.core.qwen3_kv4_fp8_quality import install_nuq4_hooks
from evaluation.calibrate_llama_nuq4 import MODEL
from evaluation.eval_qwen3_kv4_fp8_ppl import evaluate, write
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids, WIKITEXT_REVISION


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/l31-nuq4-ppl")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output.resolve() / args.phase
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "result.json").exists()
    calibration = ROOT / "results/l31-nuq4/formal/r96"
    manifest = json.loads((calibration / "manifest.json").read_text())
    complete = json.loads((calibration / "complete.json").read_text())
    assert complete["status"] == "complete" and complete["rank"] == 96
    assert manifest["phase"] == "formal" and manifest["rank"] == 96
    assert manifest["length"] == 2048 and len(manifest["starts"]) == 16
    assert manifest["calibration_split"] == "WT2 train" and manifest["model"] == str(MODEL)
    sources = [ROOT / name for name in (
        "evaluation/calibrate_llama_nuq4.py", "basisserve/core/llama_nuq4_quality.py",
        "basisserve/core/qwen3_kv4_fp8_quality.py", "evaluation/eval_qwen3_kv4_fp8_ppl.py",
        "external/KVQuant/quant/kvquant/simquant_module_quantizer.py")]
    for source in sources:
        assert source.read_bytes() == (calibration / "source" / source.name).read_bytes()
    upstream_source = sources[-1]
    sources += [Path(__file__).resolve(), ROOT / "scripts/eval_svdllm_safetensors_ppl_accelerate.py"]
    if args.phase == "formal":
        smoke = json.loads((args.output / "smoke/result.json").read_text())
        assert smoke["status"] == "complete" and smoke["protocol"]["calibration"] == str(calibration)
        for source in sources:
            assert source.read_bytes() == (args.output / "smoke/source" / source.relative_to(ROOT)).read_bytes()
    for source in sources:
        target = output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, target)
    spec = importlib.util.spec_from_file_location("kvquant_llama_ppl", upstream_source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    codes = torch.load(calibration / "quantizers.pt", map_location="cpu", weights_only=False)
    assert set(codes) == {f"{i}.{kind}" for i in range(32) for kind in ("k", "v")}
    for name, (hi, lo, lut) in codes.items():
        assert hi.shape == lo.shape and torch.isfinite(hi).all() and torch.isfinite(lo).all()
        assert (hi >= lo).all()
        poles = torch.as_tensor(lut[0]).flatten()
        assert poles.numel() == 16 and torch.isfinite(poles).all()
        if name.endswith(".k"):
            assert hi.numel() == 1024
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    test = _token_ids(tokenizer, "wikitext2", "test", None).reshape(-1)
    length = 2048
    windows = 2 if args.phase == "smoke" else test.numel() // length
    assert test.numel() >= windows * length
    protocol = dict(model=str(MODEL), factors=manifest["factors"], rank=96, arm="kv4",
        phase=args.phase, environment="basis", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        calibration=str(calibration), calibration_manifest=manifest,
        dataset="WikiText2", dataset_revision=WIKITEXT_REVISION, split="test",
        batch_size=1, tp=1, seqlen=length, windows=windows, total_test_tokens=test.numel(),
        evaluated_input_tokens=windows * length, scored_tokens=windows * (length - 1),
        unused_test_tokens=test.numel() - windows * length,
        kv="Fisher-weighted NUQ4 + 0.99 outlier rule; no rotation or first-token exclusion",
        k="static per-channel, pre-RoPE", v="dynamic per-token over active V96 coordinates across eight KV heads",
        encoder_decoder="BF16; no A8 or FP8 GEMM", other_modules="BF16; full attention; no sparse routing",
        packed_cache=False, performance_benchmark=False, factor_validation="structure only; no SHA256")
    write(output / "manifest.json", dict(protocol=protocol, command=shlex.join(sys.argv)))
    print("LOAD V96 KV4", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    model.requires_grad_(False)
    projections, modules, indices = install_llama_factors(model, Path(manifest["factors"]), 96)
    assert all(not m.fp8 and m.weight.dtype == torch.bfloat16 for m in projections.values())
    handles = install_nuq4_hooks(upstream, codes, modules, indices)
    assert len(handles) == 64
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = evaluate(model, test, length, windows, output / "progress.json")
    torch.cuda.synchronize()
    metrics["evaluation_wall_seconds"] = time.perf_counter() - started
    assert metrics["tokens"] == protocol["scored_tokens"]
    assert all(not m.fp8 and m.calls == 0 for m in projections.values())
    for handle in handles:
        handle.remove()
    for source in sources:
        assert source.read_bytes() == (output / "source" / source.relative_to(ROOT)).read_bytes()
    write(output / "result.json", dict(status="complete", protocol=protocol, metrics=metrics,
                                      source_bytes_match=True, fp8_gemm_calls=0))
    lines = ["# Llama-3.1-8B-Instruct V96 KV4 PPL", "",
        f"Phase: **{args.phase}**. Environment: `basis`, one L40S, B1/TP1.", "",
        "Smoke only; not a formal quality result." if args.phase == "smoke" else
        "WT2 test, all complete non-overlapping 2048-token windows; incomplete tail excluded.", "",
        f"- PPL: **{metrics['ppl']:.6f}**.", f"- Windows: {windows}; scored tokens: {metrics['tokens']}.",
        f"- Unused test tokens: {protocol['unused_test_tokens']}.",
        f"- Evaluation wall time: {metrics['evaluation_wall_seconds']:.2f} s (not serving latency).", "",
        "Frozen formal R96 codebooks fitted on 16 x 2048 WT2 train tokens; no recalibration.",
        "NUQ4 with outliers quantizes full K before RoPE and active V96 latent coordinates.",
        "Encoder/decoder and other projections stay BF16; no A8, FP8 GEMM, sparse routing or MLP changes.",
        "Quality uses quantize/dequantize simulation, not packed-cache serving or TP8 performance.",
        "No matched BF16 baseline was run here; do not infer delta PPL from unrelated evaluations.",
        "Structure validation and source-byte comparisons only; no SHA256. No GitHub/HF upload.",
        "", "Command (basis):", "```bash", shlex.join(sys.argv), "```", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines))
    print("COMPLETE", args.phase, metrics["ppl"], flush=True)


if __name__ == "__main__":
    main()
