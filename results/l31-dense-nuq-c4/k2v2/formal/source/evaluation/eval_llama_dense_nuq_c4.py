"""Paired BF16/low-bit NUQ C4 PPL for dense (full-rank) Llama-3.1-8B-Instruct K/V."""

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
from basisserve.core.qwen3_kv4_fp8_quality import install_nuq4_hooks
from evaluation.calibrate_llama_nuq4 import MODEL
from evaluation.eval_qwen3_kv4_fp8_ppl import fit_quantizers, evaluate, write
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids, WIKITEXT_REVISION

DATA = ROOT / "ICLR-results/llama31-8b-instruct"
WINDOWS = DATA / "calibration/c4_validation_128x2048/windows.safetensors"
REFERENCE = DATA / "quality/L31-8B-I-Dense/c4.json"
ARMS = {"k4v2": (4, 2), "k2v2": (2, 2)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    k_bits, v_bits = ARMS[args.arm]
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.output.resolve() / args.arm
    output = root / args.phase
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "result.json").exists()
    reference = json.loads(REFERENCE.read_text())
    assert reference["model"]["path"] == str(MODEL)
    assert reference["compression"]["method"] == "dense" and reference["metrics"]["tokens"] == 262016
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
        "evaluation/eval_llama_dense_nuq_c4.py", "basisserve/core/qwen3_kv4_fp8_quality.py",
        "evaluation/eval_qwen3_kv4_fp8_ppl.py", "evaluation/calibrate_llama_nuq4.py",
        "scripts/eval_svdllm_safetensors_ppl_accelerate.py", "tests/test_qwen3_kv4_fp8_quality.py",
        "external/KVQuant/quant/kvquant/simquant_module_quantizer.py")]
    if args.phase == "formal":
        smoke = json.loads((root / "smoke/result.json").read_text())
        assert smoke["status"] == "complete" and smoke["protocol"]["arm"] == args.arm
        for source in sources:
            assert source.read_bytes() == (root / "smoke/source" / source.relative_to(ROOT)).read_bytes()
    for source in sources:
        target = output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, target)
    ratio = 32 / (k_bits + v_bits)
    protocol = dict(model=str(MODEL), checkpoint="dense full-rank K/V; no C1 factors",
        historical_reference=str(REFERENCE), historical_c4_bf16_ppl=reference["metrics"]["ppl"],
        arm=args.arm, k_bits=k_bits, v_bits=v_bits, nominal_kv_compression=ratio,
        phase=args.phase, environment="basis", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        calibration_split="WT2 train", calibration_dataset_revision=WIKITEXT_REVISION,
        calibration_length=length, calibration_starts=starts,
        evaluation_split="C4 validation", evaluation_windows=str(WINDOWS),
        windows=windows, seqlen=2048, scored_tokens=windows * 2047, batch_size=1, tp=1,
        kv=f"official Fisher-weighted NUQ (K{k_bits}/V{v_bits}) + 0.99 outlier rule; no rotation or first-token exclusion",
        k="static per-channel pre-RoPE over 1024 channels", v="dynamic per-token over all 1024 dense V channels",
        other_modules="BF16; no A8, FP8 GEMM, sparse routing or MLP change",
        packed_cache=False, performance_benchmark=False)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["protocol"] == protocol
    else:
        write(manifest_path, dict(protocol=protocol, command=shlex.join(sys.argv)))
    spec = importlib.util.spec_from_file_location("kvquant_dense_c4", sources[-1])
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    print(f"LOAD dense {args.arm}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    model.requires_grad_(False)
    c = model.config
    assert (c.model_type, c.hidden_size, c.num_hidden_layers, c.num_attention_heads,
            c.num_key_value_heads) == ("llama", 4096, 32, 32, 8)
    modules, indices = {}, {}
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        assert a.k_proj.weight.shape == a.v_proj.weight.shape == (1024, 4096)
        assert a.k_proj.bias is None and a.v_proj.bias is None
        modules[f"{i}.k"], modules[f"{i}.v"] = a.k_proj, a.v_proj
        indices[f"{i}.k"] = indices[f"{i}.v"] = torch.arange(1024, device=a.k_proj.weight.device)
    bits = {name: k_bits if name.endswith(".k") else v_bits for name in modules}
    path = output / "quantizers.pt"
    if path.exists():
        codes = torch.load(path, map_location="cpu", weights_only=False)
    else:
        print("CALIBRATION WT2 train", samples, length, flush=True)
        codes = fit_quantizers(model, upstream, modules, indices, bits, train, starts, length, output)
    assert set(codes) == set(modules)
    for name, (hi, lo, lut) in codes.items():
        assert hi.shape == lo.shape and torch.isfinite(hi).all() and torch.isfinite(lo).all()
        assert (hi >= lo).all()
        poles = torch.as_tensor(lut[0]).flatten()
        assert poles.numel() == 2 ** bits[name] and torch.isfinite(poles).all()
        if name.endswith(".k"):
            assert hi.numel() == 1024
    results = {}
    for arm in ("bf16", args.arm):
        print("C4 ARM", arm, flush=True)
        handles = install_nuq4_hooks(upstream, codes, modules, indices) if arm == args.arm else []
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
    delta = results[args.arm]["ppl"] - results["bf16"]["ppl"]
    relative = 100 * delta / results["bf16"]["ppl"]
    write(output / "result.json", dict(status="complete", protocol=protocol, results=results,
        delta_ppl=delta, relative_ppl_percent=relative, source_bytes_match=True))
    lines = [f"# Llama-3.1-8B-Instruct Dense K{k_bits}V{v_bits}: C4 PPL", "",
        f"Phase: **{args.phase}**; environment: `basis`, one L40S, B1/TP1.", "",
        "Smoke calibration and evaluation only; not a formal quality result." if args.phase == "smoke" else
        "Frozen C4 validation 128 x 2048 windows; 262,016 scored tokens, no cross-document transitions.", "",
        "| Arm | C4 PPL |", "|---|---:|",
        f"| Dense BF16 | {results['bf16']['ppl']:.6f} |",
        f"| Dense K{k_bits}V{v_bits} | {results[args.arm]['ppl']:.6f} |", "",
        f"Quantization increment: **{delta:+.6f} PPL ({relative:+.3f}%)**.",
        f"Historical dense BF16 C4 reference: {reference['metrics']['ppl']:.6f}; not used to calculate the increment.", "",
        f"Nominal KV compression {ratio:.2f}x from bit width alone (32 / ({k_bits} + {v_bits})); "
        "outliers, scales and metadata are excluded.",
        "Dense full-rank K/V; no C1 factors, low-rank V, A8, FP8 GEMM, sparse routing or MLP change.",
        f"Fresh calibration: WT2 train, {samples} x {length} tokens, seed 0; no C4 validation used for fitting.",
        f"K: pre-RoPE per-channel NUQ{k_bits}; V: per-token NUQ{v_bits} over all 1024 channels.",
        "Official Fisher-weighted NUQ plus 0.99 outlier rule, no rotation or first-token exclusion.",
        "Quantize/dequantize quality simulation, not packed-cache serving performance.",
        "No SHA256 checks. Codebook checks and final source-byte comparisons passed.",
        "No GitHub/HF upload or old result overwrite.", "", "Command (basis):", "```bash",
        shlex.join(sys.argv), "```", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines))
    print("COMPLETE", args.phase, args.arm, "BF16", results["bf16"]["ppl"],
          args.arm.upper(), results[args.arm]["ppl"], flush=True)


if __name__ == "__main__":
    main()
