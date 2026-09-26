"""Short R64 order/weight-mutation audit; not a formal quality benchmark."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from basisserve.core.qwen3_kv4_fp8_quality import install_factors, install_nuq4_hooks
from evaluation.eval_qwen3_kv4_fp8_ppl import (
    BANKS, MODEL, calibrate_fp8, evaluate, write,
)
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = ROOT / "results/q3-kv4-fp8/smoke/audit"
    assert not (output / "results.json").exists()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "external/KVQuant/quant/kvquant/simquant_module_quantizer.py"
    spec = importlib.util.spec_from_file_location("kvquant_audit", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokens = _token_ids(tokenizer, "wikitext2", "test", None).reshape(-1)
    train = _token_ids(tokenizer, "wikitext2", "train", None).reshape(-1)
    starts = torch.randint(0, train.numel() - 128, (1,), generator=torch.Generator().manual_seed(0)).tolist()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").eval().cuda()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    projections, modules, indices, schedule = install_factors(model, BANKS / "Q3-8B-C1-R64", 64)
    reference = {name: m.weight.cpu().clone() for name, m in projections.items()}
    quantizers = torch.load(output.parent / "r64/quantizers.pt", map_location="cpu", weights_only=False)
    results = {}
    for arm in ("bf16_first", "bf16_repeat", "identity", "kv4", "bf16_after_kv4", "fp8", "bf16_after_fp8"):
        handles = []
        if arm == "kv4":
            handles = install_nuq4_hooks(upstream, quantizers, modules, indices)
        elif arm == "identity":
            handles = [m.register_forward_hook(lambda m, x, y: y.clone()) for m in modules.values()]
        elif arm == "fp8":
            calibrate_fp8(model, projections, train, starts, 128)
        for m in projections.values():
            m.fp8 = arm == "fp8"
        print("AUDIT", arm, flush=True)
        results[arm] = evaluate(model, tokens, 256, 2, output / "progress.json")
        for h in handles:
            h.remove()
        assert all(torch.equal(m.weight.cpu(), reference[name]) for name, m in projections.items())
    baseline = results["bf16_first"]["nll_sum"]
    deltas = {arm: r["nll_sum"] - baseline for arm, r in results.items()
              if arm.startswith("bf16") or arm == "identity"}
    status = "passed" if all(abs(v) < 1e-6 for v in deltas.values()) else "investigate"
    write(output / "results.json", dict(status=status, results=results,
        control_nll_deltas=deltas, projection_weights_unchanged=True, scope="two-window smoke audit, not full PPL"))
    print("AUDIT_COMPLETE", status, deltas, flush=True)


if __name__ == "__main__":
    main()
