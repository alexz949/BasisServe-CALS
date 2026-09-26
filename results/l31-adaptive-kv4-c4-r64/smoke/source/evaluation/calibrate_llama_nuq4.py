"""Train-only Llama NUQ4 codebooks and decoder-input A8 scales."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from basisserve.core.llama_nuq4_quality import install_llama_factors
from basisserve.core.qwen3_kv4_fp8_quality import install_nuq4_hooks
from evaluation.eval_qwen3_kv4_fp8_ppl import fit_quantizers, calibrate_fp8, evaluate, write
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids

MODEL = Path("/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--rank", type=int, choices=(64,96), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/l31-nuq4")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = args.output / args.phase / f"r{args.rank}"
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "complete.json").exists()
    source = ROOT / "external/KVQuant/quant/kvquant/simquant_module_quantizer.py"
    spec = importlib.util.spec_from_file_location("kvquant_llama", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    factor_root = ROOT / f"ICLR-results/llama31-8b-instruct/c1/factor-banks/R{args.rank}-S6-D0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    train = _token_ids(tokenizer, "wikitext2", "train", None).reshape(-1)
    length, samples = (128,1) if args.phase == "smoke" else (2048,16)
    starts = torch.randint(0,train.numel()-length,(samples,),generator=torch.Generator().manual_seed(0)).tolist()
    write(directory / "manifest.json",dict(command=shlex.join(sys.argv),environment="basis",
        phase=args.phase,rank=args.rank,model=str(MODEL),factors=str(factor_root),
        calibration_split="WT2 train",starts=starts,length=length,
        quantizer="official Fisher-weighted NUQ4, outlier threshold 0.99, no first-token exclusion",
        k="static per-channel, pre-RoPE",v="dynamic per-token, all eight KV heads",
        scale="static per-layer decoder input absmax, matching KV4, BF16 encoder/decoder",
        factor_validation="structure only; no SHA256"))
    snapshots = directory / "source"
    snapshots.mkdir(exist_ok=True)
    for name in (Path(__file__), ROOT / "basisserve/core/llama_nuq4_quality.py",
                 ROOT / "basisserve/core/qwen3_kv4_fp8_quality.py",
                 ROOT / "evaluation/eval_qwen3_kv4_fp8_ppl.py", source):
        destination = snapshots / name.name
        if destination.exists():
            assert destination.read_bytes() == name.read_bytes()
        else:
            shutil.copy2(name,destination)
    model = AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa").eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    projections, modules, indices = install_llama_factors(model,factor_root,args.rank)
    quantizer_path = directory / "quantizers.pt"
    if quantizer_path.exists():
        codes = torch.load(quantizer_path,map_location="cpu",weights_only=False)
    else:
        codes = fit_quantizers(model,upstream,modules,indices,train,starts,length,directory)
    handles = install_nuq4_hooks(upstream,codes,modules,indices)
    decoders = {n:m for n,m in projections.items() if n.endswith("decoder")}
    scales = calibrate_fp8(model,decoders,train,starts,length)
    write(directory / "a8_scales.json",scales)
    if args.phase == "smoke":
        test = _token_ids(tokenizer,"wikitext2","test",None).reshape(-1)
        metrics = evaluate(model,test,256,2,directory / "progress.json")
        write(directory / "smoke_ppl.json",metrics)
    for handle in handles:
        handle.remove()
    write(directory / "complete.json",dict(status="complete",rank=args.rank,
        quantizers="quantizers.pt",scales="a8_scales.json",calibration_only=True))
    print("COMPLETE",args.phase,args.rank,flush=True)


if __name__ == "__main__":
    main()
