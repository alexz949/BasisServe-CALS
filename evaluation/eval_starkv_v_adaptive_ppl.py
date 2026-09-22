#!/usr/bin/env python3
"""Matched WikiText2/C4 PPL for a fused STAR-KV adaptive-rank checkpoint."""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.eval_qwen3_8b_iclr_quality import (
    _atomic_json,
    _eval_ppl_fp32_loss,
    _evaluate_document_ppl,
    _load_windows,
    _sha256,
)
sys.path.insert(0, str(ROOT / "external/STAR-KV"))
from model import build_fused_from_state_dict


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    assert torch.cuda.device_count() == 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert not (args.output_dir / "result.json").exists()

    checkpoint_path = args.checkpoint.resolve()
    training = json.loads((checkpoint_path / "result.json").read_text())
    assert training["status"] == "complete"
    assert training["protocol"]["method"] == "STAR-KV V-only adaptive-rank adaptation"
    target_mean_rank = training["protocol"]["target_compressed_layer_mean_rank"]
    assert training["budget"]["exact_budget_match"]
    arm = f"star_v_adaptive_r{target_mean_rank}"

    base_manifest = json.loads(
        (ROOT / "ICLR-results/qwen3-8b/checkpoints/Q3-8B-Dense/manifest.json").read_text()
    )
    model_path = Path(base_manifest["model"]["path"])
    assert training["protocol"]["model"] == str(model_path)
    assert _sha256(model_path / "config.json") == base_manifest["model"]["config_sha256"]
    windows, provenance = _load_windows(
        ROOT / "results/calibration/qwen3_8b_c4_validation_128x2048/windows.safetensors",
        model_path=model_path,
    )

    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    artifact = checkpoint_path / "fused.pt"
    state = torch.load(artifact, map_location="cpu", mmap=True, weights_only=True)
    skip_layers = tuple(training["protocol"]["skip_layers"])
    build_fused_from_state_dict(model, model.config, state, skip_layers=skip_layers)
    model.load_state_dict(state, strict=True)
    actual_ranks = [
        layer.self_attn.v_proj.VS.out_features
        if hasattr(layer.self_attn.v_proj, "VS")
        else layer.self_attn.v_proj.out_features
        for layer in model.model.layers
    ]
    assert actual_ranks == training["budget"]["ranks"]
    del state
    gc.collect()
    model.eval()

    checkpoint = {
        "artifact": str(artifact),
        "sha256": _sha256(artifact),
        "model": base_manifest["model"],
        "training": training,
    }
    common = {
        "arm": arm,
        "command": shlex.join(sys.argv),
        "checkpoint": checkpoint,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.executable,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "gpu": torch.cuda.get_device_name(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "protocol": {
            "dtype": "bfloat16", "attention": "sdpa", "batch_size": 2,
            "sequence_length": 2048, "loss_dtype": "float32",
            "wikitext": "wikitext2 test; concatenated full 2048-token chunks",
            "c4": "128 fixed validation document windows; no cross-document transitions",
            "quality_only": True,
        },
    }
    _atomic_json(args.output_dir / "protocol.json", common)
    wiki = _eval_ppl_fp32_loss(
        model, tokenizer, dataset="wikitext2", split="test",
        seqlen=2048, batch_size=2, max_samples=None, max_tokens=None,
    )
    _atomic_json(args.output_dir / "wikitext2.json", {**common, "metrics": wiki})
    c4 = _evaluate_document_ppl(model, windows, batch_size=2, label=arm)
    _atomic_json(
        args.output_dir / "c4.json",
        {**common, "windows": provenance, "metrics": c4},
    )
    result = {
        **common, "status": "complete", "metrics": {"wikitext2": wiki, "c4": c4},
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
