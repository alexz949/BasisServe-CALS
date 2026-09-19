"""Evaluate Nemotron-H PaLU V-only checkpoints on the matched quality protocol."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results/tools/nemotron_deps"))
sys.path.insert(0, str(ROOT))

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from safetensors.torch import load_file
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss
from evaluation.eval_nemotron_h_8b_quality import _quality_windows
from evaluation.eval_qwen3_32b_c1_c4_ppl_shard import _evaluate_document_ppl
from evaluation.eval_qwen3_8b_iclr_quality import TASKS, summarize_commonsense, _write_json
from evaluation.nemotron_h_palu import CHECKPOINT_FORMAT, activate_fast_mamba
from evaluation.nemotron_h_runtime import install_mamba_device_guards
from evaluation.v96kl_common import configure, read_json, sha256
from palu.model.modules.svd_linear import HeadwiseLowRankModule

FORMAT = "basisserve.nemotron_h.palu_quality.v1"


@torch.inference_mode()
def install_palu_v(model, checkpoint):
    manifest = read_json(checkpoint / "manifest.json")
    assert manifest["status"] == "complete" and manifest["format"] == CHECKPOINT_FORMAT
    assert manifest["model"]["config_sha256"] == sha256(
        Path(manifest["model"]["path"]) / "config.json"
    )
    artifact = checkpoint / manifest["artifact"]["file"]
    assert sha256(artifact) == manifest["artifact"]["sha256"]
    payload = load_file(str(artifact), device="cpu")
    full_layers = manifest["architecture"]["full_attention_layers"]
    mamba_layers = manifest["architecture"]["mamba_layers"]
    assert manifest["compression"]["attention_o_proj"] == "dense_unchanged"
    assert manifest["compression"]["mamba2_wo"] == "dense_unchanged"
    original_o_proj = {
        layer: model.model.layers[layer].mixer.o_proj
        for layer in full_layers
    }
    original_mamba_wo = {
        layer: model.model.layers[layer].mixer.out_proj
        for layer in mamba_layers
    }
    installed = []
    expected_keys = set()
    for record in manifest["layers"]:
        layer_index = record["layer"]
        assert layer_index in full_layers
        layer = model.model.layers[layer_index]
        assert layer.block_type == "full_attention"
        projection = layer.mixer.v_proj
        assert isinstance(projection, nn.Linear) and projection.bias is None
        ranks = [int(rank) for rank in record["ranks"]]
        writer_name = f"layers.{layer_index}.v_writer.weight"
        decoder_name = f"layers.{layer_index}.v_decoder.weight"
        expected_keys.update((writer_name, decoder_name))
        writer = payload[writer_name]
        decoder = payload[decoder_name]
        group_width = projection.out_features // len(ranks)
        assert writer.shape == (sum(ranks), projection.in_features)
        assert len(set(ranks)) == 1
        assert decoder.shape == (len(ranks), group_width, ranks[0])
        replacement = HeadwiseLowRankModule(
            ranks,
            projection.in_features,
            projection.out_features,
            bias=False,
        ).to(device=projection.weight.device, dtype=projection.weight.dtype)
        replacement.VT.weight.copy_(
            writer.to(device=projection.weight.device, dtype=projection.weight.dtype)
        )
        for group, up in enumerate(replacement.U):
            up.weight.copy_(
                decoder[group].to(
                    device=projection.weight.device, dtype=projection.weight.dtype
                )
            )
        layer.mixer.v_proj = replacement
        installed.append(
            {
                "layer": layer_index,
                "ranks": ranks,
                "groups": len(ranks),
                "writer_shape": list(writer.shape),
                "decoder_shape": list(decoder.shape),
            }
        )
    assert set(payload) == expected_keys
    assert all(
        model.model.layers[layer].mixer.o_proj is projection
        for layer, projection in original_o_proj.items()
    )
    assert all(
        model.model.layers[layer].mixer.out_proj is projection
        for layer, projection in original_mamba_wo.items()
    )
    return manifest, installed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--c4-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lm-eval-batch-size", type=int, default=8)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--reserve-gib", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure()
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == args.expected_gpu_count
    assert not args.output.exists()
    fast_implementations = activate_fast_mamba()
    model_path = args.model.resolve()
    checkpoint = args.checkpoint.resolve()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="balanced",
        max_memory={
            index: torch.cuda.get_device_properties(index).total_memory
            - args.reserve_gib * 2**30
            for index in range(args.expected_gpu_count)
        },
    ).eval()
    guarded = install_mamba_device_guards(model)
    checkpoint_manifest, installation = install_palu_v(model, checkpoint)
    assert checkpoint_manifest["model"]["config_sha256"] == sha256(
        model_path / "config.json"
    )
    assert guarded == checkpoint_manifest["architecture"]["mamba_layers"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    sequences, c4_provenance = _quality_windows(args.c4_windows, model_path)
    if args.smoke:
        input_device = model.get_input_embeddings().weight.device
        logits = model(
            sequences[:1, :128].to(input_device),
            use_cache=False,
            logits_to_keep=1,
        ).logits
        assert torch.isfinite(logits).all()
        result = {
            "status": "complete",
            "format": FORMAT,
            "smoke": True,
            "run_id": args.run_id,
            "checkpoint": str(checkpoint),
            "checkpoint_manifest_sha256": sha256(checkpoint / "manifest.json"),
            "installed": installation,
            "guarded_mamba_layers": guarded,
            "mamba_wo": "dense_unchanged",
            "attention_o_proj": "dense_unchanged",
            "fast_implementations": fast_implementations,
            "logits_shape": list(logits.shape),
            "logits_max_abs": float(logits.abs().max()),
            "command": shlex.join(sys.argv),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, result)
        print("PALU QUALITY SMOKE COMPLETE", args.run_id, flush=True)
        return
    model.config.use_cache = False
    wiki = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset="wikitext2",
        split="test",
        seqlen=2048,
        batch_size=args.batch_size,
        max_samples=None,
        max_tokens=None,
    )
    print("WIKITEXT2", wiki["ppl"], flush=True)
    c4 = _evaluate_document_ppl(
        model, sequences, batch_size=args.batch_size, label=args.run_id
    )
    print("C4", c4["ppl"], flush=True)
    model.config.use_cache = True
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.lm_eval_batch_size,
        max_length=4096,
        add_bos_token=False,
    )
    raw = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(TASKS),
        num_fewshot=0,
        task_manager=TaskManager(),
        log_samples=False,
    )
    summary = summarize_commonsense(raw, TASKS)
    assert summary is not None
    tasks, average = summary
    result = {
        "status": "complete",
        "format": FORMAT,
        "run_id": args.run_id,
        "model": {
            "path": str(model_path),
            "config_sha256": sha256(model_path / "config.json"),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": sha256(checkpoint / "manifest.json"),
            "compression": checkpoint_manifest["compression"],
        },
        "protocol": {
            "wikitext2": "full test corpus, 2048-token windows, FP32 loss",
            "c4": "128 document-disjoint validation windows, 2048 tokens, FP32 loss",
            "tasks": list(TASKS),
            "num_fewshot": 0,
            "max_length": 4096,
            "metric_selection": "acc_norm when present, otherwise acc",
        },
        "metrics": {
            "wikitext2_ppl": wiki["ppl"],
            "c4_validation_128_ppl": c4["ppl"],
            "average_accuracy": average,
            "task_accuracy": tasks,
        },
        "details": {
            "wikitext2": wiki,
            "c4": c4,
            "commonsense": raw,
            "c4_windows": c4_provenance,
            "installed": installation,
            "guarded_mamba_layers": guarded,
            "mamba_wo": "dense_unchanged",
            "attention_o_proj": "dense_unchanged",
            "fast_implementations": fast_implementations,
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(args.expected_gpu_count)
            ],
            "peak_cuda_allocated_bytes": {
                str(index): torch.cuda.max_memory_allocated(index)
                for index in range(args.expected_gpu_count)
            },
        },
        "command": shlex.join(sys.argv),
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(
        "PALU QUALITY COMPLETE",
        args.run_id,
        wiki["ppl"],
        c4["ppl"],
        average,
        flush=True,
    )


if __name__ == "__main__":
    main()
