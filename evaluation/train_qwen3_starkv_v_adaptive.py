#!/usr/bin/env python3
"""Train Qwen3-8B STAR-KV V-only checkpoints at an adaptive global-rank budget."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import pyarrow.dataset as arrow_dataset
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/STAR-KV"
UPSTREAM_COMMIT = "c9f0f36e7e386eaf93099c9c9796e168ca1e6504"
sys.path.insert(0, str(UPSTREAM))
from model import DecomposeLinear, _fuse_joint
from train import CausalLMBlocks

SKIP_LAYERS = (0, 1, 31)
NUM_LAYERS = 36
FULL_V_RANK = 1024
MIN_RANK = 8
TARGET_RANKS = (64, 128, 256)


def rank_profile(model):
    ranks = []
    for block in model.model.layers:
        vp = block.self_attn.v_proj
        if isinstance(vp, DecomposeLinear):
            effective = vp.Sigma.soft_thres_layer(vp.Sigma.diag.detach().float())
            ranks.append(max(1, int((effective > 0).sum())))
        else:
            ranks.append(vp.VS.out_features if hasattr(vp, "VS") else vp.out_features)
    return ranks


def budget_report(ranks, target_mean_rank):
    compressed = [rank for layer, rank in enumerate(ranks) if layer not in SKIP_LAYERS]
    target_sum = len(compressed) * target_mean_rank
    compressed_sum = sum(compressed)
    total_sum = sum(ranks)
    return {
        "ranks": ranks,
        "compressed_layer_ranks": compressed,
        "compressed_layer_rank_sum": compressed_sum,
        "target_compressed_layer_rank_sum": target_sum,
        "exact_budget_match": compressed_sum == target_sum,
        "compressed_layer_mean_rank": compressed_sum / len(compressed),
        "target_compressed_layer_mean_rank": target_mean_rank,
        "all_layer_mean_rank": total_sum / len(ranks),
        "all_layer_v_cache_retention": total_sum / (FULL_V_RANK * len(ranks)),
        "all_layer_v_cache_compression": 1 - total_sum / (FULL_V_RANK * len(ranks)),
        "compressed_layer_v_cache_retention": compressed_sum / (FULL_V_RANK * len(compressed)),
        "compressed_layer_v_cache_compression": 1 - compressed_sum / (FULL_V_RANK * len(compressed)),
        "skip_layers": list(SKIP_LAYERS),
        "skip_layer_rank": FULL_V_RANK,
        "budget_scope": "mean joint V latent rank over non-dense layers",
    }


@torch.no_grad()
def project_adaptive_rank_budget(model, target_mean_rank, min_rank=MIN_RANK):
    """Project learned layer thresholds to an exact global budget.

    Every layer first receives ``min_rank`` directions. Remaining directions
    are assigned globally by their learned threshold margin (sigma - alpha).
    The resulting allocation is realized by moving each layer's scalar alpha
    between its last kept and first removed singular value.
    """
    modules = []
    for layer, block in enumerate(model.model.layers):
        if layer in SKIP_LAYERS:
            continue
        vp = block.self_attn.v_proj
        values = vp.Sigma.diag.detach().float()
        order = torch.argsort(values, descending=True, stable=True)
        sorted_values = values[order]
        alpha = float(vp.Sigma.soft_thres_layer.alpha.detach().float().item())
        modules.append((layer, vp, order, sorted_values, alpha))

    target_sum = len(modules) * target_mean_rank
    allocation = {layer: min_rank for layer, _, _, _, _ in modules}
    candidates = []
    for layer, _, _, values, alpha in modules:
        for position in range(min_rank, values.numel()):
            candidates.append((float(values[position].item()) - alpha, layer))
    candidates.sort(key=lambda item: item[0], reverse=True)
    extra = target_sum - min_rank * len(modules)
    for _, layer in candidates[:extra]:
        allocation[layer] += 1

    tie_breaks = []
    for layer, vp, order, values, _ in modules:
        rank = allocation[layer]
        upper = values[rank - 1]
        lower = values[rank]
        threshold = lower
        if upper == lower:
            cutoff = lower.to(dtype=vp.Sigma.diag.dtype)
            above = int((vp.Sigma.diag.detach() > cutoff).sum().item())
            needed = rank - above
            tied = order[values == lower]
            raised = torch.nextafter(
                cutoff,
                torch.full_like(cutoff, float("inf")),
            )
            vp.Sigma.diag[tied[:needed]] = raised
            threshold = cutoff.float()
            tie_breaks.append({"layer": layer, "cutoff": float(cutoff), "raised": needed})
        vp.Sigma.soft_thres_layer.alpha.copy_(
            threshold.reshape_as(vp.Sigma.soft_thres_layer.alpha)
        )

    ranks = rank_profile(model)
    measured = sum(rank for layer, rank in enumerate(ranks) if layer not in SKIP_LAYERS)
    return {
        "allocation": [allocation[layer] for layer, _, _, _, _ in modules],
        "compressed_rank_sum": measured,
        "target_compressed_rank_sum": target_sum,
        "exact": measured == target_sum,
        "priority": "learned per-direction threshold margin sigma-alpha",
        "minimum_rank_per_compressed_layer": min_rank,
        "bf16_cutoff_tie_breaks": tie_breaks,
    }


def optimizer_for(model, phase):
    no_decay = ("bias", "layer_norm.weight")
    lr = 2e-5 if phase == 1 else 5e-6
    groups = []
    for wd, select in ((0.01, False), (0.0, True)):
        params = [
            p for name, p in model.named_parameters()
            if p.requires_grad and "alpha" not in name
            and any(token in name for token in no_decay) == select
        ]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": wd})
    if phase == 1:
        groups.append({
            "params": [p for name, p in model.named_parameters() if "alpha" in name],
            "lr": 1e-2,
            "weight_decay": 0.01,
        })
    return torch.optim.AdamW(groups)


@torch.no_grad()
def fuse_v_projections_with_checks(model):
    rows = []
    for layer, block in enumerate(model.model.layers):
        if layer in SKIP_LAYERS:
            continue
        projection = block.self_attn.v_proj
        diagonal = projection.Sigma.diag.detach().float()
        effective = projection.Sigma.soft_thres_layer(diagonal)
        keep = (effective > 0).nonzero(as_tuple=False).view(-1)
        source = (projection.V.weight * projection.V.mask).detach().float()
        reconstruction = (projection.U.weight * projection.U.mask).detach().float()
        reference = (reconstruction[:, keep] * effective[keep][None, :]) @ source[keep]
        fused = _fuse_joint(projection)
        actual = fused.U.weight.detach().float() @ fused.VS.weight.detach().float()
        difference = actual - reference
        relative = float(
            (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)).item()
        )
        rows.append({
            "layer": layer,
            "rank": int(keep.numel()),
            "relative_frobenius_error": relative,
            "max_absolute_error": float(difference.abs().max().item()),
        })
        block.self_attn.v_proj = fused
        del reference, actual, difference
    return {
        "layers": rows,
        "max_relative_frobenius_error": max(row["relative_frobenius_error"] for row in rows),
        "max_absolute_error": max(row["max_absolute_error"] for row in rows),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mean-rank", type=int, choices=TARGET_RANKS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-seq-len", type=int, choices=(128, 8192), default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        print(f"[error] STAR-KV revision is {revision}, expected {UPSTREAM_COMMIT}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        print("[error] exactly one CUDA GPU is required", file=sys.stderr)
        return 2

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if result_path.exists() or (args.output_dir / "best.pt").exists():
        print("[error] output directory already contains a checkpoint", file=sys.stderr)
        return 2

    num_steps, sequence_length = (3, args.smoke_seq_len) if args.smoke else (4000, 8192)
    compression_steps = 2 if args.smoke else 3000
    log_every = 1 if args.smoke else 100
    target_sum = (NUM_LAYERS - len(SKIP_LAYERS)) * args.target_mean_rank
    settings = {
        "upstream_commit": revision,
        "paper": "STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control",
        "model": args.model,
        "command": " ".join(sys.argv),
        "conda_environment": "basis",
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "smoke": args.smoke,
        "num_samples": num_steps,
        "compression_samples": compression_steps,
        "recovery_samples": num_steps - compression_steps,
        "sequence_length": sequence_length,
        "batch_size": 1,
        "epochs": 1,
        "lr": 2e-5,
        "alpha_lr": 1e-2,
        "recovery_lr": 5e-6,
        "comp_weight_v": 0.1,
        "kd_weight": 1.0,
        "min_rank": MIN_RANK,
        "skip_layers": list(SKIP_LAYERS),
        "key_projection": "dense_but_trainable",
        "trainable_parameters": "all_student_parameters_except_masks",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "dataset_order": "official_streaming_order_no_shuffle",
        "parquet_pre_buffer": False,
        "seed": 42,
        "gradient_checkpointing": True,
        "attention_implementation": "sdpa",
        "checkpoint_selection": "lowest logged recovery KD loss",
        "method": "STAR-KV V-only adaptive-rank adaptation",
        "rank_learning": True,
        "v_decomposition": "joint across all KV heads",
        "target_compressed_layer_mean_rank": args.target_mean_rank,
        "target_compressed_layer_rank_sum": target_sum,
        "compression_phase": "3000-step STAR-KV soft-threshold training",
        "budget_projection": "at compression boundary, exact global allocation by learned sigma-alpha margins",
        "recovery": "freeze projected V thresholds, fresh AdamW at 5e-6, KD only for 1000 steps",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings), flush=True)

    started = time.monotonic()
    print("Loading student", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    config = student.config
    if (config.num_hidden_layers, config.num_key_value_heads,
            config.num_attention_heads, config.head_dim) != (36, 8, 32, 128):
        print("[error] model architecture does not match Qwen3-8B-Base", file=sys.stderr)
        return 2
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"SVD initialization: V projections except layers {SKIP_LAYERS}", flush=True)
    for layer, block in enumerate(student.model.layers):
        if layer in SKIP_LAYERS:
            print(f"skipped dense layer {layer}", flush=True)
            continue
        original = block.self_attn.v_proj
        block.self_attn.v_proj = DecomposeLinear(original.float()).to(torch.bfloat16)
        del original
        print(f"initialized layer {layer}", flush=True)

    print("Loading frozen dense teacher", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    teacher.config.use_cache = False
    teacher.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        settings["dataset"],
        name=settings["dataset_config"],
        split="train",
        streaming=True,
        fragment_scan_options=arrow_dataset.ParquetFragmentScanOptions(pre_buffer=False),
    )
    loader = DataLoader(
        CausalLMBlocks(dataset, tokenizer, block_size=sequence_length, max_blocks=num_steps),
        batch_size=1,
    )

    phase = 1
    optimizer = optimizer_for(student, phase)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )
    best_recovery_loss = float("inf")
    best_step = None
    projection = None
    progress_path = args.output_dir / "progress.jsonl"
    progress_path.write_text("")
    student.train()
    ids = None
    for step, batch in enumerate(loader):
        torch.cuda.synchronize()
        step_started = time.monotonic()
        ids = batch["input_ids"].cuda()
        mask = batch["attention_mask"].cuda()
        with torch.no_grad():
            teacher_logits = teacher(input_ids=ids, attention_mask=mask).logits
        student_logits = student(input_ids=ids, attention_mask=mask).logits
        valid = ids[:, 1:] != tokenizer.pad_token_id
        kd = F.kl_div(
            F.log_softmax(student_logits[:, :-1, :].contiguous(), dim=-1)[valid],
            F.softmax(teacher_logits[:, :-1, :].contiguous(), dim=-1)[valid],
            reduction="batchmean",
        )
        alphas = [parameter for name, parameter in student.named_parameters() if "alpha" in name]
        comp = torch.stack([torch.exp(-alpha).sum() for alpha in alphas]).sum()
        loss = kd + (0.1 * comp if phase == 1 else 0)
        if not torch.isfinite(loss).item():
            print(f"[error] non-finite loss at step {step + 1}", file=sys.stderr)
            return 2
        loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                   for parameter in alphas):
            print(f"[error] non-finite alpha gradient at step {step + 1}", file=sys.stderr)
            return 2
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.item())
        kd_value = float(kd.item())
        del teacher_logits, student_logits, kd, comp, loss

        projected_this_step = False
        if phase == 1 and step + 1 == compression_steps:
            pre_projection_ranks = rank_profile(student)
            projection = project_adaptive_rank_budget(student, args.target_mean_rank)
            if not projection["exact"]:
                print("[error] exact adaptive-rank budget projection failed", file=sys.stderr)
                return 2
            for alpha in alphas:
                alpha.requires_grad_(False)
            del optimizer, scheduler
            gc.collect()
            phase = 2
            optimizer = optimizer_for(student, phase)
            scheduler = get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=0,
                num_training_steps=max(1, num_steps - compression_steps),
            )
            projection["pre_projection_ranks"] = pre_projection_ranks
            projection["post_projection_ranks"] = rank_profile(student)
            projected_this_step = True
            print(json.dumps({"phase2_step": step + 1, "projection": projection}), flush=True)

        ranks = rank_profile(student)
        torch.cuda.synchronize()
        row = {
            "step": step + 1,
            "phase": phase,
            "loss": loss_value,
            "kd": kd_value,
            "step_seconds": time.monotonic() - step_started,
            "compressed_rank_sum": sum(
                rank for layer, rank in enumerate(ranks) if layer not in SKIP_LAYERS
            ),
            "ranks": ranks,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        }
        with progress_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if (phase == 2 and not projected_this_step and (step + 1) % log_every == 0
                and kd_value < best_recovery_loss):
            best_recovery_loss = kd_value
            best_step = step + 1
            if not args.smoke:
                torch.save(student.state_dict(), args.output_dir / "best.pt")

    if step + 1 != num_steps or projection is None or ids is None:
        print("[error] training stream ended before the requested schedule", file=sys.stderr)
        return 2
    del optimizer, scheduler, teacher
    gc.collect()
    torch.cuda.empty_cache()
    if not args.smoke:
        student.load_state_dict(
            torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
    final_projection = project_adaptive_rank_budget(student, args.target_mean_rank)
    if not final_projection["exact"]:
        print("[error] final exact adaptive-rank budget projection failed", file=sys.stderr)
        return 2
    student.eval()
    with torch.no_grad():
        before = student(input_ids=ids[:, :128]).logits.float()
        fusion_weights = fuse_v_projections_with_checks(student)
        after = student(input_ids=ids[:, :128]).logits.float()
        rmse = (before - after).square().mean().sqrt().item()
    if fusion_weights["max_relative_frobenius_error"] >= 0.01:
        print("[error] fused V factorization failed its weight check", file=sys.stderr)
        return 2
    if not torch.isfinite(torch.tensor(rmse)).item():
        print("[error] fusion produced non-finite logits", file=sys.stderr)
        return 2
    budget = budget_report(rank_profile(student), args.target_mean_rank)
    if not budget["exact_budget_match"]:
        print("[error] fused checkpoint misses its global-rank budget", file=sys.stderr)
        return 2
    if not args.smoke:
        torch.save(student.state_dict(), args.output_dir / "fused.pt")
        student.config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    result = {
        "status": "smoke_complete" if args.smoke else "complete",
        "protocol": settings,
        "projection": projection,
        "final_projection": final_projection,
        "budget": budget,
        "best_recovery_loss": best_recovery_loss,
        "best_step": best_step,
        "fusion_logit_rmse": rmse,
        "fusion_weight_check": fusion_weights,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
