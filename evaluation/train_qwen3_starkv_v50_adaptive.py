#!/usr/bin/env python3
"""STAR-KV V-only adaptive-rank training at 50% V-cache compression."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
import pyarrow.dataset as arrow_dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/STAR-KV"
UPSTREAM_COMMIT = "c9f0f36e7e386eaf93099c9c9796e168ca1e6504"
sys.path.insert(0, str(UPSTREAM))
from model import DecomposeLinear, enforce_rank_floor, fuse_and_prune
from train import CausalLMBlocks

SKIP_LAYERS = (0, 1, 31)
NUM_LAYERS = 36
FULL_V_RANK = 1024
TARGET_MEAN_V_RANK = 512
TARGET_V_RANK_SUM = NUM_LAYERS * TARGET_MEAN_V_RANK


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


def budget_report(ranks):
    total = sum(ranks)
    return {
        "ranks": ranks,
        "rank_sum": total,
        "target_rank_sum": TARGET_V_RANK_SUM,
        "within_budget": total <= TARGET_V_RANK_SUM,
        "exact_budget_match": total == TARGET_V_RANK_SUM,
        "budget_undershoot": max(0, TARGET_V_RANK_SUM - total),
        "mean_rank": total / len(ranks),
        "target_mean_rank": TARGET_MEAN_V_RANK,
        "v_cache_retention": total / (FULL_V_RANK * len(ranks)),
        "v_cache_compression": 1 - total / (FULL_V_RANK * len(ranks)),
        "skip_layers": list(SKIP_LAYERS),
        "skip_layer_rank": FULL_V_RANK,
        "budget_scope": "BF16 V-cache latent width across all layers",
    }


def optimizer_for(model, phase):
    no_decay = ("bias", "layer_norm.weight")
    lr = 2e-5 if phase == 1 else 5e-6
    groups = []
    for wd, select in ((0.01, False), (0.0, True)):
        params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "alpha" not in n
                  and any(s in n for s in no_decay) == select]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": wd})
    if phase == 1:
        groups.append({"params": [p for n, p in model.named_parameters()
                                   if "alpha" in n],
                       "lr": 1e-2, "weight_decay": 0.01})
    return torch.optim.AdamW(groups)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-seq-len", type=int, choices=(128, 8192), default=128)
    return p.parse_args()


def main():
    args = parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    assert revision == UPSTREAM_COMMIT
    assert torch.cuda.is_available()
    # Single GPU training is deliberate; TP8 specifies the deployment budget.
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    progress_path = args.output_dir / "progress.jsonl"
    # Preserve any completed run or interrupted training checkpoint.
    assert not result_path.exists()
    assert not (args.output_dir / "best.pt").exists()
    nsteps, seqlen = (3, args.smoke_seq_len) if args.smoke else (4000, 8192)
    alpha_limit = 0 if args.smoke else 3000
    log_every = 1 if args.smoke else 100
    settings = {
        "upstream_commit": revision, "model": args.model,
        "command": " ".join(sys.argv), "conda_environment": "basis",
        "python_executable": sys.executable, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "smoke": args.smoke,
        "num_samples": nsteps, "sequence_length": seqlen, "batch_size": 1,
        "epochs": 1, "alpha_samples": alpha_limit,
        "lr": 2e-5, "alpha_lr": 1e-2, "recovery_lr": 5e-6,
        "comp_weight_v": 0.1, "kd_weight": 1.0, "min_rank": 8,
        "skip_layers": list(SKIP_LAYERS), "key_projection": "dense_but_trainable",
        "trainable_parameters": "all_student_parameters_except_masks",
        "dataset": "HuggingFaceFW/fineweb-edu", "dataset_config": "sample-10BT",
        "dataset_order": "official_streaming_order_no_shuffle",
        "parquet_pre_buffer": False,
        "seed": 42, "gradient_checkpointing": True,
        "attention_implementation": "sdpa", "phase3_samples": 0,
        "checkpoint_selection": "official_best_logged_training_loss_across_phases",
        "method": "STAR-KV V-only adaptive-rank adaptation",
        "rank_learning": True,
        "v_decomposition": "joint across all KV heads",
        "nominal_v_cache_compression": 0.5,
        "target_mean_v_rank": TARGET_MEAN_V_RANK,
        "target_v_rank_sum": TARGET_V_RANK_SUM,
        "compression_stop": "first optimizer step with total V-cache rank at or below target; alpha_samples fallback",
        "recovery": "freeze V thresholds, fresh AdamW at 5e-6, KD only",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps(settings), flush=True)
    started = time.monotonic()
    print("Loading student", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True)
    cfg = student.config
    assert (cfg.num_hidden_layers, cfg.num_key_value_heads,
            cfg.num_attention_heads, cfg.head_dim) == (36, 8, 32, 128)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"SVD initialization: V projections except layers {SKIP_LAYERS}", flush=True)
    for i, layer in enumerate(student.model.layers):
        if i in SKIP_LAYERS:
            print(f"skipped dense layer {i}", flush=True)
            continue
        original = layer.self_attn.v_proj
        layer.self_attn.v_proj = DecomposeLinear(original.float()).to(torch.bfloat16)
        del original
        print(f"initialized layer {i}", flush=True)
    assert rank_profile(student) == [FULL_V_RANK] * NUM_LAYERS
    print("Loading frozen dense teacher", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True)
    teacher.config.use_cache = False
    teacher.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(settings["dataset"], name=settings["dataset_config"],
                      split="train", streaming=True,
                      fragment_scan_options=arrow_dataset.ParquetFragmentScanOptions(pre_buffer=False))
    loader = DataLoader(CausalLMBlocks(ds, tokenizer, block_size=seqlen,
                                      max_blocks=nsteps), batch_size=1)
    phase = 1
    optimizer = optimizer_for(student, phase)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0,
                              num_training_steps=nsteps)
    best_loss = float("inf")
    best_step = None
    student.train()
    for step, batch in enumerate(loader):
        torch.cuda.synchronize()
        step_started = time.monotonic()
        ids = batch["input_ids"].cuda()
        mask = batch["attention_mask"].cuda()
        with torch.no_grad():
            t_logits = teacher(input_ids=ids, attention_mask=mask).logits
        s_logits = student(input_ids=ids, attention_mask=mask).logits
        valid = ids[:, 1:] != tokenizer.pad_token_id
        # Same KD arithmetic, token masking, and reduction as the upstream trainer.
        kd = F.kl_div(F.log_softmax(s_logits[:, :-1, :].contiguous(), dim=-1)[valid],
                      F.softmax(t_logits[:, :-1, :].contiguous(), dim=-1)[valid],
                      reduction="batchmean")
        alphas = [p for n, p in student.named_parameters() if "alpha" in n]
        comp = torch.stack([torch.exp(-a).sum() for a in alphas]).sum() if alphas else kd.new_zeros(())
        loss = kd + (0.1 * comp if phase == 1 else 0)
        assert torch.isfinite(loss).item()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all().item()
                   for p in alphas)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value, kd_value = loss.item(), kd.item()
        del t_logits, s_logits, kd, comp, loss
        if phase == 1:
            enforce_rank_floor(student, 8, skip_layers=SKIP_LAYERS)
        ranks = rank_profile(student)
        if phase == 1 and (sum(ranks) <= TARGET_V_RANK_SUM or step >= alpha_limit):
            phase = 2
            for p in alphas:
                p.requires_grad_(False)
            del optimizer, scheduler
            gc.collect()
            optimizer = optimizer_for(student, 2)
            scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0,
                                      num_training_steps=max(1, nsteps - step))
            stop_reason = "V cache-rank budget reached" if sum(ranks) <= TARGET_V_RANK_SUM else "alpha_samples fallback"
            print(f"phase2 step={step + 1} rank_sum={sum(ranks)} reason={stop_reason}", flush=True)
        torch.cuda.synchronize()
        row = {"step": step + 1, "phase": phase, "loss": loss_value,
               "step_seconds": time.monotonic() - step_started,
               "kd": kd_value, "rank_sum": sum(ranks), "ranks": ranks,
               "elapsed_seconds": time.monotonic() - started,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
        with progress_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if (step + 1) % log_every == 0 and loss_value < best_loss:
            best_loss, best_step = loss_value, step + 1
            if not args.smoke:
                torch.save(student.state_dict(), args.output_dir / "best.pt")
    assert step + 1 == nsteps
    del optimizer, scheduler, teacher
    gc.collect()
    torch.cuda.empty_cache()
    if not args.smoke:
        student.load_state_dict(torch.load(args.output_dir / "best.pt",
                                          map_location="cpu", weights_only=True), strict=True)
    student.eval()
    # Compare pre/post fusion on identical real data; no TP/runtime claim.
    with torch.no_grad():
        before = student(input_ids=ids[:, :128]).logits.float()
        fuse_and_prune(student, skip_layers=SKIP_LAYERS)
        after = student(input_ids=ids[:, :128]).logits.float()
        rmse = (before - after).square().mean().sqrt().item()
    assert rmse < 0.1
    budget = budget_report(rank_profile(student))
    if not args.smoke:
        # Save the WHOLE fine-tuned student, not just its V factors.
        torch.save(student.state_dict(), args.output_dir / "fused.pt")
        student.config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    result = {"status": "smoke_complete" if args.smoke else "complete",
              "protocol": settings, "budget": budget,
              "eligible_for_v50_comparison": not args.smoke and budget["within_budget"],
              "best_logged_loss": best_loss, "best_step": best_step,
              "fusion_logit_rmse": rmse,
              "elapsed_seconds": time.monotonic() - started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
