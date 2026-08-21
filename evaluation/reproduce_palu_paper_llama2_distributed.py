#!/usr/bin/env python3
"""Reproduce the PaLU Llama-2-7B G-LRD quality anchor with data parallel calibration.

The scientific protocol follows Table 1/Table 8 of the PaLU ICLR 2025 paper:

* Llama-2-7B in fp16;
* Fisher-uniform rank allocation from 2048 WikiText-2 windows of length 1024;
* activation-aware (whitened) SVD;
* G-LRD with group size four and a 50% retained latent-rank budget; and
* WikiText-2 test perplexity with non-overlapping 2048-token chunks.

Only calibration is distributed.  Each torchrun worker owns a disjoint subset
of the deterministic official calibration windows.  Fisher sums of squared
gradients and whitening X^T X statistics are summed with all-reduce before the
non-linear square root / Cholesky operations.  This preserves the single-job
objective while fitting the experiment on 24 GiB GPUs.

This runner intentionally does not write a model checkpoint.  It writes the
rank map, metrics, and a Markdown summary from rank zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PAPER_COMMIT = "bb22666e2ef96707e8dd21d93fc00146c2e0d615"
PAPER_DENSE_PPL = 5.47
PAPER_GLRD_PPL = 6.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--compression-rate", type=float, default=0.5)
    parser.add_argument("--head-group-size", type=int, default=4)
    parser.add_argument("--rank-block-size", type=int, default=32)
    parser.add_argument("--fisher-dataset", choices=("wikitext2",), default="wikitext2")
    parser.add_argument("--fisher-samples", type=int, default=2048)
    parser.add_argument("--fisher-seqlen", type=int, default=1024)
    parser.add_argument("--whiten-dataset", choices=("wikitext2",), default="wikitext2")
    parser.add_argument("--whiten-samples", type=int, default=256)
    parser.add_argument("--whiten-seqlen", type=int, default=2048)
    parser.add_argument("--calibration-seed", type=int, default=3)
    parser.add_argument("--eval-dataset", choices=("wikitext2",), default="wikitext2")
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-chunks", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _log(message: str, *, all_ranks: bool = False) -> None:
    if all_ranks or _rank() == 0:
        print(f"[rank {_rank()}] {message}", flush=True)


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _model_layers(model: nn.Module) -> nn.ModuleList:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError(f"expected a Llama-style model, got {type(model).__name__}")
    return layers


def _kv_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    result = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (name.endswith("self_attn.k_proj") or name.endswith("self_attn.v_proj"))
    ]
    expected = 2 * len(_model_layers(model))
    if len(result) != expected:
        raise RuntimeError(f"found {len(result)} K/V linears, expected {expected}")
    return result


def _wikitext(split: str) -> str:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(str(text) for text in dataset["text"])


def _official_window_starts(text_length: int, samples: int, seqlen: int, seed: int) -> list[int]:
    """Match palu.data_utils.get_calib_data's character-index sampling."""

    generator = random.Random(seed)
    return [generator.randint(0, text_length - seqlen - 1) for _ in range(samples)]


def _local_calibration_windows(
    tokenizer: Any,
    text: str,
    *,
    samples: int,
    seqlen: int,
    seed: int,
    rank: int,
    world_size: int,
) -> list[dict[str, torch.Tensor]]:
    starts = _official_window_starts(len(text), samples, seqlen, seed)
    windows: list[dict[str, torch.Tensor]] = []
    for sample_index, start in enumerate(starts):
        if sample_index % world_size != rank:
            continue
        # This deliberately matches the official character-window heuristic.
        encoded = tokenizer(text[start : start + seqlen * 10], return_tensors="pt")
        input_ids = encoded.input_ids[:, :seqlen]
        if input_ids.shape[1] != seqlen:
            raise RuntimeError(
                f"calibration sample {sample_index} produced {input_ids.shape[1]} tokens, "
                f"expected {seqlen}"
            )
        windows.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    expected = len(range(rank, samples, world_size))
    if len(windows) != expected:
        raise AssertionError((len(windows), expected))
    return windows


def _configure_fisher_gradients(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    kv_modules = _kv_linears(model)
    selected = {parameter for _, module in kv_modules for parameter in module.parameters(recurse=False)}
    for parameter in model.parameters():
        parameter.requires_grad_(parameter in selected)
    return kv_modules


def distributed_fisher_scalars(
    model: nn.Module,
    windows: list[dict[str, torch.Tensor]],
    *,
    total_samples: int,
    device: torch.device,
) -> dict[str, float]:
    """Return the exact PaLU per-matrix mean RMS-gradient Fisher scalar."""

    kv_modules = _configure_fisher_gradients(model)
    accumulators = {
        name: torch.zeros(module.weight.shape, dtype=torch.float32, device="cpu")
        for name, module in kv_modules
    }
    model.eval()
    progress = tqdm(windows, disable=_rank() != 0, desc="Fisher local windows")
    for batch in progress:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = input_ids[:, 1:]
        outputs = model(
            input_ids=input_ids[:, :-1],
            labels=labels,
            use_cache=False,
        )
        outputs.loss.backward()
        for name, module in kv_modules:
            gradient = module.weight.grad
            if gradient is None:
                raise RuntimeError(f"missing Fisher gradient for {name}")
            accumulators[name].add_(gradient.detach().to(device="cpu", dtype=torch.float32).square())
        model.zero_grad(set_to_none=True)

    scalars: dict[str, float] = {}
    for name, _ in tqdm(kv_modules, disable=_rank() != 0, desc="Fisher all-reduce"):
        summed_squares = accumulators.pop(name).to(device)
        _all_reduce_sum(summed_squares)
        # PaLU applies sqrt before averaging the matrix/group importance.
        scalar = summed_squares.div_(float(total_samples)).sqrt_().mean()
        scalars[name] = float(scalar.item())
        del summed_squares
    return scalars


def official_fisher_uniform_rank_map(
    fisher_scalars: dict[str, float],
    *,
    output_width: int,
    num_heads: int,
    head_group_size: int,
    retained_ratio: float,
    block_size: int,
) -> tuple[dict[str, list[int]], int, int]:
    """Transcribe PaLU's fisher_uniform allocation, including its rounding order."""

    if num_heads % head_group_size:
        raise ValueError("num_heads must be divisible by head_group_size")
    group_count = num_heads // head_group_size
    total_rank = len(fisher_scalars) * output_width
    target_rank = total_rank * retained_ratio
    fisher_sum = sum(fisher_scalars.values())
    if not math.isfinite(fisher_sum) or fisher_sum <= 0:
        raise ValueError(f"invalid Fisher sum: {fisher_sum}")

    selected = {name: output_width for name in fisher_scalars}
    selected_float: dict[str, float] = {}
    indexes: list[str] = []
    for name, fisher in fisher_scalars.items():
        rank_float = target_rank * fisher / fisher_sum
        selected_float[name] = rank_float
        indexes.append(name)
        selected[name] = min(output_width, math.floor(rank_float))

    # Keep the official implementation's ascending fractional-order behavior.
    indexes.sort(key=lambda name: selected_float[name] - selected[name])
    difference = target_rank - sum(selected.values())
    while difference > 0:
        made_progress = False
        for name in indexes:
            if selected[name] == output_width:
                continue
            selected[name] += 1
            difference -= 1
            made_progress = True
            if difference == 0:
                break
        if not made_progress:
            break

    rank_map: dict[str, list[int]] = {}
    for name, full_rank in selected.items():
        per_group = full_rank // group_count
        rounded = max(1, round(per_group / block_size)) * block_size
        rank_map[name] = [int(rounded)] * group_count
    rank_sum = sum(sum(ranks) for ranks in rank_map.values())
    return rank_map, rank_sum, total_rank


class _CaptureExit(Exception):
    pass


class _FirstLayerCatcher(nn.Module):
    def __init__(self, layer: nn.Module, inputs: torch.Tensor, captured_kwargs: dict[str, Any]):
        super().__init__()
        self.layer = layer
        self.inputs = inputs
        self.captured_kwargs = captured_kwargs
        self.index = 0
        if hasattr(layer, "attention_type"):
            self.attention_type = layer.attention_type

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.inputs[self.index].copy_(hidden_states[0])
        if not self.captured_kwargs:
            for key, value in kwargs.items():
                if torch.is_tensor(value):
                    self.captured_kwargs[key] = value.detach()
                else:
                    self.captured_kwargs[key] = value
        self.index += 1
        raise _CaptureExit


def _stable_cholesky(raw: torch.Tensor) -> torch.Tensor:
    raw64 = raw.to(torch.float64)
    try:
        return torch.linalg.cholesky(raw64).to(torch.float32)
    except torch.linalg.LinAlgError:
        eigenvalues = torch.linalg.eigvalsh(raw64)
        shift = float(max(0.0, -eigenvalues[0].item()) + 1e-3)
        raw64.diagonal().add_(shift)
        return torch.linalg.cholesky(raw64).to(torch.float32)


@torch.no_grad()
def distributed_whitening_cholesky(
    model: nn.Module,
    windows: list[dict[str, torch.Tensor]],
    *,
    seqlen: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Collect per-layer K/V-input X^T X and return Cholesky factors on CPU."""

    layers = _model_layers(model)
    hidden_size = int(model.config.hidden_size)
    local_samples = len(windows)
    inputs = torch.empty(
        (local_samples, seqlen, hidden_size),
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    outputs = torch.empty_like(inputs)
    captured_kwargs: dict[str, Any] = {}
    first_layer = layers[0]
    catcher = _FirstLayerCatcher(first_layer, inputs, captured_kwargs)
    layers[0] = catcher
    try:
        for batch in tqdm(windows, disable=_rank() != 0, desc="Capture whitening inputs"):
            try:
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    use_cache=False,
                )
            except _CaptureExit:
                pass
    finally:
        layers[0] = first_layer
    if catcher.index != local_samples:
        raise RuntimeError(f"captured {catcher.index} inputs, expected {local_samples}")

    cholesky_factors: list[torch.Tensor] = []
    for layer_index, layer in enumerate(tqdm(layers, disable=_rank() != 0, desc="Whitening layers")):
        raw_xtx = torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=device)

        def accumulate_input(module: nn.Module, hook_inputs: tuple[torch.Tensor, ...]) -> None:
            del module
            hidden = hook_inputs[0].detach().reshape(-1, hidden_size).to(torch.float32)
            raw_xtx.addmm_(hidden.transpose(0, 1), hidden)

        handle = layer.self_attn.k_proj.register_forward_pre_hook(accumulate_input)
        try:
            for sample_index in range(local_samples):
                layer_kwargs = dict(captured_kwargs)
                result = layer(inputs[sample_index].unsqueeze(0), **layer_kwargs)
                outputs[sample_index].copy_(result[0][0])
        finally:
            handle.remove()

        _all_reduce_sum(raw_xtx)
        cholesky = _stable_cholesky(raw_xtx)
        cholesky_factors.append(cholesky.cpu())
        del raw_xtx, cholesky
        inputs, outputs = outputs, inputs
        _log(f"whitening layer {layer_index} complete")
    return cholesky_factors


class GroupedLowRankLinear(nn.Module):
    """Inference-equivalent PaLU G-LRD projection without cache-kernel changes."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        ranks: list[int],
        bias: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if out_features % len(ranks):
            raise ValueError("out_features must be divisible by the number of groups")
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = list(ranks)
        self.group_dim = out_features // len(ranks)
        self.VT = nn.Linear(in_features, sum(ranks), bias=False, device=device, dtype=dtype)
        self.U = nn.ModuleList(
            nn.Linear(rank, self.group_dim, bias=bias, device=device, dtype=dtype)
            for rank in ranks
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latent = self.VT(hidden_states)
        chunks = latent.split(self.ranks, dim=-1)
        return torch.cat([up(chunk) for up, chunk in zip(self.U, chunks)], dim=-1)


@torch.no_grad()
def _factor_grouped_linear(
    linear: nn.Linear,
    ranks: list[int],
    cholesky: torch.Tensor,
    *,
    device: torch.device,
) -> GroupedLowRankLinear:
    dtype = linear.weight.dtype
    grouped = GroupedLowRankLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        ranks=ranks,
        bias=linear.bias is not None,
        device=device,
        dtype=dtype,
    )
    weight_groups = linear.weight.detach().reshape(len(ranks), -1, linear.in_features)
    scale = cholesky.to(device=device, dtype=torch.float32)
    scale_inverse = torch.linalg.inv(scale)
    right_factors: list[torch.Tensor] = []
    for group_index, rank in enumerate(ranks):
        weighted = weight_groups[group_index].to(torch.float32) @ scale
        left_singular, singular_values, right_singular = torch.linalg.svd(
            weighted,
            full_matrices=False,
        )
        # Algebraically identical to the official Vt @ inv(scale) followed by truncation.
        right = right_singular[:rank] @ scale_inverse
        sigma_root = singular_values[:rank].sqrt()
        left = left_singular[:, :rank] * sigma_root.unsqueeze(0)
        right = sigma_root.unsqueeze(1) * right
        grouped.U[group_index].weight.copy_(left.to(dtype))
        if linear.bias is not None:
            bias_groups = linear.bias.detach().reshape(len(ranks), -1)
            grouped.U[group_index].bias.copy_(bias_groups[group_index])
        right_factors.append(right.to(dtype))
    grouped.VT.weight.copy_(torch.cat(right_factors, dim=0))
    return grouped


@torch.no_grad()
def apply_grouped_whitened_svd(
    model: nn.Module,
    rank_map: dict[str, list[int]],
    cholesky_factors: list[torch.Tensor],
    *,
    device: torch.device,
) -> None:
    layers = _model_layers(model)
    for layer_index, layer in enumerate(tqdm(layers, disable=_rank() != 0, desc="G-LRD factorization")):
        for projection_name in ("k_proj", "v_proj"):
            full_name = f"model.layers.{layer_index}.self_attn.{projection_name}"
            linear = getattr(layer.self_attn, projection_name)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"expected dense linear at {full_name}, got {type(linear).__name__}")
            replacement = _factor_grouped_linear(
                linear,
                rank_map[full_name],
                cholesky_factors[layer_index],
                device=device,
            )
            setattr(layer.self_attn, projection_name, replacement)


@torch.no_grad()
def distributed_official_ppl(
    model: nn.Module,
    tokenizer: Any,
    *,
    text: str,
    seqlen: int,
    max_chunks: int | None,
    device: torch.device,
) -> dict[str, float | int]:
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    chunk_count = input_ids.numel() // seqlen
    if max_chunks is not None:
        chunk_count = min(chunk_count, max_chunks)
    local_loss_sum = 0.0
    local_chunks = 0
    previous_use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    for chunk_index in tqdm(
        range(_rank(), chunk_count, _world_size()),
        disable=_rank() != 0,
        desc="WikiText-2 PPL local chunks",
    ):
        batch = input_ids[:, chunk_index * seqlen : (chunk_index + 1) * seqlen].to(device)
        hidden_states = model.model(batch, use_cache=False)[0]
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        local_loss_sum += float(loss.float().item())
        local_chunks += 1
    totals = torch.tensor([local_loss_sum, float(local_chunks)], dtype=torch.float64, device=device)
    _all_reduce_sum(totals)
    model.config.use_cache = previous_use_cache
    mean_loss = float((totals[0] / totals[1]).item())
    return {
        "ppl": math.exp(mean_loss),
        "mean_loss": mean_loss,
        "chunks": int(totals[1].item()),
        "seqlen": seqlen,
        "official_nll_scaling": "mean_cross_entropy_per_chunk (equivalent to official loss*seqlen denominator)",
    }


def _broadcast_rank_map(rank_map: dict[str, list[int]] | None) -> dict[str, list[int]]:
    payload: list[Any] = [rank_map]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if not isinstance(payload[0], dict):
        raise RuntimeError("failed to broadcast rank map")
    return payload[0]


def _write_results(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dense = result["dense_eval"]["ppl"]
    compressed = result["compressed_eval"]["ppl"]
    lines = [
        "# PaLU paper-protocol Llama-2-7B reproduction",
        "",
        "## Outcome",
        "",
        "| Endpoint | Paper | Reproduction |",
        "|---|---:|---:|",
        f"| Dense WikiText-2 PPL | {PAPER_DENSE_PPL:.2f} | {dense:.6f} |",
        f"| 50% G-LRD WikiText-2 PPL | {PAPER_GLRD_PPL:.2f} | {compressed:.6f} |",
        f"| PPL increase | {PAPER_GLRD_PPL - PAPER_DENSE_PPL:.2f} | {compressed - dense:.6f} |",
        "",
        "## Protocol",
        "",
        f"- Model: `{result['model']}`",
        f"- Fisher: WikiText-2 train, {result['fisher']['samples']} windows, length {result['fisher']['seqlen']}",
        f"- Whitening: WikiText-2 train, {result['whitening']['samples']} windows, length {result['whitening']['seqlen']}",
        f"- G-LRD head group size: {result['head_group_size']}",
        f"- Requested retained-rank ratio: {result['requested_retained_ratio']:.6f}",
        f"- Realized retained-rank ratio: {result['realized_retained_ratio']:.6f}",
        f"- Distributed calibration workers: {result['world_size']}",
        "- No LoRA, quantization, recovery, or model-checkpoint write.",
        "",
        "## Interpretation guardrail",
        "",
        "Calibration sufficient statistics are data-parallel sums. Factorization and quality semantics match PaLU, but this is a distributed implementation rather than byte-for-byte execution of the single-GPU official script.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _self_test() -> None:
    text_length = 10000
    starts = _official_window_starts(text_length, 17, 128, 3)
    shards = [starts[rank::4] for rank in range(4)]
    assert sum(len(shard) for shard in shards) == len(starts)
    reconstructed = [None] * len(starts)
    for rank, shard in enumerate(shards):
        for offset, value in enumerate(shard):
            reconstructed[rank + offset * 4] = value
    assert reconstructed == starts

    fisher = {"k0": 1.0, "v0": 3.0, "k1": 2.0, "v1": 4.0}
    ranks, rank_sum, total = official_fisher_uniform_rank_map(
        fisher,
        output_width=128,
        num_heads=8,
        head_group_size=4,
        retained_ratio=0.5,
        block_size=1,
    )
    assert set(ranks) == set(fisher)
    assert rank_sum <= total

    torch.manual_seed(0)
    gradients = torch.randn(7, 5, 3)
    direct = gradients.square().sum(dim=0).div(7).sqrt().mean()
    merged = sum(part.square().sum(dim=0) for part in gradients.tensor_split(4))
    merged = merged.div(7).sqrt().mean()
    torch.testing.assert_close(merged, direct)

    activations = torch.randn(11, 9, 5)
    direct_xtx = activations.reshape(-1, 5).T @ activations.reshape(-1, 5)
    merged_xtx = sum(
        part.reshape(-1, 5).T @ part.reshape(-1, 5)
        for part in activations.tensor_split(4)
    )
    torch.testing.assert_close(merged_xtx, direct_xtx, rtol=1e-5, atol=1e-5)

    dense = nn.Linear(5, 6, bias=False, dtype=torch.float32)
    covariance_seed = torch.randn(5, 5)
    covariance = covariance_seed @ covariance_seed.T + (0.1 * torch.eye(5))
    cholesky = torch.linalg.cholesky(covariance)
    grouped = _factor_grouped_linear(
        dense,
        [2, 2],
        cholesky,
        device=torch.device("cpu"),
    )
    reconstructed_groups = []
    inverse = torch.linalg.inv(cholesky)
    for weight_group in dense.weight.reshape(2, 3, 5):
        left, singular, right = torch.linalg.svd(weight_group @ cholesky, full_matrices=False)
        reconstructed_groups.append(
            (left[:, :2] * singular[:2].unsqueeze(0)) @ (right[:2] @ inverse)
        )
    explicit = torch.cat(reconstructed_groups, dim=0)
    module_weight = torch.cat(
        [up.weight @ down for up, down in zip(grouped.U, grouped.VT.weight.split(grouped.ranks))],
        dim=0,
    )
    torch.testing.assert_close(module_weight, explicit, rtol=1e-5, atol=1e-5)
    print("self_test=PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch this experiment with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = _rank()
    world_size = _world_size()
    if args.fisher_samples % world_size or args.whiten_samples % world_size:
        raise ValueError("calibration sample counts must be divisible by world size")
    if args.head_group_size != 4:
        raise ValueError("the paper anchor requires G-LRD head group size 4")
    if args.compression_rate != 0.5:
        raise ValueError("the paper anchor requires a 50% retained-rank ratio")

    output_dir = Path(args.output_dir).expanduser().resolve()
    exists = torch.tensor([int(output_dir.exists())], dtype=torch.int32, device=device)
    _all_reduce_sum(exists)
    if int(exists.item()) != 0:
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    _log(
        f"loaded {type(model).__name__} on {device}; workers={world_size}; "
        f"torch={torch.__version__}",
        all_ranks=True,
    )
    if model.config.model_type != "llama" or int(model.config.num_hidden_layers) != 32:
        raise ValueError("this reproduction is restricted to the paper's Llama-2-7B anchor")
    if int(model.config.num_attention_heads) != 32 or int(model.config.num_key_value_heads) != 32:
        raise ValueError("expected the Llama-2-7B 32-head MHA configuration")

    if rank == 0:
        train_text = _wikitext("train")
        test_text = _wikitext("test")
    else:
        train_text = ""
        test_text = ""
    text_payload = [train_text, test_text]
    dist.broadcast_object_list(text_payload, src=0)
    train_text, test_text = text_payload

    dense_eval = distributed_official_ppl(
        model,
        tokenizer,
        text=test_text,
        seqlen=args.eval_seqlen,
        max_chunks=args.eval_max_chunks,
        device=device,
    )
    _log(f"dense ppl={dense_eval['ppl']:.6f}")

    fisher_windows = _local_calibration_windows(
        tokenizer,
        train_text,
        samples=args.fisher_samples,
        seqlen=args.fisher_seqlen,
        seed=args.calibration_seed,
        rank=rank,
        world_size=world_size,
    )
    fisher_scalars = distributed_fisher_scalars(
        model,
        fisher_windows,
        total_samples=args.fisher_samples,
        device=device,
    )
    rank_map: dict[str, list[int]] | None = None
    rank_sum = total_rank = 0
    if rank == 0:
        rank_map, rank_sum, total_rank = official_fisher_uniform_rank_map(
            fisher_scalars,
            output_width=int(model.config.hidden_size),
            num_heads=int(model.config.num_attention_heads),
            head_group_size=args.head_group_size,
            retained_ratio=args.compression_rate,
            block_size=args.rank_block_size,
        )
        _log(f"realized retained-rank ratio={rank_sum / total_rank:.6f}")
    rank_map = _broadcast_rank_map(rank_map)
    counts = torch.tensor([rank_sum, total_rank], dtype=torch.int64, device=device)
    dist.broadcast(counts, src=0)
    rank_sum, total_rank = (int(value) for value in counts.tolist())

    # Fisher gradients and their CPU sufficient statistics are no longer needed.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del fisher_windows
    torch.cuda.empty_cache()

    whitening_windows = _local_calibration_windows(
        tokenizer,
        train_text,
        samples=args.whiten_samples,
        seqlen=args.whiten_seqlen,
        seed=args.calibration_seed,
        rank=rank,
        world_size=world_size,
    )
    cholesky_factors = distributed_whitening_cholesky(
        model,
        whitening_windows,
        seqlen=args.whiten_seqlen,
        device=device,
    )
    del whitening_windows
    torch.cuda.empty_cache()

    apply_grouped_whitened_svd(
        model,
        rank_map,
        cholesky_factors,
        device=device,
    )
    del cholesky_factors
    torch.cuda.empty_cache()
    _barrier()

    compressed_eval = distributed_official_ppl(
        model,
        tokenizer,
        text=test_text,
        seqlen=args.eval_seqlen,
        max_chunks=args.eval_max_chunks,
        device=device,
    )
    _log(f"compressed ppl={compressed_eval['ppl']:.6f}")

    if rank == 0:
        result: dict[str, Any] = {
            "protocol": "palu_iclr2025_llama2_7b_table1_table8_g_lrd_50",
            "official_code_commit": PAPER_COMMIT,
            "model": str(Path(args.model).expanduser().resolve()),
            "dtype": args.dtype,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "datasets_version": __import__("datasets").__version__,
            "world_size": world_size,
            "head_group_size": args.head_group_size,
            "rank_block_size": args.rank_block_size,
            "requested_retained_ratio": args.compression_rate,
            "realized_retained_ratio": rank_sum / total_rank,
            "rank_sum": rank_sum,
            "total_rank": total_rank,
            "rank_map": rank_map,
            "fisher_scalars": fisher_scalars,
            "fisher": {
                "dataset": args.fisher_dataset,
                "split": "train",
                "samples": args.fisher_samples,
                "seqlen": args.fisher_seqlen,
                "seed": args.calibration_seed,
                "aggregation": "all_reduce(sum(local sum(grad^2))) before sqrt and matrix mean",
            },
            "whitening": {
                "dataset": args.whiten_dataset,
                "split": "train",
                "samples": args.whiten_samples,
                "seqlen": args.whiten_seqlen,
                "seed": args.calibration_seed,
                "aggregation": "all_reduce(sum(local X^T X)) before Cholesky",
            },
            "dense_eval": dense_eval,
            "compressed_eval": compressed_eval,
            "paper_reference": {
                "dense_wikitext2_ppl": PAPER_DENSE_PPL,
                "g_lrd_50_wikitext2_ppl": PAPER_GLRD_PPL,
            },
            "elapsed_seconds": time.time() - started,
            "checkpoint_saved": False,
        }
        _write_results(output_dir, result)
        _log(f"wrote {output_dir / 'result.json'} and {output_dir / 'summary.md'}")
    _barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
