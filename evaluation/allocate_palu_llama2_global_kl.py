#!/usr/bin/env python3
"""Allocate PaLU Llama-2-7B G-LRD ranks with forward-only terminal KL.

This experiment keeps the paper's activation-aware whitening and grouped
low-rank factorization, but replaces PaLU's 2048-window parameter-Fisher
backward pass with dense-teacher forward measurements on WikiText-2
validation.  Every K/V projection is profiled over the complete 32-rank grid
while all other projections remain at the uniform r256 anchor.  An exact
dynamic program then selects a schedule at the same realized rank budget as
the reproduced PaLU Fisher schedule.

The schedule is frozen before WikiText-2 test is evaluated.  No model
checkpoint is written; rank maps, terminal-KL curves, perplexities, and a
Markdown summary are written by rank zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.global_rank_sensitivity import (  # noqa: E402
    logits_logsumexp,
    teacher_kl_sum,
)
from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)
from evaluation.reproduce_palu_paper_llama2_distributed import (  # noqa: E402
    GroupedLowRankLinear,
    _all_reduce_sum,
    _local_calibration_windows,
    _model_layers,
    _wikitext,
    distributed_official_ppl,
    distributed_whitening_cholesky,
)


FORMAT = "basisserve.palu_llama2_7b.forward_global_kl_allocation.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fisher-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-group-size", type=int, default=4)
    parser.add_argument("--rank-block-size", type=int, default=32)
    parser.add_argument(
        "--candidate-ranks",
        default="32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512",
    )
    parser.add_argument("--anchor-rank", type=int, default=256)
    parser.add_argument("--whiten-samples", type=int, default=256)
    parser.add_argument("--whiten-seqlen", type=int, default=2048)
    parser.add_argument("--whiten-seed", type=int, default=3)
    parser.add_argument("--profile-seqlen", type=int, default=512)
    parser.add_argument("--profile-windows", type=int, default=8)
    parser.add_argument("--confirmation-windows", type=int, default=16)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-chunks", type=int)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
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


def _parse_candidate_ranks(raw: str, *, block_size: int) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if (
        not values
        or values != tuple(sorted(set(values)))
        or any(value <= 0 or value % block_size for value in values)
    ):
        raise ValueError("candidate ranks must be distinct increasing block multiples")
    return values


def _projection_names(layer_count: int) -> tuple[str, ...]:
    return tuple(
        f"model.layers.{layer}.self_attn.{projection}_proj"
        for layer in range(layer_count)
        for projection in ("k", "v")
    )


def _validation_windows(
    tokenizer: Any,
    text: str,
    *,
    seqlen: int,
    profile_windows: int,
    confirmation_windows: int,
) -> tuple[Tensor, Tensor]:
    tokens = tokenizer(text, return_tensors="pt").input_ids.flatten()
    count = profile_windows + confirmation_windows
    needed = count * seqlen
    if tokens.numel() < needed:
        raise ValueError(
            f"WikiText-2 validation has {tokens.numel()} tokens, needs {needed}"
        )
    windows = tokens[:needed].reshape(count, seqlen).contiguous()
    return windows[:profile_windows], windows[profile_windows:]


@dataclass(frozen=True)
class TeacherBatch:
    input_ids: Tensor
    logits: Tensor
    logsumexp: Tensor


@torch.inference_mode()
def _capture_teacher(
    model: nn.Module,
    sequences: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    vocab_chunk_size: int,
    label: str,
) -> list[TeacherBatch]:
    result = []
    for start in range(0, len(sequences), batch_size):
        input_ids = sequences[start : start + batch_size].to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1].detach()
        result.append(
            TeacherBatch(
                input_ids=input_ids.cpu(),
                logits=logits.cpu(),
                logsumexp=logits_logsumexp(
                    logits, vocab_chunk_size=vocab_chunk_size
                ).cpu(),
            )
        )
        _log(
            f"dense teacher {label}: {min(start + batch_size, len(sequences))}/"
            f"{len(sequences)}"
        )
    return result


def _paired(values: Sequence[float]) -> dict[str, Any]:
    checked = [float(value) for value in values]
    if not checked or not all(math.isfinite(value) for value in checked):
        raise ValueError("metric values must be finite and nonempty")
    mean = statistics.fmean(checked)
    standard_error = (
        statistics.stdev(checked) / math.sqrt(len(checked))
        if len(checked) > 1
        else 0.0
    )
    return {
        "values": checked,
        "mean": mean,
        "paired_standard_error": standard_error,
        "one_standard_error_ucb": mean + standard_error,
    }


@torch.inference_mode()
def _evaluate_teacher_metrics(
    model: nn.Module,
    teacher: Sequence[TeacherBatch],
    *,
    device: torch.device,
    vocab_chunk_size: int,
) -> dict[str, Any]:
    kl_values: list[float] = []
    nll_values: list[float] = []
    for batch in teacher:
        input_ids = batch.input_ids.to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
        for index in range(input_ids.shape[0]):
            kl_sum, tokens = teacher_kl_sum(
                logits[index : index + 1],
                batch.logits[index : index + 1],
                teacher_logsumexp=batch.logsumexp[index : index + 1],
                vocab_chunk_size=vocab_chunk_size,
            )
            nll = F.cross_entropy(
                logits[index].float().reshape(-1, logits.shape[-1]),
                input_ids[index, 1:].reshape(-1),
                reduction="mean",
            )
            kl_values.append(kl_sum / tokens)
            nll_values.append(float(nll.item()))
        del logits
    return {"terminal_kl": _paired(kl_values), "nll": _paired(nll_values)}


def _paired_delta(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    result = {}
    for metric in ("terminal_kl", "nll"):
        left = candidate[metric]["values"]
        right = baseline[metric]["values"]
        if len(left) != len(right):
            raise ValueError("candidate and baseline windows are not paired")
        result[metric] = _paired(
            [a - b for a, b in zip(left, right, strict=True)]
        )
    return result


class BankedGroupedLowRankLinear(nn.Module):
    """A PaLU G-LRD projection with a reusable full whitened-SVD bank."""

    def __init__(
        self,
        *,
        left_full: Tensor,
        right_full: Tensor,
        anchor_rank: int,
        bias: Tensor | None,
    ) -> None:
        super().__init__()
        if left_full.ndim != 3 or right_full.ndim != 3:
            raise ValueError("factor banks must have [groups, rows/rank, rank/input]")
        if left_full.shape[0] != right_full.shape[0]:
            raise ValueError("left/right factor group counts differ")
        if left_full.shape[2] != right_full.shape[1]:
            raise ValueError("left/right full ranks differ")
        self.in_features = int(right_full.shape[2])
        self.out_features = int(left_full.shape[0] * left_full.shape[1])
        self.group_count = int(left_full.shape[0])
        self.group_dim = int(left_full.shape[1])
        self.maximum_rank = int(left_full.shape[2])
        self.anchor_rank = int(anchor_rank)
        self.active_rank = int(anchor_rank)
        self.register_buffer("left_full", left_full.contiguous(), persistent=False)
        self.register_buffer("right_full", right_full.contiguous(), persistent=False)
        self.register_buffer(
            "source_bias",
            None if bias is None else bias.detach().clone(),
            persistent=False,
        )
        self.anchor = self._materialize(anchor_rank)
        self.intervention: GroupedLowRankLinear | None = None

    @torch.no_grad()
    def _materialize(self, rank: int) -> GroupedLowRankLinear:
        if not 1 <= rank <= self.maximum_rank:
            raise ValueError(f"rank {rank} is outside [1, {self.maximum_rank}]")
        device = self.left_full.device
        dtype = self.left_full.dtype
        result = GroupedLowRankLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            ranks=[rank] * self.group_count,
            bias=self.source_bias is not None,
            device=device,
            dtype=dtype,
        )
        right = [self.right_full[group, :rank] for group in range(self.group_count)]
        result.VT.weight.copy_(torch.cat(right, dim=0))
        for group, up in enumerate(result.U):
            up.weight.copy_(self.left_full[group, :, :rank])
            if self.source_bias is not None:
                bias = self.source_bias.reshape(self.group_count, self.group_dim)
                up.bias.copy_(bias[group])
        return result

    @torch.no_grad()
    def set_rank(self, rank: int) -> None:
        rank = int(rank)
        if rank == self.anchor_rank:
            self.intervention = None
        elif self.intervention is None or rank != self.active_rank:
            self.intervention = self._materialize(rank)
        self.active_rank = rank

    def forward(self, hidden_states: Tensor) -> Tensor:
        module = self.anchor if self.intervention is None else self.intervention
        return module(hidden_states)


@torch.inference_mode()
def _bank_grouped_linear(
    linear: nn.Linear,
    cholesky: Tensor,
    *,
    group_count: int,
    anchor_rank: int,
    device: torch.device,
) -> BankedGroupedLowRankLinear:
    dtype = linear.weight.dtype
    group_dim = linear.out_features // group_count
    if group_dim * group_count != linear.out_features:
        raise ValueError("projection output does not divide into G-LRD groups")
    scale = cholesky.to(device=device, dtype=torch.float32)
    scale_inverse = torch.linalg.inv(scale)
    weight_groups = linear.weight.detach().reshape(
        group_count, group_dim, linear.in_features
    )
    left_groups = []
    right_groups = []
    for group in range(group_count):
        weighted = weight_groups[group].float() @ scale
        left_singular, singular_values, right_singular = torch.linalg.svd(
            weighted, full_matrices=False
        )
        sigma_root = singular_values.sqrt()
        left_groups.append(
            (left_singular * sigma_root.unsqueeze(0)).to(dtype)
        )
        right_groups.append(
            (sigma_root.unsqueeze(1) * (right_singular @ scale_inverse)).to(dtype)
        )
    return BankedGroupedLowRankLinear(
        left_full=torch.stack(left_groups),
        right_full=torch.stack(right_groups),
        anchor_rank=anchor_rank,
        bias=linear.bias,
    )


@torch.inference_mode()
def _install_factor_banks(
    model: nn.Module,
    cholesky_factors: Sequence[Tensor],
    *,
    head_group_size: int,
    anchor_rank: int,
    device: torch.device,
) -> dict[str, BankedGroupedLowRankLinear]:
    layers = _model_layers(model)
    num_heads = int(model.config.num_attention_heads)
    if num_heads % head_group_size:
        raise ValueError("attention heads do not divide by G-LRD group size")
    group_count = num_heads // head_group_size
    banks = {}
    for layer_index, layer in enumerate(
        tqdm(layers, disable=_rank() != 0, desc="Full G-LRD factor banks")
    ):
        for projection in ("k", "v"):
            attribute = f"{projection}_proj"
            dense = getattr(layer.self_attn, attribute)
            if not isinstance(dense, nn.Linear):
                raise TypeError(f"expected dense {attribute}, got {type(dense).__name__}")
            bank = _bank_grouped_linear(
                dense,
                cholesky_factors[layer_index],
                group_count=group_count,
                anchor_rank=anchor_rank,
                device=device,
            )
            name = f"model.layers.{layer_index}.self_attn.{attribute}"
            setattr(layer.self_attn, attribute, bank)
            banks[name] = bank
        _log(f"factor bank layer {layer_index} complete")
    return banks


@torch.no_grad()
def _install_schedule(
    banks: Mapping[str, BankedGroupedLowRankLinear],
    schedule: Mapping[str, int],
) -> None:
    if set(banks) != set(schedule):
        raise ValueError("rank schedule projection names differ from factor bank")
    for name, bank in banks.items():
        bank.set_rank(int(schedule[name]))


def _fisher_schedule(
    fisher_result_path: Path,
    projection_names: Sequence[str],
) -> tuple[dict[str, int], dict[str, Any]]:
    payload = json.loads(fisher_result_path.read_text(encoding="utf-8"))
    rank_map = payload["rank_map"]
    if set(rank_map) != set(projection_names):
        raise ValueError("Fisher result rank map does not match model projections")
    schedule = {}
    for name in projection_names:
        ranks = tuple(map(int, rank_map[name]))
        if not ranks or len(set(ranks)) != 1:
            raise ValueError(f"Fisher G-LRD groups do not share one rank for {name}")
        schedule[name] = ranks[0]
    expected = int(payload["rank_sum"])
    group_count = len(next(iter(rank_map.values())))
    if sum(schedule.values()) * group_count != expected:
        raise ValueError("Fisher rank sum and rank map disagree")
    return schedule, payload


def _allocate(
    records: Sequence[Mapping[str, Any]],
    projection_names: Sequence[str],
    candidate_ranks: Sequence[int],
    *,
    anchor_rank: int,
    total_rank_budget: int,
    cost_key: str,
) -> tuple[dict[str, int], float]:
    indexed = {
        (str(row["projection"]), int(row["rank"])): row for row in records
    }
    options = []
    for name in projection_names:
        coordinate = []
        for rank in candidate_ranks:
            row = indexed[(name, rank)]
            delta = row["delta_vs_anchor"]["terminal_kl"]
            coordinate.append(
                MetricRankOption(
                    option_id=f"{name}.r{rank}.{cost_key}",
                    source_family="forward_terminal_kl",
                    rank=rank,
                    scalar_cost=float(delta[cost_key]),
                    is_anchor=rank == anchor_rank,
                )
            )
        options.append(tuple(coordinate))
    allocation = allocate_metric_rank_exact(
        options,
        total_rank_budget=total_rank_budget,
        anchor_rank=anchor_rank,
    )
    schedule = {
        name: int(option.rank)
        for name, option in zip(
            projection_names, allocation.selected_options, strict=True
        )
    }
    return schedule, float(allocation.total_cost)


def _schedule_stats(
    schedule: Mapping[str, int], fisher: Mapping[str, int]
) -> dict[str, Any]:
    def values(projection: str) -> list[int]:
        return [
            int(rank)
            for name, rank in schedule.items()
            if name.endswith(f".{projection}_proj")
        ]

    fisher_values = [int(fisher[name]) for name in schedule]
    selected_values = [int(schedule[name]) for name in schedule]
    mean_fisher = statistics.fmean(fisher_values)
    mean_selected = statistics.fmean(selected_values)
    covariance = statistics.fmean(
        (left - mean_fisher) * (right - mean_selected)
        for left, right in zip(fisher_values, selected_values, strict=True)
    )
    fisher_variance = statistics.fmean(
        (value - mean_fisher) ** 2 for value in fisher_values
    )
    selected_variance = statistics.fmean(
        (value - mean_selected) ** 2 for value in selected_values
    )
    correlation = (
        covariance / math.sqrt(fisher_variance * selected_variance)
        if fisher_variance > 0 and selected_variance > 0
        else 0.0
    )
    result: dict[str, Any] = {
        "rank_sum_per_group": sum(selected_values),
        "exact_match_coordinates": sum(
            left == right
            for left, right in zip(fisher_values, selected_values, strict=True)
        ),
        "mean_absolute_difference_vs_fisher": statistics.fmean(
            abs(left - right)
            for left, right in zip(fisher_values, selected_values, strict=True)
        ),
        "pearson_correlation_vs_fisher": correlation,
    }
    for projection in ("k", "v"):
        ranks = values(projection)
        result[projection] = {
            "mean_rank": statistics.fmean(ranks),
            "minimum_rank": min(ranks),
            "maximum_rank": max(ranks),
            "histogram": {
                str(rank): ranks.count(rank) for rank in sorted(set(ranks))
            },
            "retained_ratio": statistics.fmean(ranks) / 512.0,
            "compression_ratio": 1.0 - statistics.fmean(ranks) / 512.0,
        }
    return result


def _summary(result: Mapping[str, Any]) -> str:
    selected_name = str(result["selection"]["selected_candidate"])
    selected = result["schedules"][selected_name]
    fisher = result["schedules"]["fisher"]
    uniform = result["schedules"]["uniform_r256"]
    stats = result["selection"]["selected_stats"]
    lines = [
        "# Llama-2-7B PaLU forward-only global-KL allocation",
        "",
        "## Outcome",
        "",
        "| Schedule | Rank sum/group | Validation terminal KL | WikiText-2 test PPL |",
        "|:---|---:|---:|---:|",
    ]
    for name, row in (
        ("Uniform r256", uniform),
        ("PaLU Fisher", fisher),
        (f"Global KL ({selected_name})", selected),
    ):
        lines.append(
            f"| {name} | {row['rank_sum_per_group']} | "
            f"{row['confirmation']['terminal_kl']['mean']:.8g} | "
            f"{row['test']['ppl']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Selected global-KL allocation",
            "",
            f"- K mean rank: `{stats['k']['mean_rank']:.3f}` "
            f"({100 * stats['k']['compression_ratio']:.3f}% compressed)",
            f"- V mean rank: `{stats['v']['mean_rank']:.3f}` "
            f"({100 * stats['v']['compression_ratio']:.3f}% compressed)",
            f"- Exact Fisher-coordinate matches: "
            f"`{stats['exact_match_coordinates']}/64`",
            f"- Mean absolute rank difference from Fisher: "
            f"`{stats['mean_absolute_difference_vs_fisher']:.3f}`",
            f"- Pearson rank correlation with Fisher: "
            f"`{stats['pearson_correlation_vs_fisher']:.6f}`",
            "",
            "## Protocol",
            "",
            "- Whitening: WikiText-2 train, 256 windows × 2048 tokens, seed 3.",
            "- Allocation: dense-teacher full-vocabulary terminal KL on "
            "WikiText-2 validation only.",
            "- Candidate grid: r32–r512 in increments of 32 for every K/V projection.",
            "- Selection: exact-rank dynamic programming from one-coordinate "
            "marginal curves, followed by disjoint validation confirmation.",
            "- Test: full WikiText-2 test after the schedule was frozen.",
            "- No backward pass or parameter gradients are used.",
            "",
            "## Selected ranks",
            "",
            "| Layer | Global K | Global V | Fisher K | Fisher V |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    selected_map = result["selection"]["selected_schedule"]
    fisher_map = result["selection"]["fisher_schedule"]
    for layer in range(32):
        k_name = f"model.layers.{layer}.self_attn.k_proj"
        v_name = f"model.layers.{layer}.self_attn.v_proj"
        lines.append(
            f"| {layer} | {selected_map[k_name]} | {selected_map[v_name]} | "
            f"{fisher_map[k_name]} | {fisher_map[v_name]} |"
        )
    lines.extend(["", "## Command", "", f"`{result['command']}`", ""])
    return "\n".join(lines)


def _self_test() -> None:
    torch.manual_seed(7)
    left = torch.randn(2, 4, 4)
    right = torch.randn(2, 4, 6)
    module = BankedGroupedLowRankLinear(
        left_full=left,
        right_full=right,
        anchor_rank=2,
        bias=None,
    )
    inputs = torch.randn(3, 6)
    for rank in (1, 2, 3, 4, 2):
        module.set_rank(rank)
        expected = torch.cat(
            [
                (inputs @ right[group, :rank].T) @ left[group, :, :rank].T
                for group in range(2)
            ],
            dim=-1,
        )
        torch.testing.assert_close(module(inputs), expected)

    names = ("p0", "p1")
    records = []
    for coordinate, name in enumerate(names):
        for rank in (1, 2, 3):
            cost = float((rank - (coordinate + 1)) ** 2)
            records.append(
                {
                    "projection": name,
                    "rank": rank,
                    "delta_vs_anchor": {
                        "terminal_kl": {"mean": cost, "one_standard_error_ucb": cost}
                    },
                }
            )
    schedule, _ = _allocate(
        records,
        names,
        (1, 2, 3),
        anchor_rank=2,
        total_rank_budget=3,
        cost_key="mean",
    )
    assert schedule == {"p0": 1, "p1": 2}
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
    world_size = _world_size()
    candidate_ranks = _parse_candidate_ranks(
        args.candidate_ranks, block_size=args.rank_block_size
    )
    positive = (
        args.head_group_size,
        args.rank_block_size,
        args.anchor_rank,
        args.whiten_samples,
        args.whiten_seqlen,
        args.profile_seqlen,
        args.profile_windows,
        args.confirmation_windows,
        args.eval_seqlen,
        args.vocab_chunk_size,
        args.batch_size,
    )
    if min(positive) <= 0:
        raise ValueError("all geometry and sample arguments must be positive")
    if args.anchor_rank not in candidate_ranks or max(candidate_ranks) != 512:
        raise ValueError("candidate grid must contain the anchor and full r512")
    if args.whiten_samples % world_size:
        raise ValueError("whitening samples must divide across torchrun workers")

    output_dir = args.output_dir.expanduser().resolve()
    exists = torch.tensor(int(output_dir.exists()), dtype=torch.int32, device=device)
    _all_reduce_sum(exists)
    if int(exists.item()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    started = time.time()
    model_path = Path(args.model).expanduser().resolve()
    fisher_path = args.fisher_result.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=args.local_files_only, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.config.use_cache = False
    if (
        model.config.model_type != "llama"
        or int(model.config.num_hidden_layers) != 32
        or int(model.config.num_attention_heads) != 32
        or int(model.config.num_key_value_heads) != 32
    ):
        raise ValueError("expected the PaLU Llama-2-7B MHA anchor")
    projection_names = _projection_names(32)
    fisher_schedule, fisher_payload = _fisher_schedule(
        fisher_path, projection_names
    )
    group_count = int(model.config.num_attention_heads) // args.head_group_size
    target_budget = int(fisher_payload["rank_sum"]) // group_count
    if target_budget != sum(fisher_schedule.values()):
        raise AssertionError("Fisher target budget conversion failed")

    if _rank() == 0:
        train_text = _wikitext("train")
        validation_text = _wikitext("validation")
        test_text = _wikitext("test")
    else:
        train_text = validation_text = test_text = ""
    texts = [train_text, validation_text, test_text]
    dist.broadcast_object_list(texts, src=0)
    train_text, validation_text, test_text = texts
    profile_sequences, confirmation_sequences = _validation_windows(
        tokenizer,
        validation_text,
        seqlen=args.profile_seqlen,
        profile_windows=args.profile_windows,
        confirmation_windows=args.confirmation_windows,
    )
    profile_teacher = _capture_teacher(
        model,
        profile_sequences,
        batch_size=args.batch_size,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        label="profile",
    )
    confirmation_teacher = _capture_teacher(
        model,
        confirmation_sequences,
        batch_size=args.batch_size,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        label="confirmation",
    )

    whitening_windows = _local_calibration_windows(
        tokenizer,
        train_text,
        samples=args.whiten_samples,
        seqlen=args.whiten_seqlen,
        seed=args.whiten_seed,
        rank=_rank(),
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
    banks = _install_factor_banks(
        model,
        cholesky_factors,
        head_group_size=args.head_group_size,
        anchor_rank=args.anchor_rank,
        device=device,
    )
    del cholesky_factors
    torch.cuda.empty_cache()

    uniform_schedule = {name: args.anchor_rank for name in projection_names}
    anchor_profile = _evaluate_teacher_metrics(
        model,
        profile_teacher,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    interventions = [
        (name, rank)
        for name in projection_names
        for rank in candidate_ranks
        if rank != args.anchor_rank
    ]
    local_records = []
    progress = tqdm(
        interventions[_rank() :: world_size],
        disable=_rank() != 0,
        desc="Forward terminal-KL marginals",
    )
    for name, candidate_rank in progress:
        banks[name].set_rank(candidate_rank)
        metrics = _evaluate_teacher_metrics(
            model,
            profile_teacher,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        banks[name].set_rank(args.anchor_rank)
        local_records.append(
            {
                "projection": name,
                "rank": candidate_rank,
                "metrics": metrics,
                "delta_vs_anchor": _paired_delta(metrics, anchor_profile),
            }
        )
    gathered: list[Any] | None = [None] * world_size if _rank() == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    all_records: list[dict[str, Any]] | None = None
    candidates: dict[str, dict[str, int]] | None = None
    predicted_costs: dict[str, float] | None = None
    if _rank() == 0:
        all_records = [row for shard in gathered for row in shard]
        for name in projection_names:
            all_records.append(
                {
                    "projection": name,
                    "rank": args.anchor_rank,
                    "metrics": anchor_profile,
                    "delta_vs_anchor": {
                        metric: _paired([0.0] * args.profile_windows)
                        for metric in ("terminal_kl", "nll")
                    },
                }
            )
        all_records.sort(key=lambda row: (row["projection"], row["rank"]))
        candidates = {}
        predicted_costs = {}
        for label, cost_key in (
            ("global_kl_mean", "mean"),
            ("global_kl_ucb", "one_standard_error_ucb"),
        ):
            schedule, predicted = _allocate(
                all_records,
                projection_names,
                candidate_ranks,
                anchor_rank=args.anchor_rank,
                total_rank_budget=target_budget,
                cost_key=cost_key,
            )
            candidates[label] = schedule
            predicted_costs[label] = predicted
    payloads: list[Any] = [all_records, candidates, predicted_costs]
    dist.broadcast_object_list(payloads, src=0)
    all_records, candidates, predicted_costs = payloads

    schedules = {
        "uniform_r256": uniform_schedule,
        "fisher": fisher_schedule,
        **candidates,
    }
    confirmation = {}
    for label, schedule in schedules.items():
        _install_schedule(banks, schedule)
        confirmation[label] = _evaluate_teacher_metrics(
            model,
            confirmation_teacher,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _log(
            f"confirmation {label}: KL="
            f"{confirmation[label]['terminal_kl']['mean']:.8g}"
        )
    selected_name = min(
        candidates,
        key=lambda label: (
            confirmation[label]["terminal_kl"]["mean"], label
        ),
    )
    selected_schedule = candidates[selected_name]
    # The validation-only selection is now frozen.  WikiText-2 test has not
    # been run through any compressed schedule before this point.
    test_metrics = {}
    for label in ("uniform_r256", "fisher", selected_name):
        _install_schedule(banks, schedules[label])
        test_metrics[label] = distributed_official_ppl(
            model,
            tokenizer,
            text=test_text,
            seqlen=args.eval_seqlen,
            max_chunks=args.eval_max_chunks,
            device=device,
        )
        _log(f"test {label}: PPL={test_metrics[label]['ppl']:.6f}")

    if _rank() == 0:
        schedule_rows = {}
        for label, schedule in schedules.items():
            row = {
                "rank_sum_per_group": sum(schedule.values()),
                "confirmation": confirmation[label],
                "schedule": schedule,
            }
            if label in test_metrics:
                row["test"] = test_metrics[label]
            schedule_rows[label] = row
        result = {
            "format": FORMAT,
            "command": shlex.join(os.sys.argv),
            "elapsed_seconds": time.time() - started,
            "world_size": world_size,
            "model": str(model_path),
            "dtype": "float16",
            "factorization": {
                "method": "PaLU activation-aware whitened G-LRD",
                "head_group_size": args.head_group_size,
                "whitening_dataset": "wikitext2_train",
                "whitening_samples": args.whiten_samples,
                "whitening_seqlen": args.whiten_seqlen,
                "whitening_seed": args.whiten_seed,
            },
            "profile": {
                "dataset": "wikitext2_validation",
                "seqlen": args.profile_seqlen,
                "windows": args.profile_windows,
                "candidate_ranks": list(candidate_ranks),
                "anchor_rank": args.anchor_rank,
                "anchor_metrics": anchor_profile,
                "records": all_records,
            },
            "confirmation": {
                "dataset": "wikitext2_validation",
                "disjoint_from_profile": True,
                "seqlen": args.profile_seqlen,
                "windows": args.confirmation_windows,
            },
            "selection": {
                "target_rank_budget_per_group": target_budget,
                "same_realized_budget_as_fisher": True,
                "predicted_additive_costs": predicted_costs,
                "selected_candidate": selected_name,
                "selected_schedule": selected_schedule,
                "fisher_schedule": fisher_schedule,
                "selected_stats": _schedule_stats(
                    selected_schedule, fisher_schedule
                ),
            },
            "schedules": schedule_rows,
            "fisher_source": str(fisher_path),
            "fisher_source_ppl": fisher_payload["compressed_eval"],
            "dense_source_ppl": fisher_payload["dense_eval"],
            "test_protocol": {
                "dataset": "wikitext2_test",
                "seqlen": args.eval_seqlen,
                "max_chunks": args.eval_max_chunks,
                "schedule_frozen_before_test": True,
            },
        }
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.md").write_text(
            _summary(result), encoding="utf-8"
        )
        _log(f"wrote {output_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
