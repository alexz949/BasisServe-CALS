#!/usr/bin/env python3
"""Allocate layerwise Llama-2 MHA C1 Value ranks with exact terminal KL.

Every layer keeps one rank shared by its 32 Value heads.  Candidate C1
factorizations are fitted beforehand with the same activation-aware output
initialization and full-layer joint decoder refit.  This program profiles one
layer/rank intervention at a time around the uniform anchor, solves an exact
budget dynamic program, confirms the mean and one-standard-error schedules on
disjoint C4 documents, and only then evaluates WikiText-2 test perplexity.

The dense teacher and all compressed candidates are forward-only.  Full-vocab
KL uses FP32 probabilities and FP64 accumulation; no backward, JVP, or VJP is
performed here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.allocate_palu_llama2_global_kl import (  # noqa: E402
    _allocate,
    _capture_teacher,
    _evaluate_teacher_metrics,
    _paired,
    _paired_delta,
)
from evaluation.eval_llama2_mha_v25_comparison import (  # noqa: E402
    C1_FORMAT,
    _fold_c1_weights,
)
from evaluation.reproduce_palu_paper_llama2_distributed import (  # noqa: E402
    _all_reduce_sum,
    _model_layers,
    _wikitext,
    distributed_official_ppl,
)


FORMAT = "basisserve.llama2_7b.mha_c1.forward_global_kl_allocation.v1"
SELECT_SPLIT_CODE = 1
RANK_DEPENDENT_CONFIG_KEYS = {
    "cache_rank_per_head",
    "total_v_cache_rank",
    "total_kv_retained_ratio_with_dense_k",
    "v_retained_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--factor-dir",
        action="append",
        required=True,
        metavar="RANK=PATH",
        help="Repeat once for every candidate rank, including the anchor",
    )
    parser.add_argument("--anchor-rank", type=int, default=96)
    parser.add_argument("--rank-block-size", type=int, default=16)
    parser.add_argument("--profile-windows", type=int, default=32)
    parser.add_argument("--confirmation-windows", type=int, default=32)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-chunks", type=int)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _log(message: str, *, all_ranks: bool = False) -> None:
    if all_ranks or _rank() == 0:
        print(f"[rank {_rank()}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_factor_dirs(specs: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in specs:
        rank_text, separator, path_text = raw.partition("=")
        if not separator or not rank_text.strip() or not path_text.strip():
            raise ValueError(f"factor directory must use RANK=PATH: {raw!r}")
        rank = int(rank_text)
        if rank <= 0 or rank in result:
            raise ValueError(f"invalid or duplicate factor rank: {rank}")
        result[rank] = Path(path_text).expanduser().resolve()
    return dict(sorted(result.items()))


def _rank_invariant_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in RANK_DEPENDENT_CONFIG_KEYS
    }


def _load_factor_results(
    factor_dirs: Mapping[int, Path],
    *,
    model_config_sha256: str,
    layer_count: int,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    reference_config: dict[str, Any] | None = None
    expected_layers = list(range(layer_count))
    for rank, directory in factor_dirs.items():
        result_path = directory / "results.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("format") != C1_FORMAT or payload.get("status") != "complete":
            raise ValueError(f"incomplete or incompatible C1 result: {result_path}")
        if payload.get("layers") != expected_layers:
            raise ValueError(f"C1 result does not cover every layer: {result_path}")
        config = payload.get("fit_config", {})
        if int(config.get("cache_rank_per_head", -1)) != rank:
            raise ValueError(f"factor rank disagrees with directory spec: {result_path}")
        if config.get("model_config_sha256") != model_config_sha256:
            raise ValueError(f"factor result belongs to another model: {result_path}")
        invariant = _rank_invariant_fit_config(config)
        if reference_config is None:
            reference_config = invariant
        elif invariant != reference_config:
            raise ValueError("candidate C1 fits differ in more than rank-dependent fields")
        results[rank] = payload
    return results


def _select_windows(
    path: Path,
    *,
    profile_windows: int,
    confirmation_windows: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    path = path.expanduser().resolve()
    payload = load_file(str(path), device="cpu")
    if "input_ids" not in payload or "split_codes" not in payload:
        raise ValueError("C4 window bank must contain input_ids and split_codes")
    input_ids = payload["input_ids"].to(torch.long)
    split_codes = payload["split_codes"].to(torch.long)
    if input_ids.ndim != 2 or split_codes.shape != input_ids.shape[:1]:
        raise ValueError("invalid C4 window-bank geometry")
    selected_indices = torch.nonzero(
        split_codes == SELECT_SPLIT_CODE, as_tuple=False
    ).flatten()
    needed = profile_windows + confirmation_windows
    if needed <= 0 or selected_indices.numel() < needed:
        raise ValueError(
            f"requested {needed} select windows but only "
            f"{selected_indices.numel()} are available"
        )
    selected_indices = selected_indices[:needed]
    selected = input_ids.index_select(0, selected_indices).contiguous()
    provenance = {
        "path": str(path),
        "sha256": _sha256(path),
        "split_code": SELECT_SPLIT_CODE,
        "selected_window_indices": selected_indices.tolist(),
        "sequence_length": int(input_ids.shape[1]),
    }
    return (
        selected[:profile_windows],
        selected[profile_windows:],
        provenance,
    )


@dataclass
class C1LayerBank:
    layer_index: int
    v_proj: nn.Linear
    o_proj: nn.Linear
    factor_dirs: Mapping[int, Path]
    factor_results: Mapping[int, Mapping[str, Any]]
    anchor_rank: int
    num_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.v_proj.bias is not None or self.o_proj.bias is not None:
            raise TypeError("C1 folding requires bias-free V and O projections")
        self.device = self.v_proj.weight.device
        self.model_dtype = self.v_proj.weight.dtype
        self.dense_v = self.v_proj.weight.detach().cpu().clone()
        self.dense_o = self.o_proj.weight.detach().cpu().clone()
        self._verified_paths: set[Path] = set()
        self.active_rank: int | None = None
        self._materialize(self.anchor_rank)
        self.anchor_v = self.v_proj.weight.detach().cpu().clone()
        self.anchor_o = self.o_proj.weight.detach().cpu().clone()
        self.active_rank = self.anchor_rank

    def _artifact_path(self, rank: int) -> tuple[Path, Mapping[str, Any]]:
        result = self.factor_results[rank]
        artifact = result["artifacts"][str(self.layer_index)]
        path = self.factor_dirs[rank] / artifact["file"]
        if path not in self._verified_paths:
            if _sha256(path) != artifact["sha256"]:
                raise ValueError(
                    f"C1 factor hash mismatch at layer {self.layer_index}, rank {rank}"
                )
            self._verified_paths.add(path)
        return path, artifact

    @torch.no_grad()
    def _materialize(self, rank: int) -> None:
        path, _ = self._artifact_path(rank)
        factors = load_file(str(path), device="cpu")
        if set(factors) != {
            "value_coordinate_encoders",
            "head_output_decoders",
        }:
            raise ValueError(f"unexpected tensors in {path}")
        encoders = factors["value_coordinate_encoders"]
        decoders = factors["head_output_decoders"]
        if tuple(encoders.shape) != (self.num_heads, self.head_dim, rank):
            raise ValueError(f"unexpected encoder shape in {path}")
        if tuple(decoders.shape) != (
            self.num_heads,
            rank,
            self.num_heads * self.head_dim,
        ):
            raise ValueError(f"unexpected decoder shape in {path}")
        folded_v, folded_o = _fold_c1_weights(
            self.dense_v.to(device=self.device),
            self.dense_o.to(device=self.device),
            encoders,
            decoders,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.v_proj.weight.copy_(folded_v.to(dtype=self.model_dtype))
        self.o_proj.weight.copy_(folded_o.to(dtype=self.model_dtype))
        del factors, encoders, decoders, folded_v, folded_o

    @torch.no_grad()
    def set_rank(self, rank: int) -> None:
        rank = int(rank)
        if rank not in self.factor_results:
            raise ValueError(f"rank {rank} is absent from the C1 factor bank")
        if rank == self.active_rank:
            return
        if rank == self.anchor_rank:
            self.v_proj.weight.copy_(
                self.anchor_v.to(device=self.device, dtype=self.model_dtype)
            )
            self.o_proj.weight.copy_(
                self.anchor_o.to(device=self.device, dtype=self.model_dtype)
            )
        else:
            self._materialize(rank)
        self.active_rank = rank


@torch.inference_mode()
def _install_banks(
    model: nn.Module,
    *,
    factor_dirs: Mapping[int, Path],
    factor_results: Mapping[int, Mapping[str, Any]],
    anchor_rank: int,
    num_heads: int,
    head_dim: int,
) -> list[C1LayerBank]:
    banks = []
    for layer_index, layer in enumerate(
        tqdm(_model_layers(model), disable=_rank() != 0, desc="C1 rank banks")
    ):
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        if not isinstance(v_proj, nn.Linear) or not isinstance(o_proj, nn.Linear):
            raise TypeError(f"layer {layer_index} does not use dense V/O linears")
        banks.append(
            C1LayerBank(
                layer_index=layer_index,
                v_proj=v_proj,
                o_proj=o_proj,
                factor_dirs=factor_dirs,
                factor_results=factor_results,
                anchor_rank=anchor_rank,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        )
        _log(f"C1 anchor bank layer {layer_index} complete")
        torch.cuda.empty_cache()
    return banks


def _install_schedule(banks: Sequence[C1LayerBank], schedule: Mapping[str, int]) -> None:
    expected = {f"layer_{index:03d}" for index in range(len(banks))}
    if set(schedule) != expected:
        raise ValueError("rank schedule layer names differ from the C1 banks")
    for index, bank in enumerate(banks):
        bank.set_rank(int(schedule[f"layer_{index:03d}"]))
    torch.cuda.empty_cache()


def _schedule_stats(schedule: Mapping[str, int]) -> dict[str, Any]:
    ranks = [int(schedule[name]) for name in sorted(schedule)]
    return {
        "rank_sum_per_head": sum(ranks),
        "minimum_rank": min(ranks),
        "maximum_rank": max(ranks),
        "mean_rank": statistics.fmean(ranks),
        "histogram": {
            str(rank): ranks.count(rank) for rank in sorted(set(ranks))
        },
    }


@torch.inference_mode()
def _distributed_teacher_metrics(
    model: nn.Module,
    teacher: Sequence[Any],
    *,
    device: torch.device,
    vocab_chunk_size: int,
) -> dict[str, Any]:
    """Evaluate disjoint local contexts and reconstruct one global summary."""

    local = _evaluate_teacher_metrics(
        model,
        teacher,
        device=device,
        vocab_chunk_size=vocab_chunk_size,
    )
    gathered: list[Any] | None = [None] * _world_size() if _rank() == 0 else None
    dist.gather_object(local, gathered, dst=0)
    merged: dict[str, Any] | None = None
    if _rank() == 0:
        assert gathered is not None
        merged = {
            metric: _paired(
                [
                    float(value)
                    for shard in gathered
                    for value in shard[metric]["values"]
                ]
            )
            for metric in ("terminal_kl", "nll")
        }
    broadcast: list[Any] = [merged]
    dist.broadcast_object_list(broadcast, src=0)
    if not isinstance(broadcast[0], dict):
        raise RuntimeError("failed to broadcast distributed teacher metrics")
    return broadcast[0]


def _summary(result: Mapping[str, Any]) -> str:
    selected_name = str(result["selection"]["selected_candidate"])
    selected = result["schedules"][selected_name]
    uniform = result["schedules"]["uniform_anchor"]
    encoder_sweeps = int(result["factorization"]["encoder_sweeps"])
    fit_description = (
        "activation-aware attention-output initialization plus one closed-form "
        "full-layer joint decoder refit and zero encoder sweeps"
        if encoder_sweeps == 0
        else (
            "activation-aware attention-output initialization plus full-layer "
            f"joint decoder refits and {encoder_sweeps} encoder BCD sweeps"
        )
    )
    lines = [
        "# Llama-2-7B MHA C1 forward global-KL allocation",
        "",
        "## Outcome",
        "",
        "| Schedule | Sum of per-layer head ranks | Confirmation KL | WikiText-2 PPL |",
        "|:---|---:|---:|---:|",
        (
            f"| Uniform r{result['profile']['anchor_rank']} | "
            f"{uniform['rank_sum_per_head']} | "
            f"{uniform['confirmation']['terminal_kl']['mean']:.9g} | "
            f"{uniform['test']['ppl']:.9f} |"
        ),
        (
            f"| Global KL ({selected_name}) | "
            f"{selected['rank_sum_per_head']} | "
            f"{selected['confirmation']['terminal_kl']['mean']:.9g} | "
            f"{selected['test']['ppl']:.9f} |"
        ),
        "",
        "## Protocol",
        "",
        "- K remains dense; all 32 V heads inside one layer share that layer's rank.",
        f"- Every rank candidate uses {fit_description}.",
        "- Allocation uses dense-teacher, full-vocabulary terminal KL only; no backward, JVP, or VJP.",
        "- Profile and confirmation use disjoint C4 select documents; the schedule is frozen before WikiText-2 test.",
        (
            "- The exact total V-cache budget equals uniform "
            f"V{result['profile']['anchor_rank']}."
        ),
        "",
        "## Selected ranks",
        "",
        "| Layer | Rank/head |",
        "|---:|---:|",
    ]
    for layer in range(int(result["geometry"]["num_hidden_layers"])):
        name = f"layer_{layer:03d}"
        lines.append(f"| {layer} | {result['selection']['selected_schedule'][name]} |")
    lines.extend(["", "## Command", "", f"`{result['command']}`", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch this experiment with torchrun")
    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    started = time.perf_counter()

    positive = (
        args.anchor_rank,
        args.rank_block_size,
        args.profile_windows,
        args.confirmation_windows,
        args.vocab_chunk_size,
        args.batch_size,
        args.eval_seqlen,
        args.torch_num_threads,
    )
    if min(positive) <= 0:
        raise ValueError("all rank, sample, and compute arguments must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    exists = torch.tensor(int(output_dir.exists()), dtype=torch.int32, device=device)
    _all_reduce_sum(exists)
    if int(exists.item()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    factor_dirs = _parse_factor_dirs(args.factor_dir)
    candidate_ranks = tuple(factor_dirs)
    if args.anchor_rank not in factor_dirs:
        raise ValueError("factor directories must include the anchor rank")
    if any(rank % args.rank_block_size for rank in candidate_ranks):
        raise ValueError("candidate ranks must be multiples of rank-block-size")

    model_path = Path(args.model).expanduser().resolve()
    model_config_sha256 = _sha256(model_path / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=args.local_files_only, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.config.use_cache = False
    if model.config.model_type != "llama":
        raise ValueError(f"expected Llama, got {model.config.model_type}")

    layer_count = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = int(getattr(model.config, "head_dim", 0) or hidden_size // num_heads)
    if num_heads != num_kv_heads or hidden_size != num_heads * head_dim:
        raise ValueError("this allocator requires MHA with hidden_size = heads * head_dim")
    if max(candidate_ranks) > head_dim:
        raise ValueError("candidate rank exceeds the Value head width")
    layer_names = tuple(f"layer_{layer:03d}" for layer in range(layer_count))
    target_budget = layer_count * args.anchor_rank

    factor_results = _load_factor_results(
        factor_dirs,
        model_config_sha256=model_config_sha256,
        layer_count=layer_count,
    )
    profile_sequences, confirmation_sequences, windows_provenance = _select_windows(
        args.windows,
        profile_windows=args.profile_windows,
        confirmation_windows=args.confirmation_windows,
    )
    if profile_sequences.shape[1] != confirmation_sequences.shape[1]:
        raise AssertionError("profile and confirmation sequence lengths differ")
    _log(
        f"profile={len(profile_sequences)} confirmation={len(confirmation_sequences)} "
        f"seqlen={profile_sequences.shape[1]} candidates={candidate_ranks}"
    )

    profile_teacher = _capture_teacher(
        model,
        profile_sequences,
        batch_size=args.batch_size,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        label="C4 profile",
    )
    local_confirmation_sequences = confirmation_sequences[
        _rank() :: _world_size()
    ].contiguous()
    confirmation_teacher = _capture_teacher(
        model,
        local_confirmation_sequences,
        batch_size=args.batch_size,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        label="local C4 confirmation",
    )
    banks = _install_banks(
        model,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        anchor_rank=args.anchor_rank,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    uniform_schedule = {name: args.anchor_rank for name in layer_names}
    anchor_profile: dict[str, Any] | None = None
    if _rank() == 0:
        anchor_profile = _evaluate_teacher_metrics(
            model,
            profile_teacher,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
    anchor_broadcast: list[Any] = [anchor_profile]
    dist.broadcast_object_list(anchor_broadcast, src=0)
    anchor_profile = anchor_broadcast[0]
    if not isinstance(anchor_profile, dict):
        raise RuntimeError("failed to broadcast anchor metrics")
    _log(f"anchor profile KL={anchor_profile['terminal_kl']['mean']:.9g}")

    interventions = [
        (layer, rank)
        for layer in range(layer_count)
        for rank in candidate_ranks
        if rank != args.anchor_rank
    ]
    local_records: list[dict[str, Any]] = []
    for layer, candidate_rank in tqdm(
        interventions[_rank() :: _world_size()],
        disable=_rank() != 0,
        desc="C1 terminal-KL marginals",
    ):
        banks[layer].set_rank(candidate_rank)
        metrics = _evaluate_teacher_metrics(
            model,
            profile_teacher,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        banks[layer].set_rank(args.anchor_rank)
        local_records.append(
            {
                "projection": layer_names[layer],
                "layer": layer,
                "rank": candidate_rank,
                "metrics": metrics,
                "delta_vs_anchor": _paired_delta(metrics, anchor_profile),
            }
        )
        _log(
            f"profile layer={layer} rank={candidate_rank} "
            f"KL={metrics['terminal_kl']['mean']:.9g}",
            all_ranks=True,
        )

    gathered: list[Any] | None = [None] * _world_size() if _rank() == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    all_records: list[dict[str, Any]] | None = None
    candidates: dict[str, dict[str, int]] | None = None
    predicted_costs: dict[str, float] | None = None
    if _rank() == 0:
        assert gathered is not None
        all_records = [row for shard in gathered for row in shard]
        for layer, name in enumerate(layer_names):
            all_records.append(
                {
                    "projection": name,
                    "layer": layer,
                    "rank": args.anchor_rank,
                    "metrics": anchor_profile,
                    "delta_vs_anchor": {
                        metric: _paired([0.0] * args.profile_windows)
                        for metric in ("terminal_kl", "nll")
                    },
                }
            )
        all_records.sort(key=lambda row: (row["layer"], row["rank"]))
        candidates = {}
        predicted_costs = {}
        for label, cost_key in (
            ("global_kl_mean", "mean"),
            ("global_kl_ucb", "one_standard_error_ucb"),
        ):
            schedule, predicted = _allocate(
                all_records,
                layer_names,
                candidate_ranks,
                anchor_rank=args.anchor_rank,
                total_rank_budget=target_budget,
                cost_key=cost_key,
            )
            candidates[label] = schedule
            predicted_costs[label] = predicted
    broadcast: list[Any] = [all_records, candidates, predicted_costs]
    dist.broadcast_object_list(broadcast, src=0)
    all_records, candidates, predicted_costs = broadcast
    assert candidates is not None

    schedules = {"uniform_anchor": uniform_schedule, **candidates}
    confirmation: dict[str, dict[str, Any]] = {}
    for label, schedule in schedules.items():
        _install_schedule(banks, schedule)
        confirmation[label] = _distributed_teacher_metrics(
            model,
            confirmation_teacher,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _log(
            f"confirmation {label}: "
            f"KL={confirmation[label]['terminal_kl']['mean']:.9g}"
        )
    selected_name = min(
        candidates,
        key=lambda label: (confirmation[label]["terminal_kl"]["mean"], label),
    )
    selected_schedule = candidates[selected_name]

    if _rank() == 0:
        test_text = _wikitext("test")
    else:
        test_text = ""
    texts = [test_text]
    dist.broadcast_object_list(texts, src=0)
    test_text = texts[0]
    test_metrics = {}
    for label in ("uniform_anchor", selected_name):
        _install_schedule(banks, schedules[label])
        test_metrics[label] = distributed_official_ppl(
            model,
            tokenizer,
            text=test_text,
            seqlen=args.eval_seqlen,
            max_chunks=args.eval_max_chunks,
            device=device,
        )
        _log(f"test {label}: PPL={test_metrics[label]['ppl']:.9f}")

    if _rank() == 0:
        schedule_rows: dict[str, Any] = {}
        for label, schedule in schedules.items():
            stats = _schedule_stats(schedule)
            row: dict[str, Any] = {
                **stats,
                "total_v_rank_all_heads": stats["rank_sum_per_head"] * num_heads,
                "confirmation": confirmation[label],
                "schedule": schedule,
            }
            if label in test_metrics:
                row["test"] = test_metrics[label]
            schedule_rows[label] = row
        selected_stats = _schedule_stats(selected_schedule)
        result = {
            "format": FORMAT,
            "command": shlex.join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "world_size": _world_size(),
            "model": str(model_path),
            "model_config_sha256": model_config_sha256,
            "model_dtype": "float16",
            "geometry": {
                "num_hidden_layers": layer_count,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
            },
            "factorization": {
                "method": "C1 activation-aware output initialization plus joint decoder refit",
                "encoder_sweeps": int(
                    factor_results[args.anchor_rank]["fit_config"]["encoder_sweeps"]
                ),
                "factor_dirs": {
                    str(rank): {
                        "path": str(factor_dirs[rank]),
                        "results_sha256": _sha256(factor_dirs[rank] / "results.json"),
                        "aggregate": factor_results[rank]["aggregate"],
                    }
                    for rank in candidate_ranks
                },
                "common_fit_config": _rank_invariant_fit_config(
                    factor_results[args.anchor_rank]["fit_config"]
                ),
            },
            "profile": {
                "dataset": "c4_select",
                "windows": args.profile_windows,
                "seqlen": int(profile_sequences.shape[1]),
                "candidate_ranks": list(candidate_ranks),
                "anchor_rank": args.anchor_rank,
                "anchor_metrics": anchor_profile,
                "records": all_records,
            },
            "confirmation": {
                "dataset": "c4_select",
                "windows": args.confirmation_windows,
                "seqlen": int(confirmation_sequences.shape[1]),
                "disjoint_from_profile": True,
                "windows_provenance": windows_provenance,
            },
            "selection": {
                "target_rank_budget_per_head_across_layers": target_budget,
                "target_total_v_rank_all_heads": target_budget * num_heads,
                "uniform_total_v_rank_all_heads": (
                    layer_count * args.anchor_rank * num_heads
                ),
                "budget_exact": sum(selected_schedule.values()) == target_budget,
                "predicted_additive_costs": predicted_costs,
                "selected_candidate": selected_name,
                "selected_schedule": selected_schedule,
                "selected_stats": selected_stats,
            },
            "schedules": schedule_rows,
            "test_protocol": {
                "dataset": "wikitext2_test",
                "seqlen": args.eval_seqlen,
                "max_chunks": args.eval_max_chunks,
                "schedule_frozen_before_test": True,
            },
            "numerics": {
                "teacher_and_model_dtype": "float16",
                "full_vocabulary_kl_probability_dtype": "float32",
                "terminal_kl_accumulation_dtype": "float64",
                "ppl_loss_dtype": "float32",
            },
            "environment": {
                "python_executable": sys.executable,
                "conda_environment_variable": os.environ.get("CONDA_DEFAULT_ENV"),
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
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
