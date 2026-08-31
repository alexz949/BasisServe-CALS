#!/usr/bin/env python3
"""Compare reverse-mode Fisher and exact terminal KL for C1 TP-source ranks.

The pilot starts from the uniform Llama-2 MHA C1 anchor.  It partitions the 32
Value heads into the eight contiguous four-head sources owned by TP=8, changes
all four heads of exactly one source to an adjacent candidate rank, and
analytically refits the complete ragged layer decoder with the encoders fixed.
Every candidate is then scored in two matched ways:

* exact dense-teacher terminal KL from a complete model forward; and
* a suffix reverse-mode Fisher/GGN approximation around the uniform anchor.

The reverse-mode pass is cut at the selected attention output, so no parameter
gradient or prefix graph is retained.  A differentiable teacher-KL gradient
provides the linear term.  Pseudo-label gradients sampled from the anchor
distribution provide a Monte-Carlo softmax-Fisher quadratic term.  The same
candidate attention-output displacement is used for both terms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
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
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_group_rank_llama import (  # noqa: E402
    RaggedGQATiedVOLlamaAttention,
)
from basisserve.core.decoder_closed_rank_candidates import (  # noqa: E402
    close_ragged_decoder_with_fixed_encoders,
    tensor_sha256,
)
from basisserve.core.global_rank_sensitivity import (  # noqa: E402
    differentiable_teacher_kl_mean,
    teacher_kl_sum,
)
from basisserve.core.gqa_routed_ov_joint import (  # noqa: E402
    covariance_with_trace_damping,
    evaluate_quadratic,
    quadratic_from_target,
)
from evaluation.allocate_llama2_mha_c1_global_kl import (  # noqa: E402
    _capture_teacher,
    _evaluate_teacher_metrics,
    _install_banks,
    _load_factor_results,
    _paired,
    _paired_delta,
    _parse_factor_dirs,
    _select_windows,
    _sha256,
)
from evaluation.fit_llama2_mha_c1_joint import (  # noqa: E402
    _activation_covariance_blocks,
    _load_snapshot_layer,
    _snapshot_manifest,
)
from evaluation.reproduce_palu_paper_llama2_distributed import (  # noqa: E402
    _all_reduce_sum,
    _model_layers,
)


FORMAT = "basisserve.llama2_7b.mha_c1.per_tp_source_fisher_kl_pilot.v2"
NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = 4096
TP_SIZE = 8
HEADS_PER_SOURCE = NUM_HEADS // TP_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--validation-snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--factor-dir",
        action="append",
        required=True,
        metavar="RANK=PATH",
    )
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--anchor-rank", type=int, default=64)
    parser.add_argument("--candidate-ranks", default="48,80")
    parser.add_argument("--profile-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--fit-windows", type=int, default=128)
    parser.add_argument("--validation-windows", type=int, default=64)
    parser.add_argument("--validation-window-start", type=int, default=384)
    parser.add_argument("--fisher-probes", type=int, default=2)
    parser.add_argument("--fisher-seed", type=int, default=20260819)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-7)
    parser.add_argument("--covariance-row-chunk-size", type=int, default=8192)
    parser.add_argument("--decoder-relative-jitter", type=float, default=0.0)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
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


def _parse_ranks(raw: str) -> tuple[int, ...]:
    ranks = tuple(sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()}))
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError("candidate ranks must be positive")
    return ranks


def _mean(values: Sequence[float]) -> float:
    checked = [float(value) for value in values]
    if not checked or not all(math.isfinite(value) for value in checked):
        raise ValueError("values must be finite and nonempty")
    return statistics.fmean(checked)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires matched vectors of length at least two")
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(x, y) / denominator)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(map(float, values)), key=lambda item: (item[1], item[0]))
    result = [0.0] * len(ordered)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        average = 0.5 * ((start + 1) + stop)
        for index in range(start, stop):
            result[ordered[index][0]] = average
        start = stop
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _correlations(
    records: Sequence[Mapping[str, Any]],
    *,
    actual_key: str,
    predictor_key: str,
) -> dict[str, float]:
    actual = [float(record[actual_key]) for record in records]
    predicted = [float(record[predictor_key]) for record in records]
    return {
        "count": len(records),
        "pearson": _pearson(predicted, actual),
        "spearman": _spearman(predicted, actual),
    }


@dataclass(frozen=True)
class VJPWindow:
    anchor_output: Tensor
    kl_gradient: Tensor
    fisher_gradients: tuple[Tensor, ...]
    fisher_sample_count: int


def _collective_cost(
    ranks: Sequence[int],
    *,
    tp_size: int = TP_SIZE,
) -> dict[str, Any]:
    """Account for contiguous-head TP shards and a rectangular AllGather."""

    selected = tuple(int(rank) for rank in ranks)
    if not selected or any(rank <= 0 for rank in selected):
        raise ValueError("head ranks must be positive")
    if tp_size <= 0 or len(selected) % tp_size:
        raise ValueError("head ranks must divide evenly over the TP size")
    heads_per_source = len(selected) // tp_size
    source_widths = tuple(
        sum(selected[source * heads_per_source : (source + 1) * heads_per_source])
        for source in range(tp_size)
    )
    return {
        "tp_size": tp_size,
        "heads_per_source": heads_per_source,
        "source_widths": list(source_widths),
        "ideal_allgather_width": sum(source_widths),
        "padded_allgather_width": tp_size * max(source_widths),
    }


def _source_heads(source: int, *, tp_size: int = TP_SIZE) -> tuple[int, ...]:
    if tp_size <= 0 or NUM_HEADS % tp_size:
        raise ValueError("Llama heads must divide evenly over the TP size")
    if not 0 <= source < tp_size:
        raise IndexError("TP source index is outside the collective")
    heads_per_source = NUM_HEADS // tp_size
    first = source * heads_per_source
    return tuple(range(first, first + heads_per_source))


def _fisher_curvature(
    probe_derivatives: Sequence[Tensor],
    *,
    sample_count: int,
) -> float:
    if not probe_derivatives or sample_count <= 0:
        raise ValueError("Fisher curvature needs probes and a positive sample count")
    values = torch.stack(
        [torch.as_tensor(value, dtype=torch.float64).square() for value in probe_derivatives]
    )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Fisher probe derivatives must be finite")
    return float(sample_count * values.mean())


def _load_layer_factors(
    factor_dirs: Mapping[int, Path],
    factor_results: Mapping[int, Mapping[str, Any]],
    *,
    layer: int,
    ranks: Sequence[int],
) -> dict[int, tuple[Tensor, Tensor]]:
    result = {}
    for rank in ranks:
        artifact = factor_results[rank]["artifacts"][str(layer)]
        path = factor_dirs[rank] / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"factor hash mismatch for rank {rank}, layer {layer}")
        payload = load_file(str(path), device="cpu")
        encoders = payload["value_coordinate_encoders"].contiguous()
        decoders = payload["head_output_decoders"].contiguous()
        if tuple(encoders.shape) != (NUM_HEADS, HEAD_DIM, rank):
            raise ValueError("unexpected C1 encoder shape")
        if tuple(decoders.shape) != (NUM_HEADS, rank, HIDDEN_SIZE):
            raise ValueError("unexpected C1 decoder shape")
        result[rank] = (encoders, decoders)
    return result


def _padded_candidate_factors(
    source: Mapping[int, tuple[Tensor, Tensor]],
    *,
    candidate_heads: Sequence[int],
    candidate_rank: int,
    anchor_rank: int,
    maximum_rank: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    changed = tuple(int(head) for head in candidate_heads)
    if not changed or len(changed) != len(set(changed)):
        raise ValueError("candidate heads must be nonempty and unique")
    if min(changed) < 0 or max(changed) >= NUM_HEADS:
        raise IndexError("candidate head is outside Llama MHA")
    changed_set = set(changed)
    ranks = tuple(
        candidate_rank if index in changed_set else anchor_rank
        for index in range(NUM_HEADS)
    )
    A = torch.zeros(
        NUM_HEADS,
        HEAD_DIM,
        maximum_rank,
        dtype=torch.float64,
        device=device,
    )
    D = torch.zeros(
        NUM_HEADS,
        maximum_rank,
        HIDDEN_SIZE,
        dtype=torch.float64,
        device=device,
    )
    for index, rank in enumerate(ranks):
        source_A, source_D = source[rank]
        A[index, :, :rank].copy_(source_A[index].to(device=device, dtype=torch.float64))
        D[index, :rank].copy_(source_D[index].to(device=device, dtype=torch.float64))
    return A, D, ranks


def _ragged_module(
    base_attention: nn.Module,
    *,
    dense_v_weight: Tensor,
    A: Tensor,
    D: Tensor,
    ranks: Sequence[int],
    source_group_size: int = HEADS_PER_SOURCE,
) -> RaggedGQATiedVOLlamaAttention:
    if tuple(dense_v_weight.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError("dense V weight has unexpected shape")
    model_dtype = base_attention.q_proj.weight.dtype
    device = base_attention.q_proj.weight.device
    dense_v = dense_v_weight.to(device=device, dtype=torch.float32)
    v_weights = []
    o_weights = []
    for head, rank in enumerate(map(int, ranks)):
        start = head * HEAD_DIM
        writer = (
            A[head, :, :rank].to(device=device, dtype=torch.float32).transpose(0, 1)
            @ dense_v[start : start + HEAD_DIM]
        )
        decoder = D[head, :rank].to(device=device, dtype=torch.float32).transpose(0, 1)
        v_weights.append(writer.to(dtype=model_dtype).contiguous())
        o_weights.append(decoder.to(dtype=model_dtype).contiguous())
    return RaggedGQATiedVOLlamaAttention(
        base_attention,
        v_group_weights=v_weights,
        o_group_weights=o_weights,
        source_group_size=source_group_size,
    )


def _broadcast_covariance(
    *,
    snapshot_dir: Path,
    validation_snapshot_dir: Path,
    layer: int,
    fit_windows: int,
    validation_windows: int,
    validation_window_start: int,
    row_chunk_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    shape = (NUM_HEADS, NUM_HEADS, HEAD_DIM, HEAD_DIM)
    fit_covariance = torch.empty(shape, dtype=torch.float64, device=device)
    validation_covariance = torch.empty_like(fit_covariance)
    dense_weight = torch.empty(
        HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float16, device="cpu"
    )
    if _rank() == 0:
        fit_manifest = _snapshot_manifest(snapshot_dir)
        validation_manifest = _snapshot_manifest(validation_snapshot_dir)
        fit_activation, weight, _ = _load_snapshot_layer(
            snapshot_dir, fit_manifest, layer
        )
        validation_activation, validation_weight, _ = _load_snapshot_layer(
            validation_snapshot_dir, validation_manifest, layer
        )
        if not torch.equal(weight, validation_weight):
            raise ValueError("fit and validation snapshot weights differ")
        fit_rows = fit_windows * int(fit_manifest["calibration"]["positions_per_window"])
        validation_positions = int(
            validation_manifest["calibration"]["positions_per_window"]
        )
        validation_start = validation_window_start * validation_positions
        validation_rows = validation_windows * validation_positions
        fit_covariance.copy_(
            _activation_covariance_blocks(
                fit_activation[:fit_rows],
                device=device,
                dtype=torch.float64,
                row_chunk_size=row_chunk_size,
            )
        )
        validation_covariance.copy_(
            _activation_covariance_blocks(
                validation_activation[
                    validation_start : validation_start + validation_rows
                ],
                device=device,
                dtype=torch.float64,
                row_chunk_size=row_chunk_size,
            )
        )
        dense_weight.copy_(weight.to(dtype=torch.float16))
        del fit_activation, validation_activation, weight, validation_weight
    dist.broadcast(fit_covariance, src=0)
    dist.broadcast(validation_covariance, src=0)
    dense_weight_device = dense_weight.to(device=device)
    dist.broadcast(dense_weight_device, src=0)
    return fit_covariance, validation_covariance, dense_weight_device.cpu()


def _collect_anchor_vjps(
    model: nn.Module,
    module: nn.Module,
    teacher: Sequence[Any],
    *,
    device: torch.device,
    fisher_probes: int,
    fisher_seed: int,
    vocab_chunk_size: int,
) -> tuple[VJPWindow, ...]:
    if fisher_probes <= 0:
        raise ValueError("Fisher probe count must be positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    result = []
    generator = torch.Generator(device=device)
    generator.manual_seed(int(fisher_seed))
    for window_index, batch in enumerate(teacher):
        captured: list[Tensor] = []

        def cut_hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: tuple[Tensor, ...],
        ) -> tuple[Tensor, ...]:
            leaf = output[0].detach().requires_grad_(True)
            captured.append(leaf)
            return (leaf, *output[1:])

        handle = module.register_forward_hook(cut_hook)
        input_ids = batch.input_ids.to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
        handle.remove()
        if len(captured) != 1:
            raise RuntimeError("attention cut hook did not capture exactly once")
        leaf = captured[0]
        kl = differentiable_teacher_kl_mean(
            logits,
            batch.logits,
            teacher_logsumexp=batch.logsumexp,
            vocab_chunk_size=vocab_chunk_size,
        )
        kl_gradient = torch.autograd.grad(
            kl,
            leaf,
            retain_graph=True,
            create_graph=False,
        )[0]
        probabilities = torch.softmax(logits.detach().float(), dim=-1)
        fisher_gradients = []
        for probe in range(fisher_probes):
            labels = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]),
                1,
                replacement=True,
                generator=generator,
            ).reshape(probabilities.shape[:-1])
            pseudo_nll = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="mean",
            )
            fisher_gradients.append(
                torch.autograd.grad(
                    pseudo_nll,
                    leaf,
                    retain_graph=probe + 1 < fisher_probes,
                    create_graph=False,
                )[0]
                .detach()
                .float()
                .cpu()
                .contiguous()
            )
        result.append(
            VJPWindow(
                anchor_output=leaf.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                kl_gradient=kl_gradient.detach().float().cpu().contiguous(),
                fisher_gradients=tuple(fisher_gradients),
                fisher_sample_count=int(labels.numel()),
            )
        )
        _log(f"anchor reverse-mode window {window_index + 1}/{len(teacher)}")
        del logits, probabilities, leaf, kl, kl_gradient
        torch.cuda.empty_cache()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("suffix Fisher unexpectedly populated a parameter gradient")
    return tuple(result)


@torch.inference_mode()
def _candidate_metrics_and_fisher(
    model: nn.Module,
    module: nn.Module,
    teacher: Sequence[Any],
    vjps: Sequence[VJPWindow],
    *,
    device: torch.device,
    vocab_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(teacher) != len(vjps):
        raise ValueError("teacher and reverse-mode window counts differ")
    kl_values = []
    nll_values = []
    linear_values = []
    curvature_values = []
    taylor_values = []
    displacement_rms = []
    for batch, vjp in zip(teacher, vjps, strict=True):
        captured: list[Tensor] = []

        def capture_hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: tuple[Tensor, ...],
        ) -> None:
            captured.append(output[0].detach())

        handle = module.register_forward_hook(capture_hook)
        input_ids = batch.input_ids.to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
        handle.remove()
        if len(captured) != 1:
            raise RuntimeError("candidate attention hook did not capture exactly once")
        candidate_output = captured[0]
        delta = candidate_output.float() - vjp.anchor_output.to(
            device=device, dtype=torch.float32
        )
        linear = float(
            torch.sum(delta * vjp.kl_gradient.to(device=device)).double().item()
        )
        probe_derivatives = [
            torch.sum(delta * gradient.to(device=device)).double()
            for gradient in vjp.fisher_gradients
        ]
        # Each pseudo loss is averaged over tokens.  Squaring that VJP gives
        # 1/T^2 times the sum of independent token score variances, whereas
        # the Hessian of the mean loss is 1/T times their sum.
        curvature = _fisher_curvature(
            probe_derivatives,
            sample_count=vjp.fisher_sample_count,
        )
        kl_sum, tokens = teacher_kl_sum(
            logits,
            batch.logits,
            teacher_logsumexp=batch.logsumexp,
            vocab_chunk_size=vocab_chunk_size,
        )
        nll = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
            reduction="mean",
        )
        kl_values.append(kl_sum / tokens)
        nll_values.append(float(nll.item()))
        linear_values.append(linear)
        curvature_values.append(curvature)
        taylor_values.append(linear + 0.5 * curvature)
        displacement_rms.append(float(delta.double().square().mean().sqrt().item()))
        del logits, candidate_output, delta
    metrics = {
        "terminal_kl": _paired(kl_values),
        "nll": _paired(nll_values),
    }
    fisher = {
        "linear": _paired(linear_values),
        "curvature": _paired(curvature_values),
        "taylor": _paired(taylor_values),
        "attention_output_displacement_rms": _paired(displacement_rms),
    }
    return metrics, fisher


def _top_sources(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_rank: int,
    key: str,
    limit: int = 4,
) -> list[int]:
    selected = [record for record in records if int(record["candidate_rank"]) == candidate_rank]
    return [
        int(record["source"])
        for record in sorted(
            selected,
            key=lambda item: (float(item[key]), int(item["source"])),
        )[:limit]
    ]


def _summary(result: Mapping[str, Any]) -> str:
    anchor = result["anchor"]
    lines = [
        "# Llama-2-7B C1 per-TP-source Fisher versus terminal-KL pilot",
        "",
        "## Protocol",
        "",
        (
            f"- Layer {result['configuration']['layer']}, uniform r"
            f"{result['configuration']['anchor_rank']} anchor."
        ),
        (
            "- One TP source (four contiguous heads) changes at a time to ranks "
            f"{result['configuration']['candidate_ranks']}; every candidate receives "
            "a full-layer closed-form ragged decoder refit."
        ),
        (
            f"- {result['configuration']['profile_windows']} C4 windows of length "
            f"{result['configuration']['sequence_length']}; "
            f"{result['configuration']['fisher_probes']} reverse-mode Fisher probes/window."
        ),
        "- Exact KL uses the dense teacher; Fisher is evaluated around the ragged uniform anchor.",
        "",
        "## Anchor controls",
        "",
        "| Anchor | Terminal KL | NLL |",
        "|:---|---:|---:|",
        (
            f"| Folded uniform | {anchor['folded']['terminal_kl']['mean']:.9g} | "
            f"{anchor['folded']['nll']['mean']:.9g} |"
        ),
        (
            f"| Ragged uniform | {anchor['ragged']['terminal_kl']['mean']:.9g} | "
            f"{anchor['ragged']['nll']['mean']:.9g} |"
        ),
        (
            f"| Ragged minus folded | "
            f"{anchor['ragged_minus_folded']['terminal_kl']['mean']:.9g} | "
            f"{anchor['ragged_minus_folded']['nll']['mean']:.9g} |"
        ),
        "",
        "## Rank correlation with exact candidate KL delta",
        "",
        "| Candidate subset | Predictor | Pearson | Spearman |",
        "|:---|:---|---:|---:|",
    ]
    for subset, predictors in result["analysis"]["correlations"].items():
        for predictor, metric in predictors.items():
            lines.append(
                f"| {subset} | {predictor} | {metric['pearson']:.6f} | "
                f"{metric['spearman']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Top TP sources by exact KL and Fisher-Taylor",
            "",
            "| Rank | Exact KL top-4 | Fisher-Taylor top-4 | Overlap |",
            "|---:|:---|:---|---:|",
        ]
    )
    for rank, row in result["analysis"]["top_sources"].items():
        lines.append(
            f"| {rank} | {row['exact']} | {row['fisher_taylor']} | "
            f"{row['overlap']} |"
        )
    lines.extend(["", "## Command", "", f"`{result['command']}`", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch this pilot with torchrun")
    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    candidate_ranks = _parse_ranks(args.candidate_ranks)
    if args.anchor_rank in candidate_ranks:
        raise ValueError("candidate ranks must exclude the anchor")
    required_ranks = tuple(sorted({args.anchor_rank, *candidate_ranks}))
    if min(required_ranks) <= 0 or max(required_ranks) > HEAD_DIM:
        raise ValueError("rank grid lies outside the Value-head width")
    positive = (
        args.profile_windows,
        args.sequence_length,
        args.fit_windows,
        args.validation_windows,
        args.fisher_probes,
        args.covariance_row_chunk_size,
        args.vocab_chunk_size,
        args.torch_num_threads,
    )
    if min(positive) <= 0 or not 0 <= args.layer < 32:
        raise ValueError("pilot configuration is invalid")

    output_dir = args.output_dir.expanduser().resolve()
    exists = torch.tensor(int(output_dir.exists()), dtype=torch.int32, device=device)
    _all_reduce_sum(exists)
    if int(exists.item()):
        raise FileExistsError(output_dir)

    model_path = Path(args.model).expanduser().resolve()
    factor_dirs = _parse_factor_dirs(args.factor_dir)
    if set(factor_dirs) != set(required_ranks):
        raise ValueError("factor directories must exactly cover anchor and candidates")
    factor_results = _load_factor_results(
        factor_dirs,
        model_config_sha256=_sha256(model_path / "config.json"),
        layer_count=32,
    )
    profile_sequences, _unused, windows_provenance = _select_windows(
        args.windows,
        profile_windows=args.profile_windows,
        confirmation_windows=0,
    )
    if args.sequence_length > profile_sequences.shape[1]:
        raise ValueError("requested sequence length exceeds the stored windows")
    profile_sequences = profile_sequences[:, : args.sequence_length].contiguous()

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.config.use_cache = False
    if (
        model.config.model_type != "llama"
        or int(model.config.num_attention_heads) != NUM_HEADS
        or int(model.config.num_key_value_heads) != NUM_HEADS
        or int(model.config.hidden_size) != HIDDEN_SIZE
    ):
        raise ValueError("pilot requires Llama-2-7B MHA geometry")

    teacher = _capture_teacher(
        model,
        profile_sequences,
        batch_size=1,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
        label="per-TP-source pilot",
    )
    banks = _install_banks(
        model,
        factor_dirs=factor_dirs,
        factor_results=factor_results,
        anchor_rank=args.anchor_rank,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    folded_anchor = _evaluate_teacher_metrics(
        model,
        teacher,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )

    source_factors = _load_layer_factors(
        factor_dirs,
        factor_results,
        layer=args.layer,
        ranks=required_ranks,
    )
    layer_module = _model_layers(model)[args.layer].self_attn
    anchor_A, anchor_D = source_factors[args.anchor_rank]
    anchor_ranks = (args.anchor_rank,) * NUM_HEADS
    anchor_collective_cost = _collective_cost(anchor_ranks)
    ragged_anchor = _ragged_module(
        layer_module,
        dense_v_weight=banks[args.layer].dense_v,
        A=anchor_A,
        D=anchor_D,
        ranks=anchor_ranks,
    )
    _model_layers(model)[args.layer].self_attn = ragged_anchor
    ragged_anchor_metrics = _evaluate_teacher_metrics(
        model,
        teacher,
        device=device,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    vjps = _collect_anchor_vjps(
        model,
        ragged_anchor,
        teacher,
        device=device,
        fisher_probes=args.fisher_probes,
        fisher_seed=args.fisher_seed,
        vocab_chunk_size=args.vocab_chunk_size,
    )

    fit_covariance, validation_covariance, snapshot_weight = _broadcast_covariance(
        snapshot_dir=args.snapshot_dir.expanduser().resolve(),
        validation_snapshot_dir=args.validation_snapshot_dir.expanduser().resolve(),
        layer=args.layer,
        fit_windows=args.fit_windows,
        validation_windows=args.validation_windows,
        validation_window_start=args.validation_window_start,
        row_chunk_size=args.covariance_row_chunk_size,
        device=device,
    )
    if not torch.equal(snapshot_weight, banks[args.layer].dense_o.to(dtype=torch.float16)):
        raise ValueError("snapshot O weight differs from the model")
    fit_covariance, absolute_damping = covariance_with_trace_damping(
        fit_covariance,
        relative_damping=args.covariance_damping,
    )
    target = (
        banks[args.layer]
        .dense_o.to(device=device, dtype=torch.float64)
        .transpose(0, 1)
        .reshape(NUM_HEADS, HEAD_DIM, HIDDEN_SIZE)
        .contiguous()
    )
    fit_objective = quadratic_from_target(
        covariance=fit_covariance,
        target=target,
        name=f"llama2_c1_per_head_pilot_fit_layer_{args.layer:03d}",
        trace_normalize=False,
    )
    validation_objective = quadratic_from_target(
        covariance=validation_covariance,
        target=target,
        name=f"llama2_c1_per_head_pilot_validation_layer_{args.layer:03d}",
        trace_normalize=False,
    )
    mapping = torch.arange(NUM_HEADS, dtype=torch.long, device=device)
    maximum_rank = max(required_ranks)
    interventions = [
        (source, rank)
        for source in range(TP_SIZE)
        for rank in candidate_ranks
    ]
    local_records = []
    for local_index, (source, candidate_rank) in enumerate(
        interventions[_rank() :: _world_size()]
    ):
        candidate_heads = _source_heads(source)
        initial_A, initial_D, ranks = _padded_candidate_factors(
            source_factors,
            candidate_heads=candidate_heads,
            candidate_rank=candidate_rank,
            anchor_rank=args.anchor_rank,
            maximum_rank=maximum_rank,
            device=device,
        )
        closure = close_ragged_decoder_with_fixed_encoders(
            objective=fit_objective,
            initial_A=initial_A,
            initial_D=initial_D,
            head_to_kv_group=mapping,
            group_ranks=ranks,
            relative_jitter=args.decoder_relative_jitter,
        )
        fit_loss = evaluate_quadratic(
            fit_objective, closure.A_unique, closure.D_heads, mapping
        )
        validation_loss = evaluate_quadratic(
            validation_objective, closure.A_unique, closure.D_heads, mapping
        )
        candidate_collective_cost = _collective_cost(ranks)
        candidate_module = _ragged_module(
            ragged_anchor,
            dense_v_weight=banks[args.layer].dense_v,
            A=closure.A_unique,
            D=closure.D_heads,
            ranks=ranks,
        )
        _model_layers(model)[args.layer].self_attn = candidate_module
        metrics, fisher = _candidate_metrics_and_fisher(
            model,
            candidate_module,
            teacher,
            vjps,
            device=device,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        _model_layers(model)[args.layer].self_attn = ragged_anchor
        delta = _paired_delta(metrics, ragged_anchor_metrics)
        record = {
            "source": source,
            "heads": list(candidate_heads),
            "candidate_rank": candidate_rank,
            "rank_delta": candidate_rank - args.anchor_rank,
            "rank_vector": list(ranks),
            "collective_cost": {
                **candidate_collective_cost,
                "ideal_width_delta": (
                    candidate_collective_cost["ideal_allgather_width"]
                    - anchor_collective_cost["ideal_allgather_width"]
                ),
                "padded_width_delta": (
                    candidate_collective_cost["padded_allgather_width"]
                    - anchor_collective_cost["padded_allgather_width"]
                ),
            },
            "fit_relative_mse": float(fit_loss / fit_objective.constant),
            "validation_relative_mse": float(
                validation_loss / validation_objective.constant
            ),
            "terminal_kl": metrics["terminal_kl"],
            "nll": metrics["nll"],
            "terminal_kl_delta": delta["terminal_kl"],
            "nll_delta": delta["nll"],
            "fisher": fisher,
            "terminal_kl_delta_mean": delta["terminal_kl"]["mean"],
            "nll_delta_mean": delta["nll"]["mean"],
            "fisher_linear_mean": fisher["linear"]["mean"],
            "fisher_curvature_mean": fisher["curvature"]["mean"],
            "fisher_taylor_mean": fisher["taylor"]["mean"],
            "encoder_sha256": closure.encoder_sha256_after_solve,
            "decoder_sha256": tensor_sha256(
                closure.D_heads.to(device="cpu", dtype=torch.float16)
            ),
            "decoder_solve": {
                "absolute_jitter": closure.decoder.absolute_jitters[0],
                "condition_estimate": closure.decoder.condition_estimates[0],
                "matrix_dimension": closure.decoder.matrix_dimensions[0],
                "relative_residual": closure.decoder.relative_residuals[0],
                "wall_time_seconds": closure.decoder.wall_times_seconds[0],
            },
        }
        local_records.append(record)
        _log(
            f"candidate {local_index + 1}/{len(interventions[_rank() :: _world_size()])} "
            f"source={source} heads={candidate_heads} rank={candidate_rank} "
            f"dKL={record['terminal_kl_delta_mean']:.6g} "
            f"F={record['fisher_taylor_mean']:.6g}",
            all_ranks=True,
        )
        del initial_A, initial_D, closure, candidate_module
        torch.cuda.empty_cache()

    gathered: list[Any] | None = [None] * _world_size() if _rank() == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    if _rank() == 0:
        assert gathered is not None
        records = sorted(
            [record for shard in gathered for record in shard],
            key=lambda record: (record["candidate_rank"], record["source"]),
        )
        if len(records) != len(interventions):
            raise RuntimeError("distributed pilot lost candidate records")
        subsets = {"all": records}
        for candidate_rank in candidate_ranks:
            subsets[f"rank_{candidate_rank}"] = [
                record
                for record in records
                if int(record["candidate_rank"]) == candidate_rank
            ]
        predictor_keys = {
            "Fisher linear": "fisher_linear_mean",
            "Fisher curvature": "fisher_curvature_mean",
            "Fisher Taylor": "fisher_taylor_mean",
            "held-out output MSE": "validation_relative_mse",
        }
        correlations = {
            subset: {
                label: _correlations(
                    rows,
                    actual_key="terminal_kl_delta_mean",
                    predictor_key=key,
                )
                for label, key in predictor_keys.items()
            }
            for subset, rows in subsets.items()
        }
        top_sources = {}
        for candidate_rank in candidate_ranks:
            exact = _top_sources(
                records,
                candidate_rank=candidate_rank,
                key="terminal_kl_delta_mean",
            )
            fisher = _top_sources(
                records,
                candidate_rank=candidate_rank,
                key="fisher_taylor_mean",
            )
            top_sources[str(candidate_rank)] = {
                "exact": exact,
                "fisher_taylor": fisher,
                "overlap": len(set(exact) & set(fisher)),
            }
        result = {
            "format": FORMAT,
            "command": shlex.join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "world_size": _world_size(),
            "model": str(model_path),
            "model_config_sha256": _sha256(model_path / "config.json"),
            "configuration": {
                "layer": args.layer,
                "tp_size": TP_SIZE,
                "heads_per_source": HEADS_PER_SOURCE,
                "anchor_rank": args.anchor_rank,
                "candidate_ranks": list(candidate_ranks),
                "profile_windows": args.profile_windows,
                "sequence_length": args.sequence_length,
                "fit_windows": args.fit_windows,
                "validation_windows": args.validation_windows,
                "validation_window_start": args.validation_window_start,
                "fisher_probes": args.fisher_probes,
                "fisher_seed": args.fisher_seed,
                "covariance_damping": args.covariance_damping,
                "absolute_covariance_damping": absolute_damping,
                "decoder_relative_jitter": args.decoder_relative_jitter,
                "candidate_semantics": (
                    "one changed four-head TP source plus full-layer closed-form "
                    "ragged decoder refit"
                ),
                "fisher_semantics": (
                    "exact teacher-KL VJP linear term plus pseudo-label reverse-mode "
                    "softmax-Fisher quadratic term around the ragged anchor"
                ),
                "ragged_runtime": (
                    "model-selected attention backend plus concatenated source latents "
                    "and one joint output GEMM"
                ),
            },
            "windows": {
                **windows_provenance,
                "used_sequence_length": args.sequence_length,
            },
            "factors": {
                str(rank): {
                    "path": str(factor_dirs[rank]),
                    "results_sha256": _sha256(factor_dirs[rank] / "results.json"),
                }
                for rank in required_ranks
            },
            "anchor": {
                "folded": folded_anchor,
                "ragged": ragged_anchor_metrics,
                "collective_cost": anchor_collective_cost,
                "ragged_minus_folded": _paired_delta(
                    ragged_anchor_metrics, folded_anchor
                ),
            },
            "records": records,
            "analysis": {
                "actual_target": "paired exact terminal-KL delta versus ragged uniform anchor",
                "correlations": correlations,
                "top_sources": top_sources,
            },
            "numerics": {
                "model_dtype": "float16",
                "decoder_closure_dtype": "float64",
                "factor_install_dtype": "float16",
                "KL_probability_dtype": "float32",
                "KL_accumulation_dtype": "float64",
            },
            "environment": {
                "python_executable": sys.executable,
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
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
        (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
        _log(f"wrote {output_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
