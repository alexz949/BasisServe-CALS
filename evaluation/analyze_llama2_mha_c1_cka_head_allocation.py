#!/usr/bin/env python3
"""Screen activation-CKA head packing for TP-aware Llama-2 C1 ranks.

Eight contiguous four-head TP sources are compared with equal-capacity CKA
groups learned on calibration attention outputs and checked on a disjoint
snapshot. Existing C1 encoders provide a rank-response diagnostic. The final
fixed-budget score is a compression-CKA proxy, not terminal KL or PPL.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.analysis.neuron_repartition import (  # noqa: E402
    contiguous_balanced_partition,
    validate_balanced_partition,
)
from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)

FORMAT = "basisserve.llama2_7b.mha_c1.cka_tp_head_allocation_audit.v1"
LAYERS, HEADS, HEAD_DIM, HIDDEN = 32, 32, 128, 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    run = sub.add_parser("analyze-shard")
    run.add_argument("--train-snapshot-dir", type=Path, required=True)
    run.add_argument("--validation-snapshot-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--factor-dir", action="append", required=True, metavar="RANK=PATH")
    run.add_argument("--anchor-rank", type=int, default=96)
    run.add_argument("--train-row-start", type=int, default=0)
    run.add_argument("--validation-row-start", type=int, default=49152)
    run.add_argument("--rows", type=int, default=8192)
    run.add_argument("--tp-size", type=int, default=8)
    run.add_argument("--random-partitions", type=int, default=128)
    run.add_argument("--local-search-rounds", type=int, default=256)
    run.add_argument("--seed", type=int, default=20260819)
    run.add_argument("--layer-shard-index", type=int, default=0)
    run.add_argument("--layer-shard-count", type=int, default=1)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--torch-num-threads", type=int, default=2)
    merge = sub.add_parser("merge")
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--layer-shard-count", type=int, required=True)
    return parser.parse_args()


def parse_factor_dirs(values: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        rank_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"factor directory must be RANK=PATH: {value}")
        rank, path = int(rank_text), Path(path_text).expanduser().resolve()
        if rank <= 0 or rank > HEAD_DIM or rank in result or not path.is_dir():
            raise ValueError(f"invalid factor directory: {value}")
        result[rank] = path
    return dict(sorted(result.items()))


def snapshot_manifest(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "manifest.json").read_text())
    model = payload.get("model", {})
    geometry = (
        int(model.get("num_hidden_layers", -1)),
        int(model.get("num_attention_heads", -1)),
        int(model.get("head_dim", -1)),
        int(model.get("hidden_size", -1)),
    )
    if geometry != (LAYERS, HEADS, HEAD_DIM, HIDDEN):
        raise ValueError(f"unexpected snapshot geometry: {geometry}")
    return payload


def activation_slice(
    directory: Path,
    manifest: Mapping[str, Any],
    layer: int,
    row_start: int,
    rows: int,
) -> Tensor:
    record = manifest["artifacts"][str(layer)]
    if row_start < 0 or rows < 2 or row_start + rows > int(record["activation_shape"][0]):
        raise ValueError(f"row slice is outside snapshot layer {layer}")
    with safe_open(str(directory / record["file"]), framework="pt", device="cpu") as handle:
        value = handle.get_slice("activation")[row_start : row_start + rows]
    return value.reshape(rows, HEADS, HEAD_DIM).contiguous()


@torch.no_grad()
def centered_head_covariance(
    activation: Tensor,
    *,
    device: torch.device,
    chunk_size: int = 1024,
) -> Tensor:
    """Return centered cross-head covariance blocks with shape [H,H,D,D]."""
    if activation.ndim != 3 or tuple(activation.shape[1:]) != (HEADS, HEAD_DIM):
        raise ValueError("activation must have shape [rows, 32, 128]")
    rows = int(activation.shape[0])
    mean = torch.zeros(HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    for start in range(0, rows, chunk_size):
        mean.add_(activation[start : start + chunk_size].to(device, torch.float32).sum(0))
    mean.div_(rows)
    covariance = torch.zeros(
        HEADS, HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32
    )
    for start in range(0, rows, chunk_size):
        work = activation[start : start + chunk_size].to(device, torch.float32)
        work.sub_(mean)
        covariance.add_(torch.einsum("nhi,nkj->hkij", work, work))
    return covariance / rows


def cka_matrix_from_covariance(covariance: Tensor) -> Tensor:
    if covariance.shape != (HEADS, HEADS, HEAD_DIM, HEAD_DIM):
        raise ValueError("unexpected covariance shape")
    energy = covariance.square().sum((-1, -2))
    diagonal = torch.diagonal(energy).clamp_min(torch.finfo(energy.dtype).tiny)
    result = (energy / torch.sqrt(diagonal[:, None] * diagonal[None, :])).clamp(0, 1)
    result.fill_diagonal_(1)
    return result


def compression_cka_from_covariance(covariance: Tensor, encoder: Tensor) -> float:
    """Compute centered linear CKA(U, U A) from covariance U.T@U."""
    if covariance.shape != (HEAD_DIM, HEAD_DIM) or encoder.shape[0] != HEAD_DIM:
        raise ValueError("invalid covariance or encoder shape")
    covariance, encoder = covariance.double(), encoder.double()
    cross = covariance @ encoder
    compressed = encoder.T @ cross
    denominator = torch.sqrt(covariance.square().sum() * compressed.square().sum())
    value = cross.square().sum() / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    return float(value.clamp(0, 1))


def partition_score(similarity: Tensor, groups: Sequence[Tensor]) -> float:
    groups = validate_balanced_partition(groups, width=HEADS, tp_size=len(groups))
    values = [
        float(similarity[first, second])
        for group in groups
        for offset, first in enumerate(group.tolist())
        for second in group.tolist()[offset + 1 :]
    ]
    return sum(values) / len(values)


def refine_partition(
    similarity: Tensor,
    groups: Sequence[Tensor],
    rounds: int,
) -> tuple[tuple[Tensor, ...], int]:
    current = [group.tolist() for group in groups]
    swaps = 0
    for _ in range(rounds):
        best_gain, best = 1e-12, None
        for left_group in range(len(current)):
            for right_group in range(left_group + 1, len(current)):
                left, right = current[left_group], current[right_group]
                for left_pos, left_head in enumerate(left):
                    for right_pos, right_head in enumerate(right):
                        before = sum(float(similarity[left_head, peer]) for peer in left if peer != left_head)
                        before += sum(float(similarity[right_head, peer]) for peer in right if peer != right_head)
                        after = sum(float(similarity[right_head, peer]) for peer in left if peer != left_head)
                        after += sum(float(similarity[left_head, peer]) for peer in right if peer != right_head)
                        if after - before > best_gain:
                            best_gain = after - before
                            best = left_group, left_pos, right_group, right_pos
        if best is None:
            break
        lg, lp, rg, rp = best
        current[lg][lp], current[rg][rp] = current[rg][rp], current[lg][lp]
        swaps += 1
    groups = tuple(torch.tensor(sorted(group), dtype=torch.int64) for group in current)
    groups = tuple(sorted(groups, key=lambda group: int(group[0])))
    return validate_balanced_partition(groups, width=HEADS, tp_size=len(groups)), swaps


def optimize_balanced_cka_partition(
    similarity: Tensor,
    *,
    tp_size: int,
    random_partitions: int,
    local_search_rounds: int,
    seed: int,
) -> tuple[tuple[Tensor, ...], dict[str, Any]]:
    if similarity.shape != (HEADS, HEADS) or HEADS % tp_size:
        raise ValueError("invalid similarity or TP geometry")
    contiguous = contiguous_balanced_partition(HEADS, tp_size)
    best, best_swaps = refine_partition(similarity, contiguous, local_search_rounds)
    best_score = partition_score(similarity, best)
    generator, capacity, random_scores = torch.Generator().manual_seed(seed), HEADS // tp_size, []
    for _ in range(random_partitions):
        order = torch.randperm(HEADS, generator=generator)
        groups = tuple(
            order[source * capacity : (source + 1) * capacity].sort().values
            for source in range(tp_size)
        )
        random_scores.append(partition_score(similarity, groups))
        candidate, swaps = refine_partition(similarity, groups, local_search_rounds)
        score = partition_score(similarity, candidate)
        if score > best_score:
            best, best_score, best_swaps = candidate, score, swaps
    random_tensor = torch.tensor(random_scores, dtype=torch.float64)
    return best, {
        "method": "random_restart_balanced_pair_swap",
        "train_within_source_cka": best_score,
        "random_mean": float(random_tensor.mean()),
        "random_std": float(random_tensor.std(unbiased=False)),
        "selected_swaps": best_swaps,
        "seed": seed,
    }


def factor_encoders(directory: Path, layer: int, rank: int) -> Tensor:
    path = directory / f"layer_{layer:03d}.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        encoders = handle.get_tensor("value_coordinate_encoders")
    if tuple(encoders.shape) != (HEADS, HEAD_DIM, rank):
        raise ValueError(f"unexpected encoders in {path}: {tuple(encoders.shape)}")
    return encoders


def rank_response(covariance: Tensor, directories: Mapping[int, Path], layer: int) -> dict[int, list[float]]:
    result = {}
    for rank, directory in directories.items():
        encoders = factor_encoders(directory, layer, rank)
        result[rank] = [
            compression_cka_from_covariance(covariance[head, head].cpu(), encoders[head])
            for head in range(HEADS)
        ]
    return result


def dispersion(response: Mapping[int, Sequence[float]], groups: Sequence[Tensor]) -> float:
    values = []
    for head_values in response.values():
        tensor = torch.tensor(head_values, dtype=torch.float64)
        values.extend(float(tensor[group].std(unbiased=False)) for group in groups)
    return sum(values) / len(values)


def minimum_moved(contiguous: Sequence[Tensor], candidate: Sequence[Tensor]) -> int:
    left, right = [set(group.tolist()) for group in contiguous], [set(group.tolist()) for group in candidate]
    best = max(
        sum(len(left[index] & right[permutation[index]]) for index in range(len(left)))
        for permutation in itertools.permutations(range(len(right)))
    )
    return HEADS - best


def analyze_layer(args: argparse.Namespace, layer: int, directories: Mapping[int, Path], device: torch.device) -> dict[str, Any]:
    train_dir, validation_dir = args.train_snapshot_dir.resolve(), args.validation_snapshot_dir.resolve()
    train = activation_slice(train_dir, snapshot_manifest(train_dir), layer, args.train_row_start, args.rows)
    train_cov = centered_head_covariance(train, device=device)
    train_similarity = cka_matrix_from_covariance(train_cov).cpu()
    del train, train_cov
    groups, search = optimize_balanced_cka_partition(
        train_similarity,
        tp_size=args.tp_size,
        random_partitions=args.random_partitions,
        local_search_rounds=args.local_search_rounds,
        seed=args.seed + layer,
    )
    validation = activation_slice(
        validation_dir, snapshot_manifest(validation_dir), layer, args.validation_row_start, args.rows
    )
    validation_cov = centered_head_covariance(validation, device=device)
    similarity = cka_matrix_from_covariance(validation_cov).cpu()
    response = rank_response(validation_cov, directories, layer)
    del validation, validation_cov
    contiguous = contiguous_balanced_partition(HEADS, args.tp_size)
    contiguous_cka, allocated_cka = partition_score(similarity, contiguous), partition_score(similarity, groups)
    contiguous_dispersion, allocated_dispersion = dispersion(response, contiguous), dispersion(response, groups)
    return {
        "layer": layer,
        "contiguous_groups": [group.tolist() for group in contiguous],
        "cka_groups": [group.tolist() for group in groups],
        "minimum_heads_moved": minimum_moved(contiguous, groups),
        "search": search,
        "heldout": {
            "contiguous_cka": contiguous_cka,
            "allocated_cka": allocated_cka,
            "relative_cka_gain": allocated_cka / contiguous_cka - 1,
            "contiguous_rank_response_dispersion": contiguous_dispersion,
            "allocated_rank_response_dispersion": allocated_dispersion,
            "dispersion_ratio": allocated_dispersion / contiguous_dispersion,
            "compression_cka_by_rank_and_head": {str(rank): values for rank, values in response.items()},
        },
    }


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def analyze_shard(args: argparse.Namespace) -> None:
    if args.tp_size <= 1 or HEADS % args.tp_size or not 0 <= args.layer_shard_index < args.layer_shard_count:
        raise ValueError("invalid TP or layer-shard geometry")
    torch.set_num_threads(args.torch_num_threads)
    torch.set_float32_matmul_precision("high")
    directories, device = parse_factor_dirs(args.factor_dir), torch.device(args.device)
    if args.anchor_rank not in directories:
        raise ValueError("anchor rank is absent from factor directories")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(args.layer_shard_index, LAYERS, args.layer_shard_count))
    records = []
    for offset, layer in enumerate(layers, 1):
        records.append(analyze_layer(args, layer, directories, device))
        print(f"[CKA audit] shard={args.layer_shard_index} layer={layer} {offset}/{len(layers)}", flush=True)
    configuration = {
        "train_snapshot_dir": str(args.train_snapshot_dir.resolve()),
        "validation_snapshot_dir": str(args.validation_snapshot_dir.resolve()),
        "train_row_start": args.train_row_start,
        "validation_row_start": args.validation_row_start,
        "rows": args.rows,
        "factor_dirs": {str(rank): str(path) for rank, path in directories.items()},
        "anchor_rank": args.anchor_rank,
        "tp_size": args.tp_size,
        "random_partitions": args.random_partitions,
        "local_search_rounds": args.local_search_rounds,
        "seed": args.seed,
    }
    atomic_json(
        args.output_dir / f"shard_{args.layer_shard_index:02d}.json",
        {"format": FORMAT, "status": "complete", "configuration": configuration, "records": records, "command": shlex.join(sys.argv)},
    )


def proxy_allocate(
    records: Sequence[Mapping[str, Any]],
    groups_key: str,
    ranks: Sequence[int],
    anchor_rank: int,
) -> dict[str, Any]:
    options, coordinates = [], []
    for record in records:
        response = {
            int(rank): torch.tensor(values, dtype=torch.float64)
            for rank, values in record["heldout"]["compression_cka_by_rank_and_head"].items()
        }
        groups = [torch.tensor(group) for group in record[groups_key]]
        for source, group in enumerate(groups):
            options.append(
                tuple(
                    MetricRankOption(
                        option_id=f"L{record['layer']}.S{source}.R{rank}.{groups_key}",
                        source_family="compression_cka_proxy",
                        rank=rank,
                        scalar_cost=float((1 - response[rank][group]).sum()),
                        is_anchor=rank == anchor_rank,
                    )
                    for rank in ranks
                )
            )
            coordinates.append((int(record["layer"]), source))
    tp_size = len(records[0][groups_key])
    allocation = allocate_metric_rank_exact(
        options, total_rank_budget=LAYERS * tp_size * anchor_rank, anchor_rank=anchor_rank
    )
    schedule = [[anchor_rank] * tp_size for _ in range(LAYERS)]
    for (layer, source), option in zip(coordinates, allocation.selected_options, strict=True):
        schedule[layer][source] = int(option.rank)
    return {
        "proxy_cost": float(allocation.total_cost),
        "rank_histogram": {str(rank): sum(value == rank for row in schedule for value in row) for rank in ranks},
        "schedule": schedule,
    }


def summary(result: Mapping[str, Any]) -> str:
    aggregate, proxy = result["aggregate"], result["rank_allocation_proxy"]
    return "\n".join(
        [
            "# Llama-2-7B C1 CKA TP-head allocation audit",
            "",
            "## Screening outcome",
            "",
            f"- Held-out within-source CKA: contiguous `{aggregate['contiguous_cka']:.6f}`, CKA `{aggregate['allocated_cka']:.6f}`.",
            f"- Relative held-out CKA gain: `{100 * aggregate['relative_cka_gain']:.3f}%`.",
            f"- Rank-response dispersion ratio: `{aggregate['dispersion_ratio']:.6f}` (below one is favorable).",
            f"- Fixed-budget proxy: contiguous `{proxy['contiguous']['proxy_cost']:.9f}`, CKA `{proxy['cka']['proxy_cost']:.9f}`.",
            f"- Proxy cost change: `{100 * proxy['relative_cost_change']:.3f}%` (negative is favorable).",
            "",
            "This is a held-out activation-CKA screen, not terminal KL or PPL.",
            "",
        ]
    )


def merge(args: argparse.Namespace) -> None:
    shards = [json.loads((args.output_dir / f"shard_{index:02d}.json").read_text()) for index in range(args.layer_shard_count)]
    if any(shard.get("format") != FORMAT or shard.get("status") != "complete" for shard in shards):
        raise ValueError("incomplete or incompatible shard")
    configuration = shards[0]["configuration"]
    if any(shard["configuration"] != configuration for shard in shards[1:]):
        raise ValueError("shard configurations differ")
    records = sorted([record for shard in shards for record in shard["records"]], key=lambda row: row["layer"])
    if [record["layer"] for record in records] != list(range(LAYERS)):
        raise ValueError("shards do not cover all layers")

    def average(key: str) -> float:
        return sum(float(record["heldout"][key]) for record in records) / LAYERS

    ranks, anchor = tuple(sorted(map(int, configuration["factor_dirs"]))), int(configuration["anchor_rank"])
    contiguous = proxy_allocate(records, "contiguous_groups", ranks, anchor)
    cka = proxy_allocate(records, "cka_groups", ranks, anchor)
    result = {
        "format": FORMAT,
        "status": "complete",
        "configuration": configuration,
        "records": records,
        "aggregate": {
            "contiguous_cka": average("contiguous_cka"),
            "allocated_cka": average("allocated_cka"),
            "relative_cka_gain": average("relative_cka_gain"),
            "dispersion_ratio": average("dispersion_ratio"),
            "minimum_heads_moved_mean": sum(row["minimum_heads_moved"] for row in records) / LAYERS,
        },
        "rank_allocation_proxy": {
            "contiguous": contiguous,
            "cka": cka,
            "relative_cost_change": cka["proxy_cost"] / contiguous["proxy_cost"] - 1,
        },
        "commands": [shard["command"] for shard in shards] + [shlex.join(sys.argv)],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_dir / "result.json", result)
    text = summary(result)
    (args.output_dir / "summary.md").write_text(text)
    print(text, flush=True)


def main() -> None:
    args = parse_args()
    analyze_shard(args) if args.operation == "analyze-shard" else merge(args)


if __name__ == "__main__":
    main()
