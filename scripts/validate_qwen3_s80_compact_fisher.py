#!/usr/bin/env python3
"""Compare raw and compact Store80 softmax-Fisher statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (  # noqa: E402
    S80DirectResidualData,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    compact_softmax_fisher_adapter_system,
    compact_softmax_fisher_encoder_diagonal,
    compact_softmax_fisher_loss,
    compact_softmax_fisher_roots,
    prepare_compact_softmax_fisher_routing,
    prepare_softmax_fisher_routing,
    softmax_fisher_adapter_system,
    softmax_fisher_encoder_diagonal,
    softmax_fisher_routing_loss,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-capture-dir", required=True)
    parser.add_argument("--compact-fisher-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--documents", type=int, default=1)
    parser.add_argument("--route-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--work-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mmap_prefix(
    root: Path,
    record: dict,
    *,
    documents: int,
) -> torch.Tensor:
    shape = tuple(int(size) for size in record["shape"])
    values = 1
    for size in shape:
        values *= size
    tensor = torch.from_file(
        str(root / record["file"]),
        shared=False,
        size=values,
        dtype=torch.bfloat16,
    ).reshape(shape)
    return tensor[:documents]


def _relative_error(observed: torch.Tensor, expected: torch.Tensor) -> float:
    tiny = torch.finfo(observed.dtype).tiny
    return float(
        torch.linalg.vector_norm(observed - expected)
        / torch.linalg.vector_norm(expected).clamp_min(tiny)
    )


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    direct_root = Path(args.direct_capture_dir).expanduser().resolve()
    compact_root = Path(args.compact_fisher_dir).expanduser().resolve()
    direct_manifest = _load_json(direct_root / "manifest.json")
    compact_manifest = _load_json(compact_root / "manifest.json")
    direct_records = direct_manifest["artifacts"][str(args.layer)]
    documents = min(
        args.documents,
        int(direct_records["routing_queries"]["shape"][0]),
    )
    direct = S80DirectResidualData(
        routing_queries=_mmap_prefix(
            direct_root,
            direct_records["routing_queries"],
            documents=documents,
        ),
        routing_joint_rows=_mmap_prefix(
            direct_root,
            direct_records["routing_joint_rows"],
            documents=documents,
        ),
    )
    compact_artifact = compact_manifest["artifacts"][str(args.layer)]
    compact_tensors = load_file(
        str(compact_root / compact_artifact["file"]),
        device="cpu",
    )
    device = torch.device(args.device)
    dtype = torch.float64 if args.work_dtype == "float64" else torch.float32
    value_dim = int(compact_tensors["value_dim"])
    key_dim = int(compact_tensors["key_dim"])
    mapping = compact_tensors["head_to_kv_group"]
    raw = prepare_softmax_fisher_routing(
        direct,
        head_to_kv_group=mapping,
        value_dim=value_dim,
        key_dim=key_dim,
        device=device,
        dtype=dtype,
    )
    compact = prepare_compact_softmax_fisher_routing(
        queries_by_head=compact_tensors["queries_by_head"][:, :documents],
        fisher_grams_packed_by_head=compact_tensors[
            "fisher_grams_packed_by_head"
        ][:, :documents],
        head_to_kv_group=mapping,
        value_dim=value_dim,
        key_dim=key_dim,
        scaling=float(compact_tensors["scaling"]),
        teacher_fisher_energy=float(compact_tensors["teacher_fisher_energy"]),
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    groups = int(mapping.max()) + 1
    heads = int(mapping.numel())
    joint_dim = value_dim + key_dim
    encoders = torch.randn(
        groups,
        joint_dim,
        args.route_rank,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    query_factors = torch.randn(
        heads,
        key_dim,
        args.route_rank,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    raw_loss = softmax_fisher_routing_loss(
        raw,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    compact_loss = compact_softmax_fisher_loss(
        compact,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    raw_adapter = softmax_fisher_adapter_system(
        raw,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    compact_adapter = compact_softmax_fisher_adapter_system(
        compact,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    diagonal_errors = []
    for group in range(groups):
        head_indices = torch.nonzero(mapping.to(device) == group).flatten()
        group_queries = raw.queries_by_head.index_select(0, head_indices)
        group_factors = query_factors.index_select(0, head_indices)
        projected = torch.einsum("hdk,hkr->dhr", group_queries, group_factors)
        raw_diagonal = softmax_fisher_encoder_diagonal(
            joint_rows=raw.joint_rows_by_group[group],
            projected_queries=projected,
            probabilities=raw.probabilities_by_head.index_select(
                0,
                head_indices,
            ).permute(1, 0, 2),
            scaling=raw.scaling,
        )
        compact_diagonal = compact_softmax_fisher_encoder_diagonal(
            compact,
            head_indices=head_indices,
            projected_queries=projected,
        )
        diagonal_errors.append(_relative_error(compact_diagonal, raw_diagonal))
    roots = compact_softmax_fisher_roots(compact)
    root_error = _relative_error(
        roots @ roots.mT,
        compact.fisher_grams_by_head,
    )
    report = {
        "layer": args.layer,
        "documents": documents,
        "tokens": raw.tokens,
        "dtype": args.work_dtype,
        "raw_loss": raw_loss,
        "compact_loss": compact_loss,
        "loss_relative_error": abs(compact_loss - raw_loss) / abs(raw_loss),
        "adapter_left_relative_error": _relative_error(
            compact_adapter[0],
            raw_adapter[0],
        ),
        "adapter_right_relative_error": _relative_error(
            compact_adapter[1],
            raw_adapter[1],
        ),
        "adapter_rhs_relative_error": _relative_error(
            compact_adapter[2],
            raw_adapter[2],
        ),
        "maximum_encoder_diagonal_relative_error": max(diagonal_errors),
        "fisher_root_reconstruction_relative_error": root_error,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
