#!/usr/bin/env python3
"""Convert raw Qwen3 Store80 routing captures to compact Fisher Grams."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    pack_symmetric_fisher_grams,
    softmax_fisher_gram,
)


FORMAT = "basisserve.qwen3.s80_compact_softmax_fisher.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _parse_layers(spec: str) -> tuple[int, ...]:
    selected: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(item))
    return tuple(sorted(selected))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _mmap_tensor(root: Path, record: dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(size) for size in record["shape"])
    values = 1
    for size in shape:
        values *= size
    return torch.from_file(
        str(root / record["file"]),
        shared=False,
        size=values,
        dtype=torch.bfloat16,
    ).reshape(shape)


@torch.inference_mode()
def _compact_layer(
    *,
    root: Path,
    artifacts: dict[str, Any],
    layer_index: int,
    output_path: Path,
    device: torch.device,
    work_dtype: torch.dtype,
    storage_dtype: torch.dtype,
) -> dict[str, Any]:
    records = artifacts[str(layer_index)]
    queries_raw = _mmap_tensor(root, records["routing_queries"])
    rows_raw = _mmap_tensor(root, records["routing_joint_rows"])
    documents, query_heads, key_dim = map(int, queries_raw.shape)
    _, tokens, kv_heads, joint_dim = map(int, rows_raw.shape)
    value_dim = joint_dim - key_dim
    heads_per_group = query_heads // kv_heads
    mapping = torch.arange(query_heads, dtype=torch.long) // heads_per_group
    queries = (
        queries_raw.permute(1, 0, 2)
        .contiguous()
        .to(device=device, dtype=work_dtype)
    )
    packed_width = joint_dim * (joint_dim + 1) // 2
    packed = torch.empty(
        query_heads,
        documents,
        packed_width,
        dtype=storage_dtype,
        device="cpu",
    )
    scaling = key_dim**-0.5
    teacher_energy = 0.0
    for document in range(documents):
        for group in range(kv_heads):
            first_head = group * heads_per_group
            head_indices = torch.arange(
                first_head,
                first_head + heads_per_group,
                device=device,
            )
            joint_rows = rows_raw[document, :, group].to(
                device=device,
                dtype=work_dtype,
            )
            group_queries = queries.index_select(0, head_indices)[:, document]
            grams, group_teacher_energy = softmax_fisher_gram(
                group_queries,
                joint_rows,
                value_dim=value_dim,
                scaling=scaling,
            )
            packed[first_head : first_head + heads_per_group, document].copy_(
                pack_symmetric_fisher_grams(grams).to(
                    device="cpu",
                    dtype=storage_dtype,
                )
            )
            teacher_energy += group_teacher_energy
            del joint_rows, group_queries, grams
        print(
            f"[S80 compact Fisher] layer={layer_index} "
            f"documents={document + 1}/{documents}",
            flush=True,
        )
    output_queries = queries.to(device="cpu", dtype=storage_dtype)
    _atomic_safetensors(
        output_path,
        {
            "queries_by_head": output_queries.contiguous(),
            "fisher_grams_packed_by_head": packed.contiguous(),
            "head_to_kv_group": mapping.contiguous(),
            "value_dim": torch.tensor(value_dim, dtype=torch.int64),
            "key_dim": torch.tensor(key_dim, dtype=torch.int64),
            "scaling": torch.tensor(scaling, dtype=torch.float64),
            "teacher_fisher_energy": torch.tensor(
                teacher_energy,
                dtype=torch.float64,
            ),
        },
    )
    return {
        "file": output_path.name,
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
        "documents": documents,
        "tokens": tokens,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "value_dim": value_dim,
        "key_dim": key_dim,
        "joint_dim": joint_dim,
        "packed_width": packed_width,
        "teacher_fisher_energy": teacher_energy,
    }


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    started = time.monotonic()
    root = Path(args.direct_capture_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = _parse_layers(args.layers)
    device = torch.device(args.device)
    work_dtype = torch.float64 if args.work_dtype == "float64" else torch.float32
    storage_dtype = (
        torch.float64 if args.storage_dtype == "float64" else torch.float32
    )
    artifacts = {}
    for layer_index in layers:
        output_path = output_dir / f"layer_{layer_index:03d}.safetensors"
        artifacts[str(layer_index)] = _compact_layer(
            root=root,
            artifacts=source_manifest["artifacts"],
            layer_index=layer_index,
            output_path=output_path,
            device=device,
            work_dtype=work_dtype,
            storage_dtype=storage_dtype,
        )
    manifest = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "elapsed_seconds": time.monotonic() - started,
        "source": {
            "direct_capture_manifest": str(manifest_path),
            "direct_capture_manifest_sha256": _sha256(manifest_path),
            "format": source_manifest["format"],
        },
        "geometry": {
            "layer_coverage": list(layers),
            "key_convention": "post_rope",
            "fisher_pairing": "per_document_per_query_head",
        },
        "numerics": {
            "work_dtype": args.work_dtype,
            "storage_dtype": args.storage_dtype,
            "gram": "X.T @ (Diag(p) - p p.T) @ X",
            "gram_scaling_included": False,
            "symmetric_storage": "packed_upper_triangle",
        },
        "artifacts": artifacts,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"[S80 compact Fisher] wrote {output_dir / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
