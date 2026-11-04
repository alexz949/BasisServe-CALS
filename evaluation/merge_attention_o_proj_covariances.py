#!/usr/bin/env python3
"""Merge window-sharded attention ``o_proj`` covariance snapshots into one snapshot.

Each shard (``capture_attention_o_proj_covariances.py --window-shard s --window-shards N``) holds per-layer mean
covariances over its own fit and held-out windows. The merged mean is the row-weighted sum of the shard means,
accumulated in FP64 and stored in the shards' covariance dtype, so the result matches an unsharded capture up to
floating-point summation order. Shards must come from the same model, windows file and layer set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

from safetensors.torch import load_file, save_file
import torch

SPLITS = ("fit", "heldout")
MATCHED_MODEL_KEYS = ("path", "rope", "config_sha256", "model_type", "attention_type", "num_hidden_layers",
                      "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size")
MATCHED_CALIBRATION_KEYS = ("activation_aware", "objective", "storage", "covariance_dtype", "windows_file",
                            "windows_sha256", "windows_manifest_format", "sequence_length", "positions_per_window")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True, help="per-shard snapshot directories")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    shard_dirs = [Path(p).expanduser().resolve() for p in args.shards]
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    assert not output_dir.exists() and not partial_dir.exists(), f"refusing to overwrite output: {output_dir}"
    manifests = [_manifest(d) for d in shard_dirs]
    first = manifests[0]
    for manifest in manifests[1:]:
        assert manifest["format"] == first["format"]
        assert all(manifest["model"][k] == first["model"][k] for k in MATCHED_MODEL_KEYS), "shards differ in model"
        assert manifest["layers"] == first["layers"], "shards differ in layers"
        assert all(manifest["calibration"][k] == first["calibration"][k] for k in MATCHED_CALIBRATION_KEYS), \
            "shards differ in calibration windows"
        assert manifest["calibration"]["sharding"]["window_shards"] == first["calibration"]["sharding"]["window_shards"]
    shardings = [m["calibration"]["sharding"] for m in manifests]
    assert sorted(s["window_shard"] for s in shardings) == list(range(shardings[0]["window_shards"])), \
        "merge needs every window shard exactly once"
    ids = {split: sorted(i for m in manifests for i in m["calibration"][f"{split}_window_ids"]) for split in SPLITS}
    bank = {split: shardings[0][f"bank_{split}_windows"] for split in SPLITS}
    assert ids["fit"] == list(range(bank["fit"]))
    assert ids["heldout"] == list(range(bank["fit"], bank["fit"] + bank["heldout"]))
    rows = {split: [int(m["calibration"][f"{split}_rows"]) for m in manifests] for split in SPLITS}
    total_rows = {split: sum(rows[split]) for split in SPLITS}
    sequence_length = int(first["calibration"]["sequence_length"])
    assert total_rows == {split: bank[split] * sequence_length for split in SPLITS}

    partial_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    for layer in first["layers"]:
        key = str(layer)
        merged: dict[str, torch.Tensor] = {}
        weight = None
        dtype = None
        for manifest, directory, index in zip(manifests, shard_dirs, range(len(manifests))):
            record = manifest["artifacts"][key]
            path = directory / record["file"]
            assert _sha256(path) == record["sha256"], f"shard {directory} layer {layer}: checksum differs"
            payload = load_file(str(path))
            if weight is None:
                weight = payload["weight"]
                dtype = payload["fit_covariance"].dtype
            else:
                assert torch.equal(weight, payload["weight"]), f"layer {layer}: o_proj weight differs between shards"
            for split in SPLITS:
                part = payload[f"{split}_covariance"].to(torch.float64) * (rows[split][index] / total_rows[split])
                merged[split] = part if split not in merged else merged[split] + part
        tensors = {f"{split}_covariance": merged[split].to(dtype).contiguous() for split in SPLITS}
        tensors["weight"] = weight.contiguous()
        out = partial_dir / f"layer_{layer:03d}.safetensors"
        save_file(tensors, str(out))
        artifacts[key] = {
            "file": out.name,
            "sha256": _sha256(out),
            "fit_covariance_shape": list(tensors["fit_covariance"].shape),
            "heldout_covariance_shape": list(tensors["heldout_covariance"].shape),
            "weight_shape": list(weight.shape),
            "covariance_dtype": str(dtype),
            "weight_dtype": str(weight.dtype),
        }
        print(f"[Merge] wrote layer={layer}", flush=True)

    calibration = {k: v for k, v in first["calibration"].items()
                   if k not in ("fit_window_ids", "heldout_window_ids", "sharding")}
    calibration.update(
        window_count=bank["fit"] + bank["heldout"], fit_windows=bank["fit"], heldout_windows=bank["heldout"],
        fit_rows=total_rows["fit"], heldout_rows=total_rows["heldout"], rows_per_layer=sum(total_rows.values()),
    )
    manifest = {
        "format": first["format"],
        "schema_version": first["schema_version"],
        "command": shlex.join(sys.argv),
        "model": first["model"],
        "layers": first["layers"],
        "calibration": calibration,
        "artifacts": artifacts,
        "merged_from": [
            {"directory": str(d), "manifest_sha256": _sha256(d / "manifest.json"), "sharding": m["calibration"]["sharding"],
             "fit_window_ids": m["calibration"]["fit_window_ids"], "heldout_window_ids": m["calibration"]["heldout_window_ids"],
             "elapsed_seconds": m.get("elapsed_seconds"), "environment": m.get("environment")}
            for d, m in zip(shard_dirs, manifests)
        ],
        "merge": "row-weighted FP64 sum of per-shard mean covariances, stored in the shard covariance dtype",
    }
    (partial_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(partial_dir, output_dir)
    print(f"[Merge] complete output={output_dir} layers={len(artifacts)}", flush=True)


if __name__ == "__main__":
    main()
