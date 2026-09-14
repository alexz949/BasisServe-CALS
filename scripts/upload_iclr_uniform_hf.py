#!/usr/bin/env python3
"""Prepare all completed ICLR uniform C1 checkpoints; optionally validate and upload."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from huggingface_hub import CommitOperationAdd, HfApi
from safetensors import safe_open

from upload_iclr_checkpoints_hf import file_hashes, verify_inventory

ROOT = Path(__file__).resolve().parents[1]
REPO = "alexz949/BasisServe-CALS"
REPORT = ROOT / "results/uploads/iclr-uniform-hf"
MODELS = {"qwen3-8b": "Q3-8B", "qwen3-32b": "Q3-32B",
          "llama31-8b": "L31-8B", "llama31-70b": "L31-70B", "llama2-7b": "L2-7B"}
RANKS = (32, 48, 64, 80, 96, 112)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    template = json.loads((ROOT / "ICLR-results/qwen3-8b/checkpoints/Q3-8B-C1U-R80/manifest.json").read_text())
    inventory = {}
    runs = []
    problems = []
    candidates = []
    for model, prefix in MODELS.items():
        base = ROOT / "ICLR-results" / model
        reference = json.loads((base / "checkpoints" / f"{prefix}-C1-R64/manifest.json").read_text())
        for rank in RANKS:
            bank = base / f"c1/factor-banks/R{rank}-S6"
            result_path = bank / "results.json"
            result = json.loads(result_path.read_text())
            config = result["fit_config"]
            count, heads = config["num_hidden_layers"], config["num_physical_kv_heads"]
            valid = (result["status"] == "complete" and config["cache_rank_per_head"] == rank
                     and config["encoder_sweeps"] == 6 and config["decoder_objective"] == "full_layer"
                     and config["checkpoint_policy"] == template["compression"]["checkpoint_policy"]
                     and config["model_config_sha256"] == reference["model"]["config_sha256"]
                     and len(result["records"]) == count and config["factor_dtype"] == "bfloat16")
            run = f"{prefix}-C1U-R{rank}"
            output = base / "checkpoints" / run
            layers = []
            shapes = {"value_coordinate_encoders": [heads, config["head_dim"], rank],
                      "head_output_decoders": [config["num_query_heads"], rank, config["hidden_size"]]}
            for index, record in enumerate(result["records"]):
                path = bank / record["artifact"]["file"]
                if not path.is_file():
                    problems.append(f"Missing factor: {path}")
                    continue
                with safe_open(str(path), framework="numpy") as handle:
                    actual = {k: list(handle.get_slice(k).get_shape()) for k in handle.keys()}
                valid = (valid and actual == shapes and record["layer"] == index
                         and record["checkpoint"]["sweep"] == 6
                         and record["checkpoint"]["boundary"] == "after_redecoder")
                row = dict(layer=index, ranks=[rank] * heads, file=os.path.relpath(path, output),
                           sha256=record["artifact"]["sha256"], bytes=path.stat().st_size, tensor_shapes=actual)
                layers.append(row)
                inventory[str(path.relative_to(ROOT))] = dict(bytes=row["bytes"], sha256=row["sha256"])
            if not valid or len(layers) != count:
                problems.append(f"Invalid uniform factor bank: {bank}")
                continue
            manifest = copy.deepcopy(template)
            manifest.update(format=reference["format"], model=reference["model"], run_id=run,
                            command=shlex.join(sys.argv), timestamp_utc=datetime.now(timezone.utc).isoformat())
            manifest["compression"].update(
                equivalent_rank_target=rank, layer_ranks=[[rank] * heads for _ in range(count)],
                rank_sum_across_layers=count * heads * rank,
                dense_rank_sum_across_layers=count * heads * config["head_dim"],
                realized_retained_v_ratio=rank / config["head_dim"],
                realized_v_cache_compression_ratio=1 - rank / config["head_dim"],
                num_query_heads=config["num_query_heads"], num_physical_kv_heads=heads,
                head_dim=config["head_dim"], encoder_initialization=config["encoder_initialization"])
            manifest["layers"] = layers
            manifest["artifact"] = dict(file=os.path.relpath(result_path, output), sha256=digest(result_path),
                                        bytes=result_path.stat().st_size, selected_factor_count=count,
                                        selected_factor_bytes=sum(r["bytes"] for r in layers))
            manifest["environment"] = dict(conda_environment="basis", python=sys.version)
            target = output / "manifest.json"
            if target.exists():
                existing = json.loads(target.read_text())
                if any(existing.get(k) != manifest[k] for k in ("format", "model", "compression", "layers", "artifact", "run_id", "status")):
                    problems.append(f"Existing manifest differs: {target}")
            candidates.append((target, manifest))
            inventory[str(result_path.relative_to(ROOT))] = dict(bytes=result_path.stat().st_size, sha256=digest(result_path))
            runs.append(run)
    if problems:
        print(json.dumps(problems, indent=2), flush=True)
        return None
    for target, manifest in candidates:
        if not target.exists():
            write_json(target, manifest)
        inventory[str(target.relative_to(ROOT))] = dict(bytes=target.stat().st_size, sha256=digest(target))
    return inventory, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    prepared = prepare()
    if prepared is None:
        return 1
    inventory, runs = prepared
    api = HfApi()
    before = api.model_info(REPO, files_metadata=True)
    remote = {f.rfilename: f for f in before.siblings}
    rows = []
    conflicts = []
    for path, values in sorted(inventory.items()):
        item = remote.get(path)
        if item is not None:
            sha = item.lfs.sha256 if item.lfs else None
            same = item.size == values["bytes"]
            if sha:
                same = same and sha == values["sha256"]
            else:
                _, blob = file_hashes(ROOT / path)
                same = same and blob == item.blob_id
            if not same:
                conflicts.append(path)
        rows.append(dict(path=path, **values))
    pending = [r for r in rows if r["path"] not in remote]
    plan = dict(repository=REPO, revision=before.sha, run_ids=runs, files=rows,
                file_count=len(rows), bytes=sum(r["bytes"] for r in rows),
                pending_files=[r["path"] for r in pending], pending_bytes=sum(r["bytes"] for r in pending),
                conflicts=conflicts, validation="tensor shapes, rank, layer coverage, sweep-6 metadata; full weight hashes pending")
    write_json(REPORT / "plan.json", plan)
    print(json.dumps({k:v for k,v in plan.items() if k not in {"files", "pending_files", "run_ids"}}), flush=True)
    if conflicts:
        print("Stopped: remote file conflicts.", flush=True)
        return 1
    if not args.upload:
        return 0
    for index, row in enumerate(rows):
        sha, blob = file_hashes(ROOT / row["path"])
        if sha != row["sha256"]:
            print(f"Hash mismatch: {row['path']}", flush=True)
            return 1
        row["git_blob_sha1"] = blob
        if (index + 1) % 50 == 0:
            print(f"Validated hashes: {index + 1}/{len(rows)}", flush=True)
    write_json(REPORT / "verified_inventory.json", dict(files=rows, run_ids=runs))
    for start in range(0, len(pending), 250):
        batch = pending[start:start + 250]
        api.create_commit(
            repo_id=REPO, repo_type="model",
            operations=[CommitOperationAdd(path_in_repo=r["path"], path_or_fileobj=ROOT / r["path"])
                        for r in batch],
            commit_message="Upload ICLR uniform C1 checkpoints",
        )
        print(f"Committed remaining files: {min(start + 250, len(pending))}/{len(pending)}", flush=True)
    return verify_inventory(api, rows, runs, plan["bytes"], before.sha, REPORT / "result.json")


if __name__ == "__main__":
    sys.exit(main())
