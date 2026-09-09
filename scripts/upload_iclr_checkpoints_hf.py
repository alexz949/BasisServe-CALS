#!/usr/bin/env python3
"""Upload final ICLR checkpoints and referenced factors, then verify Hub hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "alexz949/BasisServe-CALS"


def file_hashes(path):
    sha = hashlib.sha256()
    git = hashlib.sha1()
    git.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            sha.update(chunk)
            git.update(chunk)
    return sha.hexdigest(), git.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def verify_inventory(api, inventory, runs, total_bytes, baseline_revision, report_path):
    before = api.model_info(REPO_ID, revision=baseline_revision, files_metadata=True)
    after = api.model_info(REPO_ID, files_metadata=True)
    uploaded = {item.rfilename: item for item in after.siblings}
    failures = []
    for row in inventory:
        item = uploaded.get(row["path"])
        correct = item is not None and item.size == row["bytes"]
        if correct:
            correct = item.lfs.sha256 == row["sha256"] if item.lfs else item.blob_id == row["git_blob_sha1"]
        if not correct:
            failures.append(row["path"])
    changed = [item.rfilename for item in before.siblings
               if item.rfilename not in uploaded or uploaded[item.rfilename].blob_id != item.blob_id]
    attribute_additions = []
    attributes_ok = True
    if ".gitattributes" in changed:
        old = Path(hf_hub_download(REPO_ID, ".gitattributes", revision=before.sha)).read_text().splitlines()
        new = Path(hf_hub_download(REPO_ID, ".gitattributes", revision=after.sha)).read_text().splitlines()
        attribute_additions = [line for line in new if line and line not in old]
        expected_rules = {row["path"] + " filter=lfs diff=lfs merge=lfs -text" for row in inventory}
        attributes_ok = all(line in new for line in old if line) and all(line in expected_rules for line in attribute_additions)
    payload_preserved = all(name == ".gitattributes" for name in changed)
    complete = not failures and payload_preserved and attributes_ok
    result = {
        "status": "complete" if complete else "verification_failed",
        "repository": REPO_ID, "baseline_revision": before.sha, "revision": after.sha,
        "checkpoint_count": len(runs), "verified_files": len(inventory) - len(failures),
        "bytes": total_bytes, "failures": failures,
        "existing_payload_files_preserved": payload_preserved,
        "changed_existing_files": changed, "automatic_lfs_rules": attribute_additions,
        "automatic_lfs_rules_valid": attributes_ok,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(report_path, result)
    print(json.dumps(result), flush=True)
    return int(not complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--baseline-revision")
    args = parser.parse_args()
    report_dir = ROOT / "results/uploads/iclr-checkpoints-hf"
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        if not args.baseline_revision:
            print("[Stop] Verification requires --baseline-revision", flush=True)
            return 1
        inventory = json.loads((report_dir / "inventory.json").read_text())
        return verify_inventory(
            HfApi(), inventory["files"], inventory["run_ids"], inventory["bytes"],
            args.baseline_revision, report_dir / "verification.json",
        )
    manifests = sorted((ROOT / "ICLR-results").glob("*/checkpoints/*/manifest.json"))
    files = set()
    expected = {}
    runs = []
    errors = []
    for manifest in manifests:
        data = json.loads(manifest.read_text())
        if data.get("status") != "complete":
            errors.append(f"Incomplete checkpoint: {manifest}")
        runs.append(data["run_id"])
        for path in manifest.parent.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".safetensors"):
                files.add(path.resolve())
        references = [data["artifact"]] if data.get("artifact") else []
        references += [row for row in data.get("layers", []) if "file" in row]
        for record in references:
            path = (manifest.parent / record["file"]).resolve()
            if not path.is_relative_to(ROOT / "ICLR-results") or not path.is_file():
                errors.append(f"Missing or out-of-scope dependency: {path}")
                continue
            files.add(path)
            expected[path] = record["sha256"]
    if len(manifests) != 81:
        errors.append(f"Expected 81 checkpoints, found {len(manifests)}")
    if errors:
        print(json.dumps(errors), flush=True)
        return 1

    inventory = []
    for index, path in enumerate(sorted(files)):
        sha, git_sha = file_hashes(path)
        if path in expected and sha != expected[path]:
            errors.append(f"Manifest hash mismatch: {path}")
        inventory.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha, "git_blob_sha1": git_sha,
        })
        if (index + 1) % 50 == 0:
            print(f"[Validate] {index + 1}/{len(files)}", flush=True)
    plan = {
        "repository": REPO_ID, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(runs), "run_ids": runs,
        "file_count": len(inventory), "bytes": sum(row["bytes"] for row in inventory),
        "files": inventory, "validation_errors": errors,
    }
    write_json(report_dir / "inventory.json", plan)
    if errors:
        print(json.dumps(errors), flush=True)
        return 1

    api = HfApi()
    before = api.model_info(REPO_ID, files_metadata=True)
    if not before.private:
        print("[Stop] Expected an existing private repository", flush=True)
        return 1
    remote = {item.rfilename: item for item in before.siblings}

    def matches(row, item):
        if item is None or item.size != row["bytes"]:
            return False
        return item.lfs.sha256 == row["sha256"] if item.lfs else item.blob_id == row["git_blob_sha1"]

    conflicts = [row["path"] for row in inventory if row["path"] in remote and not matches(row, remote[row["path"]])]
    if conflicts:
        print("[Stop] Existing remote paths differ: " + json.dumps(conflicts), flush=True)
        return 1
    pending = [row["path"] for row in inventory if row["path"] not in remote]
    print(f"[Plan] checkpoints={len(runs)} files={len(inventory)} bytes={plan['bytes']} pending={len(pending)}", flush=True)
    if not args.upload:
        return 0
    if pending:
        api.upload_large_folder(
            repo_id=REPO_ID, repo_type="model", folder_path=ROOT,
            allow_patterns=pending, num_workers=args.workers,
            print_report=True, print_report_every=30,
        )
    return verify_inventory(api, inventory, runs, plan["bytes"], before.sha, report_dir / "result.json")


if __name__ == "__main__":
    sys.exit(main())
