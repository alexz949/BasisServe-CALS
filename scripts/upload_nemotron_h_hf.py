#!/usr/bin/env python3
"""Upload and verify final Nemotron-H C1 and PaLU checkpoints."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "alexz949/BasisServe-CALS"
TOKEN_FILE = Path.home() / ".cache/huggingface/token"
REMOTE_ROOT = Path("ICLR-results/nemotron-h")
RUNS = {
    "8b/c1-r64": ROOT / "results/nemotron_h_8b/checkpoints/r64",
    "8b/c1-r96": ROOT / "results/nemotron_h_8b/checkpoints/r96",
    "8b/palu-glrd4-r64": ROOT / "results/nemotron_h_8b/palu/checkpoints/glrd4_r64",
    "8b/palu-glrd4-r96": ROOT / "results/nemotron_h_8b/palu/checkpoints/glrd4_r96",
    "8b/palu-mlrd-r64": ROOT / "results/nemotron_h_8b/palu/checkpoints/mlrd_r64",
    "8b/palu-mlrd-r96": ROOT / "results/nemotron_h_8b/palu/checkpoints/mlrd_r96",
    "56b/c1-r64": ROOT / "results/nemotron_h_56b/checkpoints/r64",
    "56b/c1-r96": ROOT / "results/nemotron_h_56b/checkpoints/r96",
    "56b/palu-glrd4-r64": ROOT / "results/nemotron_h_56b/palu/checkpoints/glrd4_r64",
    "56b/palu-glrd4-r96": ROOT / "results/nemotron_h_56b/palu/checkpoints/glrd4_r96",
    "56b/palu-mlrd-r64": ROOT / "results/nemotron_h_56b/palu/checkpoints/mlrd_r64",
    "56b/palu-mlrd-r96": ROOT / "results/nemotron_h_56b/palu/checkpoints/mlrd_r96",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def inventory():
    rows = []
    for run, directory in RUNS.items():
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["format"] in {
            "basisserve.nemotron_h.c1_v_wo.v1",
            "basisserve.nemotron_h.palu_v_only_fisher.v1",
        }
        run_rows = []
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            run_rows.append({
                "run": run,
                "local": str(path),
                "path": (REMOTE_ROOT / run / path.relative_to(directory)).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
        assert run_rows and any(row["path"].endswith(".safetensors") for row in run_rows)
        rows.extend(run_rows)
    assert len({row["path"] for row in rows}) == len(rows)
    return rows


def remote_records(api, paths, revision=None):
    records = {}
    for offset in range(0, len(paths), 50):
        for item in api.get_paths_info(
            REPO_ID,
            paths[offset : offset + 50],
            expand=True,
            revision=revision,
            repo_type="model",
        ):
            records[item.path] = item
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--commit-message", required=True)
    args = parser.parse_args()

    rows = inventory()
    token = TOKEN_FILE.read_text().strip()
    assert token
    api = HfApi(token=token)
    before = api.repo_info(REPO_ID, repo_type="model")
    assert not before.private

    commits = []
    for run in RUNS:
        run_rows = [row for row in rows if row["run"] == run]
        remote = remote_records(api, [row["path"] for row in run_rows])
        pending = []
        for row in run_rows:
            item = remote.get(row["path"])
            if item is None:
                pending.append(row)
                continue
            assert item.size == row["bytes"], row["path"]
            if row["path"].endswith(".safetensors"):
                assert item.lfs is not None and item.lfs.sha256 == row["sha256"], row["path"]
        if pending:
            commit = api.create_commit(
                repo_id=REPO_ID,
                operations=[CommitOperationAdd(
                    path_in_repo=row["path"], path_or_fileobj=row["local"]
                ) for row in pending],
                commit_message=f"{args.commit_message}: {run}",
                repo_type="model",
                num_threads=4,
            )
            commits.append({"run": run, "revision": commit.oid, "uploaded": len(pending)})
        else:
            commits.append({"run": run, "revision": None, "uploaded": 0})
        print(commits[-1], flush=True)

    revision = api.repo_info(REPO_ID, repo_type="model").sha
    remote = remote_records(api, [row["path"] for row in rows], revision=revision)
    assert len(remote) == len(rows)
    with tempfile.TemporaryDirectory(prefix="basisserve-nemotron-hf-") as temporary:
        for row in rows:
            item = remote[row["path"]]
            assert item.size == row["bytes"]
            if row["path"].endswith(".safetensors"):
                assert item.lfs is not None and item.lfs.sha256 == row["sha256"]
            else:
                downloaded = hf_hub_download(
                    REPO_ID,
                    row["path"],
                    revision=revision,
                    repo_type="model",
                    token=token,
                    local_dir=temporary,
                )
                assert digest(Path(downloaded)) == row["sha256"]

    report = {
        "status": "complete",
        "repository": REPO_ID,
        "revision": revision,
        "baseline_revision": before.sha,
        "commit_message": args.commit_message,
        "runs": list(RUNS),
        "commits": commits,
        "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = ROOT / "results/uploads/nemotron-h-hf/result.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    main()
