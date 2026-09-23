"""Freeze three distinct real-text prompt cohorts for the TP8 benchmark grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


DEFAULT_SOURCE = Path("/workspace/runs/l31-cal128/longbench-v2/tokens.safetensors")
DEFAULT_SOURCE_MANIFEST = Path("/workspace/runs/l31-cal128/longbench-v2/manifest.json")
DEFAULT_OUTPUT = Path("/workspace/runs/l31-cal128/tp8-benchmark-prompts")
WORKLOADS = {4096: 128, 16384: 16, 65536: 16, 130048: 16}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert args.source.is_file() and args.source_manifest.is_file()
    source_manifest = json.loads(args.source_manifest.read_text())
    assert source_manifest["status"] == "complete"
    assert source_manifest["tokens_sha256"] == _sha256(args.source)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    with safe_open(str(args.source), framework="pt", device="cpu") as source:
        source_keys = sorted(source.keys())
        for prompt_tokens, maximum_batch in WORKLOADS.items():
            eligible = [
                key
                for key in source_keys
                if source.get_slice(key).get_shape()[0] >= prompt_tokens
            ]
            assert len(eligible) >= 3 * maximum_batch
            for cohort in range(3):
                sample_ids = eligible[
                    cohort * maximum_batch : (cohort + 1) * maximum_batch
                ]
                rows = torch.stack(
                    [source.get_tensor(key)[:prompt_tokens].to(torch.int32) for key in sample_ids]
                ).contiguous()
                assert tuple(rows.shape) == (maximum_batch, prompt_tokens)
                token_hashes = [_tensor_sha256(row) for row in rows]
                output_path = args.output / f"p{prompt_tokens}_c{cohort}.safetensors"
                manifest_path = args.output / f"p{prompt_tokens}_c{cohort}.json"
                assert not output_path.exists() and not manifest_path.exists()
                save_file({"input_ids": rows}, str(output_path))
                payload = {
                    "status": "complete",
                    "format": "basisserve.llama31.tp8_benchmark_prompt_cohort.v1",
                    "prompt_tokens": prompt_tokens,
                    "maximum_batch": maximum_batch,
                    "cohort": cohort,
                    "sample_ids": sample_ids,
                    "token_sha256": token_hashes,
                    "cohort_hash": hashlib.sha256(
                        "\n".join(token_hashes).encode()
                    ).hexdigest(),
                    "tokens_file": output_path.name,
                    "tokens_file_sha256": _sha256(output_path),
                    "source": str(args.source.resolve()),
                    "source_sha256": _sha256(args.source),
                    "source_manifest": str(args.source_manifest.resolve()),
                    "source_manifest_sha256": _sha256(args.source_manifest),
                    "selection": "eligible LongBench-v2 samples in stable sample-id order; three disjoint slices",
                }
                manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                records.append(payload)
                print(
                    json.dumps(
                        {
                            "prompt_tokens": prompt_tokens,
                            "cohort": cohort,
                            "batch": maximum_batch,
                            "cohort_hash": payload["cohort_hash"],
                        }
                    ),
                    flush=True,
                )
    manifest_path = args.output / "manifest.json"
    assert not manifest_path.exists()
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "format": "basisserve.llama31.tp8_benchmark_prompts.v1",
                "source_dataset": source_manifest["protocol"]["dataset"],
                "source_dataset_revision": source_manifest["protocol"]["dataset_revision"],
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
