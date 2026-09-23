"""Freeze Qwen-tokenized LongBench-v2 cohorts for the TP8 V-only memory grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer


MODEL = Path(
    "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/"
    "snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
)
SOURCE = Path(
    "/workspace/.cache/huggingface/hub/datasets--THUDM--LongBench-v2/"
    "snapshots/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json"
)
INVENTORY = Path("/workspace/runs/l31-cal128/longbench-v2/samples.json")
TEMPLATE = Path("/workspace/BasisServe-CALS/external/LongBench/prompts/0shot.txt")
OUTPUT = Path("results/system_benchmarks/tp8_external_baselines/starkv_v_only/prompts")
LENGTHS = (4096, 16384, 32768, 65536, 98304, 130048)
MAX_BATCH = 8
COHORTS = 3


def prompt_text(template: str, record: dict) -> str:
    return (
        template.replace("$DOC$", record["context"])
        .replace("$Q$", record["question"])
        .replace("$C_A$", record["choice_A"])
        .replace("$C_B$", record["choice_B"])
        .replace("$C_C$", record["choice_C"])
        .replace("$C_D$", record["choice_D"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    assert not (args.output / "manifest.json").exists()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {item["_id"]: item for item in json.loads(SOURCE.read_text())}
    inventory = json.loads(INVENTORY.read_text())
    template = TEMPLATE.read_text()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    selected = []
    for sample in inventory:
        if sample["original_prompt_tokens"] < LENGTHS[-1]:
            continue
        ids = tokenizer.encode(prompt_text(template, data[sample["source_id"]]), add_special_tokens=False)
        if len(ids) < LENGTHS[-1]:
            continue
        selected.append((f"sample_{sample['index']:03d}", sample["source_id"], len(ids), ids[:LENGTHS[-1]]))
        print(json.dumps({"selected": selected[-1][:3]}), flush=True)
        if len(selected) == MAX_BATCH * COHORTS:
            break
    assert len(selected) == MAX_BATCH * COHORTS

    records = []
    for length in LENGTHS:
        for cohort in range(COHORTS):
            rows = selected[cohort * MAX_BATCH:(cohort + 1) * MAX_BATCH]
            tokens = torch.tensor([item[3][:length] for item in rows], dtype=torch.int32)
            path = args.output / f"p{length}_c{cohort}.safetensors"
            manifest_path = args.output / f"p{length}_c{cohort}.json"
            assert not path.exists() and not manifest_path.exists()
            save_file({"input_ids": tokens}, str(path))
            record = {
                "status": "complete",
                "format": "basisserve.qwen3.tp8_memory_prompts.v1",
                "prompt_tokens": length,
                "maximum_batch": MAX_BATCH,
                "cohort": cohort,
                "sample_ids": [item[0] for item in rows],
                "source_ids": [item[1] for item in rows],
                "original_qwen_token_lengths": [item[2] for item in rows],
                "tokens_file": path.name,
                "selection": "first 24 Qwen-eligible LongBench-v2 samples in frozen inventory order; disjoint groups of eight",
            }
            manifest_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(json.dumps({"length": length, "cohort": cohort, "batch": MAX_BATCH}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "format": "basisserve.qwen3.tp8_memory_prompt_grid.v1",
        "model_revision": MODEL.name,
        "dataset_revision": SOURCE.parent.name,
        "template": str(TEMPLATE),
        "records": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
