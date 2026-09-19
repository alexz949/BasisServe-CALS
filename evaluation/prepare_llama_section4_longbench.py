"""Freeze the paired Llama-3.1-Instruct LongBench pilot for Section 4."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer

from evaluation.v96kl_common import sha256, write_json


TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "qmsum")


def bounded_tokens(tokens, total_limit, generation_cap):
    budget = total_limit - generation_cap
    assert budget > 0 and tokens
    if len(tokens) <= budget:
        return list(tokens)
    left = budget // 2
    return list(tokens[:left]) + list(tokens[-(budget - left) :])


def choose_rows(rows, task, count, seed):
    assert len(rows) >= count and len({row["_id"] for row in rows}) == len(rows)
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{task}:{row['_id']}".encode()).hexdigest(),
            row["_id"],
        ),
    )[:count]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert args.samples_per_task > 0 and args.sequence_length > 0
    assert not (args.output_dir / "manifest.json").exists()
    official = args.official_root / "LongBench"
    prompts = json.loads((official / "config/dataset2prompt.json").read_text())
    caps = json.loads((official / "config/dataset2maxlen.json").read_text())
    archive = Path(
        hf_hub_download(
            "THUDM/LongBench",
            "data.zip",
            repo_type="dataset",
            revision=args.dataset_revision,
            local_files_only=True,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = []
    tensors = {}
    source_hashes = {}
    with zipfile.ZipFile(archive) as data:
        for task in TASKS:
            names = [name for name in data.namelist() if name == f"{task}.jsonl" or name.endswith(f"/{task}.jsonl")]
            assert len(names) == 1
            raw = data.read(names[0])
            source_hashes[task] = hashlib.sha256(raw).hexdigest()
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            chosen = choose_rows(records, task, args.samples_per_task, args.seed)
            for ordinal, record in enumerate(chosen):
                prompt = prompts[task].format(**record)
                token_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )["input_ids"]
                selected = bounded_tokens(token_ids, args.sequence_length, caps[task])
                index = len(rows)
                tensor = torch.tensor(selected, dtype=torch.int32)
                tensors[f"sample_{index:03d}"] = tensor
                rows.append(
                    {
                        "index": index,
                        "task": task,
                        "ordinal": ordinal,
                        "source_id": record["_id"],
                        "answers": record["answers"],
                        "all_classes": record["all_classes"],
                        "official_length": record["length"],
                        "original_prompt_tokens": len(token_ids),
                        "prompt_tokens": len(selected),
                        "truncated": len(selected) < len(token_ids),
                        "maximum_tokens": caps[task],
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "input_ids_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                    }
                )
            print(f"prepared {task}: {len(chosen)} samples, cap={caps[task]}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_dir / "tokens.safetensors"))
    write_json(args.output_dir / "samples.json", rows)
    protocol = {
        "format": "basisserve.section4.llama_longbench_inputs.v1",
        "tasks": list(TASKS),
        "samples_per_task": args.samples_per_task,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "sample_selection": "sort SHA256(seed:task:source_id), fixed before predictions",
        "dataset_repo": "THUDM/LongBench",
        "dataset_revision": args.dataset_revision,
        "data_zip_sha256": sha256(archive),
        "task_jsonl_sha256": source_hashes,
        "official_root": str(args.official_root.resolve()),
        "official_revision": subprocess.check_output(
            ["git", "-C", str(args.official_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "official_sha256": {
            name: sha256(official / name)
            for name in ("config/dataset2prompt.json", "config/dataset2maxlen.json", "eval.py", "metrics.py")
        },
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "tokenizer_config_sha256": sha256(args.model / "tokenizer_config.json"),
        "prompt": "official task template as one user message; tokenizer chat template with generation prompt",
        "truncation": "token-level equal prefix/suffix retention; total limit includes official generation cap",
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(
        args.output_dir / "manifest.json",
        {
            "status": "complete",
            "protocol": protocol,
            "tokens_sha256": sha256(args.output_dir / "tokens.safetensors"),
            "samples_sha256": sha256(args.output_dir / "samples.json"),
        },
    )
    print(
        {
            "count": len(rows),
            "truncated": sum(row["truncated"] for row in rows),
            "minimum_tokens": min(row["prompt_tokens"] for row in rows),
            "maximum_tokens": max(row["prompt_tokens"] for row in rows),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
