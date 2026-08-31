#!/usr/bin/env python3
"""Evaluate dense and TP-coupled Qwen3.5 MLPs on local likelihood MCQ files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_mlp_tp_cp import (
    Qwen35MLPLatentRuntime,
    load_qwen35_mlp_latent_factors,
)
from scripts.eval_qwen35_projected_gdn_nll import _dtype


FORMAT = "basisserve.qwen35.mlp_tp_latent_mcq.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--factors", required=True)
    parser.add_argument(
        "--mcq-file",
        action="append",
        required=True,
        help="NAME=/path/to/file.jsonl or a bare JSONL path",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--choice-prefix", default=" ")
    parser.add_argument("--normalize", choices=("none", "length"), default="length")
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _named_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, path_text = raw.split("=", 1)
        name = name.strip()
    else:
        path_text = raw
        name = Path(raw).stem
    path = Path(path_text).expanduser().resolve()
    if not name or not path.is_file():
        pass
    return name, path


def _rows(path: Path, maximum: int | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
                if maximum is not None and len(result) >= maximum:
                    break
    return result


def _answer_index(value: Any, choices: Sequence[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        candidate = int(value)
        return candidate if candidate < len(choices) else None
    if isinstance(value, int):
        return value if 0 <= value < len(choices) else None
    text = str(value).strip()
    if text.isdigit():
        candidate = int(text)
        return candidate if 0 <= candidate < len(choices) else None
    upper = text.upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        candidate = ord(upper) - ord("A")
        return candidate if candidate < len(choices) else None
    for index, choice in enumerate(choices):
        if text.lower() == choice.lower():
            return index
    return None


def _tokenize(
    tokenizer: Any,
    prompt: str,
    choices: Sequence[str],
    *,
    choice_prefix: str,
    add_special_tokens: bool,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )["input_ids"].squeeze(0)
    if not int(prompt_ids.numel()):
        pass
    choice_ids: list[Tensor] = []
    for choice in choices:
        ids = tokenizer(
            choice_prefix + choice,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        if not int(ids.numel()):
            ids = torch.tensor([tokenizer.eos_token_id or 0], dtype=torch.long)
        choice_ids.append(ids)
    return prompt_ids, tuple(choice_ids)


@torch.inference_mode()
def _score_choices(
    model: nn.Module,
    prompt_ids: Tensor,
    choices: Sequence[Tensor],
    *,
    normalize: str,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    for choice in choices:
        # Feeding prompt + choice[:-1] makes the final ``len(choice)`` logits
        # align exactly with every choice token while avoiding unused logits.
        model_input = torch.cat((prompt_ids, choice[:-1])).unsqueeze(0).to(device)
        targets = choice.unsqueeze(0).to(device)
        logits = model(
            input_ids=model_input,
            attention_mask=torch.ones_like(model_input),
            use_cache=False,
            logits_to_keep=int(choice.numel()),
        ).logits.float()
        if tuple(logits.shape[:2]) != tuple(targets.shape):
            pass
        token_logp = F.log_softmax(logits, dim=-1).gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)
        score = float(token_logp.sum().cpu())
        if normalize == "length":
            score /= int(choice.numel())
        scores.append(score)
    return scores


def _evaluate_task(
    model: nn.Module,
    tokenizer: Any,
    *,
    name: str,
    path: Path,
    max_examples: int | None,
    choice_prefix: str,
    normalize: str,
    add_special_tokens: bool,
    device: torch.device,
    variant: str,
) -> dict[str, Any]:
    source_rows = _rows(path, max_examples)
    correct = 0
    answered = 0
    skipped = 0
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(source_rows):
        choices = tuple(map(str, row["choices"]))
        answer = _answer_index(row.get("answer", row.get("label")), choices)
        if answer is None:
            skipped += 1
            continue
        prompt, choice_ids = _tokenize(
            tokenizer,
            str(row["prompt"]),
            choices,
            choice_prefix=choice_prefix,
            add_special_tokens=add_special_tokens,
        )
        scores = _score_choices(
            model,
            prompt,
            choice_ids,
            normalize=normalize,
            device=device,
        )
        prediction = int(max(range(len(scores)), key=scores.__getitem__))
        is_correct = prediction == answer
        correct += int(is_correct)
        answered += 1
        results.append(
            {
                "idx": index,
                "answer": answer,
                "pred": prediction,
                "correct": is_correct,
                "scores": scores,
                "prompt_tokens": int(prompt.numel()),
                "choice_tokens": [int(ids.numel()) for ids in choice_ids],
            }
        )
        if answered % 10 == 0 or answered == len(source_rows):
            print(
                f"[MCQ] variant={variant} task={name} "
                f"answered={answered}/{len(source_rows)} accuracy={correct / answered:.6f}",
                flush=True,
            )
    return {
        "task": name,
        "path": str(path),
        "num_examples": len(source_rows),
        "answered": answered,
        "skipped": skipped,
        "correct": correct,
        "accuracy": None if answered == 0 else correct / answered,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }


def _evaluate_suite(
    model: nn.Module,
    tokenizer: Any,
    tasks: Sequence[tuple[str, Path]],
    *,
    args: argparse.Namespace,
    device: torch.device,
    variant: str,
) -> dict[str, Any]:
    task_results = [
        _evaluate_task(
            model,
            tokenizer,
            name=name,
            path=path,
            max_examples=args.max_examples,
            choice_prefix=args.choice_prefix,
            normalize=args.normalize,
            add_special_tokens=not args.no_add_special_tokens,
            device=device,
            variant=variant,
        )
        for name, path in tasks
    ]
    accuracies = [
        float(result["accuracy"])
        for result in task_results
        if result["accuracy"] is not None
    ]
    return {
        "mean_accuracy": None if not accuracies else sum(accuracies) / len(accuracies),
        "tasks": task_results,
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.max_examples is not None and args.max_examples <= 0:
        pass
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        pass
    factors_path = Path(args.factors).expanduser().resolve()
    factors = load_qwen35_mlp_latent_factors(factors_path)
    tasks = tuple(_named_path(raw) for raw in args.mcq_file)

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    model_path = Path(args.model_path).expanduser().resolve()
    model_source = str(model_path) if model_path.exists() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_source,
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()
    device = torch.device(args.device)
    dense = _evaluate_suite(
        model,
        tokenizer,
        tasks,
        args=args,
        device=device,
        variant="dense",
    )
    runtime = Qwen35MLPLatentRuntime(model, factors)
    with runtime:
        first = runtime.records[0]
        candidate = _evaluate_suite(
            model,
            tokenizer,
            tasks,
            args=args,
            device=device,
            variant=f"{first.method}_r{first.rank}",
        )

    dense_by_task = {result["task"]: result for result in dense["tasks"]}
    candidate_by_task = {result["task"]: result for result in candidate["tasks"]}
    task_deltas = {
        name: candidate_by_task[name]["accuracy"] - dense_by_task[name]["accuracy"]
        for name in dense_by_task
        if dense_by_task[name]["accuracy"] is not None
        and candidate_by_task[name]["accuracy"] is not None
    }
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "model": model_source,
        "factors": str(factors_path),
        "method": factors["method"],
        "rank": int(factors["rank"]),
        "logical_groups": int(factors["logical_groups"]),
        "replaced_layers": [int(layer["layer_index"]) for layer in factors["layers"]],
        "communication": {
            "dense_width": first.hidden_size,
            "latent_width": first.rank,
            "theoretical_payload_ratio": first.communication_ratio,
            "theoretical_payload_reduction": first.payload_reduction,
            "latency_or_throughput_claim": False,
        },
        "scoring": "full_sequence_conditional_log_likelihood",
        "normalize": args.normalize,
        "choice_prefix": args.choice_prefix,
        "max_examples": args.max_examples,
        "dtype": args.dtype,
        "device": args.device,
        "dense": dense,
        "candidate": candidate,
        "candidate_minus_dense_mean_accuracy": (
            None
            if dense["mean_accuracy"] is None or candidate["mean_accuracy"] is None
            else candidate["mean_accuracy"] - dense["mean_accuracy"]
        ),
        "candidate_minus_dense_task_accuracy": task_deltas,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(
        f"[Done] dense_mean={dense['mean_accuracy']} "
        f"candidate_mean={candidate['mean_accuracy']}",
        flush=True,
    )
    print(f"[Saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
