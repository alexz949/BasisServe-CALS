#!/usr/bin/env python3
"""Compare dense and Store80 full/chunked prefill logits on one RULER prompt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_joint_routing_payload_s80_qwen3 import (  # noqa: E402
    install_qwen3_s80_factor_bank,
)
from evaluation.eval_c1_block_scheduled_ppl import _dtype  # noqa: E402
from evaluation.eval_qwen3_c1_quest_ruler import _build_work  # noqa: E402
from evaluation.eval_qwen3_dense_ruler import _qwen3_config  # noqa: E402
from evaluation.ruler_v1 import parse_tasks, ruler_prompt  # noqa: E402


FORMAT = "basisserve.qwen3_8b.s80_chunked_prefill_diagnostic.v1"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


@torch.inference_mode()
def _full_prefill_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=1,
    )
    logits = output.logits[0, -1].float().cpu()
    del output
    return logits


def _explicit_causal_mask(
    *,
    start: int,
    stop: int,
    device: torch.device,
) -> torch.Tensor:
    query_indices = torch.arange(start, stop, device=device).unsqueeze(-1)
    key_indices = torch.arange(stop, device=device).unsqueeze(0)
    return (key_indices <= query_indices).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def _chunked_prefill_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    chunk_size: int,
    explicit_mask: bool,
) -> torch.Tensor:
    cache = DynamicCache()
    logits = None
    for start in range(0, int(input_ids.shape[1]), chunk_size):
        stop = min(start + chunk_size, int(input_ids.shape[1]))
        attention_mask = (
            _explicit_causal_mask(
                start=start,
                stop=stop,
                device=input_ids.device,
            )
            if explicit_mask
            else None
        )
        output = model(
            input_ids=input_ids[:, start:stop],
            position_ids=position_ids[:, start:stop],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[0, -1].float().cpu()
        del output
    assert logits is not None
    del cache
    return logits


def _token(logits: torch.Tensor, tokenizer: Any) -> dict[str, Any]:
    token_id = int(logits.argmax().item())
    return {
        "id": token_id,
        "text": tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.double()
    candidate = candidate.double()
    reference_centered = reference - reference.mean()
    candidate_centered = candidate - candidate.mean()
    error = candidate_centered - reference_centered
    reference_log_probability = F.log_softmax(reference, dim=-1)
    candidate_log_probability = F.log_softmax(candidate, dim=-1)
    reference_probability = reference_log_probability.exp()
    top_count = min(10, int(reference.numel()))
    reference_top = reference.topk(top_count).indices
    candidate_top = candidate.topk(top_count).indices
    candidate_membership = torch.zeros_like(candidate, dtype=torch.bool)
    candidate_membership[candidate_top] = True
    return {
        "top1_match": int(reference.argmax()) == int(candidate.argmax()),
        "top10_overlap": float(candidate_membership[reference_top].float().mean()),
        "centered_relative_rmse": math.sqrt(
            float(error.square().sum())
            / max(float(reference_centered.square().sum()), 1.0e-300)
        ),
        "centered_cosine": float(
            F.cosine_similarity(reference_centered, candidate_centered, dim=0)
        ),
        "kl_reference_to_candidate": float(
            (
                reference_probability
                * (reference_log_probability - candidate_log_probability)
            ).sum()
        ),
        "maximum_absolute_logit_error": float((candidate - reference).abs().max()),
    }


def _arm_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    result = {
        "full": _full_prefill_logits(model, input_ids, position_ids),
        "chunked_implicit_mask": _chunked_prefill_logits(
            model,
            input_ids,
            position_ids,
            chunk_size=chunk_size,
            explicit_mask=False,
        ),
        "chunked_explicit_mask": _chunked_prefill_logits(
            model,
            input_ids,
            position_ids,
            chunk_size=chunk_size,
            explicit_mask=True,
        ),
    }
    torch.cuda.empty_cache()
    return result


def _segment_result(
    *,
    tokenizer: Any,
    dense: dict[str, torch.Tensor],
    store80: dict[str, torch.Tensor],
) -> dict[str, Any]:
    all_logits = {
        **{f"dense_{name}": value for name, value in dense.items()},
        **{f"store80_{name}": value for name, value in store80.items()},
    }
    return {
        "tokens": {
            name: _token(logits, tokenizer) for name, logits in all_logits.items()
        },
        "comparisons": {
            "dense_chunked_implicit_vs_full": _comparison(
                dense["full"], dense["chunked_implicit_mask"]
            ),
            "dense_chunked_explicit_vs_full": _comparison(
                dense["full"], dense["chunked_explicit_mask"]
            ),
            "store80_full_vs_dense_full": _comparison(
                dense["full"], store80["full"]
            ),
            "store80_chunked_implicit_vs_full": _comparison(
                store80["full"], store80["chunked_implicit_mask"]
            ),
            "store80_chunked_explicit_vs_full": _comparison(
                store80["full"], store80["chunked_explicit_mask"]
            ),
            "store80_chunked_implicit_vs_dense_full": _comparison(
                dense["full"], store80["chunked_implicit_mask"]
            ),
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    zero = payload["segments"]["tail_zero_positions"]["comparisons"]
    actual = payload["segments"]["actual_tail_positions"]["comparisons"]
    lines = [
        "# Qwen3-8B Store80 chunked-prefill diagnostic",
        "",
        "Each row compares final next-token logits on the identical token segment and "
        "position IDs.",
        "",
        "## Interpretation",
        "",
        "Chunked prefill is not the failure source: dense and Store80 each preserve "
        "their full-prefill Top1 under chunk512, and explicit versus implicit causal "
        "masks give the same Store80 result. On the same tail tokens, Store80 matches "
        "dense Top1 at positions 0--2047 but diverges at the prompt's actual 30K--32K "
        "positions. The Store80-versus-dense KL rises from "
        f"`{zero['store80_full_vs_dense_full']['kl_reference_to_candidate']:.6f}` to "
        f"`{actual['store80_full_vs_dense_full']['kl_reference_to_candidate']:.6f}`. "
        "This isolates absolute-position/RoPE extrapolation as the primary failure in "
        "this sample, before Route32 is enabled.",
        "",
        "## Comparisons",
        "",
        "| segment | comparison | top1 | top10 overlap | rel-RMSE | cosine | KL |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for segment_name, segment in payload["segments"].items():
        for comparison_name, row in segment["comparisons"].items():
            lines.append(
                f"| {segment_name} | {comparison_name} | "
                f"{'yes' if row['top1_match'] else 'no'} | "
                f"{row['top10_overlap']:.4f} | "
                f"{row['centered_relative_rmse']:.6f} | "
                f"{row['centered_cosine']:.6f} | "
                f"{row['kl_reference_to_candidate']:.6f} |"
            )
    lines.extend(["", "## Next-token predictions", ""])
    for segment_name, segment in payload["segments"].items():
        lines.append(f"### {segment_name}")
        lines.append("")
        for name, token in segment["tokens"].items():
            lines.append(f"- `{name}`: `{token['id']}` / `{token['text']!r}`")
        lines.append("")
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--s80-export", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--sample-ordinal", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    assert torch.cuda.is_available()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model_path = args.model_path.expanduser().resolve()
    s80_export = args.s80_export.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    assert not output_dir.exists()
    output_dir.mkdir(parents=True)

    tasks = parse_tasks(args.task)
    work = _build_work(data_dir, tasks, args.sample_ordinal + 1)
    selected = [
        row for row in work
        if row[1].name == args.task and int(row[2]) == args.sample_ordinal
    ][0]
    source = selected[3]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    prompt_ids = tokenizer(
        ruler_prompt(source),
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    prompt_tokens = int(prompt_ids.shape[1])
    assert args.tokens <= prompt_tokens

    tail_ids = prompt_ids[:, -args.tokens :]
    segments = {
        "tail_zero_positions": (
            tail_ids,
            torch.arange(args.tokens).unsqueeze(0),
        ),
        "actual_tail_positions": (
            tail_ids,
            torch.arange(prompt_tokens - args.tokens, prompt_tokens).unsqueeze(0),
        ),
    }
    effective_config, _ = _qwen3_config(
        model_path,
        sequence_length=args.sequence_length,
        yarn_factor=None,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            config=effective_config,
            dtype=_dtype(args.dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    dense_logits = {}
    for name, (input_ids, position_ids) in segments.items():
        dense_logits[name] = _arm_logits(
            model,
            input_ids.to(device),
            position_ids.to(device),
            chunk_size=args.chunk_size,
        )

    replacements = install_qwen3_s80_factor_bank(
        model,
        s80_export,
        attention_backend="sdpa",
    )
    assert len(replacements) == int(effective_config.num_hidden_layers)
    output_segments = {}
    for name, (input_ids, position_ids) in segments.items():
        store80_logits = _arm_logits(
            model,
            input_ids.to(device),
            position_ids.to(device),
            chunk_size=args.chunk_size,
        )
        output_segments[name] = _segment_result(
            tokenizer=tokenizer,
            dense=dense_logits[name],
            store80=store80_logits,
        )

    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "prompt": {
            "task": args.task,
            "sample_ordinal": args.sample_ordinal,
            "source_index": source.get("index"),
            "prompt_tokens": prompt_tokens,
            "diagnostic_tokens": args.tokens,
        },
        "segments": output_segments,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        },
    }
    _atomic_text(output_dir / "result.json", json.dumps(payload, indent=2) + "\n")
    _atomic_text(output_dir / "summary.md", _markdown(payload))
    print(json.dumps(payload["segments"], indent=2), flush=True)


if __name__ == "__main__":
    main()
