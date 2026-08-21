#!/usr/bin/env python3
"""Evaluate PPL for dense or converted SVDLLM safetensors checkpoints.

If the checkpoint folder contains svdllm_config.json, the loader reconstructs
SVDLinear modules on an empty model before dispatching weights across devices.
Use --dense to evaluate a normal HuggingFace checkpoint without SVDLinear.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class SVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.v_proj = nn.Linear(self.in_features, self.rank, bias=False)
        self.u_proj = nn.Linear(self.rank, self.out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.u_proj(self.v_proj(x))
        if self.bias is not None:
            out = out + self.bias
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer", default=None, help="defaults to --checkpoint-dir")
    parser.add_argument(
        "--dense",
        action="store_true",
        help="evaluate a normal dense HF checkpoint; do not require svdllm_config.json",
    )
    parser.add_argument("--dataset", default="wikitext2", help="wikitext2, ptb, c4, or a text file")
    parser.add_argument("--split", default=None, help="dataset split override")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="cap number of seqlen chunks")
    parser.add_argument("--max-tokens", type=int, default=None, help="cap token stream before chunking")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16, float16, float32, auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
        help="attention backend; auto preserves the Transformers default",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--max-memory",
        nargs="*",
        default=None,
        help="device memory caps, e.g. 0=22GiB 1=22GiB cpu=200GiB or '0=22GiB,1=22GiB'",
    )
    parser.add_argument(
        "--no-split-module-classes",
        default="Qwen3DecoderLayer,LlamaDecoderLayer,MistralDecoderLayer,OPTDecoderLayer",
    )
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--offload-buffers", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype | None:
    normalized = (name or "auto").lower()
    if normalized in {"auto", "none"}:
        return None
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _parse_max_memory(items: list[str] | None) -> dict[int | str, str] | None:
    if not items:
        return None
    parts: list[str] = []
    for item in items:
        parts.extend(part.strip() for part in item.split(",") if part.strip())
    result: dict[int | str, str] = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            raise ValueError(f"invalid --max-memory entry: {part}")
        key = key.strip()
        value = value.strip()
        result["cpu" if key == "cpu" else int(key)] = value
    return result


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [part.strip() for part in value.split(",") if part.strip()]
    return values or None


def _load_svdllm_config(path: Path) -> dict[str, Any]:
    config_path = path / "svdllm_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def _replace_module(root: nn.Module, name: str, new_module: nn.Module) -> None:
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = name
    setattr(parent, child_name, new_module)


def _apply_svd_modules(model: nn.Module, svdllm_config: dict[str, Any]) -> None:
    specs = svdllm_config.get("svd_modules")
    if not isinstance(specs, list) or not specs:
        raise ValueError("svdllm_config.json does not contain svd_modules")

    replaced = 0
    for spec in specs:
        name = str(spec["name"])
        old = model.get_submodule(name)
        if not isinstance(old, nn.Linear):
            raise TypeError(f"expected dense Linear at {name}, got {type(old)!r}")
        in_features = int(spec["in_features"])
        out_features = int(spec["out_features"])
        if old.in_features != in_features or old.out_features != out_features:
            raise ValueError(
                f"shape mismatch for {name}: base=({old.in_features},{old.out_features}) "
                f"svd=({in_features},{out_features})"
            )
        _replace_module(
            model,
            name,
            SVDLinear(
                in_features=in_features,
                out_features=out_features,
                rank=int(spec["rank"]),
                bias=bool(spec.get("bias", False)),
            ),
        )
        replaced += 1
    print(f"[Load] replaced {replaced} modules with SVDLinear", flush=True)


def _device_from_map_value(value: Any) -> torch.device | None:
    if value is None:
        return None
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    text = str(value)
    if text == "disk":
        return None
    if text.isdigit():
        return torch.device(f"cuda:{text}")
    return torch.device(text)


def _input_device(model: nn.Module) -> torch.device:
    device_map = getattr(model, "hf_device_map", None) or {}
    for key in ("model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"):
        if key in device_map:
            device = _device_from_map_value(device_map[key])
            if device is not None:
                return device
    for value in device_map.values():
        device = _device_from_map_value(value)
        if device is not None and device.type == "cuda":
            return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_text(dataset: str, split: str | None) -> str:
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split or "test")
        return "\n\n".join(str(text) for text in ds["text"])
    if dataset == "ptb":
        try:
            ds = load_dataset("ptb_text_only", "penn_treebank", split=split or "test")
            return "\n\n".join(str(text) for text in ds["sentence"])
        except RuntimeError as exc:
            if "Dataset scripts are no longer supported" not in str(exc):
                raise
            ds = load_dataset("allenai/paloma", "ptb", split=split or "test")
            return "\n\n".join(str(text) for text in ds["text"])
    path = Path(dataset)
    if not path.exists():
        raise ValueError(f"unknown dataset or missing text file: {dataset}")
    return path.read_text(encoding="utf-8")


def _token_ids(
    tokenizer: Any,
    dataset: str,
    split: str | None,
    max_tokens: int | None,
) -> torch.Tensor:
    text = _load_text(dataset, split)
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded.input_ids
    if max_tokens is not None:
        input_ids = input_ids[:, : int(max_tokens)]
    return input_ids


def _iter_batches(input_ids: torch.Tensor, seqlen: int, batch_size: int, max_samples: int | None):
    nsamples = int(input_ids.numel() // seqlen)
    if max_samples is not None:
        nsamples = min(nsamples, int(max_samples))
    for start in range(0, nsamples, batch_size):
        end = min(nsamples, start + batch_size)
        chunks = [
            input_ids[:, sample_idx * seqlen : (sample_idx + 1) * seqlen].squeeze(0)
            for sample_idx in range(start, end)
        ]
        yield start, torch.stack(chunks, dim=0)


def load_model(args: argparse.Namespace, checkpoint_dir: Path, svdllm_config: dict[str, Any]) -> nn.Module:
    dtype = _torch_dtype(args.dtype)
    config = AutoConfig.from_pretrained(
        checkpoint_dir,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if isinstance(dtype, torch.dtype):
        config.torch_dtype = dtype

    print(f"[Load] config model_type={getattr(config, 'model_type', None)}", flush=True)
    with init_empty_weights():
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(args.trust_remote_code),
        }
        attn_implementation = getattr(args, "attn_implementation", "auto")
        if attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_config(config, **model_kwargs)
        if not args.dense:
            _apply_svd_modules(model, svdllm_config)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    max_memory = _parse_max_memory(args.max_memory)
    no_split = _split_csv(args.no_split_module_classes)
    offload_folder = args.offload_folder
    if offload_folder is not None:
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
    print(f"[Load] device_map={args.device_map}", flush=True)
    print(f"[Load] max_memory={max_memory}", flush=True)
    print(f"[Load] no_split_module_classes={no_split}", flush=True)
    return load_checkpoint_and_dispatch(
        model,
        checkpoint=str(checkpoint_dir),
        device_map=args.device_map,
        max_memory=max_memory,
        no_split_module_classes=no_split,
        offload_folder=offload_folder,
        offload_buffers=bool(args.offload_buffers),
        dtype=dtype,
        strict=bool(args.strict_load),
    )


@torch.no_grad()
def eval_ppl(
    model: nn.Module,
    tokenizer: Any,
    *,
    dataset: str,
    split: str | None,
    seqlen: int,
    batch_size: int,
    max_samples: int | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    input_ids = _token_ids(tokenizer, dataset, split, max_tokens)
    nsamples_total = int(input_ids.numel() // seqlen)
    nsamples = nsamples_total if max_samples is None else min(nsamples_total, int(max_samples))
    if nsamples <= 0:
        raise ValueError("no complete evaluation chunks; reduce --seqlen or increase --max-tokens")

    input_device = _input_device(model)
    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    model.eval()

    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nll_sum = 0.0
    token_count = 0
    print(
        f"[Eval] dataset={dataset} split={split or 'default'} seqlen={seqlen} "
        f"batch_size={batch_size} chunks={nsamples}/{nsamples_total} input_device={input_device}",
        flush=True,
    )
    for _, batch in tqdm(
        _iter_batches(input_ids, seqlen, batch_size, max_samples),
        total=math.ceil(nsamples / batch_size),
    ):
        batch = batch.to(input_device)
        outputs = model(input_ids=batch, use_cache=False)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous().to(shift_logits.device)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nll_sum += float(loss.detach().float().cpu())
        token_count += int(shift_labels.numel())

    if use_cache is not None:
        model.config.use_cache = use_cache
    ppl = math.exp(nll_sum / token_count)
    return {
        "dataset": dataset,
        "split": split,
        "seqlen": seqlen,
        "batch_size": batch_size,
        "chunks": nsamples,
        "tokens": token_count,
        "nll_sum": nll_sum,
        "ppl": ppl,
    }


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve() if args.tokenizer else checkpoint_dir

    svdllm_config = {} if args.dense else _load_svdllm_config(checkpoint_dir)
    print(f"[Config] checkpoint_dir={checkpoint_dir}", flush=True)
    print(f"[Config] tokenizer={tokenizer_path}", flush=True)
    print(f"[Config] dense={bool(args.dense)}", flush=True)
    print(f"[Config] attn_implementation={args.attn_implementation}", flush=True)
    if not args.dense:
        print(f"[Config] svd_modules={svdllm_config.get('num_svd_modules')}", flush=True)
        print(f"[Config] targets={svdllm_config.get('targets')}", flush=True)
        print(f"[Config] layers={svdllm_config.get('layers')}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = load_model(args, checkpoint_dir, svdllm_config)
    print(f"[Load] hf_device_map={getattr(model, 'hf_device_map', None)}", flush=True)

    result = eval_ppl(
        model,
        tokenizer,
        dataset=args.dataset,
        split=args.split,
        seqlen=int(args.seqlen),
        batch_size=int(args.batch_size),
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    result["attention_implementation"] = args.attn_implementation
    result["dtype"] = args.dtype
    print(f"[Result] ppl={result['ppl']:.6f}", flush=True)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[Write] {output_path}", flush=True)


if __name__ == "__main__":
    main()
