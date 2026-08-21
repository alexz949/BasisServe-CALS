#!/usr/bin/env python3
"""Collect compact Qwen3.5 GDN state/gate interface moments.

The collector hooks each Gated DeltaNet's gated RMSNorm boundary and stores
per-layer, per-value-head moments for:

* ``core``: recurrent readout before gated RMSNorm;
* ``gate``: ``SiLU(z)`` at the dynamic channel gate;
* ``post_norm``: exact gated-RMSNorm output entering ``out_proj``.

It deliberately does not alter the model or cache.  The resulting artifact is
the Phase-0 input for nested state-codec and gate-spectrum analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_gdn_state import (
    HeadwiseSecondMoment,
    Qwen35GDNGeometry,
)


FORMAT = "basisserve.qwen35.gdn_moments.v1"
SIGNALS = ("core", "gate", "post_norm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--signals",
        default=",".join(SIGNALS),
        help=f"comma-separated subset of {SIGNALS}",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--moment-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=8)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _checked_signals(raw: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(piece.strip() for piece in raw.split(",") if piece.strip()))
    if not values or any(value not in SIGNALS for value in values):
        raise ValueError(f"signals must be a nonempty subset of {SIGNALS}")
    return values


def _fixed_length_samples(
    path: Path,
    tokenizer: Any,
    *,
    text_field: str,
    sequence_length: int,
    num_samples: int,
) -> tuple[tuple[Tensor, ...], tuple[dict[str, int], ...]]:
    if sequence_length <= 0 or num_samples <= 0:
        raise ValueError("sequence length and sample count must be positive")
    samples: list[Tensor] = []
    records: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            payload = json.loads(line)
            text = payload.get(text_field)
            if not isinstance(text, str) or not text:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < sequence_length:
                continue
            samples.append(torch.tensor(token_ids[:sequence_length], dtype=torch.long))
            records.append(
                {
                    "line_index": line_index,
                    "available_tokens": len(token_ids),
                    "used_tokens": sequence_length,
                }
            )
            if len(samples) == num_samples:
                break
    if len(samples) != num_samples:
        raise ValueError(
            f"dataset supplied only {len(samples)} samples with at least "
            f"{sequence_length} tokens; requested {num_samples}"
        )
    return tuple(samples), tuple(records)


def _batches(samples: Sequence[Tensor], batch_size: int) -> Iterator[Tensor]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(samples), batch_size):
        yield torch.stack(tuple(samples[start : start + batch_size]), dim=0)


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = (
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, nn.ModuleList):
            return candidate
    raise ValueError("could not locate Qwen3.5 language-model decoder layers")


class _LayerCollector:
    def __init__(
        self,
        *,
        layer_index: int,
        module: nn.Module,
        geometry: Qwen35GDNGeometry,
        signals: Sequence[str],
        moment_dtype: torch.dtype,
    ) -> None:
        self.layer_index = int(layer_index)
        self.module = module
        self.geometry = geometry
        self.signals = tuple(signals)
        device = module.out_proj.weight.device
        self.moments = {
            signal: HeadwiseSecondMoment(
                geometry.num_value_heads,
                geometry.value_head_dim,
                device=device,
                dtype=moment_dtype,
            )
            for signal in self.signals
        }
        self.current_shape: tuple[int, int] | None = None
        self.calls = 0
        self.handle = module.norm.register_forward_hook(self._hook)

    def begin(self, batch: int, tokens: int) -> None:
        if self.current_shape is not None:
            raise RuntimeError("collector began a second batch before finishing")
        self.current_shape = (int(batch), int(tokens))

    def finish(self) -> None:
        if self.current_shape is None:
            raise RuntimeError(f"layer {self.layer_index} norm hook did not run")
        self.current_shape = None

    @torch.no_grad()
    def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
        if self.current_shape is None:
            raise RuntimeError("GDN norm hook ran outside an active collector batch")
        if len(inputs) < 2 or not isinstance(inputs[0], Tensor) or not isinstance(inputs[1], Tensor):
            raise TypeError("GDN gated norm must receive core output and gate tensors")
        batch, tokens = self.current_shape
        heads = self.geometry.num_value_heads
        width = self.geometry.value_head_dim
        expected_rows = batch * tokens * heads
        core, gate = inputs[0], inputs[1]
        if core.numel() != expected_rows * width or gate.shape != core.shape:
            raise ValueError(
                f"layer {self.layer_index} GDN norm shape mismatch: "
                f"core={tuple(core.shape)} gate={tuple(gate.shape)}"
            )
        if not isinstance(output, Tensor) or output.shape != core.shape:
            raise ValueError("GDN norm output does not match its core input")
        rows = {
            "core": core.reshape(batch * tokens, heads, width),
            "gate": F.silu(gate.float()).reshape(batch * tokens, heads, width),
            "post_norm": output.reshape(batch * tokens, heads, width),
        }
        for signal, moments in self.moments.items():
            moments.update(rows[signal])
        self.calls += 1

    def close(self) -> None:
        self.handle.remove()

    def state_dict(self) -> dict[str, Any]:
        norm_weight = getattr(self.module.norm, "weight", None)
        return {
            "layer_index": self.layer_index,
            "calls": self.calls,
            "norm_class": type(self.module.norm).__name__,
            "norm_eps": float(getattr(self.module.norm, "variance_epsilon", 1e-6)),
            "norm_weight": (
                None if norm_weight is None else norm_weight.detach().cpu().contiguous()
            ),
            "out_proj_shape": tuple(map(int, self.module.out_proj.weight.shape)),
            "moments": {
                signal: moments.state_dict() for signal, moments in self.moments.items()
            },
        }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    signals = _checked_signals(args.signals)
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite GDN moments: {output_path}")
    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer

    model_path = Path(args.model_path).expanduser().resolve()
    model_source = str(model_path) if model_path.exists() else args.model_path
    config = AutoConfig.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    geometry = Qwen35GDNGeometry.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    samples, records = _fixed_length_samples(
        dataset_path,
        tokenizer,
        text_field=args.text_field,
        sequence_length=args.sequence_length,
        num_samples=args.num_samples,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_source,
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()
    layers = _decoder_layers(model)
    collectors: list[_LayerCollector] = []
    for layer_index, layer in enumerate(layers):
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        collectors.append(
            _LayerCollector(
                layer_index=layer_index,
                module=linear_attn,
                geometry=geometry,
                signals=signals,
                moment_dtype=_dtype(args.moment_dtype),
            )
        )
    expected_linear_layers = sum(
        layer_type == "linear_attention" for layer_type in config.text_config.layer_types
    )
    if len(collectors) != expected_linear_layers:
        raise ValueError(
            f"found {len(collectors)} GDN modules, expected {expected_linear_layers}"
        )

    processed = 0
    try:
        with torch.inference_mode():
            for batch_index, input_ids in enumerate(
                _batches(samples, args.batch_size), start=1
            ):
                input_ids = input_ids.to(args.device)
                for collector in collectors:
                    collector.begin(*map(int, input_ids.shape))
                model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    logits_to_keep=1,
                )
                for collector in collectors:
                    collector.finish()
                processed += int(input_ids.shape[0])
                print(
                    f"[GDN moments] batch={batch_index} samples={processed}/{len(samples)} "
                    f"tokens={processed * args.sequence_length}",
                    flush=True,
                )
    finally:
        for collector in collectors:
            collector.close()

    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "model": {
            "source": model_source,
            "config_model_type": config.model_type,
            "commit_hash": getattr(config, "_commit_hash", None),
        },
        "geometry": {
            "num_value_heads": geometry.num_value_heads,
            "num_key_heads": geometry.num_key_heads,
            "key_head_dim": geometry.key_head_dim,
            "value_head_dim": geometry.value_head_dim,
        },
        "collection": {
            "dataset_jsonl": str(dataset_path),
            "text_field": args.text_field,
            "num_samples": processed,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "signals": list(signals),
            "model_dtype": args.dtype,
            "moment_dtype": args.moment_dtype,
            "records": list(records),
        },
        "layers": [collector.state_dict() for collector in collectors],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[Saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
