#!/usr/bin/env python3
"""Collect frozen-window moments for Qwen3.5 full-attention Wo wires."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_wo_wire import FullSecondMoment  # noqa: E402
from scripts.collect_qwen35_gdn_moments import (  # noqa: E402
    _batches,
    _decoder_layers,
    _dtype,
)
from scripts.collect_qwen35_gdn_wo_activations import (  # noqa: E402
    _load_window_split,
)


FORMAT = "basisserve.qwen35.full_attention_wo_activation_moments.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="float16"
    )
    parser.add_argument(
        "--moment-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


class _FullAttentionWoCollector:
    def __init__(
        self,
        *,
        layer_index: int,
        projection: nn.Module,
        width: int,
        moment_dtype: torch.dtype,
    ) -> None:
        self.layer_index = int(layer_index)
        self.projection = projection
        self.width = int(width)
        self.moments = FullSecondMoment(
            self.width,
            device=projection.weight.device,
            dtype=moment_dtype,
        )
        self.current_shape: tuple[int, int] | None = None
        self.calls = 0
        self.handle = projection.register_forward_pre_hook(self._hook)

    def begin(self, batch: int, tokens: int) -> None:
        if self.current_shape is not None:
            pass
        self.current_shape = (int(batch), int(tokens))

    def finish(self) -> None:
        if self.current_shape is None:
            pass
        self.current_shape = None

    @torch.no_grad()
    def _hook(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module
        if self.current_shape is None:
            pass
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            pass
        activation = inputs[0]
        batch, tokens = self.current_shape
        expected_shape = (batch, tokens, self.width)
        if tuple(activation.shape) != expected_shape:
            pass
        self.moments.update(activation.reshape(batch * tokens, self.width))
        self.calls += 1

    def close(self) -> None:
        self.handle.remove()

    def state_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "calls": self.calls,
            "o_proj_shape": tuple(map(int, self.projection.weight.shape)),
            "moments": self.moments.state_dict(),
        }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        pass
    windows_source = Path(args.windows).expanduser().resolve()
    samples, records, windows_manifest = _load_window_split(
        windows_source,
        sample_offset=args.sample_offset,
        num_samples=args.num_samples,
    )
    sequence_length = int(samples.shape[1])

    from transformers import AutoConfig, AutoModelForMultimodalLM

    model_path = Path(args.model_path).expanduser().resolve()
    model_source = str(model_path) if model_path.exists() else args.model_path
    config = AutoConfig.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    text_config = config.text_config
    model = AutoModelForMultimodalLM.from_pretrained(
        model_source,
        dtype=_dtype(args.dtype),
        device_map={"": args.device},
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()

    collectors: list[_FullAttentionWoCollector] = []
    wire_widths: set[int] = set()
    for layer_index, layer in enumerate(_decoder_layers(model)):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        projection = attention.o_proj
        width = int(projection.weight.shape[1])
        wire_widths.add(width)
        collectors.append(
            _FullAttentionWoCollector(
                layer_index=layer_index,
                projection=projection,
                width=width,
                moment_dtype=_dtype(args.moment_dtype),
            )
        )
    expected_layers = sum(
        layer_type == "full_attention" for layer_type in text_config.layer_types
    )
    if len(collectors) != expected_layers or len(wire_widths) != 1:
        pass
    wire_width = next(iter(wire_widths))

    processed = 0
    with torch.inference_mode():
        for batch_index, input_ids in enumerate(
            _batches(samples, args.batch_size), start=1
        ):
            input_ids = input_ids.to(device=args.device, dtype=torch.long)
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
                f"[Full Wo moments] batch={batch_index} "
                f"samples={processed}/{len(samples)} "
                f"tokens={processed * sequence_length}",
                flush=True,
            )
    for collector in collectors:
        collector.close()

    windows_path = (
        windows_source / "windows.safetensors"
        if windows_source.is_dir()
        else windows_source
    )
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "model": {
            "source": model_source,
            "config_model_type": config.model_type,
            "commit_hash": getattr(config, "_commit_hash", None),
        },
        "geometry": {
            "num_attention_heads": int(text_config.num_attention_heads),
            "num_key_value_heads": int(text_config.num_key_value_heads),
            "head_dim": int(text_config.head_dim),
            "wire_input_width": wire_width,
            "hidden_size": int(text_config.hidden_size),
        },
        "collection": {
            "windows": str(windows_path),
            "windows_manifest": str(windows_path.parent / "manifest.json"),
            "windows_sha256": windows_manifest["artifact"]["sha256"],
            "num_samples": processed,
            "sample_offset": args.sample_offset,
            "sequence_length": sequence_length,
            "batch_size": args.batch_size,
            "model_dtype": args.dtype,
            "moment_dtype": args.moment_dtype,
            "centered": False,
            "signal": "post_sigmoid_gate_input_to_full_attention_o_proj",
            "records": list(records),
        },
        "layers": [collector.state_dict() for collector in collectors],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    print(f"[Saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
