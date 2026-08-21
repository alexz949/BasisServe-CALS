#!/usr/bin/env python3
"""Collect full post-gate activation moments for Qwen3.5 GDN Wo wires."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_gdn_state import Qwen35GDNGeometry
from basisserve.core.qwen35_wo_wire import FullSecondMoment
from scripts.collect_qwen35_gdn_moments import (
    _batches,
    _decoder_layers,
    _dtype,
    _fixed_length_samples,
)


FORMAT = "basisserve.qwen35.gdn_wo_activation_moments.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--moment-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


class _WoActivationCollector:
    def __init__(
        self,
        *,
        layer_index: int,
        module: nn.Module,
        width: int,
        moment_dtype: torch.dtype,
    ) -> None:
        self.layer_index = int(layer_index)
        self.module = module
        self.width = int(width)
        self.moments = FullSecondMoment(
            self.width,
            device=module.out_proj.weight.device,
            dtype=moment_dtype,
        )
        self.current_shape: tuple[int, int] | None = None
        self.calls = 0
        self.handle = module.norm.register_forward_hook(self._hook)

    def begin(self, batch: int, tokens: int) -> None:
        if self.current_shape is not None:
            raise RuntimeError("Wo activation collector began twice")
        self.current_shape = (int(batch), int(tokens))

    def finish(self) -> None:
        if self.current_shape is None:
            raise RuntimeError(f"layer {self.layer_index} norm hook did not run")
        self.current_shape = None

    @torch.no_grad()
    def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
        if self.current_shape is None:
            raise RuntimeError("GDN norm hook ran outside an active collector batch")
        if not isinstance(output, Tensor):
            raise TypeError("GDN gated norm output must be a tensor")
        batch, tokens = self.current_shape
        expected = batch * tokens * self.width
        if output.numel() != expected:
            raise ValueError(
                f"layer {self.layer_index} post-gate output has {output.numel()} values, "
                f"expected {expected}"
            )
        rows = output.reshape(batch * tokens, self.width)
        self.moments.update(rows)
        self.calls += 1

    def close(self) -> None:
        self.handle.remove()

    def state_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "calls": self.calls,
            "out_proj_shape": tuple(map(int, self.module.out_proj.weight.shape)),
            "moments": self.moments.state_dict(),
        }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Wo activation moments: {output_path}")
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
    width = geometry.num_value_heads * geometry.value_head_dim
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
    collectors: list[_WoActivationCollector] = []
    for layer_index, layer in enumerate(_decoder_layers(model)):
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        collectors.append(
            _WoActivationCollector(
                layer_index=layer_index,
                module=linear_attn,
                width=width,
                moment_dtype=_dtype(args.moment_dtype),
            )
        )
    expected_layers = sum(
        layer_type == "linear_attention" for layer_type in config.text_config.layer_types
    )
    if len(collectors) != expected_layers:
        raise ValueError(f"found {len(collectors)} GDN modules, expected {expected_layers}")

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
                    f"[Wo moments] batch={batch_index} samples={processed}/{len(samples)} "
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
            "value_head_dim": geometry.value_head_dim,
            "wire_input_width": width,
            "hidden_size": int(config.text_config.hidden_size),
        },
        "collection": {
            "dataset_jsonl": str(dataset_path),
            "text_field": args.text_field,
            "num_samples": processed,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "model_dtype": args.dtype,
            "moment_dtype": args.moment_dtype,
            "centered": False,
            "signal": "post_gated_rmsnorm_input_to_out_proj",
            "records": list(records),
        },
        "layers": [collector.state_dict() for collector in collectors],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[Saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
