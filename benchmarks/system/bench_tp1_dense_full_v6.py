"""TP1 Llama-3.1-8B dense attention with local, K-only, or KV offload."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from basisserve.kernels.mapped_host_paged_attention import _load_extension
from evaluation.chunked_prefill_mlp import ChunkedTokenwise
from benchmarks.system.chunked_prefill_rope import apply_prefill_rope_


MODEL_PATH = Path(
    "/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
TOKENS_PATH = Path("/workspace/runs/l31-cal128/calibration/windows.safetensors")
LAYERS = 32
KV_HEADS = 8
HEAD_DIM = 128


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


class TP1DenseAttention(torch.nn.Module):
    """Exact Llama attention with local KV or per-token host K/KV transfer."""

    def __init__(
        self,
        original: torch.nn.Module,
        *,
        capacity: int,
        storage: str,
        host_extension: object | None,
        shared_staging: dict[str, torch.Tensor] | None,
        profile_components: bool,
        validate: bool,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.config = original.config
        self.layer_idx = int(original.layer_idx)
        self.head_dim = int(original.head_dim)
        self.scaling = float(original.scaling)
        self.attention_dropout = float(original.attention_dropout)
        self.capacity = int(capacity)
        self.storage = storage
        self.host_extension = host_extension
        self.shared_staging = shared_staging
        self.profile_components = bool(profile_components)
        self.validate = bool(validate)
        self.validated = False
        self.length = 0

        cache_shape = (batch_size, KV_HEADS, capacity, HEAD_DIM)
        if storage == "local":
            self.key_cache = torch.empty(
                cache_shape, device="cuda", dtype=torch.bfloat16
            )
            self.value_cache = torch.empty_like(self.key_cache)
            self.host_key = None
            self.host_value = None
        elif storage == "offload":
            assert host_extension is not None and shared_staging is not None
            self.key_cache = None
            self.value_cache = None
            self.host_key = host_extension.mapped_host_bf16_empty(
                batch_size, KV_HEADS, capacity, HEAD_DIM
            )
            self.host_value = host_extension.mapped_host_bf16_empty(
                batch_size, KV_HEADS, capacity, HEAD_DIM
            )
        else:
            assert storage == "k_offload"
            assert host_extension is not None and shared_staging is not None
            self.key_cache = None
            self.value_cache = torch.empty(
                cache_shape, device="cuda", dtype=torch.bfloat16
            )
            self.host_key = host_extension.mapped_host_bf16_empty(
                batch_size, KV_HEADS, capacity, HEAD_DIM
            )
            self.host_value = None

        self.profile_events = {}
        if self.profile_components:
            names = ["attention_block", "cache_append", "dense_attention"]
            if self.storage != "local":
                names.append("host_to_gpu")
            for name in names:
                self.profile_events[name] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )

    def _start(self, name: str) -> None:
        if self.profile_components:
            self.profile_events[name][0].record()

    def _end(self, name: str) -> None:
        if self.profile_components:
            self.profile_events[name][1].record()

    def component_times(self) -> dict[str, float]:
        if not self.profile_components:
            return {}
        return {
            name: begin.elapsed_time(end)
            for name, (begin, end) in self.profile_events.items()
        }

    def _append(self, key: torch.Tensor, value: torch.Tensor, start: int) -> None:
        if self.storage == "local":
            stop = start + int(key.shape[2])
            self.key_cache[:, :, start:stop].copy_(key)
            self.value_cache[:, :, start:stop].copy_(value)
        elif self.storage == "offload":
            self.host_extension.append(self.host_key, key, int(start))
            self.host_extension.append(self.host_value, value, int(start))
        else:
            stop = start + int(key.shape[2])
            self.host_extension.append(self.host_key, key, int(start))
            self.value_cache[:, :, start:stop].copy_(value)

    def _decode_cache(self, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.storage == "local":
            return self.key_cache[:, :, :end], self.value_cache[:, :, :end]
        self.shared_staging["key"].copy_(self.host_key, non_blocking=True)
        if self.storage == "offload":
            self.shared_staging["value"].copy_(self.host_value, non_blocking=True)
            value = self.shared_staging["value"][:, :, :end]
        else:
            value = self.value_cache[:, :, :end]
        return (
            self.shared_staging["key"][:, :, :end],
            value,
        )

    def _validate_prefill_cache(
        self, key: torch.Tensor, value: torch.Tensor, end: int
    ) -> None:
        if self.storage == "local":
            assert torch.equal(self.key_cache[:, :, :end], key)
            assert torch.equal(self.value_cache[:, :, :end], value)
        elif self.storage == "offload":
            torch.cuda.synchronize()
            assert torch.equal(self.host_key[:, :, :end], key.cpu())
            assert torch.equal(self.host_value[:, :, :end], value.cpu())
        else:
            torch.cuda.synchronize()
            assert torch.equal(self.host_key[:, :, :end], key.cpu())
            assert torch.equal(self.value_cache[:, :, :end], value)

    def _validate_staging(
        self, key: torch.Tensor, value: torch.Tensor, end: int
    ) -> None:
        if self.storage != "local":
            assert torch.equal(key.cpu(), self.host_key[:, :, :end])
            if self.storage == "offload":
                assert torch.equal(value.cpu(), self.host_value[:, :, :end])
            else:
                assert torch.equal(value, self.value_cache[:, :, :end])

    def _validate_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        repeated_key = key.repeat_interleave(4, dim=1)
        repeated_value = value.repeat_interleave(4, dim=1)
        score = torch.matmul(query.float(), repeated_key.transpose(-1, -2).float())
        expected = torch.matmul(
            torch.softmax(score * self.scaling, dim=-1), repeated_value.float()
        )
        torch.testing.assert_close(output.float(), expected, rtol=0.025, atol=0.025)
        self.validated = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, past_key_values, kwargs
        batch, tokens, _ = hidden_states.shape
        assert position_embeddings is not None
        start = self.length
        end = start + tokens
        assert end <= self.capacity
        hidden_shape = (batch, tokens, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        if start == 0:
            query, key = apply_prefill_rope_(query, key, cos, sin)
        else:
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if start == 0:
            self._append(key, value, 0)
            if self.validate:
                self._validate_prefill_cache(key, value, end)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attention = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                    scale=self.scaling,
                    enable_gqa=True,
                )
        else:
            assert tokens == 1
            self._start("attention_block")
            self._start("cache_append")
            self._append(key, value, start)
            self._end("cache_append")
            if self.storage != "local":
                self._start("host_to_gpu")
            full_key, full_value = self._decode_cache(end)
            if self.storage != "local":
                self._end("host_to_gpu")
            self._start("dense_attention")
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attention = F.scaled_dot_product_attention(
                    query,
                    full_key,
                    full_value,
                    is_causal=False,
                    scale=self.scaling,
                    enable_gqa=True,
                )
            self._end("dense_attention")
            self._end("attention_block")
            if self.validate and not self.validated:
                self._validate_staging(full_key, full_value, end)
                self._validate_attention(query, full_key, full_value, attention)

        self.length = end
        output = attention.transpose(1, 2).reshape(batch, tokens, -1).contiguous()
        return self.o_proj(output), None


def _load_tokens(context: int, decode_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors.torch import load_file

    windows = load_file(str(TOKENS_PATH))["input_ids"]
    assert windows.ndim == 2 and windows.shape[0] >= 2
    assert windows.shape[1] >= context and windows.shape[1] >= decode_steps
    return windows[0, :context].clone(), windows[1, :decode_steps].clone()


def _install_dense_attention(
    model: torch.nn.Module,
    *,
    capacity: int,
    storage: str,
    host_extension: object | None,
    profile_components: bool,
    validate: bool,
    batch_size: int = 1,
) -> list[TP1DenseAttention]:
    shared_staging = None
    if storage != "local":
        cache_shape = (batch_size, KV_HEADS, capacity, HEAD_DIM)
        shared_staging = {
            "key": torch.empty(cache_shape, device="cuda", dtype=torch.bfloat16),
        }
        if storage == "offload":
            shared_staging["value"] = torch.empty(
                cache_shape, device="cuda", dtype=torch.bfloat16
            )
    installed = []
    for layer in model.model.layers:
        replacement = TP1DenseAttention(
            layer.self_attn,
            capacity=capacity,
            storage=storage,
            host_extension=host_extension,
            shared_staging=shared_staging,
            profile_components=profile_components,
            validate=validate,
            batch_size=batch_size,
        )
        layer.self_attn = replacement
        layer.mlp = ChunkedTokenwise(layer.mlp, chunk_size=1024)
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm, chunk_size=2048)
        layer.post_attention_layernorm = ChunkedTokenwise(
            layer.post_attention_layernorm, chunk_size=2048
        )
        installed.append(replacement)
    model.model.norm = ChunkedTokenwise(model.model.norm, chunk_size=2048)
    return installed


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage", choices=("local", "offload", "k_offload"), required=True
    )
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--measure-steps", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--tag", default="smoke")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/system_benchmarks/tp1_dense_full_v6"),
    )
    args = parser.parse_args()
    assert args.length >= 4096 and args.warmup_steps >= 0 and args.measure_steps > 0
    total_steps = args.warmup_steps + args.measure_steps
    capacity = args.length + total_steps
    output = args.output_root / (
        f"{args.tag}_dense_{args.storage}_t{args.length}_r{args.repeat}"
    )
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    log_path.write_text("status=running\n", encoding="utf-8")

    def emit(payload: dict) -> None:
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (8, 9)
    assert MODEL_PATH.is_dir() and TOKENS_PATH.is_file()
    torch.manual_seed(20260921)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    emit(
        {
            "status": "starting",
            "storage": args.storage,
            "context": args.length,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "profile_components": args.profile_components,
            "gpu": torch.cuda.get_device_name(),
        }
    )

    host_extension = None
    if args.storage != "local":
        host_extension = _load_extension(
            value_dim=128,
            queries_per_kv=4,
            page_size=1,
            base_rank=16,
            residual_rank=16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    layers = _install_dense_attention(
        model,
        capacity=capacity,
        storage=args.storage,
        host_extension=host_extension,
        profile_components=args.profile_components,
        validate=args.validate,
    )
    prefill_tokens, decode_tokens = _load_tokens(args.length, total_steps)
    prefill_tokens = prefill_tokens[None].to("cuda")
    decode_tokens = decode_tokens.to("cuda")
    full_kv_bytes = LAYERS * 2 * KV_HEADS * capacity * HEAD_DIM * 2
    context_kv_bytes = LAYERS * 2 * KV_HEADS * args.length * HEAD_DIM * 2
    staged_tensors = 0 if args.storage == "local" else (
        2 if args.storage == "offload" else 1
    )
    stage_bytes = staged_tensors * KV_HEADS * capacity * HEAD_DIM * 2
    host_kv_bytes = full_kv_bytes if args.storage == "offload" else (
        full_kv_bytes // 2 if args.storage == "k_offload" else 0
    )
    gpu_kv_bytes = full_kv_bytes - host_kv_bytes + stage_bytes
    emit(
        {
            "status": "model_ready",
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "exact_kv_gib": full_kv_bytes / 2**30,
            "host_exact_kv_gib": host_kv_bytes / 2**30,
            "gpu_staging_gib": stage_bytes / 2**30,
        }
    )

    torch.cuda.reset_peak_memory_stats()
    prefill_begin = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_begin.record()
    outputs = model.model(input_ids=prefill_tokens, use_cache=False)
    last_hidden = outputs.last_hidden_state[:, -1:].contiguous()
    prefill_end.record()
    prefill_end.synchronize()
    prefill_ms = prefill_begin.elapsed_time(prefill_end)
    del outputs, last_hidden, prefill_tokens
    emit(
        {
            "status": "prefill_complete",
            "prefill_ms": prefill_ms,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    )

    cuda_ms = []
    wall_ms = []
    component_samples = {
        "attention_block_ms": [],
        "cache_append_ms": [],
        "host_to_gpu_ms": [],
        "dense_attention_ms": [],
    }
    argmax_tokens = []
    finite = []
    for step in range(total_steps):
        input_id = decode_tokens[step].view(1, 1)
        position_id = torch.tensor([[args.length + step]], device="cuda")
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        wall_start = time.perf_counter()
        hidden = model.model(
            input_ids=input_id,
            position_ids=position_id,
            use_cache=False,
        ).last_hidden_state
        logits = model.lm_head(hidden[:, -1:])
        end.record()
        end.synchronize()
        elapsed_cuda = begin.elapsed_time(end)
        elapsed_wall = (time.perf_counter() - wall_start) * 1000.0
        components = {name: 0.0 for name in component_samples}
        if args.profile_components:
            for layer in layers:
                for name, value in layer.component_times().items():
                    components[f"{name}_ms"] += value
        token = int(logits.argmax(dim=-1).item())
        is_finite = bool(torch.isfinite(logits).all().item())
        argmax_tokens.append(token)
        finite.append(is_finite)
        if step >= args.warmup_steps:
            cuda_ms.append(elapsed_cuda)
            wall_ms.append(elapsed_wall)
            if args.profile_components:
                for name, value in components.items():
                    component_samples[name].append(value)
        emit(
            {
                "status": "decode_step",
                "step": step,
                "measured": step >= args.warmup_steps,
                "cuda_ms": elapsed_cuda,
                "wall_ms": elapsed_wall,
                "components": components,
                "argmax": token,
                "finite": is_finite,
            }
        )

    component_summary = {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
        }
        for name, values in component_samples.items()
        if values
    }
    result = {
        "schema": "tp1-dense-full-v6",
        "method": f"dense-{args.storage}",
        "storage": args.storage,
        "context_tokens": args.length,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measure_steps,
        "repeat": args.repeat,
        "prefill_backend": "torch-flash-sdpa",
        "decode_backend": "torch-flash-sdpa",
        "prefill_ms": prefill_ms,
        "decode_cuda_ms": cuda_ms,
        "decode_wall_ms": wall_ms,
        "decode_cuda_median_ms": statistics.median(cuda_ms),
        "decode_cuda_mean_ms": statistics.fmean(cuda_ms),
        "decode_cuda_p95_ms": _percentile(cuda_ms, 0.95),
        "decode_wall_median_ms": statistics.median(wall_ms),
        "throughput_tokens_per_second": 1000.0 / statistics.fmean(wall_ms),
        "profile_components": args.profile_components,
        "components": component_summary,
        "exact_kv_bytes": full_kv_bytes,
        "exact_kv_bytes_at_context": context_kv_bytes,
        "gpu_resident_kv_bytes": gpu_kv_bytes,
        "host_resident_kv_bytes": host_kv_bytes,
        "host_to_gpu_bytes_per_decode_token": host_kv_bytes,
        "all_logits_finite": all(finite),
        "argmax_tokens": argmax_tokens,
        "validated_layers": sum(int(layer.validated) for layer in layers),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "model_path": str(MODEL_PATH),
        "tokens_path": str(TOKENS_PATH),
    }
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit(
        {
            "status": "complete",
            "decode_cuda_median_ms": result["decode_cuda_median_ms"],
            "decode_wall_median_ms": result["decode_wall_median_ms"],
            "validated_layers": result["validated_layers"],
            "result": str(output / "benchmark.json"),
        }
    )


if __name__ == "__main__":
    main()
