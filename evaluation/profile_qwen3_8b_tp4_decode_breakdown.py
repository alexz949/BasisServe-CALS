#!/usr/bin/env python3
"""Profile real Qwen3-8B TP4 one-token decode at fixed context lengths.

The model and tensor-parallel collectives are the same implementations used by
``benchmark_qwen3_8b_tp4_decode.py``.  A static KV cache is seeded to each
requested context length so the benchmark can sample a real decode step at
4096 tokens without first executing 4095 untimed model steps.

Uninstrumented iterations provide the trusted end-to-end latency.  Four short
CUDA-event passes then measure non-overlapping macro stages, attention
internals, MLP internals, and the Transformers row-wise AllReduce calls.  The
instrumented total and its overhead are reported separately rather than being
silently substituted for the end-to-end result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from basisserve.core.qwen3_8b_tp4_decode import (  # noqa: E402
    Qwen3TP4C1DecodeAttention,
    Qwen3TP4DenseDecodeAttention,
    TP_SIZE,
    close_qwen3_tp4_packed_communicator,
    configure_qwen3_tp4_caches,
    install_qwen3_tp4_decode_attention,
)
from evaluation.benchmark_qwen3_8b_tp4_decode import (  # noqa: E402
    _atomic_json,
    _distributed_greedy,
    _global_max_float,
    _global_max_int,
    _timing_summary,
)


FORMAT = "basisserve.qwen3_8b.tp4_decode_breakdown.v1"
ARMS = ("dense", "c1_mean_dp", "c1_uniform_r64")


def _parse_positive_csv(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("expected nonempty positive comma-separated integers")
    if len(set(values)) != len(values):
        raise ValueError("comma-separated integers must be unique")
    return values


class _CudaEventRecorder:
    """Collect nested or disjoint CUDA intervals without per-event syncs."""

    def __init__(self) -> None:
        self.iteration: int | None = None
        self.records: list[
            tuple[str, int | None, int, torch.cuda.Event, torch.cuda.Event]
        ] = []

    def begin(self, iteration: int) -> None:
        if self.iteration is not None:
            raise RuntimeError("a profiling iteration is already active")
        self.iteration = int(iteration)

    def finish(self) -> None:
        if self.iteration is None:
            raise RuntimeError("no profiling iteration is active")
        self.iteration = None

    def start(self) -> torch.cuda.Event:
        if self.iteration is None:
            raise RuntimeError("CUDA interval recorded outside an active iteration")
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(
        self,
        label: str,
        layer: int | None,
        start: torch.cuda.Event,
    ) -> None:
        if self.iteration is None:
            raise RuntimeError("CUDA interval recorded outside an active iteration")
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.records.append((label, layer, self.iteration, start, end))

    def call(
        self,
        label: str,
        layer: int | None,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        start = self.start()
        output = function(*args, **kwargs)
        self.stop(label, layer, start)
        return output

    def summarize(
        self,
        *,
        iterations: int,
        device: torch.device,
    ) -> dict[str, Any]:
        torch.cuda.synchronize(device)
        by_category: dict[str, list[list[float]]] = defaultdict(
            lambda: [[] for _ in range(iterations)]
        )
        by_layer: dict[tuple[str, int], list[list[float]]] = defaultdict(
            lambda: [[] for _ in range(iterations)]
        )
        for label, layer, iteration, start, end in self.records:
            elapsed = float(start.elapsed_time(end))
            by_category[label][iteration].append(elapsed)
            if layer is not None:
                by_layer[(label, layer)][iteration].append(elapsed)

        category_names = sorted(by_category)
        layer_keys = sorted(by_layer)
        local_rows: list[list[float]] = []
        for label in category_names:
            local_rows.append([sum(values) for values in by_category[label]])
        for key in layer_keys:
            local_rows.append([sum(values) for values in by_layer[key]])
        if not local_rows:
            raise RuntimeError("profiling pass recorded no CUDA events")
        critical = torch.tensor(local_rows, dtype=torch.float64, device=device)
        dist.all_reduce(critical, op=dist.ReduceOp.MAX)
        rows = critical.cpu().tolist()
        category_rows = rows[: len(category_names)]
        layer_rows = rows[len(category_names) :]
        categories = {
            label: _timing_summary(values)
            for label, values in zip(category_names, category_rows, strict=True)
        }
        layers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (label, layer), values in zip(layer_keys, layer_rows, strict=True):
            layers[label].append({"layer": layer, **_timing_summary(values)})
        return {"categories": categories, "by_layer": dict(layers)}


def _register_module_interval(
    recorder: _CudaEventRecorder,
    module: nn.Module,
    *,
    label: str,
    layer: int | None,
) -> list[Any]:
    pending: list[torch.cuda.Event] = []

    def before(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        pending.append(recorder.start())

    def after(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if not pending:
            raise RuntimeError(f"missing start event for {label}")
        recorder.stop(label, layer, pending.pop(),)
        return output

    return [
        module.register_forward_pre_hook(before),
        module.register_forward_hook(after),
    ]


def _wrap_method_interval(
    recorder: _CudaEventRecorder,
    owner: Any,
    method_name: str,
    *,
    label: str,
    layer: int | None,
) -> Callable[[], None]:
    original = getattr(owner, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return recorder.call(label, layer, original, *args, **kwargs)

    setattr(owner, method_name, wrapped)

    def restore() -> None:
        setattr(owner, method_name, original)

    return restore


def _remove_instrumentation(cleanups: Iterable[Any]) -> None:
    for cleanup in reversed(tuple(cleanups)):
        if callable(cleanup):
            cleanup()
        else:
            cleanup.remove()


def _seed_cache(
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    batch: int,
    context_length: int,
) -> int:
    for module in modules:
        module.clear_cache()
    torch.cuda.empty_cache()
    cache_bytes = configure_qwen3_tp4_caches(
        modules,
        batch_size=batch,
        capacity=context_length,
    )
    prefix = context_length - 1
    for module in modules:
        if module.key_cache is None or module.value_cache is None:
            raise AssertionError("static cache allocation failed")
        module.key_cache[:, :, :prefix].zero_()
        module.value_cache[:, :, :prefix].zero_()
        module._cache_length = prefix
    return cache_bytes


def _restore_cache_length(
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    context_length: int,
) -> None:
    for module in modules:
        module._cache_length = context_length - 1


def _decode_parts(
    model: nn.Module,
    token_ids: Tensor,
    *,
    position: int,
    recorder: _CudaEventRecorder | None,
) -> Tensor:
    batch = int(token_ids.shape[0])
    position_ids = torch.full(
        (1, 1),
        int(position),
        dtype=torch.int64,
        device=token_ids.device,
    )

    def backbone() -> Tensor:
        output = model.model(
            input_ids=token_ids.reshape(batch, 1),
            position_ids=position_ids,
            use_cache=False,
        )
        return output.last_hidden_state[:, -1, :]

    def complete() -> Tensor:
        if recorder is None:
            hidden = backbone()
            local_logits = F.linear(hidden, model.lm_head.weight)
            return _distributed_greedy(local_logits)
        hidden = recorder.call("backbone_total", None, backbone)
        local_logits = recorder.call(
            "lm_head",
            None,
            F.linear,
            hidden,
            model.lm_head.weight,
        )
        return recorder.call(
            "distributed_greedy",
            None,
            _distributed_greedy,
            local_logits,
        )

    if recorder is None:
        return complete()
    return recorder.call("decode_total", None, complete)


def _run_uninstrumented(
    model: nn.Module,
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    batch: int,
    context_length: int,
    prompt_token_id: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    token = torch.full(
        (batch,),
        prompt_token_id,
        dtype=torch.int64,
        device=device,
    )
    for _ in range(warmup):
        _restore_cache_length(modules, context_length)
        _decode_parts(
            model,
            token,
            position=context_length - 1,
            recorder=None,
        )
    torch.cuda.synchronize(device)
    dist.barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        _restore_cache_length(modules, context_length)
        start.record()
        _decode_parts(
            model,
            token,
            position=context_length - 1,
            recorder=None,
        )
        end.record()
    torch.cuda.synchronize(device)
    timings = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(timings, op=dist.ReduceOp.MAX)
    return _timing_summary(timings.cpu().tolist())


def _run_profile_pass(
    model: nn.Module,
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    batch: int,
    context_length: int,
    prompt_token_id: int,
    iterations: int,
    device: torch.device,
    install: Callable[[_CudaEventRecorder], Sequence[Any]],
) -> dict[str, Any]:
    recorder = _CudaEventRecorder()
    cleanups = tuple(install(recorder))
    token = torch.full(
        (batch,),
        prompt_token_id,
        dtype=torch.int64,
        device=device,
    )
    try:
        dist.barrier()
        for iteration in range(iterations):
            _restore_cache_length(modules, context_length)
            recorder.begin(iteration)
            _decode_parts(
                model,
                token,
                position=context_length - 1,
                recorder=recorder,
            )
            recorder.finish()
        torch.cuda.synchronize(device)
    finally:
        _remove_instrumentation(cleanups)
    return recorder.summarize(iterations=iterations, device=device)


def _install_macro(
    model: nn.Module,
    recorder: _CudaEventRecorder,
) -> list[Any]:
    cleanups: list[Any] = []
    cleanups.extend(
        _register_module_interval(
            recorder, model.model.embed_tokens, label="embedding", layer=None
        )
    )
    cleanups.extend(
        _register_module_interval(
            recorder, model.model.rotary_emb, label="rotary_embedding", layer=None
        )
    )
    for layer_index, layer in enumerate(model.model.layers):
        for module, label in (
            (layer.input_layernorm, "input_rmsnorm"),
            (layer.self_attn, "attention_total"),
            (layer.post_attention_layernorm, "post_attention_rmsnorm"),
            (layer.mlp, "mlp_total"),
        ):
            cleanups.extend(
                _register_module_interval(
                    recorder,
                    module,
                    label=label,
                    layer=layer_index,
                )
            )
    cleanups.extend(
        _register_module_interval(
            recorder, model.model.norm, label="final_rmsnorm", layer=None
        )
    )
    return cleanups


def _install_attention(
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    recorder: _CudaEventRecorder,
) -> list[Any]:
    cleanups: list[Any] = []
    for module in modules:
        layer = module.layer_idx
        for child, label in (
            (module.q_proj, "attention_q_proj"),
            (module.k_proj, "attention_k_proj"),
            (module.q_norm, "attention_q_norm"),
            (module.k_norm, "attention_k_norm"),
        ):
            cleanups.extend(
                _register_module_interval(recorder, child, label=label, layer=layer)
            )
        for method, label in (
            ("_project_values", "attention_v_proj"),
            ("_attention", "attention_core"),
            ("_project_output", "attention_output_path"),
        ):
            cleanups.append(
                _wrap_method_interval(
                    recorder,
                    module,
                    method,
                    label=label,
                    layer=layer,
                )
            )
        if isinstance(module, Qwen3TP4C1DecodeAttention):
            cleanups.append(
                _wrap_method_interval(
                    recorder,
                    module,
                    "_decode_gathered",
                    label="attention_c1_decoder",
                    layer=layer,
                )
            )
    return cleanups


def _install_mlp(model: nn.Module, recorder: _CudaEventRecorder) -> list[Any]:
    cleanups: list[Any] = []
    for layer_index, layer in enumerate(model.model.layers):
        for module, label in (
            (layer.mlp.gate_proj, "mlp_gate_proj"),
            (layer.mlp.up_proj, "mlp_up_proj"),
            (layer.mlp.down_proj, "mlp_down_proj"),
        ):
            cleanups.extend(
                _register_module_interval(
                    recorder,
                    module,
                    label=label,
                    layer=layer_index,
                )
            )
    return cleanups


def _install_collectives(model: nn.Module, recorder: _CudaEventRecorder) -> list[Any]:
    import transformers.integrations.tensor_parallel as tensor_parallel

    original = tensor_parallel.all_reduce_forward
    phase: dict[str, Any] = {"label": None, "layer": None}
    cleanups: list[Any] = []

    def timed_all_reduce(hidden: Tensor, device_mesh: Any) -> Tensor:
        label = phase["label"] or "unexpected_all_reduce"
        return recorder.call(label, phase["layer"], original, hidden, device_mesh)

    tensor_parallel.all_reduce_forward = timed_all_reduce

    def restore_function() -> None:
        tensor_parallel.all_reduce_forward = original

    cleanups.append(restore_function)

    def add_phase(module: nn.Module, *, label: str, layer: int) -> None:
        def before(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            if phase["label"] is not None:
                raise RuntimeError("nested row-wise collective phase")
            phase.update(label=label, layer=layer)

        def after(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            phase.update(label=None, layer=None)
            return output

        cleanups.append(module.register_forward_pre_hook(before))
        cleanups.append(module.register_forward_hook(after))

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if isinstance(attention, Qwen3TP4DenseDecodeAttention):
            add_phase(
                attention.o_proj,
                label="attention_all_reduce",
                layer=layer_index,
            )
        add_phase(
            layer.mlp.down_proj,
            label="mlp_all_reduce",
            layer=layer_index,
        )
    return cleanups


def _install_collective_ablation(
    model: nn.Module,
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    skip_attention: bool,
    skip_mlp: bool,
) -> list[Any]:
    """Bypass selected TP collectives while retaining every local kernel."""

    import transformers.integrations.tensor_parallel as tensor_parallel

    original_all_reduce = tensor_parallel.all_reduce_forward
    phase: dict[str, Any] = {"kind": None}
    cleanups: list[Any] = []

    def selective_all_reduce(hidden: Tensor, device_mesh: Any) -> Tensor:
        kind = phase["kind"]
        if (kind == "attention" and skip_attention) or (
            kind == "mlp" and skip_mlp
        ):
            return hidden
        return original_all_reduce(hidden, device_mesh)

    tensor_parallel.all_reduce_forward = selective_all_reduce

    def restore_all_reduce() -> None:
        tensor_parallel.all_reduce_forward = original_all_reduce

    cleanups.append(restore_all_reduce)

    def add_phase(module: nn.Module, kind: str) -> None:
        def before(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            if phase["kind"] is not None:
                raise RuntimeError("nested row-wise collective ablation phase")
            phase["kind"] = kind

        def after(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            phase["kind"] = None
            return output

        cleanups.append(module.register_forward_pre_hook(before))
        cleanups.append(module.register_forward_hook(after))

    for layer in model.model.layers:
        attention = layer.self_attn
        if isinstance(attention, Qwen3TP4DenseDecodeAttention):
            add_phase(attention.o_proj, "attention")
        add_phase(layer.mlp.down_proj, "mlp")

    c1_modules = tuple(
        module
        for module in modules
        if isinstance(module, Qwen3TP4C1DecodeAttention)
    )
    if skip_attention and c1_modules:
        communicators = {
            id(module.communicator): module.communicator for module in c1_modules
        }
        if len(communicators) != 1:
            raise RuntimeError("C1 layers must share one communicator")
        communicator = next(iter(communicators.values()))
        original_gather = communicator.gather
        workspace = communicator._shared_direct_workspace
        if workspace is None:
            raise RuntimeError("C1 direct workspace is not configured")
        workspace.zero_()

        def local_only_gather(
            local: Tensor,
            plan: Any,
            *,
            backend: str = "feature_direct",
            local_is_feature_major: bool = False,
        ) -> Tensor:
            if backend != "feature_direct" or not local_is_feature_major:
                raise RuntimeError(
                    "collective ablation supports the direct decode slot only"
                )
            selected = communicator._direct_workspace(
                local,
                plan,
                local_is_feature_major=True,
            )
            local_slot = selected.narrow(
                0,
                plan.offsets[communicator.rank],
                plan.source_widths[communicator.rank],
            )
            if local.data_ptr() != local_slot.data_ptr():
                local_slot.copy_(local)
            return selected

        communicator.gather = local_only_gather

        def restore_gather() -> None:
            communicator.gather = original_gather

        cleanups.append(restore_gather)
    return cleanups


def _run_collective_ablations(
    model: nn.Module,
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    baseline: dict[str, float],
    batch: int,
    context_length: int,
    prompt_token_id: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    variants: dict[str, dict[str, float]] = {"baseline": baseline}
    for name, skip_attention, skip_mlp in (
        ("without_attention_collective", True, False),
        ("without_mlp_collective", False, True),
        ("without_attention_and_mlp_collectives", True, True),
    ):
        cleanups = _install_collective_ablation(
            model,
            modules,
            skip_attention=skip_attention,
            skip_mlp=skip_mlp,
        )
        try:
            variants[name] = _run_uninstrumented(
                model,
                modules,
                batch=batch,
                context_length=context_length,
                prompt_token_id=prompt_token_id,
                warmup=warmup,
                iterations=iterations,
                device=device,
            )
        finally:
            _remove_instrumentation(cleanups)
    baseline_ms = float(baseline["mean_ms"])
    attention_cost = baseline_ms - float(
        variants["without_attention_collective"]["mean_ms"]
    )
    mlp_cost = baseline_ms - float(variants["without_mlp_collective"]["mean_ms"])
    all_cost = baseline_ms - float(
        variants["without_attention_and_mlp_collectives"]["mean_ms"]
    )
    return {
        "variants": variants,
        "attention_collective_marginal_e2e_ms": attention_cost,
        "mlp_collective_marginal_e2e_ms": mlp_cost,
        "all_main_collectives_e2e_ms": all_cost,
        "all_main_collectives_fraction_of_e2e": all_cost / baseline_ms,
        "nonadditive_interaction_ms": attention_cost + mlp_cost - all_cost,
        "method": (
            "Uninstrumented end-to-end counterfactuals retain all local kernels "
            "and tensor shapes while bypassing the selected communication call."
        ),
    }


def _category_mean(profile: dict[str, Any], label: str) -> float:
    entry = profile["categories"].get(label)
    return 0.0 if entry is None else float(entry["mean_ms"])


def _derive_breakdown(
    *,
    arm: str,
    e2e: dict[str, float],
    macro: dict[str, Any],
    attention: dict[str, Any],
    mlp: dict[str, Any],
    collectives: dict[str, Any],
    collective_ablations: dict[str, Any],
) -> dict[str, Any]:
    macro_labels = (
        "embedding",
        "rotary_embedding",
        "input_rmsnorm",
        "attention_total",
        "post_attention_rmsnorm",
        "mlp_total",
        "final_rmsnorm",
        "lm_head",
        "distributed_greedy",
    )
    macro_ms = {label: _category_mean(macro, label) for label in macro_labels}
    profiled_total = _category_mean(macro, "decode_total")
    macro_ms["residual_launch_and_framework"] = profiled_total - sum(
        macro_ms.values()
    )

    attention_labels = (
        "attention_q_proj",
        "attention_k_proj",
        "attention_v_proj",
        "attention_q_norm",
        "attention_k_norm",
        "attention_core",
        "attention_output_path",
    )
    attention_ms = {
        label: _category_mean(attention, label) for label in attention_labels
    }
    attention_ms["attention_rope_cache_and_dispatch"] = (
        macro_ms["attention_total"] - sum(attention_ms.values())
    )

    mlp_labels = ("mlp_gate_proj", "mlp_up_proj", "mlp_down_proj")
    mlp_ms = {label: _category_mean(mlp, label) for label in mlp_labels}
    mlp_ms["mlp_activation_multiply_and_dispatch"] = (
        macro_ms["mlp_total"] - sum(mlp_ms.values())
    )

    attention_all_reduce = _category_mean(collectives, "attention_all_reduce")
    mlp_all_reduce = _category_mean(collectives, "mlp_all_reduce")
    c1_decoder = _category_mean(attention, "attention_c1_decoder")
    output_path = attention_ms["attention_output_path"]
    if arm == "dense":
        output_detail = {
            "instrumented_dense_o_proj_local_gemm_and_dispatch_ms": output_path
            - attention_all_reduce,
            "instrumented_dense_attention_all_reduce_interval_ms": attention_all_reduce,
        }
    else:
        output_detail = {
            "instrumented_c1_feature_all_gather_and_dispatch_interval_ms": output_path
            - c1_decoder,
            "instrumented_c1_decoder_gemm_ms": c1_decoder,
        }
    output_detail.update(
        {
            "instrumented_mlp_down_local_gemm_and_dispatch_ms": mlp_ms[
                "mlp_down_proj"
            ]
            - mlp_all_reduce,
            "instrumented_mlp_all_reduce_interval_ms": mlp_all_reduce,
        }
    )
    e2e_mean = float(e2e["mean_ms"])
    return {
        "trusted_uninstrumented_e2e_ms": e2e_mean,
        "instrumented_macro_total_ms": profiled_total,
        "instrumentation_overhead_percent": 100.0 * (profiled_total / e2e_mean - 1.0),
        "macro_stage_ms": macro_ms,
        "attention_internal_ms": attention_ms,
        "mlp_internal_ms": mlp_ms,
        "instrumented_output_path_detail_ms": output_detail,
        "collective_ablation": collective_ablations,
        "notes": [
            "Derived residuals subtract independent instrumented passes and may be slightly negative within CUDA-event overhead/noise.",
            "Per-collective CUDA-event intervals perturb small-message synchronization and are diagnostic only.",
            "Communication conclusions use uninstrumented collective-removal counterfactuals, not sums of event intervals.",
        ],
    }


@torch.inference_mode()
def _run_configuration(
    model: nn.Module,
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    arm: str,
    batch: int,
    context_length: int,
    prompt_token_id: int,
    warmup: int,
    iterations: int,
    profile_iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    cache_bytes = _seed_cache(
        modules,
        batch=batch,
        context_length=context_length,
    )
    cache_bytes = _global_max_int(cache_bytes, device=device)
    e2e = _run_uninstrumented(
        model,
        modules,
        batch=batch,
        context_length=context_length,
        prompt_token_id=prompt_token_id,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    macro = _run_profile_pass(
        model,
        modules,
        batch=batch,
        context_length=context_length,
        prompt_token_id=prompt_token_id,
        iterations=profile_iterations,
        device=device,
        install=lambda recorder: _install_macro(model, recorder),
    )
    attention = _run_profile_pass(
        model,
        modules,
        batch=batch,
        context_length=context_length,
        prompt_token_id=prompt_token_id,
        iterations=profile_iterations,
        device=device,
        install=lambda recorder: _install_attention(modules, recorder),
    )
    mlp = _run_profile_pass(
        model,
        modules,
        batch=batch,
        context_length=context_length,
        prompt_token_id=prompt_token_id,
        iterations=profile_iterations,
        device=device,
        install=lambda recorder: _install_mlp(model, recorder),
    )
    collectives = _run_profile_pass(
        model,
        modules,
        batch=batch,
        context_length=context_length,
        prompt_token_id=prompt_token_id,
        iterations=profile_iterations,
        device=device,
        install=lambda recorder: _install_collectives(model, recorder),
    )
    collective_ablations = _run_collective_ablations(
        model,
        modules,
        baseline=e2e,
        batch=batch,
        context_length=context_length,
        prompt_token_id=prompt_token_id,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    return {
        "batch_size": batch,
        "context_length_including_current_token": context_length,
        "static_kv_cache_bytes_per_rank": cache_bytes,
        "e2e": e2e,
        "profiles": {
            "macro": macro,
            "attention": attention,
            "mlp": mlp,
            "collectives": collectives,
        },
        "derived": _derive_breakdown(
            arm=arm,
            e2e=e2e,
            macro=macro,
            attention=attention,
            mlp=mlp,
            collectives=collectives,
            collective_ablations=collective_ablations,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir")
    parser.add_argument("--batch-sizes", default="1,8,64")
    parser.add_argument("--context-lengths", default="128,2048,4096")
    parser.add_argument("--prompt-token-id", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.arm == "dense" and args.factor_dir is not None:
        raise ValueError("dense arm must not receive --factor-dir")
    if args.arm != "dense" and args.factor_dir is None:
        raise ValueError("C1 arm requires --factor-dir")
    if min(args.warmup, args.iterations, args.profile_iterations) <= 0:
        raise ValueError("warmup and iteration counts must be positive")
    if args.torch_num_threads <= 0:
        raise ValueError("torch thread count must be positive")
    batches = _parse_positive_csv(args.batch_sizes)
    contexts = _parse_positive_csv(args.context_lengths)
    if max(contexts) > 32768:
        raise ValueError("context length exceeds Qwen3-8B positional capacity")

    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from transformers import AutoModelForCausalLM
    from transformers.distributed import DistributedConfig

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=TP_SIZE),
        local_files_only=True,
    ).eval()
    if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
        raise RuntimeError(f"decode profiler requires TP{TP_SIZE}")
    modules = install_qwen3_tp4_decode_attention(
        model,
        factor_dir=args.factor_dir,
        c1_decode_attention_backend=(None if args.arm == "dense" else "cuda"),
    )
    torch.cuda.synchronize(device)
    load_seconds = _global_max_float(time.perf_counter() - load_started, device=device)

    model_bytes = _global_max_int(
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        + sum(
            buffer.numel() * buffer.element_size()
            for buffer in model.buffers()
            if buffer is not None
        ),
        device=device,
    )
    records: list[dict[str, Any]] = []
    try:
        for batch in batches:
            for context_length in contexts:
                record = _run_configuration(
                    model,
                    modules,
                    arm=args.arm,
                    batch=batch,
                    context_length=context_length,
                    prompt_token_id=args.prompt_token_id,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    profile_iterations=args.profile_iterations,
                    device=device,
                )
                records.append(record)
                if dist.get_rank() == 0:
                    print(
                        json.dumps(
                            {
                                "event": "configuration_complete",
                                "arm": args.arm,
                                "batch": batch,
                                "context": context_length,
                                "e2e_mean_ms": record["e2e"]["mean_ms"],
                                "attention_ms": record["derived"]["macro_stage_ms"]["attention_total"],
                                "mlp_ms": record["derived"]["macro_stage_ms"]["mlp_total"],
                                "collective_ablation_fraction": record["derived"]["collective_ablation"]["all_main_collectives_fraction_of_e2e"],
                            }
                        ),
                        flush=True,
                    )

        if dist.get_rank() == 0:
            factor_records = [
                {
                    "layer": module.layer_idx,
                    "rank": module.source_rank,
                    "path": module.factor_path,
                    "sha256": module.factor_sha256,
                }
                for module in modules
                if isinstance(module, Qwen3TP4C1DecodeAttention)
            ]
            payload = {
                "format": FORMAT,
                "status": "complete",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join(sys.argv),
                "arm": args.arm,
                "model": str(Path(args.model).expanduser().resolve()),
                "factor_dir": (
                    None
                    if args.factor_dir is None
                    else str(Path(args.factor_dir).expanduser().resolve())
                ),
                "protocol": {
                    "tp_size": TP_SIZE,
                    "dtype": "bfloat16",
                    "attention_backend": (
                        "dense static-cache SDPA"
                        if args.arm == "dense"
                        else "C1 direct-slot CUDA attention + feature-major NCCL AllGather + BF16 decoder"
                    ),
                    "batch_sizes": list(batches),
                    "context_lengths_including_current_token": list(contexts),
                    "cache_seed": "zero K/V prefix; current token executes the real model path",
                    "trusted_e2e_iterations": args.iterations,
                    "warmup_iterations": args.warmup,
                    "profile_iterations_per_pass": args.profile_iterations,
                    "profile_passes": ["macro", "attention", "mlp", "collectives"],
                    "collective_ablation_variants": [
                        "without_attention_collective",
                        "without_mlp_collective",
                        "without_attention_and_mlp_collectives",
                    ],
                    "collectives_per_decode_step": {
                        "attention": 36,
                        "mlp": 36,
                        "lm_head_argmax_all_gather": 2,
                    },
                },
                "environment": {
                    "world_size": dist.get_world_size(),
                    "gpu": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "model_load_seconds": load_seconds,
                    "model_and_factor_bytes_per_rank": model_bytes,
                },
                "factors": factor_records,
                "records": records,
            }
            output = Path(args.output_json).expanduser().resolve()
            _atomic_json(output, payload)
            print(json.dumps({"event": "result_written", "path": str(output)}), flush=True)
        dist.barrier()
    finally:
        close_qwen3_tp4_packed_communicator(modules)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
