"""Runtime-only adapters for the official ShadowKV and LRQK sources.

The paper baselines use their upstream cache, routing, transfer, and attention
implementations unchanged.  The installed vLLM wheel already contains
FlashAttention-2 for SM89, but exposes its variable-length entry point instead
of the historical ``flash_attn`` Python package imported by both repositories.
This module supplies that import surface without replacing either baseline's
algorithm.
"""

from __future__ import annotations

import importlib
import importlib.machinery
from pathlib import Path
import sys
import types

import torch


_CU_SEQLENS: dict[tuple[str, int, int], torch.Tensor] = {}


def _cu_seqlens(device: torch.device, batch: int, length: int) -> torch.Tensor:
    key = (str(device), batch, length)
    value = _CU_SEQLENS.get(key)
    if value is None:
        value = torch.arange(
            0,
            (batch + 1) * length,
            length,
            device=device,
            dtype=torch.int32,
        )
        _CU_SEQLENS[key] = value
    return value


def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] | list[int] | None = None,
) -> torch.Tensor:
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    assert query.ndim == key.ndim == value.ndim == 4
    assert query.shape[0] == key.shape[0] == value.shape[0]
    assert key.shape[1] == value.shape[1]
    batch, query_length, query_heads, head_dim = query.shape
    key_length = int(key.shape[1])
    assert batch == 1, "paper-faithful TP1 baseline is defined for B1"
    query_flat = query.reshape(batch * query_length, query_heads, head_dim)
    key_flat = key.reshape(batch * key_length, key.shape[2], head_dim)
    value_flat = value.reshape(batch * key_length, value.shape[2], value.shape[3])
    normalized_window = None if window_size in (None, (-1, -1)) else list(window_size)
    output = flash_attn_varlen_func(
        query_flat,
        key_flat,
        value_flat,
        query_length,
        _cu_seqlens(query.device, batch, query_length),
        key_length,
        _cu_seqlens(key.device, batch, key_length),
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=normalized_window,
        fa_version=2,
    )
    return output.reshape(batch, query_length, query_heads, value.shape[-1])


def install_flash_attn_adapter() -> None:
    """Expose the historical flash_attn functions through vLLM's FA2 build."""

    if "flash_attn" in sys.modules:
        return

    module = types.ModuleType("flash_attn")
    module.__file__ = str(Path(__file__).resolve())
    module.__version__ = "vllm-fa2-adapter"
    module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None, is_package=True
    )
    module.__path__ = []

    def flash_attn_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] | list[int] | None = None,
        **_: object,
    ) -> torch.Tensor:
        return _flash_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )

    def flash_attn_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] | list[int] | None = None,
        **_: object,
    ) -> torch.Tensor:
        assert k is None and v is None
        return _flash_attention(
            q,
            k_cache,
            v_cache,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )

    module.flash_attn_func = flash_attn_func
    module.flash_attn_with_kvcache = flash_attn_with_kvcache
    sys.modules["flash_attn"] = module


def install_shadowkv_import_adapters(upstream_root: Path) -> object:
    """Install inactive MInference stubs and load the official CUDA extension."""

    install_flash_attn_adapter()
    vllm = importlib.import_module("vllm")
    vllm_ops = importlib.import_module("vllm._custom_ops")
    if not hasattr(vllm_ops, "silu_and_mul"):
        vllm_ops.silu_and_mul = torch.ops._C.silu_and_mul
    vllm._custom_ops = vllm_ops
    kernels = importlib.import_module("kernels")
    kernel_root = upstream_root / "kernels"
    sys.path.insert(0, str(kernel_root))
    extension = importlib.import_module("shadowkv")

    kernels.shadowkv = extension
    sys.modules["kernels.shadowkv"] = extension

    def inactive_minference(*_: object, **__: object) -> None:
        assert False, "MInference is disabled for the paper-faithful decode baseline"

    minference = types.ModuleType("minference")
    minference.vertical_slash_sparse_attention = inactive_minference
    minference.block_sparse_attention = inactive_minference
    minference.streaming_forward = inactive_minference
    configs = types.ModuleType("minference.configs")
    model2path = types.ModuleType("minference.configs.model2path")
    model2path.MODEL2PATH = {}
    sys.modules["minference"] = minference
    sys.modules["minference.configs"] = configs
    sys.modules["minference.configs.model2path"] = model2path
    return extension
