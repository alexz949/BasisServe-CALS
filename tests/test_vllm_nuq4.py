from dataclasses import replace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheTensor, create_kv_cache_views
from vllm.v1.kv_cache_layout import KVCacheLayout

from basisserve.kernels.nuq4_cache import NUQ4Layout, NUQ4PagedCache, nuq4_value_stats
from basisserve.vllm.nuq4_attention import nuq4_backend, serving_layout


@pytest.mark.parametrize("width", [32, 48, 64, 80, 96, 112, 128])
def test_canonical_packed_allocation_and_strided_pages(width):
    backend = nuq4_backend(width)
    original = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=128,
                                 head_size_v=width, dtype=torch.bfloat16)
    spec = backend.customize_spec(original)
    kl, vl = serving_layout(128), serving_layout(width)
    assert spec.num_states == 1 and spec.dtype == torch.uint8
    assert spec.page_size_bytes == kl.page_bytes + vl.page_bytes
    stride = serving_layout(128).page_bytes * 2
    padded = replace(spec, page_size_padded=stride)
    raw = torch.zeros(4*stride, device="cuda", dtype=torch.int8)
    views = create_kv_cache_views(raw, padded, 4, KVCacheLayout.LBNHC,
        KVCacheTensor(size=raw.numel(), layers=["a"], layer_stride=raw.numel(), block_stride=stride))
    assert len(views) == 1 and views[0].shape == (4, 1, 1, spec.page_size_bytes)
    assert views[0].stride(0) == stride
    lo, hi = torch.full((128,), -4.0, device="cuda"), torch.full((128,), 4.0, device="cuda")
    lut = torch.linspace(-1, 1, 16, device="cuda")
    kc = NUQ4PagedCache(0, kl, lo, hi, lut)
    vc = NUQ4PagedCache(0, vl, lo[:width], hi[:width], lut, dynamic=True)
    kc.bind(views[0][:, 0, 0, :kl.page_bytes])
    vc.bind(views[0][:, 0, 0, kl.page_bytes:])
    slots = torch.tensor([32, 33, 0, 1, 2, 16], device="cuda")
    k = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(6, 8*width, device="cuda", dtype=torch.bfloat16)
    stats = nuq4_value_stats(v)
    kc.append(k, slots)
    vc.append(v[:, :width], slots, stats)
    kcopy = NUQ4PagedCache(4, kl, lo, hi, lut)
    vcopy = NUQ4PagedCache(4, vl, lo[:width], hi[:width], lut, dynamic=True)
    kcopy.append(k, slots)
    vcopy.append(v[:, :width], slots, stats)
    torch.testing.assert_close(kc.gather_for_validation(slots), kcopy.gather_for_validation(slots), rtol=0, atol=0)
    torch.testing.assert_close(vc.gather_for_validation(slots), vcopy.gather_for_validation(slots), rtol=0, atol=0)
