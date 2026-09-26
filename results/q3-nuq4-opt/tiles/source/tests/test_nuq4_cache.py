import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from basisserve.kernels.nuq4_cache import NUQ4Layout, NUQ4PagedCache, nuq4_value_stats

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def upstream():
    spec = importlib.util.spec_from_file_location("kvquant_cache_reference",
        ROOT / "external/KVQuant/quant/kvquant/simquant_module_quantizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(upstream, values, lower, upper, lut, dynamic):
    data = values.float()
    mask = (upstream.get_outliers_dynamic(data, thresh=0.99) if dynamic else
        upstream.get_outliers(data, channel=0, outlier_threshold_lower=lower,
                              outlier_threshold_upper=upper))
    return upstream.quant_fn_nuq_recon(data, bits=4, qchannel=-1 if dynamic else 0,
        dynamicquantization=dynamic, include_sparse=True, outlier_mask=mask,
        minval=lower, maxval=upper, lut=(lut.cpu().numpy(),),
        first_few_fp16=-1).to(torch.bfloat16)


@pytest.mark.parametrize("width", [16, 32, 48, 64, 80, 96, 112, 128])
@pytest.mark.parametrize("dynamic", [False, True])
def test_official_reconstruction_and_incremental_pages(upstream, width, dynamic):
    torch.manual_seed(9)
    values = torch.randn(29, width * (8 if dynamic else 1), device="cuda", dtype=torch.bfloat16)
    values[0, 0], values[7, 3] = 12, -12
    lower = torch.full((width,), -3.0, device="cuda")
    upper = torch.full((width,), 3.0, device="cuda")
    # Deliberately unsorted: ties must follow original LUT order.
    lut = torch.tensor(np.random.default_rng(2).permutation(np.linspace(-1, 1, 16)),
                       device="cuda", dtype=torch.float32)
    stats = nuq4_value_stats(values) if dynamic else None
    source = values[:, :width]
    cache = NUQ4PagedCache(4, NUQ4Layout(width), lower, upper, lut, dynamic=dynamic)
    slots = torch.cat((torch.arange(32, 48), torch.arange(0, 13))).cuda()
    cache.append(source[:19], slots[:19], stats[:19] if dynamic else None)
    cache.append(source[19:], slots[19:], stats[19:] if dynamic else None)
    expected = reference(upstream, values, lower, upper, lut, dynamic)[:, :width]
    actual = cache.gather_for_validation(slots)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert cache.storage.dtype == torch.uint8
    assert cache.storage.numel() == 4 * cache.layout.page_bytes
    # Reusing an allocated physical page must reset its sparse arena.
    cache.append(source[:16], slots[:16], stats[:16] if dynamic else None)
    torch.testing.assert_close(cache.gather_for_validation(slots[:16]), expected[:16], atol=0, rtol=0)


def test_ties_degenerate_range_and_overflow(upstream):
    width = 64
    values = torch.ones((16, width * 8), device="cuda", dtype=torch.bfloat16)
    stats = nuq4_value_stats(values)
    lo, hi = torch.zeros(width, device="cuda"), torch.ones(width, device="cuda")
    lut = torch.linspace(-1, 1, 16, device="cuda")
    slots = torch.arange(16, device="cuda")
    cache = NUQ4PagedCache(1, NUQ4Layout(width, exceptions_per_token=width), lo, hi, lut, dynamic=True)
    cache.append(values[:, :width], slots, stats)
    torch.testing.assert_close(cache.gather_for_validation(slots), values[:, :width], atol=0, rtol=0)
    small = NUQ4PagedCache(1, NUQ4Layout(width), lo, hi, lut, dynamic=True)
    small.append(values[:, :width], slots, stats)
    assert int(small.failed) > small.layout.capacity


@pytest.mark.parametrize("width", [16, 64, 96, 128])
def test_all_bitmap_byte_patterns(width):
    d = torch.arange(width, device="cuda")[None, :]
    patterns = (torch.arange(256, device="cuda")[:, None] + (d // 8) * 37) % 256
    exceptions = ((patterns >> (d % 8)) & 1).bool()
    source = torch.where(exceptions, 2 + d.float() / 32, 0).to(torch.bfloat16)
    lower = torch.full((width,), -1.0, device="cuda")
    lut = torch.zeros(16, device="cuda")
    cache = NUQ4PagedCache(16, NUQ4Layout(width, exceptions_per_token=width), lower, -lower, lut)
    slots = torch.arange(256, device="cuda")
    cache.append(source, slots)
    torch.testing.assert_close(cache.gather_for_validation(slots), source, atol=0, rtol=0)


def test_padding_and_cuda_graph(upstream):
    width = 96
    source = torch.randn((9, width), device="cuda", dtype=torch.bfloat16)
    lo, hi = torch.full((width,), -4.0, device="cuda"), torch.full((width,), 4.0, device="cuda")
    lut = torch.linspace(-1, 1, 16, device="cuda")
    slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, -1], device="cuda")
    cache = NUQ4PagedCache(1, NUQ4Layout(width), lo, hi, lut)
    cache.append(source, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.append(source, slots)
    graph.replay()
    actual = cache.gather_for_validation(slots)
    expected = reference(upstream, source[:8], lo, hi, lut, False)
    torch.testing.assert_close(actual[:8], expected, atol=0, rtol=0)
    assert not actual[-1].count_nonzero()


@pytest.mark.parametrize("rank", [64, 96])
def test_frozen_qwen_codebooks(upstream, rank):
    codes = torch.load(ROOT / f"results/q3-kv4-fp8/formal/r{rank}/quantizers.pt",
                       weights_only=False, map_location="cpu")
    torch.manual_seed(rank)
    for layer in (0, 18, 35):
        hi, lo, lut = codes[f"{layer}.k"]
        lower, upper = lo.flatten()[:128].cuda().float(), hi.flatten()[:128].cuda().float()
        poles = torch.as_tensor(lut[0], device="cuda", dtype=torch.float32).flatten()
        values = ((upper + lower)[None, :] / 2 + torch.randn(16, 128, device="cuda")
                   * (upper - lower)[None, :] / 6).to(torch.bfloat16)
        slots = torch.arange(16, device="cuda")
        cache = NUQ4PagedCache(1, NUQ4Layout(128), lower, upper, poles)
        cache.append(values, slots)
        expected = reference(upstream, values, lower, upper, poles, False)
        torch.testing.assert_close(cache.gather_for_validation(slots), expected, atol=0, rtol=0)
