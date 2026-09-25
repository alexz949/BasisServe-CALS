"""Streamed Loki Key-PCA moments (evaluation/fit_qwen35_128k_loki.py): sharded k_norm hooks reproduce the exact centered
PCA of the concatenated keys, and the evaluator accepts the 'loki' arm."""
import torch
from torch import nn

from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.eval_qwen35_128k_ruler import ARM_PATTERN
from evaluation.fit_qwen35_128k_loki import HEAD_DIM, KV_HEADS, KeyMoments


class Stub(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(max(layers) + 1):
            block = nn.Module()
            block.self_attn = nn.Module()
            block.self_attn.k_norm = nn.Identity()
            self.model.layers.append(block)


def test_sharded_moments_match_exact_centered_pca():
    torch.manual_seed(0)
    layers = (1, 3)
    keys = {layer: torch.randn(3, 500, KV_HEADS, HEAD_DIM, dtype=torch.float64) * (1 + layer) + layer for layer in layers}
    shards = []
    for shard in ((0,), (1, 2)):
        stub = Stub(layers)
        collector = KeyMoments(stub, layers, 'cpu')
        for window in shard:
            for slot, layer in enumerate(layers):
                stub.model.layers[layer].self_attn.k_norm(keys[layer][window:window + 1].bfloat16())
        collector.remove()
        assert collector.rows == 500 * len(shard)
        shards.append(collector)
    rows = sum(c.rows for c in shards)
    sums = sum(c.sums for c in shards)
    grams = sum(c.grams for c in shards)
    projector, mean, spectrum, retained = _fit_pca(torch.tensor([rows] * len(layers), dtype=torch.float64), sums, grams, rank=32)
    for slot, layer in enumerate(layers):
        flat = keys[layer].bfloat16().double().reshape(-1, KV_HEADS, HEAD_DIM).transpose(0, 1)  # [heads, rows, dim]
        centered = flat - flat.mean(1, keepdim=True)
        torch.testing.assert_close(mean[slot], flat.mean(1), atol=1e-9, rtol=1e-9)
        singular = torch.linalg.svdvals(centered) ** 2
        torch.testing.assert_close(spectrum[slot][:, :32], singular[:, :32], atol=1e-6, rtol=1e-8)
        torch.testing.assert_close(retained[slot], singular[:, :32].sum(-1) / singular.sum(-1), atol=1e-9, rtol=1e-9)
        # The projector spans the top-32 right singular subspace: projecting onto it keeps exactly the retained energy.
        kept = (centered @ projector[slot]).pow(2).sum((-1, -2))
        torch.testing.assert_close(kept, singular[:, :32].sum(-1), atol=1e-6, rtol=1e-8)


def test_arm_pattern_accepts_loki():
    match = ARM_PATTERN.match('loki')
    assert match and match.group(2) is None
    assert ARM_PATTERN.match('b16r16').group(2) == '16'
