"""Shape and chunk-equivalence checks for Qwen3 TP8 V-only placement."""

import torch
from torch import nn

from basisserve.core.qwen3_tp8_v_only import (
    NUM_LAYERS,
    SKIP_LAYERS,
    StarFusedValue,
    VOnlyTP8Attention,
    star_tp_plan,
)


def test_star_tp_plan_preserves_shared_latent():
    plan = star_tp_plan([512] * NUM_LAYERS)
    assert plan["model.layers.*.self_attn.v_proj"] == "colwise"
    assert plan["model.layers.*.self_attn.v_proj.U"] == "colwise"
    assert "model.layers.*.self_attn.v_proj.VS" not in plan


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 2
        self.scaling = 128 ** -0.5
        self.q_proj = nn.Linear(16, 4 * 128, bias=False)
        self.k_proj = nn.Linear(16, 128, bias=False)
        self.v_proj = StarFusedValue(16, 9, 128)
        self.o_proj = nn.Linear(4 * 128, 16, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


def test_star_chunked_prefill_matches_one_shot():
    assert torch.cuda.is_available()
    torch.manual_seed(7)
    source = TinyAttention().to(device="cuda", dtype=torch.bfloat16)
    full = VOnlyTP8Attention(source, arm="star_v_adaptive", batch=1, capacity=7,
                            rank=0, factor_root=None)
    chunked = VOnlyTP8Attention(TinyAttention().to(device="cuda", dtype=torch.bfloat16),
                               arm="star_v_adaptive", batch=1, capacity=7,
                               rank=0, factor_root=None)
    chunked.load_state_dict(full.state_dict(), strict=False)
    hidden = torch.randn(1, 6, 16, device="cuda", dtype=torch.bfloat16)

    def rope(tokens):
        shape = (1, tokens, 128)
        return (torch.ones(shape, device="cuda", dtype=torch.bfloat16),
                torch.zeros(shape, device="cuda", dtype=torch.bfloat16))

    expected, _ = full(hidden, rope(6))
    first, _ = chunked(hidden[:, :3], rope(3))
    second, _ = chunked(hidden[:, 3:], rope(3))
    actual = torch.cat((first, second), dim=1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.03, rtol=0.02)
    torch.testing.assert_close(chunked.key_cache[:, :, :6], full.key_cache[:, :, :6])
    torch.testing.assert_close(chunked.value_cache[:, :, :6], full.value_cache[:, :, :6])
    assert chunked.state_bytes()["value_cache"] == 1 * 7 * 9 * 2
    assert chunked.state_bytes()["dense_key_cache"] == 1 * 7 * 128 * 2
