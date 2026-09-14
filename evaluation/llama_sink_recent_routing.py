"""Llama Page32 routing with sink32 and recent64 inside a hard 2048 budget."""
import torch
from basisserve.core.c1_conditional_page_attention import _selected_pages


def page_support(scores):
    batch, groups, heads, length = scores.shape
    assert length > 0
    if length <= 2048:
        ids = torch.arange(length, device=scores.device).expand(batch, groups, -1)
        return ids, torch.ones_like(ids, dtype=torch.bool)
    historical = length - 64
    proxy = scores[..., :historical].reshape(batch, groups * heads, 1, historical)
    pages, _ = _selected_pages(proxy, torch.ones_like(proxy, dtype=torch.bool),
        kv_heads=groups, page_size=32, page_budget=62, pinned_prefix_pages=1)
    pages = pages.squeeze(-2).sort(-1).values
    ids = (pages[..., None] * 32 + torch.arange(32, device=scores.device)).flatten(-2)
    recent = torch.arange(historical, length, device=scores.device).expand(batch, groups, -1)
    valid = torch.cat((ids < historical, torch.ones_like(recent, dtype=torch.bool)), -1)
    return torch.cat((ids, recent), -1), valid
