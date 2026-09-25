"""Page routing with one pinned sink page and recent64 inside a hard budget; page size 32 (default), 16, 8, 4, 2 or 1."""
import torch
from basisserve.core.c1_conditional_page_attention import _selected_pages


def page_support(scores, budget=2048, page_size=32, pinned=1):
    batch, groups, heads, length = scores.shape
    assert length > 0
    assert page_size in (1, 2, 4, 8, 16, 32) and pinned >= 0 and budget >= 64 + (pinned + 1) * page_size and budget % page_size == 0
    if length <= budget:
        ids = torch.arange(length, device=scores.device).expand(batch, groups, -1)
        return ids, torch.ones_like(ids, dtype=torch.bool)
    historical = length - 64
    proxy = scores[..., :historical].reshape(batch, groups * heads, 1, historical)
    pages, _ = _selected_pages(proxy, torch.ones_like(proxy, dtype=torch.bool),
        kv_heads=groups, page_size=page_size, page_budget=(budget-64)//page_size, pinned_prefix_pages=pinned)
    pages = pages.squeeze(-2).sort(-1).values
    ids = (pages[..., None] * page_size + torch.arange(page_size, device=scores.device)).flatten(-2)
    recent = torch.arange(historical, length, device=scores.device).expand(batch, groups, -1)
    valid = torch.cat((ids < historical, torch.ones_like(recent, dtype=torch.bool)), -1)
    return torch.cat((ids, recent), -1), valid
