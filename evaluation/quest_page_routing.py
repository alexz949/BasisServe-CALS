"""Quest page selection (Tang et al., ICML 2024) on post-RoPE Keys for the routing evaluators.
Metadata: per page (page_size tokens) and channel, the minimum and maximum Key value. Criticality of a page for a query q is
the upper bound sum_i max(q_i m_i, q_i M_i) = relu(q).M + (-relu(-q)).m on q.k over the page. Every query head keeps the
top-K pages (K = token budget / page_size) of its own bound and attends exactly to their tokens; no recent window and no
pinned pages. GQA heads of one group share the Keys, so the physical union over the group's heads is measured, not capped."""
import math

import torch


def page_minmax(keys, page_size):
    """keys: [1, kv_heads, tokens, d] -> (kmin, kmax) FP32 [1, kv_heads, pages, d]; the last page may be partial."""
    batch, heads, tokens, width = keys.shape
    pages = math.ceil(tokens / page_size)
    x = keys.float()
    pad = pages * page_size - tokens
    if pad:
        x = torch.cat((x, x[:, :, -1:].expand(batch, heads, pad, width)), 2)   # copies of the last token leave min/max unchanged
    x = x.reshape(batch, heads, pages, page_size, width)
    return x.amin(3), x.amax(3)


def append_minmax(state, key, page_size, previous):
    """Extend (kmin, kmax) covering `previous` tokens by one token key [1, kv_heads, 1, d]; in place when the page is open."""
    kmin, kmax = state
    x = key.float()
    if previous % page_size == 0:
        return torch.cat((kmin, x), 2), torch.cat((kmax, x), 2)
    kmin[:, :, -1] = torch.minimum(kmin[:, :, -1], x[:, :, 0])
    kmax[:, :, -1] = torch.maximum(kmax[:, :, -1], x[:, :, 0])
    return kmin, kmax


def quest_scores(query, kmin, kmax):
    """query: [1, heads, 1, d]; kmin/kmax: [1, kv_heads, pages, d] -> upper bounds [1, heads, pages] (head i uses KV head i // g)."""
    batch, heads, _, width = query.shape
    kv_heads = kmin.shape[1]
    q = query.float().reshape(batch, kv_heads, heads // kv_heads, width)
    return (torch.einsum('bhgd,bhpd->bhgp', q.clamp_min(0), kmax) + torch.einsum('bhgd,bhpd->bhgp', q.clamp_max(0), kmin)).reshape(batch, heads, -1)


def quest_select(query, state, page_size, budget, length):
    """Token ids [1, heads, K * page_size] of every query head's top-K pages (-1 beyond `length`)."""
    kmin, kmax = state
    scores = quest_scores(query, kmin, kmax)
    k = min(budget // page_size, scores.shape[-1])
    top = scores.topk(k, dim=-1).indices
    ids = (top[..., None] * page_size + torch.arange(page_size, device=query.device)).flatten(-2)
    return ids.masked_fill(ids >= length, -1)


def union_per_group(ids, kv_heads):
    """ids: [1, heads, n] -> distinct valid token count per KV group (physical support of the group's heads)."""
    heads = ids.shape[1]
    g = heads // kv_heads
    out = []
    for group in range(kv_heads):
        block = ids[0, group * g:(group + 1) * g]
        out.append(int(torch.unique(block[block >= 0]).numel()))
    return out
