"""Full-dimensional FP32 QK routing scores with globally normalized Page4 selection."""

import torch


@torch.inference_mode()
def exact_grouped_scores(query, key, *, scale, chunk_size=16384):
    """Read every stored BF16 Key, resident or losslessly offloaded, without chunk normalization."""
    assert query.ndim == key.ndim == 4 and query.shape[2] == 1
    batch, heads, _, width = query.shape
    assert key.shape[0] == batch and key.shape[-1] == width and heads % key.shape[1] == 0
    assert chunk_size > 0 and key.shape[2] > 0
    groups = key.shape[1]
    q = query[:, :, 0].float().reshape(batch, groups, heads // groups, width)
    scores = torch.empty((batch, groups, heads // groups, key.shape[2]), device=query.device, dtype=torch.float32)
    for start in range(0, key.shape[2], chunk_size):
        stop = min(start + chunk_size, key.shape[2])
        block = key[:, :, start:stop].to(device=query.device, non_blocking=True).float()
        scores[..., start:stop] = (q @ block.transpose(-1, -2)) * scale
    return scores


@torch.inference_mode()
def exact_page4_support(query, key, *, scale, chunk_size=16384):
    from evaluation.llama_sink_recent_routing_p16 import page_support
    scores = exact_grouped_scores(query,key,scale=scale,chunk_size=chunk_size)
    return page_support(scores,budget=2112,page_size=4,pinned=0)
