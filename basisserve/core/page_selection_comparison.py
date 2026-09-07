"""Audit physical page sets against an exact-QK GQA-max selector."""

import torch

from basisserve.core.c1_conditional_page_attention import _selected_pages


def selection_details(scores, *, page_size=32, page_budget=64, pinned=1):
    heads, tokens = scores.shape
    assert tokens % page_size == 0 and 0 <= pinned < page_budget <= tokens // page_size
    logits = scores.float().reshape(heads, -1, page_size).logsumexp(-1)
    full_mass = logits.softmax(-1)
    routed = logits.clone()
    routed[:, :pinned] = -torch.inf
    non_sink = routed.softmax(-1)
    group_score, owner = non_sink.max(0)
    group_score = group_score.clone()
    group_score[:pinned] = -torch.inf
    ids, valid = _selected_pages(scores[None, :, None], torch.ones_like(scores[None, :, None], dtype=torch.bool),
                                kv_heads=1, page_size=page_size, page_budget=page_budget,
                                pinned_prefix_pages=pinned)
    assert valid.all()
    selected = torch.zeros(logits.shape[-1], dtype=torch.bool, device=scores.device)
    selected[ids.flatten()] = True
    # Rank intervals, not arbitrary tie-breaking: pinned pages have rank zero.
    ordered = group_score[pinned:].sort().values.contiguous()
    values = group_score[pinned:].contiguous()
    n = ordered.numel()
    rank_min = torch.zeros_like(group_score, dtype=torch.int32)
    rank_max = torch.zeros_like(rank_min)
    rank_min[pinned:] = n - torch.searchsorted(ordered, values, right=True).int() + 1
    rank_max[pinned:] = n - torch.searchsorted(ordered, values, right=False).int()
    cutoff = ordered[-(page_budget - pinned)]
    return dict(selected=selected, ids=ids.flatten(), full_mass=full_mass,
                non_sink_mass=non_sink, group_score=group_score, owner=owner.int(),
                rank_min=rank_min, rank_max=rank_max, cutoff=cutoff)


def compare_selection(teacher, proxy, *, pinned=1):
    exact, predicted = teacher["selected"], proxy["selected"]
    assert exact.shape == predicted.shape and int(exact.sum()) == int(predicted.sum())
    intersection = exact & predicted
    missed, extra = exact & ~predicted, predicted & ~exact
    assert int(missed.sum()) == int(extra.sum())
    routed = torch.arange(exact.numel(), device=exact.device) >= pinned
    full = teacher["full_mass"].mean(0)
    non_sink = teacher["non_sink_mass"].mean(0)
    assert torch.all(exact[:pinned]) and torch.all(predicted[:pinned])
    return {
        "intersection": intersection, "missed": missed, "extra": extra,
        "page_recall": float(intersection.sum() / exact.sum()),
        "non_sink_page_recall": float((intersection & routed).sum() / (exact & routed).sum()),
        "exact_mass": float(full[exact].sum()), "proxy_mass": float(full[predicted].sum()),
        "exact_non_sink_mass": float(non_sink[exact].sum()),
        "proxy_non_sink_mass": float(non_sink[predicted].sum()),
        "missed_mass": float(full[missed].sum()), "extra_mass": float(full[extra].sum()),
        "missed_count": int(missed.sum()), "extra_count": int(extra.sum()),
        "missed_exact_top16": int((missed & (teacher["rank_min"] > 0) & (teacher["rank_min"] <= 16)).sum()),
    }
