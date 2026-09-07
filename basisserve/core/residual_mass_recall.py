"""Teacher-query attention-mass diagnostics for the deployed page selector."""

from __future__ import annotations

from contextlib import ExitStack
import math

import torch
from torch import Tensor
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from basisserve.core.c1_conditional_page_attention import (
    _head_to_kv, _selected_pages, _valid_attention_support,
)
from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar, conditional_routing_query_projector,
)


@torch.inference_mode()
def capture_teacher_queries(model, tokens: Tensor, cache) -> tuple[Tensor, list[dict]]:
    """Observe the existing Q normalization; do not rerun or replace projections."""
    records = [{} for _ in model.model.layers]

    def query_hook(index):
        def hook(_module, _args, output):
            records[index]["normalized_query"] = output.detach().transpose(1, 2)
        return hook

    def attention_hook(index):
        def hook(module, args, kwargs, _output):
            assert module.key_projector is None
            context = dict(kwargs)
            if args:
                assert len(args) == 1
                context["hidden_states"] = args[0]
            context.pop("past_key_values", None)
            q = records[index].pop("normalized_query")
            cos, sin = context["position_embeddings"]
            post_query, _ = apply_rotary_pos_emb(q, q, cos, sin)
            records[index].update(query=post_query, context=context)
        return hook

    with ExitStack() as stack:
        for index, layer in enumerate(model.model.layers):
            module = layer.self_attn
            stack.callback(module.q_norm.register_forward_hook(query_hook(index)).remove)
            stack.callback(module.register_forward_hook(
                attention_hook(index), with_kwargs=True,
            ).remove)
        hidden = model.model(input_ids=tokens, past_key_values=cache, use_cache=True).last_hidden_state
    assert all("query" in row and "context" in row for row in records)
    return hidden, records


def rank_sidecar(tensors: dict, rank: int, prefix_layer, full_layer,
                 prefix_rotary: tuple[Tensor, Tensor], suffix_rotary: tuple[Tensor, Tensor]):
    """Build prefix and suffix separately, matching the cached routing path."""
    prefix_length = int(prefix_layer.keys.shape[-2])
    pieces = []
    for values, keys, rotary in (
        (prefix_layer.values, prefix_layer.keys, prefix_rotary),
        (full_layer.values[:, :, prefix_length:], full_layer.keys[:, :, prefix_length:], suffix_rotary),
    ):
        pieces.append(build_conditional_routing_sidecar(
            values, keys, base_left=tensors["base_left_b16"],
            base_right=tensors["base_right_b16"], base_bias=tensors["base_bias_b16"],
            residual_encoder=tensors[f"residual_encoder_b16_r{rank}"], cos=rotary[0], sin=rotary[1],
        ))
    projector = conditional_routing_query_projector(
        tensors[f"residual_query_b16_r{rank}"].to(device=pieces[0].device, dtype=pieces[0].dtype),
    )
    return torch.cat(pieces, dim=-2), projector


@torch.inference_mode()
def teacher_page_probabilities(query: Tensor, exact_key: Tensor, *, page_size: int,
                               pinned_prefix_pages: int, scale: float,
                               query_block_size: int, attention_mask: Tensor | None = None) -> dict[str, Tensor]:
    """FP32 exact-QK probabilities, with a separately normalized non-sink view.

    Separate non-sink normalization is equivalent to conditioning the full
    distribution, and avoids subtracting a sink mass numerically close to one.
    No-support non-sink rows are flagged rather than counted as zero recall.
    """
    batch, heads, query_length, _ = query.shape
    kv_heads, sequence_length = exact_key.shape[1:3]
    assert 0 <= pinned_prefix_pages <= math.ceil(sequence_length / page_size)
    expanded_key = exact_key.index_select(1, _head_to_kv(heads, kv_heads, query.device)).float()
    blocks = {name: [] for name in ("mass", "non_sink_mass", "non_sink_valid")}
    for start in range(0, query_length, query_block_size):
        stop = min(start + query_block_size, query_length)
        valid = _valid_attention_support(
            attention_mask, batch=batch, query_start=start, query_stop=stop,
            query_length=query_length, sequence_length=sequence_length, device=query.device,
        ).expand(batch, heads, -1, -1)
        assert bool(valid.any(dim=-1).all())
        scores = torch.matmul(query[:, :, start:stop].float(), expanded_key.transpose(-1, -2)).mul_(scale)
        scores.masked_fill_(~valid, -torch.inf)
        padding = (-sequence_length) % page_size
        if padding:
            scores = torch.nn.functional.pad(scores, (0, padding), value=-torch.inf)
            valid = torch.nn.functional.pad(valid, (0, padding), value=False)
        page_logits = torch.logsumexp(scores.unflatten(-1, (-1, page_size)), dim=-1)
        page_valid = valid.unflatten(-1, (-1, page_size)).any(dim=-1)
        blocks["mass"].append(page_logits.softmax(dim=-1))
        non_sink_valid = page_valid.clone()
        non_sink_valid[..., :pinned_prefix_pages] = False
        has_non_sink = non_sink_valid.any(dim=-1)
        non_sink_logits = page_logits.masked_fill(~non_sink_valid, -torch.inf)
        non_sink_logits = torch.where(has_non_sink[..., None], non_sink_logits, 0.0)
        blocks["non_sink_mass"].append(non_sink_logits.softmax(dim=-1).masked_fill(~non_sink_valid, 0.0))
        blocks["non_sink_valid"].append(has_non_sink)
    return {name: torch.cat(parts, dim=2) for name, parts in blocks.items()}


@torch.inference_mode()
def routing_mass_recall(query: Tensor, sidecar: Tensor, projector: Tensor,
                        teacher: dict[str, Tensor], *, page_size: int,
                        exact_token_budget: int, pinned_prefix_pages: int,
                        scale: float, query_block_size: int,
                        attention_mask: Tensor | None = None,
                        return_pages: bool = False) -> dict[str, Tensor]:
    """Reuse the native BF16 proxy contractions and actual physical-page selector."""
    batch, heads, query_length, _ = query.shape
    kv_heads, sequence_length = sidecar.shape[1:3]
    head_to_kv = _head_to_kv(heads, kv_heads, query.device)
    expanded_sidecar = sidecar.index_select(1, head_to_kv)
    projector = projector.to(device=query.device, dtype=query.dtype)
    blocks = {name: [] for name in ("mass", "non_sink_mass")}
    page_ids, page_valids = [], []
    for start in range(0, query_length, query_block_size):
        stop = min(start + query_block_size, query_length)
        valid = _valid_attention_support(
            attention_mask, batch=batch, query_start=start, query_stop=stop,
            query_length=query_length, sequence_length=sequence_length, device=query.device,
        ).expand(batch, heads, -1, -1)
        code = torch.einsum("bhqd,hdr->bhqr", query[:, :, start:stop], projector)
        scores = torch.matmul(code, expanded_sidecar.transpose(-1, -2)).mul_(scale)
        scores.masked_fill_(~valid, -torch.inf)
        ids, selected_valid = _selected_pages(
            scores, valid, kv_heads=kv_heads, page_size=page_size,
            page_budget=math.ceil(exact_token_budget / page_size),
            pinned_prefix_pages=pinned_prefix_pages,
        )
        head_ids = ids.index_select(1, head_to_kv)
        head_valid = selected_valid.index_select(1, head_to_kv)
        for name in blocks:
            mass = teacher[name][:, :, start:stop].gather(-1, head_ids).masked_fill(~head_valid, 0.0).sum(dim=-1)
            assert bool(torch.isfinite(mass).all())
            assert float(mass.min()) >= -1e-6 and float(mass.max()) <= 1.00001
            blocks[name].append(mass)
        if return_pages:
            page_ids.append(ids)
            page_valids.append(selected_valid)
    result = {name: torch.cat(parts, dim=2) for name, parts in blocks.items()}
    if return_pages:
        result.update(page_ids=torch.cat(page_ids, dim=2), page_valid=torch.cat(page_valids, dim=2))
    return result
