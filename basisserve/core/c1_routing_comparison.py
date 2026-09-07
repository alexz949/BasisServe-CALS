"""Matched-budget routing baselines with the existing C1 attention module."""

from contextlib import ExitStack
import math
from unittest.mock import patch

import torch

from basisserve.core.c1_conditional_page_attention import c1_conditional_page_topk_attention
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig
from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
from basisserve.core.c1_loki_attention import c1_loki_pca_topk_attention
from basisserve.core.residual_kl_replay import fork_routing_prefix


BASELINES = ("quest_page", "loki_page_r32", "loki_token_r32")


class QuestPageBounds:
    """Exact BF16 page extrema, updating only appended tokens and the tail page."""

    def __init__(self, keys, page_size):
        self.page_size = page_size
        self.length = keys.shape[-2]
        pages = math.ceil(self.length / page_size)
        padding = pages * page_size - self.length
        shape = (*keys.shape[:-2], pages, page_size, keys.shape[-1])
        self.minimum = torch.nn.functional.pad(keys, (0, 0, 0, padding), value=torch.inf).reshape(shape).amin(-2)
        self.maximum = torch.nn.functional.pad(keys, (0, 0, 0, padding), value=-torch.inf).reshape(shape).amax(-2)

    def update(self, keys):
        assert keys.shape[-2] >= self.length
        for position in range(self.length, keys.shape[-2]):
            token = keys[..., position:position + 1, :]
            if position % self.page_size == 0:
                self.minimum = torch.cat((self.minimum, token), dim=-2)
                self.maximum = torch.cat((self.maximum, token), dim=-2)
            else:
                self.minimum[..., -1:, :] = torch.minimum(self.minimum[..., -1:, :], token)
                self.maximum[..., -1:, :] = torch.maximum(self.maximum[..., -1:, :], token)
        self.length = keys.shape[-2]

    def scores(self, query):
        assert query.shape[-2] == 1
        groups = self.minimum.shape[1]
        head_map = torch.arange(groups, device=query.device).repeat_interleave(query.shape[1] // groups)
        low = self.minimum.index_select(1, head_map).float()
        high = self.maximum.index_select(1, head_map).float()
        return torch.maximum(query.float() * low, query.float() * high).sum(-1) / math.sqrt(query.shape[-1])

    def nbytes(self):
        return (self.minimum.numel() + self.maximum.numel()) * self.minimum.element_size()


def quest_page_ids(scores, *, kv_heads, page_budget, pinned_prefix_pages):
    """Existing physical-shared QUEST convention: max of raw bounds across GQA."""
    batch, query_heads, pages = scores.shape
    groups = scores.reshape(batch, kv_heads, query_heads // kv_heads, pages).amax(2)
    selected = min(page_budget, pages)
    pinned = min(pinned_prefix_pages, selected)
    prefix = torch.arange(pinned, device=scores.device).view(1, 1, pinned).expand(batch, kv_heads, pinned)
    routed = groups[..., pinned:].topk(selected - pinned, sorted=False).indices + pinned
    return torch.cat((prefix, routed), dim=-1).unsqueeze(2)


def exact_page_attention(query, keys, values, page_ids, page_size):
    """Native BF16 attention, with the same score/softmax arithmetic as C1 routing."""
    batch, heads, queries, width = query.shape
    assert queries == 1
    groups, length = keys.shape[1:3]
    token_ids = (page_ids[..., None] * page_size + torch.arange(page_size, device=query.device)).flatten(-2)
    valid = token_ids < length
    token_ids = token_ids.clamp_max(length - 1)
    head_map = torch.arange(groups, device=query.device).repeat_interleave(heads // groups)
    ids = token_ids.index_select(1, head_map)
    head_valid = valid.index_select(1, head_map)
    selected_key = keys.index_select(1, head_map)[:, :, None].gather(
        3, ids[..., None].expand(batch, heads, 1, ids.shape[-1], width))
    scores = torch.einsum("bhqd,bhqkd->bhqk", query, selected_key).mul_(width ** -0.5)
    scores.masked_fill_(~head_valid, -torch.inf)
    probabilities = scores.float().softmax(-1).to(query.dtype).masked_fill_(~head_valid, 0.0)
    selected_value = values.index_select(1, head_map)[:, :, None].gather(
        3, ids[..., None].expand(batch, heads, 1, ids.shape[-1], values.shape[-1]))
    return torch.einsum("bhqk,bhqkv->bhqv", probabilities, selected_value), valid


class RoutingBaseline:
    """Temporarily replace only attention selection; QKV/RoPE/cache/decoder stay shared."""

    def __init__(self, model, prefix, projector, arm, *, page_size=32, budget=2048, pinned=1):
        assert arm in BASELINES
        self.arm, self.page_size, self.budget, self.pinned = arm, page_size, budget, pinned
        self.cache = fork_routing_prefix(prefix)
        self.bounds = {}
        self.metadata_bytes = {}
        self.selected = 0.0
        self.group_queries = 0
        self.calls = 0
        self.modules = [layer.self_attn for layer in model.model.layers]
        for layer, module in enumerate(self.modules):
            module.set_reverse_shadow_config(None)
            module.conditional_base_left = None
            module.conditional_base_right = None
            module.conditional_base_bias = None
            module.conditional_residual_encoder = None
            module.set_routing_projectors(None, None)
            module.set_conditional_page_query_block_size(None)
            module.set_loki_query_block_size(None)
            module.attention_backend = "native"
            if arm == "quest_page":
                self.bounds[layer] = QuestPageBounds(prefix.layers[layer].keys, page_size)
                self.metadata_bytes[layer] = self.bounds[layer].nbytes()
                config = ReverseShadowConfig(page_size=page_size, exact_token_budget=budget,
                    selector="quest_minmax", quest_support="physical_shared", pinned_prefix_pages=pinned)
            else:
                module.set_routing_projectors(projector[layer], projector[layer])
                self.cache._ensure_routing_layer(layer)
                self.cache._routing_sidecars[layer] = build_routing_sidecar(prefix.layers[layer].keys, module.routing_key_projector)
                sidecar = self.cache.routing_sidecar(layer)
                self.metadata_bytes[layer] = sidecar.numel() * sidecar.element_size()
                token_arm = arm == "loki_token_r32"
                config = ReverseShadowConfig(page_size=1 if token_arm else page_size,
                    exact_token_budget=budget, selector="kq_svd",
                    quest_support="per_query_head" if token_arm else "physical_shared",
                    pinned_prefix_pages=0 if token_arm else pinned)
                if token_arm:
                    module.set_loki_query_block_size(1, collect_statistics=False)
            module.set_reverse_shadow_config(config)

    def __enter__(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch("basisserve.checkpoint.gqa_vo_qwen3.c1_k_reverse_shadow_block_attention", self.page_attention))
        self.stack.enter_context(patch("basisserve.checkpoint.gqa_vo_qwen3.c1_loki_pca_topk_attention", self.token_attention))
        return self

    def __exit__(self, *args):
        result = self.stack.__exit__(*args)
        for module in self.modules:
            module.set_reverse_shadow_config(None)
            module.set_routing_projectors(None, None)
            module.set_loki_query_block_size(None)
            module.attention_backend = "sdpa"
        return result

    def record(self, query, keys, selected):
        self.calls += 1
        self.selected += float(selected)
        self.group_queries += query.shape[0] * keys.shape[1] * query.shape[-2]

    def page_attention(self, query, keys, values, config, attention_mask, *,
                       routing_key_projector=None, routing_query_projector=None,
                       routing_sidecar=None, layer_idx=None, **kwargs):
        assert query.shape[-2] == 1 and attention_mask is not None
        assert attention_mask.dtype == torch.bool and bool(attention_mask.all())
        if self.arm == "quest_page":
            bounds = self.bounds[layer_idx]
            bounds.update(keys)
            ids = quest_page_ids(bounds.scores(query), kv_heads=keys.shape[1],
                page_budget=math.ceil(self.budget / self.page_size), pinned_prefix_pages=self.pinned)
            assert bool((ids.sort(-1).values.diff(dim=-1) > 0).all())
            if self.pinned:
                assert bool((ids[..., 0] == 0).all())
            output, valid = exact_page_attention(query, keys, values, ids, self.page_size)
            self.record(query, keys, valid.sum())
            self.metadata_bytes[layer_idx] = bounds.nbytes()
        else:
            assert self.arm == "loki_page_r32" and routing_sidecar is not None
            heads_per_group = query.shape[1] // keys.shape[1]
            q_projector = routing_query_projector.repeat_interleave(heads_per_group, dim=0)
            result = c1_conditional_page_topk_attention(query, keys, values,
                routing_sidecar, q_projector, page_size=self.page_size,
                exact_token_budget=self.budget, pinned_prefix_pages=self.pinned,
                scale=query.shape[-1] ** -0.5, query_block_size=1,
                attention_mask=attention_mask, collect_statistics=True)
            output = result.output
            self.record(query, keys, result.statistics["selected_tokens"])
            self.metadata_bytes[layer_idx] = result.statistics["resident_selector_metadata_bytes"]
        return output, []

    def token_attention(self, query, keys, values, *args, **kwargs):
        assert self.arm == "loki_token_r32" and kwargs["attention_mask"] is not None
        kwargs["collect_statistics"] = True
        result = c1_loki_pca_topk_attention(query, keys, values, *args, **kwargs)
        self.record(query, keys, result.statistics["selected_tokens"])
        return result

    def statistics(self):
        if self.arm != "quest_page":
            self.metadata_bytes = {layer: x.numel() * x.element_size()
                for layer, x in enumerate(self.cache._routing_sidecars) if x is not None}
        return {"selector_calls": self.calls, "physical_selected_tokens_sum": self.selected,
            "kv_group_query_count": self.group_queries,
            "mean_physical_tokens_per_kv_group": self.selected / self.group_queries if self.group_queries else None,
            "final_resident_routing_metadata_bytes": sum(self.metadata_bytes.values())}
