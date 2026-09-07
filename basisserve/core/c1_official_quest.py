"""Call the unmodified upstream QUEST accuracy forward on Qwen3 C1 tensors."""

import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import torch

from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig
from basisserve.core.residual_kl_replay import fork_routing_prefix


def load_quest(source):
    spec = importlib.util.spec_from_file_location("upstream_quest_accuracy", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def official_attention(upstream, query, keys, values, *, page_size, budget, observe=None):
    """Bridge model interfaces only; upstream computes bounds, support and attention.

    Q/K have already undergone Qwen3 normalization and RoPE. Identity projections
    and identity RoPE hand them to upstream unchanged. Its legacy cache receives
    all but the last token; upstream appends that token exactly once. Zero-padding
    V accommodates upstream's equal K/V width assertion without changing payload.
    """
    batch, heads, queries, width = query.shape
    groups, length, value_width = values.shape[1:]
    assert queries == 1 and length > 1 and value_width <= width
    padded_value = torch.nn.functional.pad(values, (0, width - value_width))
    flatten = lambda tensor: tensor.transpose(1, 2).reshape(batch, 1, -1)
    view = SimpleNamespace(
        layer_id=2, num_heads=heads, head_dim=width, num_key_value_heads=groups,
        num_key_value_groups=heads // groups, hidden_size=heads * width,
        chunk_size=page_size, token_budget=budget,
        q_proj=lambda _: flatten(query), k_proj=lambda _: flatten(keys[..., -1:, :]),
        v_proj=lambda _: flatten(padded_value[..., -1:, :]),
        o_proj=lambda tensor: tensor,
        rotary_emb=lambda tensor, positions: (
            torch.ones(batch, 1, width, dtype=query.dtype, device=query.device),
            torch.zeros(batch, 1, width, dtype=query.dtype, device=query.device)),
    )
    select = upstream.local_heavy_hitter_mask

    def capture(*args, **kwargs):
        mask = select(*args, **kwargs)
        if observe is not None:
            observe(mask)
        return mask

    rotate = upstream.apply_rotary_pos_emb
    # Upstream passes position_ids as argument five; current Transformers removed
    # that argument. The identity cos/sin below still leave Q/K unchanged.
    with patch.object(upstream, "local_heavy_hitter_mask", capture), patch.object(
        upstream, "apply_rotary_pos_emb", lambda q, k, c, s, positions: rotate(q, k, c, s)
    ):
        output, _, _ = upstream.forward(
            view, query.new_zeros(batch, 1, 1),
            position_ids=torch.full((batch, 1), length - 1, device=query.device, dtype=torch.long),
            past_key_value=(keys[..., :-1, :], padded_value[..., :-1, :]),
            use_cache=True,
        )
    return output.reshape(batch, 1, heads, width).transpose(1, 2)[..., :value_width]


class OfficialQuest:
    """Upstream per-query-head page selection; layers 0 and 1 remain full."""

    def __init__(self, model, prefix, upstream, *, page_size=32, budget=2048):
        self.cache = fork_routing_prefix(prefix)
        self.upstream, self.page_size, self.budget = upstream, page_size, budget
        self.modules = [layer.self_attn for layer in model.model.layers]
        self.layer_calls = [0] * len(self.modules)
        self.selected = self.group_queries = 0
        for layer, module in enumerate(self.modules):
            module.set_reverse_shadow_config(None)
            module.set_routing_projectors(None, None)
            module.conditional_base_left = module.conditional_base_right = None
            module.conditional_base_bias = module.conditional_residual_encoder = None
            module.set_conditional_page_query_block_size(None)
            module.set_loki_query_block_size(None)
            module.attention_backend = "sdpa" if layer < 2 else "native"
            if layer >= 2:
                module.set_reverse_shadow_config(ReverseShadowConfig(
                    page_size=page_size, exact_token_budget=budget, selector="quest_minmax",
                    quest_support="per_query_head", pinned_prefix_pages=0))

    def __enter__(self):
        self.patch = patch("basisserve.checkpoint.gqa_vo_qwen3.c1_k_reverse_shadow_block_attention", self.attention)
        self.patch.__enter__()
        return self

    def __exit__(self, *args):
        result = self.patch.__exit__(*args)
        for module in self.modules:
            module.set_reverse_shadow_config(None)
            module.attention_backend = "sdpa"
        return result

    def attention(self, query, keys, values, config, attention_mask, *, layer_idx, **kwargs):
        assert layer_idx >= 2 and query.shape[-2] == 1
        assert attention_mask.dtype == torch.bool and bool(attention_mask.all())

        def observe(mask):
            batch, heads, queries, length = mask.shape
            groups = keys.shape[1]
            count = min(length, max(3, self.budget // self.page_size) * self.page_size)
            assert bool((mask.sum(-1) <= count).all())
            physical = mask.reshape(batch, groups, heads // groups, queries, length).any(2)
            self.selected += int(physical.sum())
            self.group_queries += batch * groups * queries
            self.layer_calls[layer_idx] += 1

        output = official_attention(self.upstream, query, keys, values,
            page_size=self.page_size, budget=self.budget, observe=observe)
        return output, []

    def statistics(self):
        return {"layer_calls": self.layer_calls, "physical_selected_tokens_sum": self.selected,
            "kv_group_query_count": self.group_queries,
            "mean_physical_tokens_per_sparse_kv_group": self.selected / self.group_queries if self.group_queries else None}
