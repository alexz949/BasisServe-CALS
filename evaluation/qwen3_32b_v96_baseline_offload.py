"""Sparse payload offload for one-GPU Qwen3-32B V96 evaluation.

Loki and LRQK retain V96 on the GPU and fetch only selected exact-Key rows
from pinned host memory. ShadowKV retains reconstructed-Key state on the GPU,
stores no dense exact-Key cache, and fetches only selected V96 rows.
"""

from types import MethodType

import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.core.c1_k_offload import (
    PinnedCPUExactKeyPageStore,
    PreparedQueryKeyFetch,
)
from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
from basisserve.core.c1_loki_attention import c1_loki_recent_selection
from basisserve.core.c1_lrqk import LRQKState, decode_factors, select_tokens
from basisserve.core.c1_shadowkv import C1ShadowKVState
from basisserve.kernels.compressed_v_decode_attention import (
    compressed_v_decode_attention_triton,
    compressed_v_prefill_attention,
)
from basisserve.kernels.indexed_sparse_decode_attention import (
    gqa_indexed_sparse_decode_attention_triton,
)
from evaluation.chunked_prefill_mlp import (
    install_chunked_prefill_mlps,
    install_chunked_prefill_norms,
)


ARMS = ("shadowkv", "lrqk", "loki")


class PinnedSequence:
    def __init__(self, value, capacity):
        assert value.ndim == 4 and value.shape[2] <= capacity
        self.storage = torch.empty(
            (*value.shape[:2], capacity, value.shape[-1]),
            dtype=value.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.length = 0
        self.append(value)

    def append(self, value):
        assert value.shape[:2] == self.storage.shape[:2]
        assert value.shape[-1] == self.storage.shape[-1]
        stop = self.length + value.shape[2]
        assert stop <= self.storage.shape[2]
        self.storage[:, :, self.length : stop].copy_(value.detach())
        self.length = stop

    def view(self):
        return self.storage[:, :, : self.length]


class SelectedValueFetch:
    def __init__(self):
        self.host_ids = None
        self.staging = None
        self.destination = None
        self.last_bytes = 0

    def __call__(self, store, ids, device):
        assert ids.ndim == 3 and ids.shape[:2] == store.storage.shape[:2]
        batch, heads, count = ids.shape
        dim = store.storage.shape[-1]
        capacity = 1 << max(count - 1, 0).bit_length()
        shape = (batch, heads, capacity, dim)
        if self.staging is None or self.staging.shape != shape:
            self.host_ids = torch.empty(
                batch, heads, capacity, dtype=torch.int64, pin_memory=True
            )
            self.staging = torch.empty(
                shape,
                dtype=store.storage.dtype,
                device="cpu",
                pin_memory=True,
            )
            self.destination = torch.empty_like(self.staging, device=device)
        self.host_ids[:, :, :count].copy_(ids)
        torch.gather(
            store.storage,
            2,
            self.host_ids[:, :, :count, None].expand(batch, heads, count, dim),
            out=self.staging[:, :, :count],
        )
        self.destination[:, :, :count].copy_(
            self.staging[:, :, :count], non_blocking=True
        )
        self.last_bytes = batch * heads * count * dim * store.storage.element_size()
        return self.destination[:, :, :count]


class SelectiveCache(DynamicCache):
    def __init__(self, config, capacity):
        super().__init__(config=config)
        self.capacity = int(capacity)
        self.lengths = {}
        self.statistics = {}

    def get_seq_length(self, layer_idx=0):
        return self.lengths.get(layer_idx, 0)


class ShadowSelectiveCache(SelectiveCache):
    def __init__(self, config, capacity):
        super().__init__(config, capacity)
        self.shadow_states = {}
        self.host_values = {}
        self.value_fetch = SelectedValueFetch()

    def store_prefill(self, layer, value):
        store = PinnedSequence(value, self.capacity)
        self.host_values[layer] = store
        self.lengths[layer] = store.length
        self.layers[layer].keys = torch.empty(
            *value.shape[:2], 0, 1, dtype=value.dtype, device="cpu"
        )
        self.layers[layer].values = store.view()

    def append_value(self, layer, value):
        self.host_values[layer].append(value)
        self.lengths[layer] = self.host_values[layer].length
        self.layers[layer].values = self.host_values[layer].view()

    def selected_value(self, layer, ids, device):
        return self.value_fetch(self.host_values[layer], ids, device)


class KeySelectiveCache(SelectiveCache):
    def __init__(self, config, capacity, *, groups):
        super().__init__(config, capacity)
        self.groups = int(groups)
        self.host_keys = {}
        self.host_values = {}
        self.resident_values = {}
        self.key_fetch = None

    def store_prefill(self, layer, key, value):
        key_store = PinnedSequence(key, self.capacity)
        value_store = PinnedSequence(value, self.capacity)
        self.host_keys[layer] = key_store
        self.host_values[layer] = value_store
        self.lengths[layer] = key_store.length
        self.layers[layer].keys = key_store.view()
        self.layers[layer].values = value_store.view()

    def append_key_and_value(self, layer, key, value):
        self.host_keys[layer].append(key)
        if layer not in self.resident_values:
            host = self.host_values[layer]
            resident = torch.empty(
                host.storage.shape,
                dtype=host.storage.dtype,
                device=value.device,
            )
            resident[:, :, : host.length].copy_(host.view(), non_blocking=True)
            self.resident_values[layer] = resident
            del self.host_values[layer]
        resident = self.resident_values[layer]
        previous = self.lengths[layer]
        resident[:, :, previous : previous + 1].copy_(value)
        self.lengths[layer] = previous + 1
        self.layers[layer].keys = self.host_keys[layer].view()
        self.layers[layer].values = resident[:, :, : previous + 1]

    def value(self, layer):
        return self.resident_values[layer][:, :, : self.lengths[layer]]

    def fetch_exact(self, layer, ids, device):
        batch, query_heads, count = ids.shape
        kv_heads = self.host_keys[layer].storage.shape[1]
        assert query_heads == kv_heads * self.groups
        grouped = ids.reshape(batch, kv_heads, self.groups, count).contiguous()
        if self.key_fetch is None or self.key_fetch.shape[-1] != count:
            source = PinnedCPUExactKeyPageStore(
                self.host_keys[layer].storage, layer_idx=layer
            )
            self.key_fetch = PreparedQueryKeyFetch(
                source,
                groups=self.groups,
                tokens_per_query=count,
                device=device,
            )
        self.key_fetch.bank = self.host_keys[layer].storage.reshape(
            batch * kv_heads * self.capacity, -1
        )
        self.key_fetch(grouped)
        return self.key_fetch.destination, self.key_fetch.inverse_ids.reshape(
            batch, query_heads, count
        )


class LokiSelectiveCache(KeySelectiveCache):
    def __init__(self, config, capacity):
        groups = config.num_attention_heads // config.num_key_value_heads
        super().__init__(config, capacity, groups=groups)
        self.sidecar_storage = {}

    def store_sidecar(self, layer, sidecar):
        storage = torch.empty(
            (*sidecar.shape[:2], self.capacity, sidecar.shape[-1]),
            dtype=sidecar.dtype,
            device=sidecar.device,
        )
        storage[:, :, : sidecar.shape[2]].copy_(sidecar)
        self.sidecar_storage[layer] = storage

    def append_sidecar(self, layer, sidecar, position):
        self.sidecar_storage[layer][:, :, position : position + 1].copy_(sidecar)

    def sidecar(self, layer):
        return self.sidecar_storage[layer][:, :, : self.lengths[layer]]


class LRQKSelectiveCache(KeySelectiveCache):
    def __init__(self, config, capacity):
        groups = config.num_attention_heads // config.num_key_value_heads
        super().__init__(config, capacity, groups=groups)
        self.lrqk_states = {}
        self.host_codes = {}
        self.code_buffer = None

    def store_codes(self, layer, state):
        store = PinnedSequence(state.ak, self.capacity)
        self.host_codes[layer] = store
        state.ak = store.view()

    def load_codes(self, layer, device):
        store = self.host_codes[layer]
        if self.code_buffer is None:
            self.code_buffer = torch.empty(
                store.storage.shape,
                dtype=store.storage.dtype,
                device=device,
            )
        self.code_buffer[:, :, : store.length].copy_(
            store.view(), non_blocking=True
        )
        return self.code_buffer[:, :, : store.length]

    def append_code(self, layer, code):
        self.host_codes[layer].append(code)
        self.lrqk_states[layer].ak = self.host_codes[layer].view()


def create_cache(config, arm, capacity):
    assert arm in ARMS and capacity > 0
    if arm == "shadowkv":
        return ShadowSelectiveCache(config, capacity)
    if arm == "loki":
        return LokiSelectiveCache(config, capacity)
    return LRQKSelectiveCache(config, capacity)


def _qkv(attention, hidden_states, position_embeddings):
    batch, length, _ = hidden_states.shape
    query = attention.q_norm(
        attention.q_proj(hidden_states).view(
            batch, length, attention.num_attention_heads, attention.head_dim
        )
    ).transpose(1, 2)
    pre_key = attention.k_norm(
        attention.k_proj(hidden_states).view(
            batch, length, attention.num_key_value_heads, attention.head_dim
        )
    ).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(
        batch, length, attention.num_key_value_heads, attention.value_head_dim
    ).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, pre_key, cos, sin)
    return query, pre_key, key, value, cos, sin


def _validate(attention_mask, query, previous):
    if attention_mask is None:
        return
    assert attention_mask.ndim == 4
    length = query.shape[2]
    expected = torch.arange(previous + length, device=query.device)[None, :] <= (
        previous + torch.arange(length, device=query.device)[:, None]
    )
    valid = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    assert torch.equal(
        valid.expand(query.shape[0], 1, length, previous + length)[0, 0], expected
    )


@torch.inference_mode()
def shadow_forward(self, hidden_states, position_embeddings, attention_mask=None,
                   past_key_values=None, **kwargs):
    assert isinstance(past_key_values, ShadowSelectiveCache)
    assert hidden_states.shape[0] == 1 and not kwargs.get("output_attentions", False)
    query, pre_key, key, value, cos, sin = _qkv(
        self, hidden_states, position_embeddings
    )
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or hidden_states.shape[1] == 1
    _validate(attention_mask, query, previous)
    if previous == 0:
        state = C1ShadowKVState(pre_key, key, cos, sin)
        past_key_values.shadow_states[self.layer_idx] = state
        output = compressed_v_prefill_attention(query, key, value, scale=self.scaling)
        past_key_values.store_prefill(self.layer_idx, value)
    else:
        past_key_values.append_value(self.layer_idx, value)
        state = past_key_values.shadow_states[self.layer_idx]
        selected_key, selected_ids = state.route(query, key)
        selected_value = past_key_values.selected_value(
            self.layer_idx, selected_ids, query.device
        )
        output = compressed_v_decode_attention_triton(
            query, selected_key, selected_value, scale=self.scaling
        )
        past_key_values.statistics[self.layer_idx] = {
            **state.statistics(),
            "exact_key_storage": "not_stored",
            "value_storage": "pinned_cpu",
            "value_bytes_fetched": past_key_values.value_fetch.last_bytes,
        }
    output = output.transpose(1, 2).contiguous().reshape(
        hidden_states.shape[0], hidden_states.shape[1], -1
    )
    return self.o_proj(output), None


@torch.inference_mode()
def loki_forward(self, hidden_states, position_embeddings, attention_mask=None,
                 past_key_values=None, **kwargs):
    assert isinstance(past_key_values, LokiSelectiveCache)
    assert hidden_states.shape[0] == 1 and not kwargs.get("output_attentions", False)
    query, _, key, value, _, _ = _qkv(self, hidden_states, position_embeddings)
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or hidden_states.shape[1] == 1
    _validate(attention_mask, query, previous)
    current_sidecar = build_routing_sidecar(key, self._loki_projector)
    if previous == 0:
        output = compressed_v_prefill_attention(query, key, value, scale=self.scaling)
        past_key_values.store_prefill(self.layer_idx, key, value)
        past_key_values.store_sidecar(self.layer_idx, current_sidecar)
    else:
        past_key_values.append_key_and_value(self.layer_idx, key, value)
        past_key_values.append_sidecar(self.layer_idx, current_sidecar, previous)
        selected, statistics = c1_loki_recent_selection(
            query,
            self._loki_projector,
            past_key_values.sidecar(self.layer_idx),
            length=previous + 1,
            top_k=856,
            recent_tokens=0,
            scale=self.scaling,
        )
        packed_key, packed_rows = past_key_values.fetch_exact(
            self.layer_idx, selected, query.device
        )
        output = gqa_indexed_sparse_decode_attention_triton(
            query,
            packed_key,
            past_key_values.value(self.layer_idx),
            selected,
            selected_key_rows=packed_rows,
            scale=self.scaling,
        )
        past_key_values.statistics[self.layer_idx] = {
            **statistics,
            **past_key_values.key_fetch.traffic(),
            "exact_key_storage": "pinned_cpu",
            "value_storage": "cuda",
            "routing_state_storage": "cuda",
        }
    output = output.transpose(1, 2).contiguous().reshape(
        hidden_states.shape[0], hidden_states.shape[1], -1
    )
    return self.o_proj(output), None


@torch.inference_mode()
def lrqk_forward(self, hidden_states, position_embeddings, attention_mask=None,
                 past_key_values=None, **kwargs):
    assert isinstance(past_key_values, LRQKSelectiveCache)
    assert hidden_states.shape[0] == 1 and not kwargs.get("output_attentions", False)
    query, _, key, value, _, _ = _qkv(self, hidden_states, position_embeddings)
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or hidden_states.shape[1] == 1
    _validate(attention_mask, query, previous)
    config = self._lrqk_config
    if previous == 0:
        state = LRQKState(query, key, config, self.layer_idx)
        past_key_values.lrqk_states[self.layer_idx] = state
        output = compressed_v_prefill_attention(query, key, value, scale=self.scaling)
        past_key_values.store_prefill(self.layer_idx, key, value)
        past_key_values.store_codes(self.layer_idx, state)
    else:
        past_key_values.append_key_and_value(self.layer_idx, key, value)
        state = past_key_values.lrqk_states[self.layer_idx]
        assert state.length == previous and state.selected.max() < previous
        codes = past_key_values.load_codes(self.layer_idx, query.device)
        prior_key, prior_rows = past_key_values.fetch_exact(
            self.layer_idx, state.selected, query.device
        )
        active_key = prior_key[prior_rows].reshape(
            query.shape[0], query.shape[1], state.selected.shape[-1], query.shape[-1]
        )
        active_code = codes.gather(
            2,
            state.selected[..., None].expand(-1, -1, -1, config.rank),
        )
        current_key = key.repeat_interleave(
            query.shape[1] // key.shape[1], dim=1
        )
        factors = decode_factors(
            state.bq.float(),
            active_code.float(),
            state.bk.float(),
            active_key.float(),
            query.float(),
            current_key.float(),
            config.decode_iterations,
            config.tolerance,
        )
        state.bq, state.bk, query_code, key_code = [
            tensor.to(query.dtype) for tensor in factors
        ]
        past_key_values.append_code(self.layer_idx, key_code)
        past_key_values.code_buffer[:, :, previous : previous + 1].copy_(key_code)
        state.length = previous + 1
        state.steps += 1
        full_codes = past_key_values.code_buffer[:, :, : state.length]
        state.selected = select_tokens(query_code, full_codes, config)
        selected_key, selected_rows = past_key_values.fetch_exact(
            self.layer_idx, state.selected, query.device
        )
        output = gqa_indexed_sparse_decode_attention_triton(
            query,
            selected_key,
            past_key_values.value(self.layer_idx),
            state.selected,
            selected_key_rows=selected_rows,
            scale=self.scaling,
        )
        past_key_values.statistics[self.layer_idx] = {
            **state.statistics(self.num_key_value_heads),
            **past_key_values.key_fetch.traffic(),
            "exact_key_storage": "pinned_cpu",
            "value_storage": "cuda",
            "key_code_storage": "pinned_cpu_with_shared_cuda_workspace",
        }
    output = output.transpose(1, 2).contiguous().reshape(
        hidden_states.shape[0], hidden_states.shape[1], -1
    )
    return self.o_proj(output), None


def install(model, arm):
    assert arm in ARMS
    install_chunked_prefill_mlps(model)
    install_chunked_prefill_norms(model)
    selected = {
        "shadowkv": shadow_forward,
        "lrqk": lrqk_forward,
        "loki": loki_forward,
    }[arm]
    for _, attention in c1_attention_layers(model):
        attention.forward = MethodType(selected, attention)


def audit_selective_residency(cache, arm, layers):
    assert arm in ARMS and len(cache.lengths) == layers
    for layer in range(layers):
        assert cache.lengths[layer] > 0
        if arm == "shadowkv":
            assert cache.layers[layer].keys.shape[2] == 0
            assert cache.layers[layer].values.device.type == "cpu"
            state = cache.shadow_states[layer]
            assert all(
                value.device.type == "cuda"
                for value in vars(state).values()
                if isinstance(value, torch.Tensor)
            )
        else:
            assert cache.layers[layer].keys.device.type == "cpu"
            if layer in cache.resident_values:
                assert cache.layers[layer].values.device.type == "cuda"
            else:
                assert not cache.statistics
                assert cache.layers[layer].values.device.type == "cpu"
            if arm == "loki":
                assert cache.sidecar_storage[layer].device.type == "cuda"
            else:
                state = cache.lrqk_states[layer]
                assert state.ak.device.type == "cpu"
                assert state.bq.device.type == state.bk.device.type == "cuda"


def cache_shapes(cache, arm, layer):
    length = cache.lengths[layer]
    value = cache.layers[layer].values
    key = None if arm == "shadowkv" else cache.layers[layer].keys
    return None if key is None else tuple(key.shape), tuple(value.shape), length


def residency(arm):
    if arm == "shadowkv":
        return {
            "exact_key": "not_stored",
            "value": "pinned_cpu_selected_rows_only",
            "routing_state": "gpu",
        }
    if arm == "loki":
        return {
            "exact_key": "pinned_cpu_selected_rows_only",
            "value": "gpu",
            "routing_state": "gpu",
        }
    return {
        "exact_key": "pinned_cpu_selected_rows_only",
        "value": "gpu",
        "routing_state": "hybrid_cpu_codes_shared_gpu_workspace",
    }
