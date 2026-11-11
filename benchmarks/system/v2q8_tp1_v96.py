"""V96 Full-K, LRQK and ShadowKV TP1 runtimes on the paper's Table 2 quality code path.

Model surgery and per-layer attention are the quality evaluator's own (evaluation/eval_k_routing_ruler.install,
basisserve/checkpoint/c1_lrqk_qwen3.py, basisserve/checkpoint/c1_shadowkv_qwen3.py, basisserve/core/c1_*.py).
Two storage changes remove per-step whole-cache copies without changing any arithmetic:
  * PreallocatedLayer replaces DynamicLayer's torch.cat by in-place writes into storage sized once;
  * PreallocatedLRQKState writes each decode step's Key code into preallocated storage instead of torch.cat.
ShadowKV reconstructs historical Keys from its factors, so its cache keeps no exact historical Keys
(the quality reference kept them resident but never read them during decode).
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache
from transformers.cache_utils import DynamicLayer

from basisserve.core import c1_lrqk
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk_adapter
from basisserve.checkpoint import c1_shadowkv_qwen3 as shadowkv_adapter
from evaluation import eval_k_routing_ruler as ruler
from evaluation.chunked_prefill_mlp import install_chunked_prefill_mlps

LRQK_TOPK, SHADOWKV_BUDGET = 832, 2048  # paper Table 2 / Appendix C.1
CAPACITY = {"tokens": 0}  # set by build(); sizes LRQK Key-code storage


class PreallocatedLayer(DynamicLayer):
    """DynamicLayer semantics with storage allocated on first use; update writes in place and returns views."""

    def __init__(self, capacity, keep_keys):
        super().__init__()
        self.capacity, self.keep_keys, self.length = capacity, keep_keys, 0

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        batch, heads = key_states.shape[:2]
        self.value_storage = torch.empty(batch, heads, self.capacity, value_states.shape[-1],
                                         dtype=self.dtype, device=self.device)
        self.key_storage = (torch.empty(batch, heads, self.capacity, key_states.shape[-1],
                                        dtype=self.dtype, device=self.device) if self.keep_keys else None)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        start, stop = self.length, self.length + key_states.shape[-2]
        assert stop <= self.capacity
        self.value_storage[:, :, start:stop].copy_(value_states)
        self.values = self.value_storage[:, :, :stop]
        if self.keep_keys:
            self.key_storage[:, :, start:stop].copy_(key_states)
            self.keys = self.key_storage[:, :, :stop]
        else:
            self.keys = key_states
        self.length = stop
        return self.keys, self.values

    def get_seq_length(self):
        return self.length

    def reset(self):
        self.length = 0


def preallocate(cache, capacity, keep_keys):
    cache.layers = [PreallocatedLayer(capacity, keep_keys) for _ in cache.layers]
    return cache


class PreallocatedLRQKState(c1_lrqk.LRQKState):
    """c1_lrqk.LRQKState with the Key-code history in preallocated storage; decode arithmetic is unchanged."""

    def __init__(self, q, k, config, layer=0):
        super().__init__(q, k, config, layer)
        self.code_storage = torch.empty(*self.ak.shape[:2], CAPACITY["tokens"], self.ak.shape[-1],
                                        dtype=self.ak.dtype, device=self.ak.device)
        self.code_storage[:, :, :self.length].copy_(self.ak)
        self.ak = self.code_storage[:, :, :self.length]

    @torch.inference_mode()
    def decode(self, q, k, v, scale):
        assert q.shape[2] == 1 and k.shape[2] == v.shape[2] == self.length + 1
        assert self.selected.max() < self.length and self.ak.shape[2] == self.length
        active_k = c1_lrqk.gather_heads(k, self.selected)
        active_ak = self.ak.gather(2, self.selected[..., None].expand(-1, -1, -1, self.config.rank))
        current_k = k[:, :, -1:].repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        factors = c1_lrqk.decode_factors(self.bq.float(), active_ak.float(), self.bk.float(),
            active_k.float(), q.float(), current_k.float(), self.config.decode_iterations, self.config.tolerance)
        self.bq, self.bk, qcode, kcode = [t.to(q.dtype) for t in factors]
        assert all(torch.isfinite(t).all() for t in (self.bq, self.bk, qcode, kcode))
        self.code_storage[:, :, self.length:self.length + 1].copy_(kcode)
        self.length += 1
        self.steps += 1
        self.ak = self.code_storage[:, :, :self.length]
        self.selected = c1_lrqk.select_tokens(qcode, self.ak, self.config)
        output = c1_lrqk.selected_attention(q, k, v, self.selected, scale)
        assert torch.isfinite(output).all()
        return output


def build(arm, *, model_path, value_checkpoint, capacity, preallocated=True):
    """arm in {"fullk_v96", "lrqk_v96", "shadowkv_v96"}; returns the timing-harness runtime namespace.

    preallocated=False keeps the quality evaluator's DynamicCache and reference LRQKState (equivalence checks)."""
    from types import SimpleNamespace

    assert arm in ("fullk_v96", "lrqk_v96", "shadowkv_v96")
    CAPACITY["tokens"] = capacity
    ruler.LRQK_TOPK = LRQK_TOPK
    shadowkv_adapter.SHADOWKV_BUDGET = SHADOWKV_BUDGET
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).cuda().eval()
    install_chunked_prefill_mlps(model)
    checkpoint = Path(value_checkpoint)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    quality_arm = {"fullk_v96": "full", "lrqk_v96": "lrqk", "shadowkv_v96": "shadowkv"}[arm]
    ruler.install(model, checkpoint, manifest, quality_arm, None)
    if preallocated:
        # Same prefill kernel as the BasisKV arm: FlashAttention with V zero-padded to the Key width (exact;
        # the evaluator's own flash_prefill_attention). The Triton compressed-V prefill is under half its speed.
        for owner in (ruler, lrqk_adapter, shadowkv_adapter):
            owner.compressed_v_prefill_attention = ruler.flash_prefill_attention
    if arm == "lrqk_v96" and preallocated:
        # install() rebinds the adapter's state class to the reference one; swap in the preallocated subclass after it.
        lrqk_adapter.LRQKState = PreallocatedLRQKState
    kind, keep_keys, states_name = {
        "lrqk_v96": (lambda: lrqk_adapter.C1LRQKCache(config=model.config), True, "lrqk_states"),
        "shadowkv_v96": (lambda: shadowkv_adapter.C1ShadowKVCache(model.config), False, "shadow_states"),
        "fullk_v96": (lambda: ruler.RoutingCache(model.config), True, "statistics")}[arm]
    holder = {}

    def new_cache():
        cache = kind()
        holder["cache"] = preallocate(cache, capacity, keep_keys) if preallocated else cache

    new_cache()

    def prefill(tokens):
        return model(input_ids=tokens, past_key_values=holder["cache"], use_cache=True, logits_to_keep=1).logits

    def decode(token):
        cache = holder["cache"]
        mask = torch.ones(1, 1, 1, cache.get_seq_length() + 1, device=token.device, dtype=torch.bool)
        return model(input_ids=token, past_key_values=cache, attention_mask=mask, use_cache=True,
                     logits_to_keep=1).logits

    def reset():
        if not preallocated:
            new_cache()
            return
        for layer in holder["cache"].layers:
            layer.reset()
        getattr(holder["cache"], states_name).clear()

    configuration = dict(
        storage="GPU V96; GPU factors" if arm == "shadowkv_v96" else "GPU-local K/V96",
        value="compressed V96 (paper Table 2 shared checkpoint)", value_checkpoint=str(checkpoint),
        quality_code_path=f"evaluation/eval_k_routing_ruler.install arm={quality_arm}",
        storage_change="preallocated cache storage; no per-step whole-cache torch.cat",
        prefill_attention="FlashAttention, V zero-padded to 128 (same as BasisKV)" if preallocated else
                          "evaluator Triton compressed-V prefill",
        exact_keys="not stored after prefill (reconstructed from factors)" if arm == "shadowkv_v96" else "GPU-resident",
        values="GPU-resident V96")
    if arm == "lrqk_v96":
        configuration.update(rank=32, topk_per_query_head=LRQK_TOPK, recent=64, iterations=(2, 2), tolerance=0.01)
    if arm == "shadowkv_v96":
        configuration.update(rank=160, chunk=8, routed=SHADOWKV_BUDGET, outlier_chunks=48, local_chunks=4)
    return SimpleNamespace(prefill=prefill, decode=decode, place=lambda: None, reset=reset,
                           state=lambda: holder["cache"], model=model,
                           states=lambda: getattr(holder["cache"], states_name), configuration=configuration)
