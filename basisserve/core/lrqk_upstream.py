"""The upstream LRQK runtime (tenghuilee/LRQK@caf16293: load_model_hack + DynamicLRQKCache +
LightAttentionIndicesOffloadPrefill), imported from external/LRQK without edits, with one requested placement
deviation: Values stay on the GPU. Both placements order prefill cache parking after its producer and record
the source tensors on the copy stream; the upstream parking method omits these stream dependencies.

Upstream keeps the exact K and V of every token in one pinned CPU pool (AutoIncreaseTensorSharedKV, K and V
concatenated) and fetches missed K and V into the GPU hit cache. The V2Q8 design offloads Keys only:
  * SplitKV takes the pool's place with the interface the factory uses (append(k, v), .data): Keys in an
    upstream AutoIncreaseTensor in pinned host memory, Values in an upstream PreAllocatedTensor on the GPU;
  * SplitKV holds the GPU Values in upstream's PreAllocatedTensor at the cache's capacity; the Key pool keeps
    upstream's 1.5x growth policy on the host;
  * KOnlyFactory.decode is upstream LightAttentionIndicesFactory.decode with its miss fetch split. The linear
    indices are computed once, with upstream's compiled helper, as upstream's take_along_dim_with_mask_python
    does. Missed Keys go through cpp_kernel.d0_index_select on the host and then H2D; missed Values are gathered
    on the GPU with index_select.
Everything else is upstream: torch.compile on its solvers and index helper, the ALS prefill fit and decode
updates, hit/miss logic, parking the hit cache and A_K in host memory after each layer's prefill with the restore
at that layer's first decode step, torch.cuda.empty_cache() after every decode layer, and the whole batch
prefilled in one forward.

Tensor parallelism (W > 1): the HF tensor-parallel model, with upstream's attention forward bound by
load_model_hack; each attention module carries its local head counts (upstream reads num_heads and
num_key_value_heads from the module before the config). LRQK has no cross-head communication, so each rank's
cache holds its local heads. Two upstream whole-tensor operations then cover the local heads only: the random
A_Q/A_K initialisation draws them with the same seed on every rank, and the early-exit delta is their mean.
"""

from contextlib import nullcontext
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(rank=32, topk=2048, lite=64, iterations=(2, 2), tolerance=(1e-2, 1e-2))  # paper RULER-128K
# Breakdown timers for the split miss fetch (set by the benchmark; a no-op otherwise).
PHASE = lambda label: nullcontext()


def upstream():
    """The official module; its flash_attn import is served by vLLM's FlashAttention-2 build."""
    if "lrqk_attention" not in sys.modules:
        from benchmarks.system.paper_faithful_compat import install_flash_attn_adapter
        install_flash_attn_adapter()
        sys.path.insert(0, str(ROOT / "external/LRQK"))
        sys.path.insert(0, str(ROOT / "external/LRQK/cpp_kernel"))
    import lrqk_attention
    return lrqk_attention


def ordered_offload_prefill():
    """Upstream cache parking with producer ordering and allocator lifetime tracking."""
    module = upstream()

    class OrderedOffloadPrefill(module.LightAttentionIndicesOffloadPrefill):
        def prefill(self, query_states, key_states, value_states):
            module.LightAttentionIndicesFactory.prefill(self, query_states, key_states, value_states)
            self.stream_temp = torch.cuda.Stream(query_states.device)
            self.stream_temp.wait_stream(torch.cuda.current_stream(query_states.device))
            with torch.cuda.stream(self.stream_temp):
                self.Kgpu_temp = self.Kgpu.to("cpu", non_blocking=True)
                self.Vgpu_temp = self.Vgpu.to("cpu", non_blocking=True)
                self._A_K_data_temp = self._A_K.data.to("cpu", non_blocking=True)
                for tensor in (self.Kgpu, self.Vgpu, self._A_K.data):
                    tensor.record_stream(self.stream_temp)
                self._A_K.data = None
                self.Kgpu = None
                self.Vgpu = None

    return OrderedOffloadPrefill


def make_classes():
    module = upstream()
    repeat_kv = module.modeling_qwen2.repeat_kv

    class SplitKV:
        """In place of upstream AutoIncreaseTensorSharedKV: Keys on pinned host (the upstream pool and its 1.5x
        growth policy), Values on GPU in upstream's PreAllocatedTensor at the cache's capacity. The 1.5x headroom
        is upstream's host-memory policy; applied to a GPU Value store it would reserve 50% more than the Values
        need (about 4 GiB at 130K on one GPU)."""

        def __init__(self, capacity, scaling_ratio):
            self.keys = module.AutoIncreaseTensor(capacity=capacity, dim=2, scaling_ratio=scaling_ratio,
                                                  device="cpu", pin_memory=True)
            self.values = module.PreAllocatedTensor(capacity, dim=2)

        def append(self, k, v):
            self.keys.append(k)
            self.values.append(v)

        @property
        def data(self):
            return self.keys.data

        def __len__(self):
            return self.keys.current_len

    class KOnlyFactory(module.LightAttentionIndicesFactory):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.KVcpu = SplitKV(self.KVcpu.capacity, self.KVcpu.scaling_ratio)

        def decode(self, query_states, key_states, value_states):
            """Upstream LightAttentionIndicesFactory.decode; only the miss fetch at the end differs."""
            self._append(key_states, value_states)

            rep_key_states = repeat_kv(
                key_states,
                self.num_key_value_groups,
            )

            if self.current_len <= self.gpu_capacity:
                # need to update B_Q, B_K, hat_q_t, hat_k_t
                B_Q, B_K, hat_q_t, hat_k_t = module.cast_lrqk_decode(
                    B_Q=self.B_Q,
                    A_K=self.A_K,
                    B_K=self.B_K,
                    K__=self.Kgpu.narrow(self.dim, 0, self.current_len-1),
                    q=query_states,
                    k=rep_key_states,
                    lambda_dec_q=self.weights[2],
                    lambda_dec_k=self.weights[3],
                    max_iter=self.max_iter[1],
                    tol=self.tol[1],
                )
                self.sync_cache(B_Q, hat_k_t, B_K)

                return (
                    self.Kgpu.narrow(self.dim, 0, self.current_len),
                    self.Vgpu.narrow(self.dim, 0, self.current_len),
                )
            else:
                # need to update B_Q, B_K, hat_q_t, hat_k_t
                B_Q, B_K, hat_q_t, hat_k_t = module.cast_lrqk_decode(
                    B_Q=self.B_Q,
                    A_K=self._get_A_K_hitted_with_lite(),
                    B_K=self.B_K,
                    K__=self.Kgpu,
                    q=query_states,
                    k=rep_key_states,
                    lambda_dec_q=self.weights[2],
                    lambda_dec_k=self.weights[3],
                    max_iter=self.max_iter[1],
                    tol=self.tol[1],
                )
                self.sync_cache(B_Q, hat_k_t, B_K)

            hit_indices = self.compute_hit_indices(
                query_states=hat_q_t,
                key_states=self.A_K.narrow(
                    self.dim,
                    0,
                    self.lite_indices[self.lite_len] - 1,
                )
            )

            hit_indices, dst_mask = self.only_indices_in_cpu(
                self.hit_indices, hit_indices)
            self.hit_indices = hit_indices

            # manual fetch by hit
            dstK = self.Kgpu.narrow(self.dim, 0, self.attn_topk)
            dstV = self.Vgpu.narrow(self.dim, 0, self.attn_topk)

            bsz, kvheads, _, hidden_size = key_states.shape

            _indices = hit_indices.view(
                bsz, kvheads, self.num_key_value_groups, -1)
            _mask = dst_mask.view(bsz, kvheads, self.num_key_value_groups, -1)

            # Upstream take_along_dim_with_mask_python on its shared pool: linear indices once (compiled helper),
            # then cpp_kernel.d0_index_select on the host and H2D. Here the indices are computed once for the GPU
            # Value pool and remapped to the host Key pool, whose sequence capacity differs.
            keys, values = self.KVcpu.data, self.KVcpu.values.data
            with PHASE("miss_indices"):
                linear = module._take_along_dim_with_mask_python_indices(
                    values.shape[1], values.shape[2], _indices, _mask)
            with PHASE("key_miss_fetch_cpu_gather_h2d"):
                rows, positions = linear // values.shape[2], linear % values.shape[2]
                out = module.cpp_kernel.d0_index_select(
                    keys.view(-1, keys.size(-1)), (rows * keys.shape[2] + positions).to(keys.device)
                ).to(_indices.device)

            if out.numel() > 0:
                with PHASE("value_miss_gather_gpu"):
                    _out_v = values.view(-1, values.size(-1)).index_select(0, linear)

                dstK[dst_mask] = out
                dstV[dst_mask] = _out_v

            return self.Kgpu, self.Vgpu

    class KOnlyOffloadPrefill(ordered_offload_prefill(), KOnlyFactory):
        """Upstream LightAttentionIndicesOffloadPrefill (prefill parking, first-decode restore) over KOnlyFactory."""

    return module, SplitKV, KOnlyFactory, KOnlyOffloadPrefill


def install(model, *, world):
    """Bind upstream's attention forward (load_model_hack); under TP give each module its local head counts."""
    module = upstream()
    model.config._attn_implementation = "flash_attention_2"  # as system_benchmarks/tp1_offline: no 4D mask
    model = module.load_model_hack(model.config._name_or_path or "llama", device=None, base_model=model)
    if world > 1:
        for layer in model.model.layers:
            layer.self_attn.num_heads = model.config.num_attention_heads // world
            layer.self_attn.num_key_value_heads = model.config.num_key_value_heads // world
    return model


def new_cache(factory_class, *, max_sequence_length):
    """An upstream DynamicLRQKCache, configured as the paper's RULER-128K setting."""
    module = upstream()
    return module.DynamicLRQKCache(
        num_key_value_groups=4, r=CONFIG["rank"], num_active_tokens=CONFIG["topk"], lite_tokens=CONFIG["lite"],
        max_iter=CONFIG["iterations"], tol=CONFIG["tolerance"], max_sequence_length=max_sequence_length,
        lwattn_factory=factory_class, init_aq_ak_method=module.InitAQAK.randn)
