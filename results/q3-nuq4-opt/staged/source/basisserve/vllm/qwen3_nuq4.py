"""Qwen3-8B-Base TP8, adaptive C1, packed NUQ4 and A8/W8 decoder."""

import torch
from torch import nn
from vllm.distributed import get_pp_group, get_tp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.utils.torch_utils import direct_register_custom_op

from basisserve.core.qwen3_nuq4_artifacts import QwenNUQ4Artifacts
from basisserve.kernels.fp8_wire import quantize_e4m3_tensorwise_col_major
from basisserve.kernels.nuq4_cache import NUQ4PagedCache
from basisserve.kernels.nuq4_prefill import staged_prefill
from basisserve.kernels.nuq4_decode import nuq4_decode
from basisserve.vllm.qwen3_8b_c1 import C1QKVParallelLinear
from basisserve.vllm.nuq4_attention import nuq4_backend, serving_layout
from basisserve.vllm.nuq4_boundary import NUQ4Boundary


class NUQ4QKV(C1QKVParallelLinear):
    def _load_value(self, param, loaded_weight):
        assert param is self.weight and self.num_kv_heads == 1
        local = loaded_weight.narrow(0, self.tp_rank * 128, 128).to(self.encoder.device)
        compact = (self.encoder.float().T @ local.float()).to(param.dtype)
        offset = (self.num_heads + 1) * 128
        param.data.narrow(0, offset, self.value_rank).copy_(compact)
        self.value_loaded = True


class NUQ4Attention(nn.Module):
    def __init__(self, base, data, boundary, config, tp_rank):
        super().__init__()
        ranks = data["value_ranks"]
        assert len(set(ranks)) == 1, "Current archived schedules are uniform within each layer"
        self.value_rank = ranks[tp_rank]
        assert base.num_heads == 4 and base.num_kv_heads == 1
        assert base.dual_chunk_attention_config is None
        name = base.attn.layer_name
        assert config.compilation_config.static_forward_context.pop(name) is base.attn
        self.qkv_proj = NUQ4QKV(data["encoder"], name[:-5] + ".qkv_proj", 4096, 32)
        self.q_norm, self.k_norm, self.rotary_emb = base.q_norm, base.k_norm, base.rotary_emb
        self.attn = Attention(4, 128, base.scaling, num_kv_heads=1,
            cache_config=config.cache_config, prefix=name, head_size_v=self.value_rank,
            attn_backend=nuq4_backend(self.value_rank), value_rank=self.value_rank)
        object.__setattr__(self.attn, "_nuq4_owner", self)
        self.boundary = boundary
        fp8, scale = quantize_e4m3_tensorwise_col_major(data["decoder"])
        self.register_buffer("decoder_fp8", fp8, persistent=False)
        self.register_buffer("decoder_scale", scale, persistent=False)
        self.register_buffer("a8_scale", data["a8_scale"], persistent=False)
        self.kcache = NUQ4PagedCache(0, serving_layout(128), data["k_lower"], data["k_upper"], data["k_lut"])
        zeros = torch.zeros(self.value_rank, device=data["encoder"].device, dtype=torch.float32)
        self.vcache = NUQ4PagedCache(0, serving_layout(self.value_rank), zeros, zeros, data["v_lut"], dynamic=True)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((512, 128, self.value_rank), dim=-1)
        q = self.q_norm(q.reshape(-1, 4, 128)).reshape(-1, 512)
        k = self.k_norm(k.reshape(-1, 1, 128)).reshape(-1, 128)
        q, _ = self.rotary_emb(positions, q, None)
        return torch.ops.vllm.basisserve_nuq4_output(q, k, v, self.attn.layer_name)

    def run_attention(self, query, key, value, metadata, cache, slots):
        n = query.shape[0]
        stats = self.boundary.value_stats(value)
        output = self.boundary.attention_output(n, self.value_rank)
        if metadata is not None:
            assert not metadata.use_cascade
            assert cache.dtype == torch.uint8 and cache.shape[1:3] == (1, 1)
            raw = cache[:, 0, 0, :]
            split = self.kcache.layout.page_bytes
            self.kcache.bind(raw[:, :split])
            self.vcache.bind(raw[:, split:split+self.vcache.layout.page_bytes])
            actual = metadata.num_actual_tokens
            assert slots is not None and actual <= n
            self.kcache.append(key[:actual], slots[:actual])
            self.vcache.append(value[:actual], slots[:actual], stats[:actual])
            rope = self.rotary_emb._match_cos_sin_cache_dtype(query)
            if metadata.max_query_len > 1:
                staged_prefill(query[:actual].view(-1, 4, 128), self.kcache, self.vcache,
                    rope, metadata.query_start_loc, metadata.seq_lens, metadata.block_table,
                    metadata.nuq4_prefill_chunks, self.boundary.prefill_scratch, out=output[:actual])
            nuq4_decode(query[:actual].view(-1, 4, 128), self.kcache, self.vcache,
                rope, metadata.query_start_loc, metadata.seq_lens, metadata.block_table,
                out=output[:actual], workspace=self.boundary.decode_workspace(metadata.seq_lens.numel(), self.value_rank))
        return self.boundary.decode(output.flatten(1), self)


def nuq4_output(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                layer_name: str) -> torch.Tensor:
    metadata, layer, cache, slots = get_attention_context(layer_name)
    return layer._nuq4_owner.run_attention(query, key, value, metadata, cache, slots)


def nuq4_output_fake(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     layer_name: str) -> torch.Tensor:
    return query.new_empty((query.shape[0], 4096))


direct_register_custom_op(op_name="basisserve_nuq4_output", op_func=nuq4_output,
                          fake_impl=nuq4_output_fake)


class BasisServeQwen3NUQ4ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        cfg = vllm_config.model_config.hf_config
        assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.head_dim, cfg.num_hidden_layers) == (4096, 32, 8, 128, 36)
        group = get_tp_group()
        assert group.world_size == 8 and get_pp_group().world_size == 1
        assert vllm_config.parallel_config.decode_context_parallel_size == 1
        assert vllm_config.parallel_config.data_parallel_size == 1
        assert vllm_config.model_config.dtype == torch.bfloat16
        assert vllm_config.quant_config is None and vllm_config.lora_config is None
        assert vllm_config.speculative_config is None
        assert not vllm_config.cache_config.enable_prefix_caching
        assert vllm_config.cache_config.block_size == 16
        assert vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        artifacts = QwenNUQ4Artifacts(cfg.basisserve_nuq4_prior, cfg.basisserve_nuq4_rank)
        assert all(len(set(row)) == 1 for row in artifacts.schedule)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        device = next(self.parameters()).device
        self.nuq4_boundary = NUQ4Boundary(group.device_group, device,
            vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.model_config.max_model_len)
        for index, layer in enumerate(self.model.layers):
            data = artifacts.layer(index, group.rank_in_group, device)
            layer.self_attn = NUQ4Attention(layer.self_attn, data, self.nuq4_boundary, vllm_config, group.rank_in_group)

    def load_weights(self, weights):
        skips = [f"model.layers.{i}.self_attn.o_proj." for i in range(36)]
        if self.config.tie_word_embeddings:
            skips.append("lm_head.")
        return AutoWeightsLoader(self).load_weights(
            (name, tensor) for name, tensor in weights if not name.startswith(tuple(skips)))

    def c1_statistics(self):
        layers = [layer.self_attn for layer in self.model.layers]
        return dict(loaded_value_layers=sum(layer.qkv_proj.value_loaded for layer in layers),
            capture_calls=self.nuq4_boundary.capture_calls, eager_calls=self.nuq4_boundary.eager_calls,
            overflow_layers=[i for i, layer in enumerate(layers) if int(layer.kcache.failed) or int(layer.vcache.failed)],
            overflow_demand=[dict(k=int(layer.kcache.failed), v=int(layer.vcache.failed)) for layer in layers],
            exceptions_per_token=layers[0].kcache.layout.exceptions_per_token,
            value_ranks=[layer.value_rank for layer in layers],
            page_bytes=[layer.kcache.layout.page_bytes + layer.vcache.layout.page_bytes for layer in layers],
            allocated_page_strides=[layer.kcache.storage.stride(0) for layer in layers],
            packed_kv=True, encoder="bf16", latent="e4m3", decoder="w8a8",
            decode_kernel="nuq4_split_k",
            prefill_kernel="bounded_dequant_diffkv",
            prefill_workspace_bytes=self.nuq4_boundary.prefill_scratch.nbytes,
            v_statistics="global BF16 AllGather then per-token NUQ4 stats",
            prepared_shapes=len(self.nuq4_boundary.prepared))
