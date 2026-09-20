"""Qwen3-8B TP8 compact V64 serving with prepared NCCL and CUDA Graphs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn

from basisserve.core.qwen3_8b_vllm_c1 import (
    HEAD_DIM, NUM_KV_HEADS, TP_SIZE, VALUE_RANK,
    fold_value_weight, load_layer, load_manifest,
)
from basisserve.vllm.prepared_c1_output import PreparedC1Output
from basisserve.vllm.diffkv_attention import C1DiffKVBackend
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.v1.attention.backends.triton_attn_diffkv import TritonAttentionDiffKVBackend


class C1QKVParallelLinear(QKVParallelLinear):
    def __init__(self, encoder: Tensor, prefix: str, hidden_size: int, num_query_heads: int):
        super().__init__(hidden_size, HEAD_DIM, num_query_heads, NUM_KV_HEADS,
                         bias=False, prefix=prefix, v_head_size=VALUE_RANK)
        self.register_buffer("encoder", encoder, persistent=False)
        self.value_loaded = False

    def _load_value(self, param, loaded_weight):
        assert param is self.weight and self.num_kv_heads == 1
        local = loaded_weight.narrow(0, self.tp_rank * HEAD_DIM, HEAD_DIM).to(self.encoder.device)
        compact = fold_value_weight(local, self.encoder)
        offset = (self.num_heads + self.num_kv_heads) * self.head_size
        param.data.narrow(0, offset, VALUE_RANK).copy_(compact)
        self.value_loaded = True

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in ("q", "k", "v")
        if loaded_shard_id == "v":
            self._load_value(param, loaded_weight)
        else:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in ("q", "k", "v")
        if loaded_shard_id == "v":
            self._load_value(param, loaded_weight)
        else:
            super().weight_loader(param, loaded_weight, loaded_shard_id)


class C1Attention(nn.Module):
    def __init__(self, base, encoder, decoder, boundary, config):
        super().__init__()
        assert base.num_heads in (4, 8) and base.num_kv_heads == 1
        self.num_heads = base.num_heads
        assert base.dual_chunk_attention_config is None
        name = base.attn.layer_name
        assert name.endswith(".attn")
        assert config.compilation_config.static_forward_context.pop(name) is base.attn
        self.qkv_proj = C1QKVParallelLinear(encoder, name[:-5] + ".qkv_proj",
                                          config.model_config.hf_config.hidden_size,
                                          config.model_config.hf_config.num_attention_heads)
        self.q_norm, self.k_norm, self.rotary_emb = base.q_norm, base.k_norm, base.rotary_emb
        TritonAttentionDiffKVBackend.set_head_size_v(VALUE_RANK)
        self.attn = Attention(
            self.num_heads, HEAD_DIM, base.scaling, num_kv_heads=1,
            cache_config=config.cache_config, prefix=name,
            attn_backend=C1DiffKVBackend, head_size_v=VALUE_RANK,
        )
        object.__setattr__(self.attn, "_basisserve_c1_output", boundary)
        self.register_buffer("decoder", decoder, persistent=False)

    def forward(self, positions: Tensor, hidden_states: Tensor) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.num_heads * HEAD_DIM, HEAD_DIM, VALUE_RANK), dim=-1)
        q = self.q_norm(q.reshape(-1, self.num_heads, HEAD_DIM)).reshape(-1, self.num_heads * HEAD_DIM)
        k = self.k_norm(k.reshape(-1, 1, HEAD_DIM)).reshape(-1, HEAD_DIM)
        q, k = self.rotary_emb(positions, q, k)
        coordinates = self.attn(q, k, v)
        return torch.ops.vllm.basisserve_prepared_c1_output(
            coordinates, self.decoder, self.attn.layer_name,
        )


class BasisServeQwen3_8BC1ForCausalLM(Qwen3ForCausalLM):
    model_geometry = (4096, 32, 8, 128, 36)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        cfg = vllm_config.model_config.hf_config
        assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.head_dim, cfg.num_hidden_layers) == self.model_geometry
        assert not cfg.attention_bias
        group = get_tp_group()
        assert group.world_size == TP_SIZE and get_pp_group().world_size == 1
        assert vllm_config.parallel_config.decode_context_parallel_size == 1
        assert vllm_config.parallel_config.data_parallel_size == 1
        assert vllm_config.quant_config is None and vllm_config.lora_config is None
        assert vllm_config.model_config.dtype == torch.bfloat16
        assert vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        assert vllm_config.speculative_config is None
        root = Path(cfg.basisserve_c1_factor_dir).expanduser().resolve()
        manifest = load_manifest(root, cfg.basisserve_c1_result_sha256)
        fit = manifest["fit_config"]
        assert (fit["hidden_size"], fit["num_query_heads"], fit["num_hidden_layers"]) == (
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        device = next(self.parameters()).device
        self.c1_boundary = PreparedC1Output(
            group.device_group, device, cfg.num_attention_heads // TP_SIZE * VALUE_RANK,
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        for index, layer in enumerate(self.model.layers):
            encoder, decoder = load_layer(root, manifest, index, group.rank_in_group,
                                           device=device, dtype=torch.bfloat16)
            layer.self_attn = C1Attention(layer.self_attn, encoder, decoder,
                                          self.c1_boundary, vllm_config)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        skips = [f"model.layers.{layer}.self_attn.o_proj." for layer in range(self.config.num_hidden_layers)]
        if self.config.tie_word_embeddings:
            skips.append("lm_head.")
        filtered = ((name, tensor) for name, tensor in weights if not name.startswith(tuple(skips)))
        return AutoWeightsLoader(self).load_weights(filtered)

    def c1_statistics(self):
        return self.c1_boundary.statistics() | {
            "loaded_value_layers": sum(layer.self_attn.qkv_proj.value_loaded for layer in self.model.layers),
            "attention_backend": self.model.layers[0].self_attn.attn.attn_backend.get_name(),
            "attention_impl": type(self.model.layers[0].self_attn.attn.impl).__name__,
            "prefill_specialization": "sm89_qk128_v64",
            "key_head_size": HEAD_DIM,
            "value_head_size": VALUE_RANK,
        }
