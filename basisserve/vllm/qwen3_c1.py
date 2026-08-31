"""Qwen3-32B uniform-C1 model implementation for vLLM 0.18.

Each TP4 rank folds its two C1 encoders into the local Value projection, stores
K128/V64 in vLLM's paged cache, runs compact attention, AllGathers the four
rank-local coordinate blocks, and applies the replicated global decoder.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from basisserve.core.qwen3_32b_tp4_decode import (
    FACTOR_FORMAT,
    HEAD_DIM,
    HIDDEN_SIZE,
    KV_HEADS_PER_PROCESS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    TP_SIZE,
    file_sha256,
    fold_ragged_local_c1_value_projection,
    load_qwen3_32b_tp4_c1_factor_layer,
    load_qwen3_32b_tp4_c1_manifest,
)
from basisserve.core.qwen3_32b_vllm_c1 import (
    Qwen3_32BTP4UniformC1OutputFactors,
    decode_qwen3_32b_c1_coordinates,
    pack_qwen3_32b_tp4_uniform_c1_output_factors,
)
from basisserve.vllm.compact_v_backend import (
    BasisServeCompactVBackend,
    VALUE_HEAD_SIZE,
)

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader


_CONFIG_FACTOR_DIR = "basisserve_c1_factor_dir"
_CONFIG_RESULT_SHA256 = "basisserve_c1_result_sha256"
_EXPECTED_UNIFORM_RANK = VALUE_HEAD_SIZE


def _required_config_string(config: Any, name: str) -> str:
    value = getattr(config, name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Qwen3 C1 serving requires a nonempty {name!r} HF override")
    return value.strip()


def _validate_qwen3_32b_config(config: Any) -> None:
    observed = (
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        int(config.num_hidden_layers),
    )
    expected = (
        HIDDEN_SIZE,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        NUM_LAYERS,
    )
    if observed != expected:
        raise ValueError(
            f"BasisServe Qwen3 C1 supports Qwen3-32B geometry {expected}, got {observed}"
        )
    if bool(getattr(config, "attention_bias", False)):
        raise ValueError("BasisServe compact QKV loading requires bias-free Qwen3 attention")


class BasisServeC1QKVParallelLinear(QKVParallelLinear):
    """Load dense Q/K normally and fold each local dense V shard to V64."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        local_encoders: Tensor,
        prefix: str,
    ) -> None:
        super().__init__(
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            bias=False,
            quant_config=None,
            prefix=prefix,
            v_head_size=VALUE_HEAD_SIZE,
        )
        expected = (KV_HEADS_PER_PROCESS, HEAD_DIM, VALUE_HEAD_SIZE)
        if tuple(local_encoders.shape) != expected:
            raise ValueError(
                f"local uniform-C1 encoders must have shape {expected}, got "
                f"{tuple(local_encoders.shape)}"
            )
        self.register_buffer(
            "local_encoders",
            local_encoders.detach().to(device=self.weight.device).contiguous(),
        )

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        if loaded_shard_id != "v":
            if loaded_shard_id is None:
                raise NotImplementedError(
                    "BasisServe requires separate q/k/v checkpoint tensors"
                )
            return super().weight_loader_v2(param, loaded_weight, loaded_shard_id)

        output_dim = getattr(param, "output_dim", None)
        if output_dim != 0 or param is not self.weight:
            raise ValueError("BasisServe compact V loader only supports unquantized weights")

        dense_local_width = self.num_kv_heads * self.head_size
        shard_rank = self.tp_rank // self.num_kv_head_replicas
        dense_local = loaded_weight.narrow(
            output_dim,
            shard_rank * dense_local_width,
            dense_local_width,
        ).to(device=self.local_encoders.device)
        compact_weight, _ = fold_ragged_local_c1_value_projection(
            dense_local,
            tuple(self.local_encoders.unbind(0)),
        )

        compact_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        compact_width = self.num_kv_heads * self.v_head_size
        destination = param.data.narrow(output_dim, compact_offset, compact_width)
        destination.copy_(compact_weight.to(dtype=destination.dtype))


class BasisServeQwen3C1Attention(nn.Module):
    """Folded V64 paged attention followed by the C1 TP output boundary."""

    def __init__(
        self,
        base_attention: Qwen3Attention,
        packed: Qwen3_32BTP4UniformC1OutputFactors,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = base_attention.hidden_size
        self.total_num_heads = base_attention.total_num_heads
        self.num_heads = base_attention.num_heads
        self.total_num_kv_heads = base_attention.total_num_kv_heads
        self.num_kv_heads = base_attention.num_kv_heads
        self.head_dim = base_attention.head_dim
        self.q_size = base_attention.q_size
        self.kv_size = base_attention.kv_size
        self.compact_v_size = self.num_kv_heads * VALUE_HEAD_SIZE
        self.scaling = base_attention.scaling
        self.dual_chunk_attention_config = base_attention.dual_chunk_attention_config
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("BasisServe compact-V excludes dual-chunk attention")

        attention_prefix = base_attention.attn.layer_name
        if not attention_prefix.endswith(".attn"):
            raise ValueError(f"unexpected vLLM attention prefix {attention_prefix!r}")
        self_attention_prefix = attention_prefix[: -len(".attn")]

        static_context = vllm_config.compilation_config.static_forward_context
        registered_attention = static_context.pop(attention_prefix, None)
        if registered_attention is not base_attention.attn:
            raise RuntimeError("vLLM static attention context does not match Qwen3 layer")

        self.qkv_proj = BasisServeC1QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            local_encoders=packed.local_encoders,
            prefix=f"{self_attention_prefix}.qkv_proj",
        )
        self.rotary_emb = base_attention.rotary_emb
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=attention_prefix,
            attn_backend=BasisServeCompactVBackend,
            head_size_v=VALUE_HEAD_SIZE,
        )

        self.layer_index = packed.layer_index
        self.process_rank = packed.process_rank
        self.source_rank = packed.source_rank
        self.local_wire_width = packed.local_wire_width
        self.global_wire_width = packed.global_wire_width
        self.manifest_sha256 = packed.manifest_sha256
        self.artifact_sha256 = packed.artifact_sha256
        self.register_buffer("decoder_weight", packed.decoder_weight)

    def forward(self, positions: Tensor, hidden_states: Tensor) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, compact_v = qkv.split(
            [self.q_size, self.kv_size, self.compact_v_size],
            dim=-1,
        )
        q_by_head = q.view(
            *q.shape[:-1],
            q.shape[-1] // self.head_dim,
            self.head_dim,
        )
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(
            *k.shape[:-1],
            k.shape[-1] // self.head_dim,
            self.head_dim,
        )
        k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        local_coordinates = self.attn(q, k, compact_v)
        global_coordinates = tensor_model_parallel_all_gather(
            local_coordinates,
            dim=-1,
        )
        return decode_qwen3_32b_c1_coordinates(
            global_coordinates,
            self.decoder_weight,
        )


class BasisServeQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3-32B vLLM model with compact V64 cache and C1 output collective."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        _validate_qwen3_32b_config(config)
        if get_tensor_model_parallel_world_size() != TP_SIZE:
            raise ValueError(f"Qwen3-32B C1 serving requires tensor parallel TP{TP_SIZE}")
        if get_pp_group().world_size != 1:
            raise ValueError("Qwen3-32B C1 serving does not support pipeline parallelism")
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise ValueError("Qwen3-32B C1 serving does not support DCP")
        if vllm_config.quant_config is not None:
            raise ValueError("Qwen3-32B compact-C1 serving requires BF16 weights")
        if vllm_config.lora_config is not None:
            raise ValueError("Qwen3-32B compact-C1 serving does not support LoRA")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Qwen3-32B compact-C1 serving requires bfloat16")
        if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("Qwen3-32B compact-C1 serving requires a BF16 KV cache")

        factor_dir = Path(
            _required_config_string(config, _CONFIG_FACTOR_DIR)
        ).expanduser().resolve()
        expected_result_sha256 = _required_config_string(
            config,
            _CONFIG_RESULT_SHA256,
        )
        result_path = factor_dir / "result.json"
        observed_result_sha256 = file_sha256(result_path)
        if observed_result_sha256 != expected_result_sha256:
            raise ValueError(
                "Qwen3 C1 manifest hash mismatch: "
                f"expected {expected_result_sha256}, observed {observed_result_sha256}"
            )
        manifest = load_qwen3_32b_tp4_c1_manifest(factor_dir)
        if manifest.get("format") != FACTOR_FORMAT:
            raise ValueError("Qwen3 C1 factor format is incompatible")
        schedule = manifest["selection"]["selected_schedule"]
        if any(
            tuple(map(int, layer_ranks)) != (_EXPECTED_UNIFORM_RANK,) * NUM_KV_HEADS
            for layer_ranks in schedule
        ):
            raise ValueError(
                "the vLLM compact-C1 path requires uniform rank 64 "
                "for all physical KV sources and layers"
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        process_rank = get_tensor_model_parallel_rank()
        for layer_index, layer in enumerate(self.model.layers):
            base_attention = layer.self_attn
            reference = next(base_attention.parameters())
            factors = load_qwen3_32b_tp4_c1_factor_layer(
                factor_dir,
                layer_index,
                manifest=manifest,
            )
            packed = pack_qwen3_32b_tp4_uniform_c1_output_factors(
                factors,
                process_rank=process_rank,
                manifest_sha256=observed_result_sha256,
                device=reference.device,
                dtype=reference.dtype,
            )
            layer.self_attn = BasisServeQwen3C1Attention(
                base_attention,
                packed,
                vllm_config,
            )

        self.c1_factor_dir = str(factor_dir)
        self.c1_result_sha256 = observed_result_sha256

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
    ) -> set[str]:
        skip_prefixes = [
            f"model.layers.{layer}.self_attn.o_proj."
            for layer in range(NUM_LAYERS)
        ]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        return AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        ).load_weights(weights)


__all__ = [
    "BasisServeC1QKVParallelLinear",
    "BasisServeQwen3C1Attention",
    "BasisServeQwen3ForCausalLM",
]
