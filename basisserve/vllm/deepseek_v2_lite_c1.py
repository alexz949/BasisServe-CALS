"""DeepSeek-V2-Lite uniform-R128 C1 model for vLLM 0.18."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from basisserve.core.deepseek_v2_lite_tp4_c1 import (
    HIDDEN_SIZE,
    LOCAL_ATTENTION_WIDTH,
    NUM_ATTENTION_HEADS,
    NUM_LAYERS,
    SOURCE_WIDTH,
    TP_SIZE,
    UNIFORM_SOURCE_RANK,
    decode_deepseek_v2_lite_c1_coordinates,
    encode_deepseek_v2_lite_local_sources,
    file_sha256,
    load_deepseek_v2_lite_uniform_c1_factors,
    pack_deepseek_v2_lite_tp4_uniform_c1_output_factors,
)

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM


_CONFIG_FACTOR_DIR = "basisserve_c1_factor_dir"
_CONFIG_RESULT_SHA256 = "basisserve_c1_result_sha256"


def _required_config_string(config: Any, name: str) -> str:
    value = getattr(config, name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"DeepSeek-V2-Lite C1 serving requires a nonempty {name!r} override"
        )
    return value.strip()


def _validate_deepseek_v2_lite_config(config: Any) -> None:
    observed = (
        str(config.model_type),
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.v_head_dim),
        int(config.num_hidden_layers),
    )
    expected = (
        "deepseek_v2",
        HIDDEN_SIZE,
        NUM_ATTENTION_HEADS,
        SOURCE_WIDTH // 2,
        NUM_LAYERS,
    )
    if observed != expected:
        raise ValueError(
            f"BasisServe DeepSeek-V2-Lite C1 expects geometry {expected}, got "
            f"{observed}"
        )


class BasisServeDeepseekV2LiteC1OutputProjection(nn.Module):
    """Encode two local MLA sources, AllGather R128, and decode hidden output."""

    def __init__(
        self,
        *,
        local_encoders: Tensor,
        decoder_weight: Tensor,
        layer_index: int,
        process_rank: int,
        manifest_sha256: str,
        artifact_sha256: str,
    ) -> None:
        super().__init__()
        expected_encoders = (2, SOURCE_WIDTH, UNIFORM_SOURCE_RANK)
        expected_decoder = (
            HIDDEN_SIZE,
            8 * UNIFORM_SOURCE_RANK,
        )
        if tuple(local_encoders.shape) != expected_encoders:
            raise ValueError(
                f"local encoders must have shape {expected_encoders}, got "
                f"{tuple(local_encoders.shape)}"
            )
        if tuple(decoder_weight.shape) != expected_decoder:
            raise ValueError(
                f"decoder must have shape {expected_decoder}, got "
                f"{tuple(decoder_weight.shape)}"
            )
        self.layer_index = int(layer_index)
        self.process_rank = int(process_rank)
        self.source_rank = UNIFORM_SOURCE_RANK
        self.local_wire_width = 2 * UNIFORM_SOURCE_RANK
        self.global_wire_width = 8 * UNIFORM_SOURCE_RANK
        self.manifest_sha256 = str(manifest_sha256)
        self.artifact_sha256 = str(artifact_sha256)
        self.register_buffer("local_encoders", local_encoders.contiguous())
        self.register_buffer("decoder_weight", decoder_weight.contiguous())

    def forward(self, local_attention_output: Tensor) -> tuple[Tensor, None]:
        leading_shape = tuple(local_attention_output.shape[:-1])
        local_coordinates = encode_deepseek_v2_lite_local_sources(
            local_attention_output,
            self.local_encoders,
        ).reshape(-1, self.local_wire_width)
        global_coordinates = tensor_model_parallel_all_gather(
            local_coordinates,
            dim=-1,
        )
        decoded = decode_deepseek_v2_lite_c1_coordinates(
            global_coordinates,
            self.decoder_weight,
        )
        return decoded.reshape(*leading_shape, HIDDEN_SIZE), None


class BasisServeDeepseekV2LiteForCausalLM(DeepseekV2ForCausalLM):
    """Native vLLM MLA/MoE with a uniform-R128 C1 output boundary."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        _validate_deepseek_v2_lite_config(config)
        if get_tensor_model_parallel_world_size() != TP_SIZE:
            raise ValueError(
                f"DeepSeek-V2-Lite C1 serving requires tensor parallel TP{TP_SIZE}"
            )
        if get_pp_group().world_size != 1:
            raise ValueError("DeepSeek-V2-Lite C1 does not support pipeline parallelism")
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise ValueError("DeepSeek-V2-Lite C1 does not support DCP")
        if vllm_config.quant_config is not None:
            raise ValueError("DeepSeek-V2-Lite C1 currently requires BF16 weights")
        if vllm_config.lora_config is not None:
            raise ValueError("DeepSeek-V2-Lite C1 does not support LoRA")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("DeepSeek-V2-Lite C1 requires bfloat16")

        factor_dir = Path(
            _required_config_string(config, _CONFIG_FACTOR_DIR)
        ).expanduser().resolve()
        expected_manifest_sha256 = _required_config_string(
            config,
            _CONFIG_RESULT_SHA256,
        )
        manifest_path = factor_dir / "results.json"
        observed_manifest_sha256 = file_sha256(manifest_path)
        if observed_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "DeepSeek C1 manifest hash mismatch: "
                f"expected {expected_manifest_sha256}, "
                f"observed {observed_manifest_sha256}"
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        factors = load_deepseek_v2_lite_uniform_c1_factors(factor_dir)
        process_rank = get_tensor_model_parallel_rank()
        for layer_index, (layer, factor) in enumerate(
            zip(self.model.layers, factors, strict=True)
        ):
            base_output = layer.self_attn.mla_attn.o_proj
            weight = getattr(base_output, "weight", None)
            expected_weight = (HIDDEN_SIZE, LOCAL_ATTENTION_WIDTH)
            if weight is None or tuple(weight.shape) != expected_weight:
                raise ValueError(
                    f"layer {layer_index} o_proj is not a TP4 row shard: "
                    f"{None if weight is None else tuple(weight.shape)}"
                )
            packed = pack_deepseek_v2_lite_tp4_uniform_c1_output_factors(
                factor,
                process_rank=process_rank,
                manifest_sha256=observed_manifest_sha256,
                device=weight.device,
                dtype=weight.dtype,
            )
            replacement = BasisServeDeepseekV2LiteC1OutputProjection(
                local_encoders=packed.local_encoders,
                decoder_weight=packed.decoder_weight,
                layer_index=layer_index,
                process_rank=process_rank,
                manifest_sha256=observed_manifest_sha256,
                artifact_sha256=packed.artifact_sha256,
            ).eval()
            del layer.self_attn.o_proj
            layer.self_attn.mla_attn.o_proj = replacement

        self.c1_factor_dir = str(factor_dir)
        self.c1_result_sha256 = observed_manifest_sha256

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
    ) -> set[str]:
        retained = (
            (name, weight)
            for name, weight in weights
            if ".self_attn.o_proj." not in name
        )
        return super().load_weights(retained)


__all__ = [
    "BasisServeDeepseekV2LiteC1OutputProjection",
    "BasisServeDeepseekV2LiteForCausalLM",
]
