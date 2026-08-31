"""vLLM plugin registration for BasisServe models."""

from __future__ import annotations


MODEL_ARCHITECTURE = "BasisServeQwen3ForCausalLM"
DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE = (
    "BasisServeDeepseekV2LiteForCausalLM"
)


def register() -> None:
    """Register the out-of-tree model without importing CUDA-facing code."""

    from vllm import ModelRegistry

    if MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            MODEL_ARCHITECTURE,
            "basisserve.vllm.qwen3_c1:BasisServeQwen3ForCausalLM",
        )
    if DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE,
            "basisserve.vllm.deepseek_v2_lite_c1:"
            "BasisServeDeepseekV2LiteForCausalLM",
        )


__all__ = [
    "DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE",
    "MODEL_ARCHITECTURE",
    "register",
]
