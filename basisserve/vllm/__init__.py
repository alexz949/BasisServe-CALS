"""vLLM plugin registration for BasisServe models."""

from __future__ import annotations


MODEL_ARCHITECTURE = "BasisServeQwen3ForCausalLM"
DENSE_DIFFKV_MODEL_ARCHITECTURE = "BasisServeQwen3DenseDiffKVForCausalLM"
DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE = "BasisServeDeepseekV2LiteForCausalLM"
QWEN3_8B_FOLDED_C1_MODEL_ARCHITECTURE = "BasisServeQwen3_8BFoldedC1ForCausalLM"
QWEN3_8B_FOLDED_PALU_MODEL_ARCHITECTURE = "BasisServeQwen3_8BFoldedPaLUForCausalLM"
QWEN3_8B_SPARSE_C1_MODEL_ARCHITECTURE = "BasisServeQwen3_8BSparseC1ForCausalLM"
QWEN3_8B_C1_MODEL_ARCHITECTURE = "BasisServeQwen3_8BC1ForCausalLM"
QWEN3_32B_TP8_C1_MODEL_ARCHITECTURE = "BasisServeQwen3_32BTP8C1ForCausalLM"


def register() -> None:
    """Register the out-of-tree model without importing CUDA-facing code."""

    from vllm import ModelRegistry

    if QWEN3_32B_TP8_C1_MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            QWEN3_32B_TP8_C1_MODEL_ARCHITECTURE,
            "basisserve.vllm.qwen3_32b_tp8_c1:BasisServeQwen3_32BTP8C1ForCausalLM",
        )

    if QWEN3_8B_C1_MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            QWEN3_8B_C1_MODEL_ARCHITECTURE,
            "basisserve.vllm.qwen3_8b_c1:BasisServeQwen3_8BC1ForCausalLM",
        )

    if "BasisServeQwen35HybridForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "BasisServeQwen35HybridForCausalLM",
            "basisserve.vllm.qwen35_hybrid:BasisServeQwen35HybridForCausalLM",
        )

    if "BasisServeQwen3_32BFoldedForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "BasisServeQwen3_32BFoldedForCausalLM",
            "basisserve.vllm.qwen3_32b_folded:BasisServeQwen3_32BFoldedForCausalLM",
        )

    if MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            MODEL_ARCHITECTURE,
            "basisserve.vllm.qwen3_c1:BasisServeQwen3ForCausalLM",
        )
    if DENSE_DIFFKV_MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            DENSE_DIFFKV_MODEL_ARCHITECTURE,
            "basisserve.vllm.qwen3_dense_diffkv:BasisServeQwen3DenseDiffKVForCausalLM",
        )
    if DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE,
            "basisserve.vllm.deepseek_v2_lite_c1:BasisServeDeepseekV2LiteForCausalLM",
        )
    if QWEN3_8B_FOLDED_C1_MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            QWEN3_8B_FOLDED_C1_MODEL_ARCHITECTURE,
            "basisserve.vllm.qwen3_8b_folded_c1:BasisServeQwen3_8BFoldedC1ForCausalLM",
        )
    if (
        QWEN3_8B_FOLDED_PALU_MODEL_ARCHITECTURE
        not in ModelRegistry.get_supported_archs()
    ):
        ModelRegistry.register_model(
            QWEN3_8B_FOLDED_PALU_MODEL_ARCHITECTURE,
            "basisserve.vllm.qwen3_8b_folded_palu:"
            "BasisServeQwen3_8BFoldedPaLUForCausalLM",
        )
    if QWEN3_8B_SPARSE_C1_MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            QWEN3_8B_SPARSE_C1_MODEL_ARCHITECTURE,
            "basisserve.vllm.qwen3_8b_sparse_c1:BasisServeQwen3_8BSparseC1ForCausalLM",
        )


__all__ = [
    "DENSE_DIFFKV_MODEL_ARCHITECTURE",
    "DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE",
    "MODEL_ARCHITECTURE",
    "QWEN3_8B_FOLDED_C1_MODEL_ARCHITECTURE",
    "QWEN3_8B_FOLDED_PALU_MODEL_ARCHITECTURE",
    "QWEN3_8B_SPARSE_C1_MODEL_ARCHITECTURE",
    "QWEN3_8B_C1_MODEL_ARCHITECTURE",
    "QWEN3_32B_TP8_C1_MODEL_ARCHITECTURE",
    "register",
]
