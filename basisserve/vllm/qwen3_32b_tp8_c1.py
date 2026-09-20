"""Qwen3-32B uses the same folded-V64, prepared-NCCL TP8 path as Qwen3-8B."""

from basisserve.vllm.qwen3_8b_c1 import BasisServeQwen3_8BC1ForCausalLM


class BasisServeQwen3_32BTP8C1ForCausalLM(BasisServeQwen3_8BC1ForCausalLM):
    model_geometry = (5120, 64, 8, 128, 64)
