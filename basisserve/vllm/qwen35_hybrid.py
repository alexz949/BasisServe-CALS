"""TP1 vLLM Qwen3.5 gated V+Wo quality runtime with padded V cache slots."""

from pathlib import Path

import torch
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

from basisserve.core.qwen35_vllm_hybrid import install_vllm_hybrid_layers
from evaluation.qwen35_hybrid_common import load_bank, sha256


class BasisServeQwen35HybridForCausalLM(Qwen3_5ForCausalLM):
    # The text-only base does not advertise the hybrid cache contract itself;
    # the native multimodal wrapper supplies it. Preserve that contract here.
    is_hybrid = True
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        assert not mm_features, 'This adapter supports text-only evaluation'
        return Qwen3_5ForConditionalGeneration._get_mrope_input_positions(
            input_tokens, mm_features, self.model_config.hf_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config, prefix=''):
        assert get_tensor_model_parallel_world_size() == 1
        assert get_pp_group().world_size == 1
        assert vllm_config.quant_config is None and vllm_config.lora_config is None
        assert vllm_config.model_config.dtype == torch.bfloat16
        config = vllm_config.model_config.hf_config
        bank_path = getattr(config, 'basisserve_v_bank', None)
        self.v_bank_path = Path(bank_path) if bank_path else None
        self.v_bank_sha = getattr(config, 'basisserve_v_bank_sha256', None)
        self.wo_bank_path = getattr(config, 'basisserve_wo_bank', None)
        self.wo_bank_sha = getattr(config, 'basisserve_wo_bank_sha256', None)
        self.wo_scope = getattr(config, 'basisserve_wo_scope', 'all')
        if self.v_bank_path:
            assert sha256(self.v_bank_path) == self.v_bank_sha
        if self.wo_bank_path:
            assert sha256(self.wo_bank_path) == self.wo_bank_sha
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        # The original multimodal checkpoint has a text-module prefix. Keep
        # original files immutable and load only its text and lm_head weights.
        text_weights = ((name.replace('model.language_model.', 'model.', 1), value)
                        for name, value in weights
                        if not name.startswith(('model.visual.', 'mtp.')))
        loaded = super().load_weights(text_weights)
        if self.v_bank_path is None:
            assert self.wo_bank_path is None
            return loaded
        native_parameters = {name for name, _ in self.named_parameters()}
        assert sha256(self.v_bank_path) == self.v_bank_sha
        bank = load_bank(self.v_bank_path)
        wo = None
        if self.wo_bank_path:
            assert sha256(self.wo_bank_path) == self.wo_bank_sha
            wo = torch.load(self.wo_bank_path, map_location='cpu', weights_only=True)
        install_vllm_hybrid_layers(self.model.layers, bank, wo, self.wo_scope)
        bank_parameters = {name for name, _ in self.named_parameters()} - native_parameters
        return loaded | bank_parameters
