"""The text plugin must retain the native hybrid cache-planning contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import importlib.util
import unittest
import torch

def check_native_hybrid_state_contract():
    from basisserve.vllm.qwen35_hybrid import BasisServeQwen35HybridForCausalLM
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    cls = BasisServeQwen35HybridForCausalLM
    assert cls.is_hybrid and cls.has_inner_state
    text = json.loads(Path('results/q35_hybrid/model/config.json').read_text())['text_config']
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(**text), dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype='auto', mamba_ssm_cache_dtype='auto'),
        speculative_config=None,
    )
    assert cls.get_mamba_state_shape_from_config(config) == Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(config)
    assert cls.get_mamba_state_dtype_from_config(config) == Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(config)
    assert cls.get_mamba_state_copy_func() == Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
    assert cls.supports_mrope
    original = json.loads(Path('results/q35_hybrid/model/config.json').read_text())
    original['vision_config'] = SimpleNamespace(**original['vision_config'])
    mock = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**original)))
    positions, delta = cls.get_mrope_input_positions(mock, [1, 2, 3, 4], [])
    torch.testing.assert_close(positions, torch.arange(4).expand(3, -1))
    assert delta == 0


@unittest.skipUnless(importlib.util.find_spec('vllm'), 'requires vLLM environment')
class HybridContractTest(unittest.TestCase):
    def test_native_hybrid_state_contract(self):
        check_native_hybrid_state_contract()


if __name__ == '__main__':
    unittest.main()
