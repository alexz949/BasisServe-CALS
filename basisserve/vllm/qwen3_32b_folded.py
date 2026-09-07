"""Tensor-parallel Qwen3-32B quality evaluation with authenticated factors."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

from basisserve.core.qwen3_8b_vllm_folded_c1 import (
    file_sha256, folded_c1_projection_weights,
)
from basisserve.core.qwen3_8b_vllm_folded_palu import reconstructed_palu_value_projection


class BasisServeQwen3_32BFoldedForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        assert (config.hidden_size, config.num_attention_heads,
                config.num_key_value_heads, config.head_dim,
                config.num_hidden_layers) == (5120, 64, 8, 128, 64)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        assert self.tp_size in (1, 2)
        assert get_pp_group().world_size == 1
        assert vllm_config.model_config.dtype == torch.bfloat16
        assert vllm_config.quant_config is None
        self.factor_dir = Path(config.basisserve_c1_checkpoint_dir)
        manifest_path = self.factor_dir / "manifest.json"
        assert file_sha256(manifest_path) == config.basisserve_c1_manifest_sha256
        self.factor_manifest = json.loads(manifest_path.read_text())
        m = self.factor_manifest
        assert m["format"] == "basisserve.qwen3_32b.iclr_v_factors.v1"
        assert m["status"] == "complete"
        assert m["compression"]["method"] in ("c1-two-sided-kl", "palu-fisher")
        assert m["compression"]["equivalent_rank_target"] == 80
        artifact = m["artifact"]
        assert file_sha256(self.factor_dir / artifact["file"]) == artifact["sha256"]
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @torch.no_grad()
    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        m = self.factor_manifest
        is_c1 = m["compression"]["method"] == "c1-two-sided-kl"
        factors = None if is_c1 else load_file(str(self.factor_dir / m["artifact"]["file"]))
        for index, layer in enumerate(self.model.layers):
            attention = layer.self_attn
            qkv = attention.qkv_proj.weight
            assert tuple(qkv.shape) == (10240 // self.tp_size, 5120)
            value = qkv[9216 // self.tp_size:10240 // self.tp_size]
            source_start = self.tp_rank * (8 // self.tp_size)
            source_end = source_start + 8 // self.tp_size
            head_start = self.tp_rank * (64 // self.tp_size)
            head_end = head_start + 64 // self.tp_size
            if is_c1:
                record = m["layers"][index]
                assert record["layer"] == index
                path = self.factor_dir / record["file"]
                assert file_sha256(path) == record["sha256"]
                payload = load_file(str(path))
                ranks = tuple(record["ranks"])
                assert tuple(payload["source_ranks"].tolist()) == ranks
                ranks = ranks[source_start:source_end]
                maximum_rank = max(ranks)
                folded_v, folded_o = folded_c1_projection_weights(
                    value, payload["value_coordinate_encoders"][source_start:source_end, :, :maximum_rank],
                    payload["head_output_decoders"][head_start:head_end, :maximum_rank], ranks,
                )
                value.copy_(folded_v)
                attention.o_proj.weight.copy_(folded_o)
            else:
                ranks = tuple(m["compression"]["layer_ranks"][index])
                name = f"layers.{index}.v_"
                reconstructed = reconstructed_palu_value_projection(
                    factors[name + "writer.weight"].to(value.device),
                    factors[name + "decoder.weight"].to(value.device), ranks,
                )
                value.copy_(reconstructed[source_start * 128:source_end * 128])
        print(f"[Folded] authenticated {m['run_id']} layers=64 TP={self.tp_size} rank={self.tp_rank}", flush=True)
        return loaded
