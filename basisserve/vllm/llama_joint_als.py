"""Llama with a uniform low-rank Value latent, native vLLM paged attention and latent TP AllGather."""

import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch import nn

from vllm.distributed import get_pp_group, get_tp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.v1.attention.backends.triton_attn_diffkv import TritonAttentionDiffKVBackend


MODEL_ARCHITECTURE = "BasisServeLlamaJointALSForCausalLM"


class JointQKV(QKVParallelLinear):
    def __init__(self, encoder, prefix):
        super().__init__(4096, 128, 32, 8, bias=False, prefix=prefix, v_head_size=encoder.shape[-1])
        self.rank = encoder.shape[-1]
        self.register_buffer("encoder", encoder, persistent=False)
        self.value_loaded = False

    def _load_value(self, param, source):
        assert param is self.weight
        assert source.shape == (8 * 128, 4096)
        count = self.num_kv_heads
        local = source.reshape(8, 128, 4096)[
            self.tp_rank * count:(self.tp_rank + 1) * count
        ].to(device=self.encoder.device, dtype=torch.float32)
        folded = (self.encoder.transpose(1, 2).float() @ local).flatten(0, 1)
        offset = (self.num_heads + count) * 128
        param.data.narrow(0, offset, count * self.rank).copy_(folded)
        self.value_loaded = True

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in ("q", "k", "v")
        if loaded_shard_id == "v":
            self._load_value(param, loaded_weight)
        else:
            super().weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in ("q", "k", "v")
        if loaded_shard_id == "v":
            self._load_value(param, loaded_weight)
        else:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)


class JointAttention(nn.Module):
    def __init__(self, original, encoder, decoder, config, router=None):
        super().__init__()
        self.num_heads = original.num_heads
        self.num_kv_heads = original.num_kv_heads
        self.group = get_tp_group()
        name = original.attn.layer_name
        assert config.compilation_config.static_forward_context.pop(name) is original.attn
        self.rank = encoder.shape[-1]
        self.qkv_proj = JointQKV(encoder, name[:-5] + ".qkv_proj")
        self.rotary_emb = original.rotary_emb
        TritonAttentionDiffKVBackend.set_head_size_v(self.rank)
        backend = TritonAttentionDiffKVBackend
        if router is not None:
            from basisserve.vllm.llama_joint_backend import JointRoutingBackend
            backend = JointRoutingBackend
        self.attn = Attention(
            self.num_heads, 128, original.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=config.cache_config, prefix=name,
            attn_backend=backend, head_size_v=self.rank,
        )
        self.register_buffer("decoder", decoder, persistent=False)
        if router is not None:
            from basisserve.kernels.paged_joint_router import PagedJointRouter
            from basisserve.kernels.paged_joint_slots import PagedJointSlots
            batch = config.scheduler_config.max_num_seqs
            capacity = config.model_config.max_model_len
            self.attn.joint_cache = None
            self.attn.joint_host_budget = config.model_config.hf_config.basisserve_host_budget_bytes_per_rank
            self.attn.joint_router = PagedJointRouter(batch, self.num_kv_heads, capacity, encoder.device)
            self.attn.joint_slots = PagedJointSlots(batch, self.num_kv_heads, capacity, encoder.device)
            self.attn.register_buffer("joint_rope", self.rotary_emb.cos_sin_cache.bfloat16(), persistent=False)
            for name, tensor in router.items():
                self.attn.register_buffer(name, tensor.to(encoder.device, torch.bfloat16), persistent=False)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            (self.num_heads * 128, self.num_kv_heads * 128, self.num_kv_heads * self.rank),
            dim=-1,
        )
        q, k = self.rotary_emb(positions, q, k)
        coordinates = self.attn(q, k, v)
        gathered = self.group.all_gather(coordinates, dim=-1)
        return gathered @ self.decoder


class BasisServeLlamaJointALSForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        cfg = vllm_config.model_config.hf_config
        assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.num_hidden_layers) == (4096, 32, 8, 32)
        group = get_tp_group()
        assert group.world_size in (4, 8) and get_pp_group().world_size == 1
        assert vllm_config.parallel_config.data_parallel_size == 1
        assert vllm_config.parallel_config.decode_context_parallel_size == 1
        assert vllm_config.quant_config is None and vllm_config.lora_config is None
        assert vllm_config.speculative_config is None
        assert vllm_config.model_config.dtype == torch.bfloat16
        root = Path(cfg.basisserve_joint_factor_dir)
        manifest_bytes = (root / "manifest.json").read_bytes()
        assert hashlib.sha256(manifest_bytes).hexdigest() == cfg.basisserve_joint_manifest_sha256
        manifest = json.loads(manifest_bytes)
        assert manifest["format"] == "basisserve.uniform_c1_als.v1"
        rank = manifest["compression"]["rank"]
        assert rank in (64, 96)
        assert len(manifest["layers"]) == 32
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        device = next(self.parameters()).device
        count = 8 // group.world_size
        start = group.rank_in_group * count
        router_dir = getattr(cfg, "basisserve_joint_router_dir", None)
        self.router_hashes = []
        for index, layer in enumerate(self.model.layers):
            record = manifest["layers"][index]
            path = root / record["file"]
            assert record["layer"] == index
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            factors = load_file(str(path))
            assert factors["value_coordinate_encoders"].shape == (8, 128, rank)
            assert factors["head_output_decoders"].shape == (32, rank, 4096)
            assert "source_ranks" not in factors or bool((factors["source_ranks"] == rank).all())
            encoder = factors["value_coordinate_encoders"].to(device=device, dtype=torch.float32)
            decoder = factors["head_output_decoders"].to(device=device, dtype=torch.float32)
            local_router = None
            if router_dir is not None:
                from basisserve.core.routing_basis import make_routing_basis
                router_path = Path(router_dir) / f"layer_{index:03d}.safetensors"
                record = json.loads(router_path.with_suffix(".json").read_text())
                digest = hashlib.sha256(router_path.read_bytes()).hexdigest()
                assert record["status"] == "complete" and record["layer"] == index
                assert digest == record["sha256"]
                self.router_hashes.append(digest)
                bank = load_file(str(router_path))
                basis = make_routing_basis(bank["base_left_b16"].to(device))
                encoder = encoder @ basis.transform.float()
                decoder = basis.inverse.float().repeat_interleave(4, dim=0) @ decoder
                local_router = {
                    "base_right": bank["base_right_b16"][start:start + count],
                    "base_bias": bank["base_bias_b16"][start:start + count],
                    "residual_encoder": bank["residual_encoder_b16_r16"][start:start + count],
                    "residual_query": bank["residual_query_b16_r16"][start * 4:(start + count) * 4],
                }
            layer.self_attn = JointAttention(
                layer.self_attn, encoder[start:start + count].contiguous(),
                decoder.reshape(32 * rank, 4096).bfloat16(), vllm_config, local_router,
            )

    def load_weights(self, weights):
        skips = tuple(f"model.layers.{i}.self_attn.o_proj." for i in range(32))
        if self.config.tie_word_embeddings:
            skips += ("lm_head.",)
        return AutoWeightsLoader(self).load_weights(
            (name, value) for name, value in weights if not name.startswith(skips)
        )

    def joint_statistics(self):
        routed = bool(self.router_hashes)
        result = dict(
            value_rank=self.model.layers[0].self_attn.rank, key_rank=128, decoder_replicated=True,
            attention_collective="vllm_all_gather", key_placement="pinned_host" if routed else "gpu",
            loaded_value_layers=sum(layer.self_attn.qkv_proj.value_loaded for layer in self.model.layers),
            attention_impl=type(self.model.layers[0].self_attn.attn.impl).__name__,
        )
        if routed:
            layers = [layer.self_attn.attn for layer in self.model.layers]
            tensors = [tensor for layer in layers for state in (layer.joint_router, layer.joint_slots)
                       for tensor in vars(state).values() if isinstance(tensor, torch.Tensor)]
            result.update(
                router_layer_sha256=self.router_hashes, gpu_base_copy_bytes=0,
                host_key_pool_bytes=sum(layer.joint_cache.host.numel() * 2 for layer in layers),
                gpu_v_r_bounds_pool_bytes=sum(layer.joint_cache.gpu.numel() * 2 for layer in layers),
                gpu_slot_router_workspace_bytes=sum(t.numel() * t.element_size() for t in tensors),
                key_slot_limit_per_request_kv_group=2048,
                last_step_slot_counts=[layer.joint_slots.counts.tolist() for layer in layers],
            )
        return result
