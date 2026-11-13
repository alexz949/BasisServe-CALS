"""Llama-3.1-8B with PaLU G-LRD4 Value latents, replicated on every rank of a head group.

Under head-partitioned tensor parallelism a G-LRD4 group (four KV heads) spans several ranks.
Each rank applies the complete group encoder, caches the whole group latent next to its own
dense Keys, attends with its own heads, and applies only the decoder rows of its KV heads,
folded into the output projection. Communication is the ordinary output AllReduce.
"""

import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch import nn
import torch.nn.functional as F

from vllm.distributed import get_pp_group, get_tp_group, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader

from basisserve.vllm.palu_group_attention import palu_group_backend
from basisserve.vllm.palu_collectives import latent_group
from basisserve.vllm.palu_shard_attention import palu_shard_backend

MODEL_ARCHITECTURE = "BasisServeLlamaPaluG4ForCausalLM"
SHARDED_MODEL_ARCHITECTURE = "BasisServeLlamaPaluG4ShardedForCausalLM"
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, GROUP_HEADS = 4096, 32, 8, 128, 4


class GroupLatentQKV(QKVParallelLinear):
    """Column-parallel Q and K; the V rows hold the complete group encoder instead."""

    def __init__(self, writer, prefix):
        # Per-rank V rows must total the group rank: rank = (KV heads per rank) * v_head_size.
        super().__init__(HIDDEN, HEAD_DIM, HEADS, KV_HEADS, bias=False, prefix=prefix,
                         v_head_size=writer.shape[0] * get_tp_group().world_size // KV_HEADS)
        self.writer = writer
        self.value_loaded = False

    def _load_value(self, param):
        offset = (self.num_heads + self.num_kv_heads) * HEAD_DIM
        param.data.narrow(0, offset, self.writer.shape[0]).copy_(self.writer)
        self.value_loaded = True

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        if loaded_shard_id == "v":
            self._load_value(param)
        else:
            super().weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        if loaded_shard_id == "v":
            self._load_value(param)
        else:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)


class FoldedOutput(nn.Module):
    """Output projection whose per-head columns are folded with the local decoder rows."""

    def __init__(self, decoder, heads, rank, tp_rank, device):
        super().__init__()
        self.decoder, self.heads, self.rank, self.tp_rank = decoder, heads, rank, tp_rank
        self.weight = nn.Parameter(torch.empty(HIDDEN, heads * rank, dtype=torch.bfloat16,
                                               device=device), requires_grad=False)
        self.weight.weight_loader = self.fold
        self.loaded = False

    def fold(self, param, loaded_weight):
        local = loaded_weight.to(device=param.device, dtype=torch.float32).view(
            HIDDEN, HEADS, HEAD_DIM)[:, self.tp_rank * self.heads:(self.tp_rank + 1) * self.heads]
        per_kv = self.heads // (self.decoder.shape[0] // HEAD_DIM)
        blocks = self.decoder.view(-1, HEAD_DIM, self.rank).repeat_interleave(per_kv, dim=0)
        param.data.copy_(torch.einsum("ohd,hdr->ohr", local, blocks).reshape(HIDDEN, -1))
        self.loaded = True


class PaluGroupAttention(nn.Module):
    def __init__(self, original, writer, decoder, config):
        super().__init__()
        self.num_heads, self.num_kv_heads = original.num_heads, original.num_kv_heads
        self.rank = writer.shape[0]
        name = original.attn.layer_name
        assert config.compilation_config.static_forward_context.pop(name) is original.attn
        self.qkv_proj = GroupLatentQKV(writer, name[:-5] + ".qkv_proj")
        self.rotary_emb = original.rotary_emb
        self.attn = Attention(
            self.num_heads, HEAD_DIM, original.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=config.cache_config, prefix=name,
            attn_backend=palu_group_backend(self.rank), head_size_v=self.rank,
            value_rank=self.rank)
        self.o_proj = FoldedOutput(decoder, self.num_heads, self.rank,
                                   get_tp_group().rank_in_group, writer.device)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, latent = qkv.split(
            (self.num_heads * HEAD_DIM, self.num_kv_heads * HEAD_DIM, self.rank), dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # The cache keeps one latent per token; the layer API wants one per KV head.
        weighted = self.attn(q, k, torch.cat((latent,) * self.num_kv_heads, dim=-1))
        return tensor_model_parallel_all_reduce(F.linear(weighted, self.o_proj.weight))


class BasisServeLlamaPaluG4ForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        cfg = vllm_config.model_config.hf_config
        assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.num_hidden_layers) == (HIDDEN, HEADS, KV_HEADS, 32)
        group = get_tp_group()
        assert group.world_size in (2, 4, 8) and get_pp_group().world_size == 1
        assert vllm_config.model_config.dtype == torch.bfloat16
        root = Path(cfg.basisserve_palu_factor_dir)
        manifest_bytes = (root / "manifest.json").read_bytes()
        assert hashlib.sha256(manifest_bytes).hexdigest() == cfg.basisserve_palu_manifest_sha256
        manifest = json.loads(manifest_bytes)
        compression = manifest["compression"]
        assert compression["method_label"] == "PaLU G-LRD4"
        assert compression["head_group_size"] == GROUP_HEADS and compression["groups"] == 2
        artifact = root / manifest["artifact"]["file"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifact"]["sha256"]
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        device = next(self.parameters()).device
        count = KV_HEADS // group.world_size
        start = group.rank_in_group * count
        self.group_index = start // GROUP_HEADS
        assert (start + count - 1) // GROUP_HEADS == self.group_index
        factors = load_file(str(artifact))
        self.layer_ranks = []
        for index, layer in enumerate(self.model.layers):
            ranks = compression["layer_ranks"][index]
            rank = ranks[self.group_index]
            offset = sum(ranks[:self.group_index])
            writer = factors[f"layers.{index}.v_writer.weight"][offset:offset + rank]
            decoder = factors[f"layers.{index}.v_decoder.weight"][self.group_index]
            first = (start - self.group_index * GROUP_HEADS) * HEAD_DIM
            local_decoder = decoder[first:first + count * HEAD_DIM, :rank]
            layer.self_attn = self.attention_module(
                layer.self_attn, writer.to(device=device, dtype=torch.bfloat16).contiguous(),
                local_decoder.to(device=device, dtype=torch.float32).contiguous(), vllm_config)
            self.layer_ranks.append(rank)
        self.factor_sha256 = manifest["artifact"]["sha256"]

    def attention_module(self, original, writer, decoder, config):
        return PaluGroupAttention(original, writer, decoder, config)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)

    def deployment_statistics(self):
        attention = [layer.self_attn for layer in self.model.layers]
        kv_heads = attention[0].num_kv_heads
        return dict(
            method="PaLU G-LRD4, group latent replicated per rank", group_index=self.group_index,
            tp_rank=get_tp_group().rank_in_group, layer_ranks=self.layer_ranks,
            factor_sha256=self.factor_sha256,
            key_bytes_per_token=kv_heads * HEAD_DIM * 2,
            latent_bytes_per_token=[rank * 2 for rank in self.layer_ranks],
            latent_replicas_per_group=GROUP_HEADS // kv_heads,
            attention_collective="all_reduce",
            loaded_value_layers=sum(a.qkv_proj.value_loaded for a in attention),
            folded_output_layers=sum(a.o_proj.loaded for a in attention),
            attention_impl=type(attention[0].attn.impl).__name__)


class PaluShardAttention(nn.Module):
    """One encoder shard per rank; attention reconstructs outputs inside the head group."""

    def __init__(self, original, writer, decoder, config):
        super().__init__()
        self.num_heads, self.num_kv_heads = original.num_heads, original.num_kv_heads
        self.rank = writer.shape[0]
        group = latent_group(get_tp_group(), KV_HEADS, GROUP_HEADS)
        group_size = GROUP_HEADS // self.num_kv_heads
        assert self.rank % group_size == 0
        self.shard = self.rank // group_size
        side = group[1]
        name = original.attn.layer_name
        assert config.compilation_config.static_forward_context.pop(name) is original.attn
        self.qkv_proj = GroupLatentQKV(writer[side * self.shard:(side + 1) * self.shard].contiguous(),
                                       name[:-5] + ".qkv_proj")
        self.rotary_emb = original.rotary_emb
        self.attn = Attention(
            self.num_heads, HEAD_DIM, original.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=config.cache_config, prefix=name,
            attn_backend=palu_shard_backend(self.rank, group_size), head_size_v=self.rank,
            value_rank=self.rank, latent_group=group)
        self.o_proj = FoldedOutput(decoder, self.num_heads, self.rank,
                                   get_tp_group().rank_in_group, writer.device)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, shard = qkv.split(
            (self.num_heads * HEAD_DIM, self.num_kv_heads * HEAD_DIM, self.shard), dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # The layer API sizes value by KV heads times the full rank; only the shard is real.
        value = torch.zeros(shard.shape[0], self.num_kv_heads * self.rank, device=shard.device,
                            dtype=shard.dtype)
        value[:, :self.shard] = shard
        weighted = self.attn(q, k, value)
        return tensor_model_parallel_all_reduce(F.linear(weighted, self.o_proj.weight))


class BasisServeLlamaPaluG4ShardedForCausalLM(BasisServeLlamaPaluG4ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        assert get_tp_group().world_size in (4, 8)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def attention_module(self, original, writer, decoder, config):
        return PaluShardAttention(original, writer, decoder, config)

    def deployment_statistics(self):
        result = super().deployment_statistics()
        impls = [layer.self_attn.attn.impl for layer in self.model.layers]
        traffic = {key: sum(impl.traffic[key] for impl in impls) for key in impls[0].traffic}
        result.update(
            method="PaLU G-LRD4, group latent sharded across its TP ranks",
            latent_bytes_per_token=[impl.shard * 2 for impl in impls],
            latent_replicas_per_group=1, latent_group_rank=impls[0].group_rank,
            latent_group_size=impls[0].group_size,
            attention_collective="all_reduce plus group weight all_gather and partial-output all_to_all",
            group_traffic_bytes_since_load=traffic,
            traffic_scope="logical bytes sent to other subgroup ranks, excluding self and transport overhead",
            exchange_dtype="bfloat16 attention weights and partial outputs")
        return result
