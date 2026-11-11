"""The upstream ShadowKV runtime (ByteDance-Seed/ShadowKV@e51904cd: models.llama.Llama with ShadowKVCache_CPU),
imported from external/ShadowKV without edits, and its minimal tensor-parallel port.

W = 1: the upstream Llama runs unchanged, imported as system_benchmarks/tp1_offline did (flash_attn served by
vLLM's FlashAttention-2 build, MInference disabled).

W > 1: TPLlama subclasses the upstream Llama and changes only what tensor parallelism requires:
  * weights: after the upstream init_parameters, rank r keeps query heads [r*32/W, (r+1)*32/W) and KV heads
    [r*8/W, (r+1)*8/W) (rows of wqkv), the matching wo columns, and its 1/W slice of the MLP intermediate
    (gate_up rows, down_proj columns). Embedding, norms and lm_head stay replicated, as upstream holds them;
  * the head counts the upstream layer code reads (num_heads, num_key_value_heads, and hidden_size, which upstream
    uses only to reshape the attention output) are the local ones, and the cache is the upstream
    ShadowKVCache_CPU built for the local heads;
  * post_attention_compute is the upstream body with an AllReduce after wo and after down_proj;
  * layer_compute is the upstream body with one line changed: the chunked post-attention output buffer (prompts
    with batch x length > 64K) is shaped like the residual instead of the attention output, which is narrower
    under TP;
  * get_svd is the upstream body. Before the SVD, the layer's owner (layer % W) receives every rank's pre-RoPE
    Keys, so the decomposition is upstream's global [T, 8*128] one; afterwards U (shared by all heads) and each
    rank's SV rows are sent to every rank. The SVD runs inline on one rank per layer, as upstream runs it inline.
layer_compute, inference, batch_prefill (request groups of 4, or 8 for 120K-200K prompts), RoPE, the FlashAttention
calls, landmarks, outliers, Value offload, selection, Key reconstruction, Value fetch and H2D are upstream code.
"""

from contextlib import nullcontext
import gc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external/ShadowKV"
SPARSE_BUDGET, RANK, CHUNK = 2048, 160, 8
# Breakdown timers for the TP-only communication inside get_svd (set by the benchmark; a no-op otherwise).
PHASE = lambda label: nullcontext()


def import_upstream():
    """Upstream modules, with the import-surface adapters of system_benchmarks/tp1_offline."""
    if "models.llama" not in sys.modules:
        from benchmarks.system.paper_faithful_compat import install_shadowkv_import_adapters
        install_shadowkv_import_adapters(UPSTREAM)
        from transformers import LlamaConfig, LlamaForCausalLM
        if not hasattr(LlamaConfig, "rope_theta"):
            LlamaConfig.rope_theta = property(lambda c: c.rope_parameters.get("rope_theta", c.default_theta))
        original = LlamaForCausalLM.from_pretrained

        def load(*positional, **keywords):
            # Newer transformers keep the rotary embedding on the model; upstream reads it from layer 0.
            loaded = original(*positional, **keywords)
            for layer in loaded.model.layers:
                layer.self_attn.rotary_emb = loaded.model.rotary_emb
            return loaded

        LlamaForCausalLM.from_pretrained = load
        package = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("models", loader=None,
                                                                                  is_package=True))
        package.__path__ = [str(UPSTREAM / "models")]
        sys.modules["models"] = package
    import models.base as base
    import models.llama as llama
    import models.tensor_op as tensor_op
    return base, llama, tensor_op


def _exchange(operations):
    for work in dist.batch_isend_irecv(operations) if operations else ():
        work.wait()


def tp_get_svd(self, new_k_cache, layer_idx):
    """Upstream ShadowKVCache_CPU.get_svd on the local heads, with the TP gather and scatter around the SVD."""
    world, rank = dist.get_world_size(), dist.get_rank()
    # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]  (upstream; local heads here)
    if new_k_cache.shape[1] <= 32:
        k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads * self.head_dim)
    else:
        k_cache = new_k_cache

    if layer_idx == 0 and self.prefilled_batch == 0:
        # init U, SV  (upstream: on CPU until H2D)
        self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device='cpu',
                             dtype=self.dtype)
        self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.head_dim, self.rank,
                              device='cpu', dtype=self.dtype)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    bsz, length, device = k_cache.shape[0], k_cache.shape[1], k_cache.device
    owner = layer_idx % world
    with PHASE("k_gather"):
        shard = k_cache.contiguous()
        if rank == owner:
            parts = [shard if peer == owner else torch.empty_like(shard) for peer in range(world)]
            _exchange([dist.P2POp(dist.irecv, parts[peer], peer) for peer in range(world) if peer != owner])
            # Global KV heads in rank order: upstream's [bsz, prefill, 8*128].
            k_cache = torch.cat(parts, dim=-1)
            del parts
        else:
            _exchange([dist.P2POp(dist.isend, shard, owner)])
        del shard

    if rank == owner:
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1, 2)
        u = u[:, :, :self.rank].to(self.dtype)
        # [bsz, 160, 1024] --> (bsz, 8, 160, 128) --> (bsz, 8, 128, 160)  (upstream, all 8 heads)
        temp_sv = torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(
            bsz, -1, world * self.num_key_value_heads, self.head_dim).transpose(1, 2)
        temp_sv = temp_sv.transpose(-1, -2)
        shards = temp_sv.reshape(bsz, world, self.num_key_value_heads, self.head_dim, self.rank).transpose(
            0, 1).contiguous()
        del s, v, temp_sv, k_cache
    else:
        u = torch.empty(bsz, length, self.rank, device=device, dtype=self.dtype)

    # Non-owners wait here (in the U broadcast) until the owner's SVD is done: the label includes that wait.
    with PHASE("factor_redistribution_incl_owner_svd_wait"):
        u = u.contiguous()
        dist.broadcast(u, src=owner)
        if rank == owner:
            local_sv = shards[owner]
            _exchange([dist.P2POp(dist.isend, shards[peer], peer) for peer in range(world) if peer != owner])
        else:
            local_sv = torch.empty(bsz, self.num_key_value_heads, self.head_dim, self.rank, device=device,
                                   dtype=self.dtype)
            _exchange([dist.P2POp(dist.irecv, local_sv, owner)])

    self.U[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(u)  # [bsz, 128k, 160]
    self.SV[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(local_sv)  # [bsz, 8/W, 128, 160]
    del u, local_sv
    if rank == owner:
        del shards


def tp_class(Llama, layer_norm, base):
    import vllm

    class TPLlama(Llama):
        """Upstream Llama on this rank's heads and MLP slice (see the module docstring)."""

        def init_parameters(self):
            super().init_parameters()
            world, rank = dist.get_world_size(), dist.get_rank()
            q_local = self.num_heads * self.head_dim // world
            kv_local = self.num_key_value_heads * self.head_dim // world
            for layer in self.layers:
                q, k, v = layer.wqkv.split([layer.q_size, layer.kv_size, layer.kv_size], dim=0)
                layer.wqkv = torch.cat((q[rank * q_local:(rank + 1) * q_local], k[rank * kv_local:(rank + 1) * kv_local],
                                        v[rank * kv_local:(rank + 1) * kv_local]), dim=0).contiguous()
                layer.wo = layer.wo[:, rank * q_local:(rank + 1) * q_local].contiguous()
                gate, up = layer.gate_up_proj.chunk(2, dim=0)
                inter = gate.shape[0] // world
                layer.gate_up_proj = torch.cat((gate[rank * inter:(rank + 1) * inter],
                                                up[rank * inter:(rank + 1) * inter]), dim=0).contiguous()
                layer.down_proj = layer.down_proj[:, rank * inter:(rank + 1) * inter].contiguous()
                layer.q_size, layer.kv_size = q_local, kv_local
                del q, k, v, gate, up
            gc.collect()
            torch.cuda.empty_cache()
            self.num_heads //= world
            self.num_key_value_heads //= world
            self.hidden_size = self.num_heads * self.head_dim

        def init_kv_cache(self, sparse_budget, rank, chunk_size, config):
            local = SimpleNamespace(hidden_size=self.num_heads * self.head_dim, num_attention_heads=self.num_heads,
                                    num_key_value_heads=self.num_key_value_heads,
                                    num_hidden_layers=config.num_hidden_layers)
            super().init_kv_cache(sparse_budget, rank, chunk_size, local)
            self.kv_cache.get_svd = MethodType(tp_get_svd, self.kv_cache)

        @torch.inference_mode()
        def layer_compute(self, buffer, layer_idx, hidden_states, position_ids):
            """Upstream LLM.layer_compute (models/base.py), unchanged except one line: the chunked post-attention
            output buffer is shaped like the residual. Upstream shapes it like the attention output, which has the
            model width on one GPU but only this rank's head width under TP. Upstream globals are read from
            models.base at call time."""
            residual = hidden_states
            bsz, q_len, _ = hidden_states.size()
            query_states, key_states, value_states = self.pre_attention_compute(
                hidden_states,
                buffer,
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim
            )

            if isinstance(self.kv_cache, base.KV_Cache):
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)

                if self.minference == True and q_len > 1:
                    hidden_states = base.minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                else:
                    hidden_states = base.flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

            elif isinstance(self.kv_cache, base.ShadowKVCache) or isinstance(self.kv_cache, base.ShadowKVCache_CPU):

                if q_len > 4*1024: # prefill
                    # svd unrope key and save
                    self.kv_cache.get_svd(key_states, layer_idx=layer_idx)
                    query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                    self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])

                    if self.minference == True:
                        hidden_states = base.minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                    else:
                        hidden_states = base.flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

                else: # decode
                    # rope query and key
                    query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

                    # update kv cache to buffer
                    self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)

                    # get retrieval idx
                    position_ids = self.kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states)

                    # multi-stream
                    curr_stream = torch.cuda.current_stream()
                    get_value_stream = self.kv_cache.copy_stream

                    with torch.cuda.stream(get_value_stream):
                        get_value_stream.wait_stream(curr_stream)
                        value_states = self.kv_cache.get_value_cache(layer_idx, position_ids)

                    # gather key cache from GPU and RoPE it (should be hide by CPU offloading time)
                    key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single, cos_sin_cache=self.cos_sin_cache)

                    curr_stream.wait_stream(get_value_stream)

                    # flash attention
                    hidden_states = base.flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

            else:
                raise ValueError(f"Invalid attention mode {self.attn_mode}")

            hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)

            if bsz*q_len > 64*1024: # [bsz, seq, 128]
                output = torch.empty_like(residual)  # TP: upstream has torch.empty_like(hidden_states)
                prop_iter = bsz * q_len // (8*1024)
                prefill_chunk_size = bsz * q_len // prop_iter
                prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
                for i in range(prefill_iter):
                    start = i*prefill_chunk_size
                    end = (i+1)*prefill_chunk_size
                    output[:, start:end] = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)

                hidden_states = output

            else:
                hidden_states = self.post_attention_compute(hidden_states, residual, buffer)

            return hidden_states

        def post_attention_compute(self, attn_output, residual, buffer):
            # Upstream body; TP: AllReduce of the row-parallel wo and down_proj outputs.
            hidden_states = F.linear(attn_output, buffer.wo)
            dist.all_reduce(hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon,
                                       buffer.post_attention_layernorm_weight)

            hidden_states = F.linear(hidden_states, buffer.gate_up_proj)
            d = hidden_states.shape[-1] // 2
            output_shape = (hidden_states.shape[:-1] + (d, ))
            out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            vllm._custom_ops.silu_and_mul(out, hidden_states)

            hidden_states = F.linear(out, buffer.down_proj)
            dist.all_reduce(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

    return TPLlama


def build(model_path, *, batch, max_length, device):
    """Upstream Llama (W=1) or its TP port (W>1), attn_mode shadowkv_cpu, budget 2048, rank 160, chunk 8."""
    base, llama, tensor_op = import_upstream()
    world = dist.get_world_size() if dist.is_initialized() else 1
    cls = llama.Llama if world == 1 else tp_class(llama.Llama, tensor_op.layer_norm, base)
    return cls(model_name=str(model_path), device=str(device), batch_size=batch, max_length=max_length,
               attn_mode="shadowkv_cpu", sparse_budget=SPARSE_BUDGET, rank=RANK, chunk_size=CHUNK, minference=False)


def fresh_cache(model, *, length, output_tokens):
    """A new upstream cache (as system_benchmarks/tp1_offline): request state and the original CPU placement of
    the scratch buffers that H2D moves. The generation buffers are sized for the actual initial support plus the
    requested output length (upstream sizes them for 128 generated tokens), rounded up to 16; capacity only."""
    model.kv_cache = None
    gc.collect()
    model.init_kv_cache(SPARSE_BUDGET, RANK, CHUNK, model.config)
    cache = model.kv_cache
    chunks = length // cache.chunk_size - cache.local_chunk
    chunks -= chunks % 8
    local_tokens = length - chunks * cache.chunk_size
    required = cache.sparse_budget + cache.outlier_chunk * cache.chunk_size + local_tokens + output_tokens
    # Rounded up to 16 so that the batched decode attention can read each row's buffer as one paged-KV block
    # (paper_faithful_compat); upstream's own default, 2592 tokens, is already a multiple of 16.
    required = -(-required // 16) * 16
    for name in ("k_cache_buffer", "v_cache_buffer"):
        buffer = getattr(cache, name)
        shape = list(buffer.shape)
        shape[-2] = max(shape[-2], required)
        if shape != list(buffer.shape):
            setattr(cache, name, torch.zeros(shape, device=buffer.device, dtype=buffer.dtype))
    return cache
