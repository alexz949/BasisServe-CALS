"""Load the upstream CPU-offload cache and its CUDA wrappers without model extras.

The selected upstream definitions are compiled unchanged. Unrelated imports for
MInference, FlashInfer normalization and the upstream model runner are omitted;
the evaluator supplies its existing Llama model and attention kernels.
"""
import ast
import gc
import importlib.util
import math
from pathlib import Path
from types import MethodType

import torch


ROOT = Path(__file__).resolve().parents[1]/'external/ShadowKV'


def load_cache_class():
    extensions = list((ROOT/'kernels').glob('shadowkv*.so'))
    assert len(extensions) == 1, extensions
    spec = importlib.util.spec_from_file_location('shadowkv', extensions[0])
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    namespace = dict(torch=torch, nn=torch.nn, math=math, gc=gc, shadowkv=extension,
                     __name__='official_shadowkv_cpu')
    definitions = {
        'models/tensor_op.py': {'apply_rotary_pos_emb_cuda_push_cache',
                               'batch_gather_gemm_rotary_pos_emb_cuda'},
        'models/kv_cache.py': {'ShadowKVCache_CPU'},
    }
    for filename, names in definitions.items():
        path = ROOT/filename
        parsed = ast.parse(path.read_text(), filename=str(path))
        selected = [node for node in parsed.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert {node.name for node in selected} == names
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ShadowKVCache_CPU']


@torch.inference_mode()
def attention_forward(self, hidden_states, position_embeddings, **kwargs):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from basisserve.kernels.compressed_v_decode_attention import (
        compressed_v_prefill_attention, compressed_v_decode_attention_triton)

    cache = self._official_shadow_cache
    batch, length, _ = hidden_states.shape
    heads, groups, dim = cache.num_attention_heads, cache.config.num_key_value_heads, cache.head_dim
    assert batch == 1 and dim == 128
    q = self.q_proj(hidden_states).view(batch, length, heads, dim).transpose(1, 2)
    pre_key = self.k_proj(hidden_states).view(batch, length, groups, dim).transpose(1, 2)
    value = self.v_proj(hidden_states).view(batch, length, groups, dim).transpose(1, 2)
    cos, sin = position_embeddings
    prefill = length > 1
    if prefill:
        cache.get_svd(pre_key, self.layer_idx)
    key = pre_key
    for start in range(0, length, 1024):
        stop = start + 1024
        qr, kr = apply_rotary_pos_emb(q[:, :, start:stop], key[:, :, start:stop],
                                     cos[:, start:stop], sin[:, start:stop])
        q[:, :, start:stop], key[:, :, start:stop] = qr, kr
    del qr, kr, pre_key
    if prefill:
        cache.prefill_kv_cache(value, self.layer_idx, key, q[:, :, -1:])
        output = compressed_v_prefill_attention(q, key, value, scale=self.scaling)
    else:
        cache.update_kv_cache(key, value, self.layer_idx)
        positions = cache.get_retrieval_position_ids(self.layer_idx, q)
        current = torch.cuda.current_stream()
        with torch.cuda.stream(cache.copy_stream):
            cache.copy_stream.wait_stream(current)
            selected_value = cache.get_value_cache(self.layer_idx, positions)
        selected_key = cache.get_key_cache(self.layer_idx, positions, None, self._shadow_cos_sin)
        current.wait_stream(cache.copy_stream)
        output = compressed_v_decode_attention_triton(q, selected_key, selected_value, scale=self.scaling)
    return self.o_proj(output.transpose(1, 2).contiguous().reshape(batch, length, -1)), None


def install(model):
    from evaluation.chunked_prefill_mlp import ChunkedTokenwise
    for layer in model.model.layers:
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm)
        layer.post_attention_layernorm = ChunkedTokenwise(layer.post_attention_layernorm)
        layer.self_attn.forward = MethodType(attention_forward, layer.self_attn)
    model.model.norm = ChunkedTokenwise(model.model.norm)


def make_cache(cache_class, config, length, cap, device):
    # The upstream CPU gather kernel uses the prompt's chunk-aligned head stride.
    cache = cache_class(config, batch_size=1, max_length=length,
                        device=str(device), dtype=torch.bfloat16, sparse_budget=2048, rank=160, chunk_size=8)
    chunks = length//cache.chunk_size-cache.local_chunk
    chunks -= chunks % 8
    local = length-chunks*cache.chunk_size
    capacity = local+cache.outlier_chunk*cache.chunk_size+cache.sparse_budget+cap
    shape = (*cache.k_cache_buffer.shape[:-2], capacity, cache.head_dim)
    cache.k_cache_buffer = cache.k_cache_buffer.new_zeros(shape)
    cache.v_cache_buffer = cache.v_cache_buffer.new_zeros(shape)
    return cache


@torch.inference_mode()
def generate(model, tokenizer, tokens, cap, cache_class):
    torch.manual_seed(0)
    device = model.get_input_embeddings().weight.device
    length = len(tokens)
    cache = make_cache(cache_class, model.config, length, cap, device)
    assert cache.v_cache_cpu.device.type == 'cpu' and cache.v_cache_cpu.is_pinned()
    assert cache.v_cache_buffer.shape[-1] == 128
    positions = torch.arange(length+cap, device=device)[None]
    cos, sin = model.model.rotary_emb(torch.empty(1, device=device, dtype=torch.bfloat16), positions)
    cos_sin = torch.cat((cos[0, :, :64], sin[0, :, :64]), -1).contiguous()
    for layer in model.model.layers:
        layer.self_attn._official_shadow_cache = cache
        layer.self_attn._shadow_cos_sin = cos_sin
    output = model(input_ids=tokens.to(device)[None], position_ids=positions[:, :length],
                   use_cache=False, logits_to_keep=1)
    first = output.logits[0, -1].float().cpu()
    assert torch.isfinite(first).all() and cache.get_kv_len() == length
    ids = [int(first.argmax())]
    del output
    cache.H2D()
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    while len(ids) < cap and ids[-1] not in eos:
        offset = length + len(ids) - 1
        output = model(input_ids=torch.tensor([[ids[-1]]], device=device),
                       position_ids=positions[:, offset:offset+1], use_cache=False, logits_to_keep=1)
        assert torch.isfinite(output.logits).all()
        ids.append(int(output.logits[0, -1].argmax()))
        del output
    assert cache.v_cache_cpu.device.type == 'cpu'
    assert cache.get_kv_len() == length+len(ids)-1
    statistics = dict(value_storage='pinned CPU V128; selected V transferred by official CUDA kernel',
        rank=cache.rank, routed_tokens=cache.sparse_budget, outlier_tokens=cache.outlier_chunk*cache.chunk_size,
        local_tokens=cache.prefill_local, generated_tokens=cache.gen_offset,
        cpu_value_bytes=cache.v_cache_cpu.numel()*cache.v_cache_cpu.element_size())
    for layer in model.model.layers:
        del layer.self_attn._official_shadow_cache, layer.self_attn._shadow_cos_sin
    return ids, first, statistics, ids[-1] in eos
