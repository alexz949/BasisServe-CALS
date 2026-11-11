"""Dense, Full-K V96, LRQK V96 and ShadowKV V96 attention for V2Q8 tensor-parallel runs (rank r of W).

Rank r owns KV heads [2r, 2r+2) and query heads [8r, 8r+8) at TP4. Batch rows are prefilled one request at a
time (set `prefill_row`), then decoded together. Output: dense Values use the row-parallel o_proj with an
AllReduce; compressed V96 uses the same source-local AllGather + replicated decoder as the BasisKV arm.

LRQK and ShadowKV keep the paper Table 2 equations (basisserve/core/c1_lrqk.py, c1_shadowkv.py). Tensor
parallelism changes no per-request arithmetic:
  * LRQK random initialisation draws all 32 heads from the per-layer seed and keeps the local slice;
    convergence deltas are global means (AllReduce over ranks) and are tested per request, so a batch row
    stops iterating exactly when it would alone (rows that stopped keep their earlier iterate).
  * ShadowKV gathers every KV head's pre-RoPE Keys and computes the one global rank-160 SVD per request
    (redundantly on every rank); each rank keeps U and its own heads' SV rows. Landmarks, outliers and
    selection are per KV head, as in the single-GPU equations.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from basisserve.core import c1_lrqk
from basisserve.core.c1_shadowkv import C1ShadowKVState, chunk_tokens, gather_group
from basisserve.kernels.compressed_v_decode_attention import compressed_v_decode_attention_triton
from benchmarks.system.v2q8_basis import GQA, HEAD_DIM, KV_HEADS, QUERY_HEADS, local

LRQK = c1_lrqk.LRQKConfig(topk=832, recent=64)  # paper Table 2 / Appendix C.1
SHADOWKV = dict(rank=160, budget=2048, chunk=8, outliers=48)


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def global_mean(tensor):
    """Mean over ranks of a per-rank mean taken over equally sized local shards."""
    if world_size() > 1:
        dist.all_reduce(tensor)
        tensor /= world_size()
    return tensor


def flash_prefill(query, key, value, scale):
    padded = F.pad(value, (0, HEAD_DIM - value.shape[-1]))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(query, key, padded, is_causal=True, scale=scale,
                                              enable_gqa=True)[..., :value.shape[-1]]


class TPAttention(torch.nn.Module):
    """Local projections and the output path shared by every baseline."""

    def __init__(self, original, *, value_weights, batch, capacity, rank, world):
        super().__init__()
        self.kv_heads, self.kv_start = KV_HEADS // world, rank * (KV_HEADS // world)
        self.query_heads, self.world, self.batch, self.capacity = self.kv_heads * GQA, world, batch, capacity
        self.layer_idx, self.scaling = int(original.layer_idx), float(original.scaling)
        self.q_weight = local(original.q_proj.weight.detach())
        self.k_weight = local(original.k_proj.weight.detach())
        if value_weights is None:
            self.value_dim, self.gather_output = HEAD_DIM, False
            self.v_weight = local(original.v_proj.weight.detach())
            self.o_weight = local(original.o_proj.weight.detach())
        else:
            self.v_weight, self.o_weight, self.value_dim = value_weights
            self.gather_output = world > 1
        self.length, self.prefill_row = 0, 0

    def project(self, attention):
        batch, tokens = attention.shape[0], attention.shape[2]
        output = attention.transpose(1, 2).reshape(batch * tokens, -1).contiguous()
        if not self.gather_output:
            projected = F.linear(output, self.o_weight)
            if self.world > 1:
                dist.all_reduce(projected)
            return projected.view(batch, tokens, -1)
        gathered = torch.empty(self.world, *output.shape, device="cuda", dtype=output.dtype)
        dist.all_gather_into_tensor(gathered, output)
        return F.linear(gathered.permute(1, 0, 2).reshape(batch * tokens, -1), self.o_weight).view(batch, tokens, -1)

    def projections(self, hidden_states, position_embeddings):
        batch, tokens, _ = hidden_states.shape
        query = F.linear(hidden_states, self.q_weight).view(batch, tokens, self.query_heads, HEAD_DIM).transpose(1, 2)
        pre_key = F.linear(hidden_states, self.k_weight).view(batch, tokens, self.kv_heads, HEAD_DIM).transpose(1, 2)
        value = F.linear(hidden_states, self.v_weight).view(batch, tokens, self.kv_heads, self.value_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, pre_key, cos, sin)
        return query, pre_key, key, value, cos, sin

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        query, pre_key, key, value, cos, sin = self.projections(hidden_states, position_embeddings)
        tokens = hidden_states.shape[1]
        if tokens > 1:
            assert hidden_states.shape[0] == 1 and tokens <= self.capacity
            attention = self.prefill(query, pre_key, key, value, cos, sin, self.prefill_row)
            self.length = tokens
        else:
            assert hidden_states.shape[0] == self.batch and self.length < self.capacity
            attention = self.decode(query, key, value)
            self.length += 1
        return self.project(attention), None

    def reset(self):
        self.length, self.prefill_row = 0, 0


class DenseAttention(TPAttention):
    """Exact attention. Dense V128 decodes with FlashAttention-2 (vLLM build, one cache block per request);
    compressed V96 (Full-K) decodes with the evaluator's Triton compressed-V kernel."""

    def __init__(self, original, **keywords):
        super().__init__(original, **keywords)
        shape = (self.batch, self.kv_heads, self.capacity)
        if self.value_dim == HEAD_DIM:
            # FlashAttention paged layout [blocks, block_size, heads, dim]; block_size must be a multiple of 256.
            self.capacity = math.ceil(self.capacity / 256) * 256
            self.key_cache = torch.empty(self.batch, self.capacity, self.kv_heads, HEAD_DIM, device="cuda",
                                         dtype=torch.bfloat16)
            self.value_cache = torch.empty_like(self.key_cache)
            self.block_table = torch.arange(self.batch, device="cuda", dtype=torch.int32)[:, None]
            self.cu_queries = torch.arange(self.batch + 1, device="cuda", dtype=torch.int32)
            self.lengths = torch.empty(self.batch, device="cuda", dtype=torch.int32)
        else:
            self.key_cache = torch.empty(*shape, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            self.value_cache = torch.empty(*shape, self.value_dim, device="cuda", dtype=torch.bfloat16)

    def prefill(self, query, pre_key, key, value, cos, sin, row):
        tokens = key.shape[2]
        if self.value_dim == HEAD_DIM:
            self.key_cache[row, :tokens].copy_(key[0].transpose(0, 1))
            self.value_cache[row, :tokens].copy_(value[0].transpose(0, 1))
        else:
            self.key_cache[row:row + 1, :, :tokens].copy_(key)
            self.value_cache[row:row + 1, :, :tokens].copy_(value)
        return flash_prefill(query, key, value, self.scaling)

    def decode(self, query, key, value):
        start = self.length
        if self.value_dim == HEAD_DIM:
            from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
            self.key_cache[:, start].copy_(key[:, :, 0])
            self.value_cache[:, start].copy_(value[:, :, 0])
            self.lengths.fill_(start + 1)
            output = flash_attn_varlen_func(
                query[:, :, 0], self.key_cache, self.value_cache, 1, self.cu_queries, start + 1,
                seqused_k=self.lengths, softmax_scale=self.scaling, causal=True,
                block_table=self.block_table, fa_version=2)
            return output[:, :, None]
        self.key_cache[:, :, start:start + 1].copy_(key)
        self.value_cache[:, :, start:start + 1].copy_(value)
        return compressed_v_decode_attention_triton(query, self.key_cache[:, :, :start + 1],
                                                    self.value_cache[:, :, :start + 1], scale=self.scaling)


def lrqk_prefill_factors(q, k, aq, ak, iterations, tolerance):
    """c1_lrqk.prefill_factors for one request with the convergence delta averaged over ranks."""
    solve = c1_lrqk._solve
    for _ in range(iterations):
        qq, kk = aq.transpose(-1, -2) @ aq, ak.transpose(-1, -2) @ ak
        bq, bk = solve(qq, aq.transpose(-1, -2) @ q), solve(kk, ak.transpose(-1, -2) @ k)
        new_aq = q @ solve(kk + bq @ bq.transpose(-1, -2), k.transpose(-1, -2) @ ak + bq.transpose(-1, -2), left=False)
        qq = new_aq.transpose(-1, -2) @ new_aq
        new_ak = k @ solve(qq + bk @ bk.transpose(-1, -2), q.transpose(-1, -2) @ new_aq + bk.transpose(-1, -2), left=False)
        delta = global_mean(torch.stack((F.mse_loss(aq, new_aq), F.mse_loss(ak, new_ak)))).max()
        aq, ak = new_aq, new_ak
        if delta < tolerance:
            break
    bq = solve(aq.transpose(-1, -2) @ aq, aq.transpose(-1, -2) @ q)
    bk = solve(ak.transpose(-1, -2) @ ak, ak.transpose(-1, -2) @ k)
    return aq, bq, ak, bk


def lrqk_decode_factors(bq, ak, bk, active_k, q, k, iterations, tolerance):
    """c1_lrqk.decode_factors with per-request convergence: a row keeps the iterate at which it would stop."""
    solve = c1_lrqk._solve
    kb, bbk = k @ bk.transpose(-1, -2), bk @ bk.transpose(-1, -2)
    rhs = q @ bq.transpose(-1, -2) + (q @ active_k.transpose(-1, -2)) @ ak
    normal = bq @ bq.transpose(-1, -2) + ak.transpose(-1, -2) @ ak
    qk = q @ k.transpose(-1, -2)
    new_k = solve(bbk, kb, left=False)
    rows = q.shape[0]
    done = torch.zeros(rows, dtype=torch.bool, device=q.device)
    final_q, final_k = None, None
    for _ in range(max(iterations, 1)):
        new_q = solve(normal + new_k.transpose(-1, -2) @ new_k, rhs + qk @ new_k, left=False)
        updated_k = solve(bbk + new_q.transpose(-1, -2) @ new_q, kb + qk @ new_q, left=False)
        delta = global_mean((new_k - updated_k).square().reshape(rows, -1).mean(-1))
        keep = done.view(rows, 1, 1, 1)
        final_q = new_q if final_q is None else torch.where(keep, final_q, new_q)
        final_k = updated_k if final_k is None else torch.where(keep, final_k, updated_k)
        new_k = updated_k
        done |= delta < tolerance
        if bool(done.all()):
            break
    dbq = final_q.transpose(-1, -2) @ (final_q @ bq - q)
    dbk = final_k.transpose(-1, -2) @ (final_k @ bk - k)
    step = c1_lrqk._gradient_step
    return bq - step(dbq, final_q) * dbq, bk - step(dbk, final_k) * dbk, final_q, final_k


class LRQKAttention(TPAttention):
    """LRQK (rank 32, top-832 per query head + 64 recent) on the shared V96 checkpoint; exact K/V resident."""

    def __init__(self, original, **keywords):
        super().__init__(original, **keywords)
        shape = (self.batch, self.kv_heads, self.capacity)
        self.key_cache = torch.empty(*shape, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        self.value_cache = torch.empty(*shape, self.value_dim, device="cuda", dtype=torch.bfloat16)
        self.code_cache = torch.empty(self.batch, self.query_heads, self.capacity, LRQK.rank, device="cuda",
                                      dtype=torch.bfloat16)
        self.row_factors, self.bq, self.bk, self.selected = {}, None, None, None
        self.statistics = {}

    def prefill(self, query, pre_key, key, value, cos, sin, row):
        tokens = key.shape[2]
        self.key_cache[row:row + 1, :, :tokens].copy_(key)
        self.value_cache[row:row + 1, :, :tokens].copy_(value)
        generator = torch.Generator(device="cuda").manual_seed(LRQK.seed + self.layer_idx)
        heads = slice(self.kv_start * GQA, self.kv_start * GQA + self.query_heads)
        aq = torch.randn((1, QUERY_HEADS, tokens, LRQK.rank), device="cuda", dtype=torch.float32, generator=generator)
        ak = torch.randn((1, QUERY_HEADS, tokens, LRQK.rank), device="cuda", dtype=torch.float32, generator=generator)
        aq, bq, ak, bk = [t.to(query.dtype) for t in lrqk_prefill_factors(
            query.float(), key.repeat_interleave(GQA, dim=1).float(), aq[:, heads].contiguous(),
            ak[:, heads].contiguous(), LRQK.prefill_iterations, LRQK.tolerance)]
        self.code_cache[row:row + 1, :, :tokens].copy_(ak)
        self.row_factors[row] = (bq, bk, c1_lrqk.select_tokens(aq[:, :, -1:], ak, LRQK))
        return flash_prefill(query, key, value, self.scaling)

    def decode(self, query, key, value):
        if self.row_factors:
            rows = [self.row_factors.pop(row) for row in range(self.batch)]
            self.bq, self.bk, self.selected = (torch.cat(parts) for parts in zip(*rows))
        start = self.length
        self.key_cache[:, :, start:start + 1].copy_(key)
        self.value_cache[:, :, start:start + 1].copy_(value)
        keys, values = self.key_cache[:, :, :start + 1], self.value_cache[:, :, :start + 1]
        codes = self.code_cache[:, :, :start]
        active_k = c1_lrqk.gather_heads(keys, self.selected)
        active_ak = codes.gather(2, self.selected[..., None].expand(-1, -1, -1, LRQK.rank))
        current_k = key.repeat_interleave(GQA, dim=1)
        factors = lrqk_decode_factors(self.bq.float(), active_ak.float(), self.bk.float(), active_k.float(),
                                      query.float(), current_k.float(), LRQK.decode_iterations, LRQK.tolerance)
        self.bq, self.bk, qcode, kcode = [t.to(query.dtype) for t in factors]
        self.code_cache[:, :, start:start + 1].copy_(kcode)
        self.selected = c1_lrqk.select_tokens(qcode, self.code_cache[:, :, :start + 1], LRQK)
        return c1_lrqk.selected_attention(query, keys, values, self.selected, self.scaling)

    def reset(self):
        super().reset()
        self.row_factors.clear()
        self.bq = self.bk = self.selected = None


class ShadowKVState(C1ShadowKVState):
    """C1ShadowKVState for local KV heads with the global SVD factors supplied (U replicated, local SV rows)."""

    def __init__(self, pre_key, post_key, cos, sin, u, sv, *, rank, budget, chunk, outliers):
        b, h, length, d = post_key.shape
        self.rank, self.budget, self.chunk = rank, budget, chunk
        self.outliers, self.prompt_length, self.length, self.steps = outliers, length, length, 0
        self.chunks = length // chunk - 4
        assert self.chunks > outliers + budget // chunk
        self.context_end = self.chunks * chunk
        self.u, self.sv = u, sv
        self.cos, self.sin = cos.clone(), sin.clone()
        ctx = post_key[:, :, :self.context_end].reshape(b, h, self.chunks, chunk, d)
        landmarks = ctx.mean(-2)
        similarity = F.cosine_similarity(landmarks.unsqueeze(3).expand_as(ctx), ctx, dim=-1)
        outlier_ids = similarity.min(-1).values.topk(outliers, largest=False).indices
        all_ids = torch.arange(self.chunks, device=post_key.device).view(1, 1, -1).expand(b, h, -1)
        mask = torch.ones_like(all_ids, dtype=torch.bool).scatter(-1, outlier_ids, False)
        self.landmark_ids = all_ids.masked_select(mask).view(b, h, -1)
        self.landmarks = landmarks.gather(2, self.landmark_ids[..., None].expand(b, h, self.chunks - outliers, d)).contiguous()
        local_ids = torch.arange(self.context_end, length, device=post_key.device).view(1, 1, -1).expand(b, h, -1)
        self.fixed_ids = torch.cat((local_ids, chunk_tokens(outlier_ids, chunk)), -1)
        self.fixed_key = gather_group(post_key, self.fixed_ids).clone()
        self.generated_key = post_key[:, :, :0].clone()
        self.selected_ids = None

    @classmethod
    def merge(cls, states):
        merged = object.__new__(cls)
        merged.__dict__.update(states[0].__dict__)
        for name in ("u", "sv", "landmark_ids", "landmarks", "fixed_ids", "fixed_key", "generated_key"):
            setattr(merged, name, torch.cat([getattr(state, name) for state in states]))
        return merged


class ShadowKVAttention(TPAttention):
    """ShadowKV (rank-160 global pre-RoPE SVD, 8-token chunks, 48 outlier chunks, 2048 routed tokens, exact
    local tail) on the shared V96 checkpoint; V96 resident, historical Keys reconstructed (not stored)."""

    def __init__(self, original, **keywords):
        super().__init__(original, **keywords)
        self.value_cache = torch.empty(self.batch, self.kv_heads, self.capacity, self.value_dim, device="cuda",
                                       dtype=torch.bfloat16)
        self.row_states, self.state = {}, None

    def prefill(self, query, pre_key, key, value, cos, sin, row):
        tokens = key.shape[2]
        self.value_cache[row:row + 1, :, :tokens].copy_(value)
        if self.world > 1:
            gathered = torch.empty(self.world, *pre_key.shape, device="cuda", dtype=pre_key.dtype)
            dist.all_gather_into_tensor(gathered, pre_key.contiguous())
            every_head = gathered.permute(1, 0, 2, 3, 4).reshape(1, KV_HEADS, tokens, HEAD_DIM)
        else:
            every_head = pre_key
        flat = every_head.transpose(1, 2).reshape(1, tokens, KV_HEADS * HEAD_DIM)
        u, s, v = torch.svd(flat.float())
        rank = SHADOWKV["rank"]
        sv = (torch.diag_embed(s[:, :rank]) @ v.transpose(1, 2)[:, :rank]).to(pre_key.dtype).view(
            1, rank, KV_HEADS, HEAD_DIM).transpose(1, 2)
        self.row_states[row] = ShadowKVState(
            pre_key, key, cos, sin, u[:, :, :rank].to(pre_key.dtype).contiguous(),
            sv[:, self.kv_start:self.kv_start + self.kv_heads].contiguous(), **SHADOWKV)
        return flash_prefill(query, key, value, self.scaling)

    def decode(self, query, key, value):
        if self.row_states:
            self.state = ShadowKVState.merge([self.row_states.pop(row) for row in range(self.batch)])
        start = self.length
        self.value_cache[:, :, start:start + 1].copy_(value)
        return self.state.decode(query, key, self.value_cache[:, :, :start + 1], self.scaling)

    def reset(self):
        super().reset()
        self.row_states.clear()
        self.state = None


ATTENTION = {"dense": DenseAttention, "fullk_v96": DenseAttention, "lrqk_v96": LRQKAttention,
             "shadowkv_v96": ShadowKVAttention}


def install(model, arm, *, value_checkpoint, batch, capacity, rank, world, chunked):
    """Replace every attention layer with the arm's TP module; chunked is ChunkedTokenwise."""
    from pathlib import Path
    from benchmarks.system.v2q8_basis import compressed_value_weights

    kv_count = KV_HEADS // world
    layers = []
    for index, layer in enumerate(model.model.layers):
        weights = None if arm == "dense" else compressed_value_weights(
            layer.self_attn, Path(value_checkpoint), index, rank * kv_count, kv_count)
        replacement = ATTENTION[arm](layer.self_attn, value_weights=weights, batch=batch, capacity=capacity,
                                     rank=rank, world=world)
        layer.self_attn = replacement
        layer.mlp = chunked(layer.mlp, chunk_size=1024)
        layer.input_layernorm = chunked(layer.input_layernorm, chunk_size=2048)
        layer.post_attention_layernorm = chunked(layer.post_attention_layernorm, chunk_size=2048)
        layers.append(replacement)
    model.model.norm = chunked(model.model.norm, chunk_size=2048)
    return layers
