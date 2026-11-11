"""Tensor-parallel adapter around the official LRQK cache (tenghuilee/LRQK@caf16293; TP4, TP8, or one GPU),
deployed with exact Keys offloaded to host and Values resident on GPU (the V2Q8 experiment design).

LRQK is head-parallel: the prefill factorisation Q~A_Q B_Q, K~A_K B_K, the decode factor updates, the proxy
scores A_K q_hat, top-k, the GPU hit cache and the CPU pool are all per (request, query head), with KV repeated
over each GQA group. Rank r of W therefore runs the upstream factory (LightAttentionIndicesFactory) on its query
heads [8r, 8r+8) and KV heads [2r, 2r+2) at TP4, with no communication beyond the attention output collective.
Upstream couples heads (and batch rows) in exactly two places, and both are preserved:
  * random initialisation draws A_Q and A_K for every head at once: each rank draws all 32 heads from the same
    seed and keeps its slice (the RNG is reseeded per request so each request starts as it would alone);
  * the early-exit tests use a mean over the whole tensor: deltas are AllReduced over ranks and, in batched
    decode, tested per request (a request that would have stopped keeps that iterate).

Placement (requested): upstream keeps the full K and V in one pinned CPU pool and fetches missed K and V for the
GPU hit cache. Here the CPU pool holds exact Keys only; Values stay in a GPU-resident cache and missed Values are
gathered on GPU. Selection, hit/miss bookkeeping, the CPU-side Key miss gather (cpp_kernel.d0_index_select) and
the attention support are upstream's. The Value width is that of v_proj (dense 128, or a compressed checkpoint).

Changes that do not alter LRQK's arithmetic:
  * storage is allocated once at its exact final size (upstream grows 1.5x and calls empty_cache), and the
    per-layer-per-step torch.cuda.empty_cache() of upstream DynamicLRQKCache.update is not issued;
  * batch rows are prefilled one request at a time into their rows (upstream prefills a batch at once);
  * the GPU hit cache and Key codes stay on GPU from prefill on (upstream LightAttentionIndicesOffloadPrefill
    parks them in host memory during prefill and restores them at the first decode step);
  * upstream's torch.compile(max_autotune) on its helpers is optional (LRQK_TORCH_COMPILE=1 keeps it; they
    graph-break on nonzero/boolean indexing); smoke measures both and the formal run uses the faster.
"""

import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260924
CONFIG = dict(rank=32, topk=2048, lite=64, iterations=(2, 2), tolerance=(1e-2, 1e-2))  # paper RULER-128K


def upstream():
    """Import the official module (its flash_attn import is served by vLLM's FlashAttention-2 build)."""
    if "lrqk_attention" not in sys.modules:
        from benchmarks.system.paper_faithful_compat import install_flash_attn_adapter
        import torch._dynamo
        install_flash_attn_adapter()
        sys.path.insert(0, str(ROOT / "external/LRQK"))
        sys.path.insert(0, str(ROOT / "external/LRQK/cpp_kernel"))
        torch._dynamo.config.disable = os.environ.get("LRQK_TORCH_COMPILE") != "1"
    import lrqk_attention
    return lrqk_attention


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def global_mean(value):
    """Mean over ranks of per-rank means taken over equally sized head shards (upstream's whole-tensor mean)."""
    if world() > 1:
        dist.all_reduce(value)
        value /= world()
    return value


def prefill_factors(Q, K, A_Q, A_K, max_iter, tol):
    """Upstream _lrqk_prefill_inv_w1 for one request (the ALS fit); the early-exit delta is the mean over all heads."""
    for _ in range(max_iter):
        AQTAQ = A_Q.transpose(-1, -2) @ A_Q
        AKTAK = A_K.transpose(-1, -2) @ A_K
        B_Q = torch.linalg.solve_ex(AQTAQ, A_Q.transpose(-1, -2) @ Q)[0]
        B_K = torch.linalg.solve_ex(AKTAK, A_K.transpose(-1, -2) @ K)[0]
        BQBQT = B_Q @ B_Q.transpose(-1, -2)
        new_A_Q = Q @ torch.linalg.solve_ex(AKTAK + BQBQT, K.transpose(-1, -2) @ A_K + B_Q.transpose(-1, -2),
                                            left=False)[0]
        AQTAQ = new_A_Q.transpose(-1, -2) @ new_A_Q
        BKBKT = B_K @ B_K.transpose(-1, -2)
        new_A_K = K @ torch.linalg.solve_ex(AQTAQ + BKBKT, Q.transpose(-1, -2) @ new_A_Q + B_K.transpose(-1, -2),
                                            left=False)[0]
        # F.mse_loss's value ((a - b) * (a - b), then mean()), computed in the old factors, which are dead after
        # this line, instead of in a new factor-sized temporary (0.5 GiB at 128K on one GPU).
        delta = global_mean(torch.stack((A_Q.sub_(new_A_Q).square_().mean(),
                                         A_K.sub_(new_A_K).square_().mean()))).max()
        A_Q, A_K = new_A_Q, new_A_K
        if delta < tol:
            break
    AQTAQ = A_Q.transpose(-1, -2) @ A_Q
    AKTAK = A_K.transpose(-1, -2) @ A_K
    B_Q = torch.linalg.solve_ex(AQTAQ, A_Q.transpose(-1, -2) @ Q)[0]
    B_K = torch.linalg.solve_ex(AKTAK, A_K.transpose(-1, -2) @ K)[0]
    return A_Q, B_Q, A_K, B_K


def decode_factors(B_Q, A_K, B_K, K__, q, k, max_iter, tol):
    """Upstream _lrqk_decode_inv_w1 (FP32); the early-exit delta is each request's mean over all heads."""
    learning_rate = upstream()._lrqk_decode_gd_B_lr
    max_iter = max(max_iter, 1)
    ktBKT = k @ B_K.transpose(-1, -2)
    BKBKT = B_K @ B_K.transpose(-1, -2)
    qtBQT_qtKTAK = q @ B_Q.transpose(-1, -2) + (q @ K__.transpose(-1, -2)) @ A_K
    BQBQT_AKTAK = B_Q @ B_Q.transpose(-1, -2) + A_K.transpose(-1, -2) @ A_K
    qtktT = q @ k.transpose(-1, -2)
    hat_k_t = torch.linalg.solve_ex(BKBKT, ktBKT, left=False)[0]
    rows = q.shape[0]
    done = torch.zeros(rows, dtype=torch.bool, device=q.device)
    final_q = final_k = None
    for _ in range(max_iter):
        hat_q_t = torch.linalg.solve_ex(BQBQT_AKTAK + hat_k_t.transpose(-1, -2) @ hat_k_t,
                                        qtBQT_qtKTAK + qtktT @ hat_k_t, left=False)[0]
        new_hat_k_t = torch.linalg.solve_ex(BKBKT + hat_q_t.transpose(-1, -2) @ hat_q_t,
                                            ktBKT + qtktT @ hat_q_t, left=False)[0]
        delta = global_mean((hat_k_t - new_hat_k_t).square().reshape(rows, -1).mean(-1))
        keep = done.view(rows, 1, 1, 1)
        final_q = hat_q_t if final_q is None else torch.where(keep, final_q, hat_q_t)
        final_k = new_hat_k_t if final_k is None else torch.where(keep, final_k, new_hat_k_t)
        hat_k_t = new_hat_k_t
        done |= delta < tol
        if bool(done.all()):
            break
    dBQ = final_q.transpose(-1, -2) @ (final_q @ B_Q - q)
    dBK = final_k.transpose(-1, -2) @ (final_k @ B_K - k)
    B_Q = B_Q - learning_rate(dBQ, final_q) * dBQ
    B_K = B_K - learning_rate(dBK, final_k) * dBK
    return B_Q, B_K, final_q, final_k


if os.environ.get("LRQK_TORCH_COMPILE") == "1":
    # Same options upstream gives its own solvers (lrqk_attention.py @torch.compile on _lrqk_*_inv_w1).
    _COMPILE = dict(fullgraph=False, options={"epilogue_fusion": True, "max_autotune": True})
    prefill_factors = torch.compile(prefill_factors, **_COMPILE)
    decode_factors = torch.compile(decode_factors, **_COMPILE)


def cast_decode(B_Q, A_K, B_K, K__, q, k, lambda_dec_q=1.0, lambda_dec_k=1.0, max_iter=8, tol=1e-5):
    """Replacement for upstream cast_lrqk_decode (lambda = 1, the paper setting)."""
    assert lambda_dec_q == 1.0 and lambda_dec_k == 1.0
    out = decode_factors(*(t.float() for t in (B_Q, A_K, B_K, K__, q, k)), max_iter, tol)
    return (o.to(dtype=k.dtype) for o in out)


def make_factory_class():
    module = upstream()
    if not hasattr(module, "upstream_cast_lrqk_decode"):
        module.upstream_cast_lrqk_decode = module.cast_lrqk_decode  # kept for reference checks
    module.cast_lrqk_decode = cast_decode

    class TPFactory(module.LightAttentionIndicesFactory):
        """Upstream factory with batch storage allocated once, one-request-at-a-time prefill into rows, exact Keys
        in a pinned CPU pool and Values in a GPU-resident cache."""

        def __init__(self, *, batch, query_heads, kv_heads, value_dim, capacity, head_start, device):
            super().__init__(num_lite_tokens=CONFIG["lite"], attn_topk=CONFIG["topk"], num_key_value_groups=4,
                             r=CONFIG["rank"], max_iter=CONFIG["iterations"], tol=CONFIG["tolerance"],
                             capacity=capacity, init_aq_ak_method=module.InitAQAK.randn, device=device,
                             dtype=torch.bfloat16)
            rank, dim = CONFIG["rank"], 128
            # Lite tokens flush to the CPU pool 64 at a time; the margin keeps every flush inside the allocation.
            capacity += 2 * CONFIG["lite"]
            self.head_start, self.batch = head_start, batch
            self._A_K.data = torch.zeros(batch, query_heads, capacity, rank, device=device, dtype=torch.bfloat16)
            self._A_K.capacity = capacity
            self.KVcpu = module.AutoIncreaseTensor(capacity=capacity, dim=2, device="cpu", dtype=torch.bfloat16,
                                                   pin_memory=True)
            self.KVcpu.data = torch.zeros(batch, kv_heads, capacity, dim, dtype=torch.bfloat16, pin_memory=True)
            self.value_cache = torch.zeros(batch, kv_heads, capacity, value_dim, device=device, dtype=torch.bfloat16)
            self.Kgpu = torch.zeros(batch, query_heads, self.gpu_capacity, dim, device=device, dtype=torch.bfloat16)
            self.Vgpu = torch.zeros(batch, query_heads, self.gpu_capacity, value_dim, device=device,
                                    dtype=torch.bfloat16)
            # Same memory layout as upstream's B (torch.linalg.solve_ex returns column-major [r, dim] matrices); the
            # layout decides which GEMM kernels the decode solver runs, and so its rounding.
            self.B_Q = torch.zeros(batch, query_heads, dim, rank, device=device, dtype=torch.bfloat16).transpose(-1, -2)
            self.B_K = torch.zeros(batch, query_heads, dim, rank, device=device, dtype=torch.bfloat16).transpose(-1, -2)
            self.hit_indices = torch.zeros(batch, query_heads, self.attn_topk, 1, device=device, dtype=torch.long)
            self.timers = None  # set by the breakdown profile: callable(label) -> context manager
            self.missed_slots = torch.zeros((), device=device, dtype=torch.long)

        def _timed(self, label):
            from contextlib import nullcontext
            return self.timers(label) if self.timers else nullcontext()

        def prefill_row(self, row, query_states, key_states, value_states):
            """Upstream prefill (sequence longer than the GPU cache) for one request, written into its row."""
            _, heads, length, _ = query_states.shape
            assert length > self.gpu_capacity
            groups = self.num_key_value_groups
            with self._timed("als_factor_fit"):
                # Same FP32 values as upstream's repeat_kv(K).float(); the BF16 repeated copy that upstream keeps
                # alive through the fit is not materialized (1 GiB at 128K on one GPU).
                Q32, K32 = query_states.float(), module.repeat_kv(key_states.float(), groups)
                # Upstream _init_randn_AQ_AK over all 32 query heads, this rank's slice kept.
                if self._first_layer:
                    torch.manual_seed(SEED)  # each request starts its RNG stream as it would alone
                shape = (1, heads * world(), length, self.r)
                own = slice(self.head_start, self.head_start + heads)
                A_Q = torch.randn(shape, dtype=torch.float32, device=Q32.device)[:, own].contiguous()
                A_K = torch.randn(shape, dtype=torch.float32, device=Q32.device)[:, own].contiguous()
                A_Q, B_Q, A_K, B_K = (t.to(torch.bfloat16) for t in prefill_factors(
                    Q32, K32, A_Q, A_K, self.max_iter[0], self.tol[0]))
                del Q32, K32
            with self._timed("cache_build_and_key_offload"):
                self.prefill_length = self.current_len = length
                self.B_Q[row:row + 1].copy_(B_Q)
                self.B_K[row:row + 1].copy_(B_K)
                self._A_K.data[row:row + 1, :, :length].copy_(A_K)
                self.KVcpu.data[row:row + 1, :, :length].copy_(key_states)
                self.value_cache[row:row + 1, :, :length].copy_(value_states)
                hit = self.compute_hit_indices(query_states=A_Q[:, :, -1:, :],
                                               key_states=A_K[:, :, 0:length - self.num_lite_tokens, :])
                self.hit_indices[row:row + 1].copy_(hit)
                self.fetch_by_hit(dst=self.Kgpu[row:row + 1, :, :self.attn_topk], src=key_states, hit_indices=hit)
                self.fetch_by_hit(dst=self.Vgpu[row:row + 1, :, :self.attn_topk], src=value_states, hit_indices=hit)
                begin = length - self.num_lite_tokens
                lite = slice(self.attn_topk, self.attn_topk + self.num_lite_tokens)
                self.Kgpu[row:row + 1, :, lite] = module.repeat_kv(key_states.narrow(2, begin, self.num_lite_tokens),
                                                                   groups)
                self.Vgpu[row:row + 1, :, lite] = module.repeat_kv(value_states.narrow(2, begin, self.num_lite_tokens),
                                                                   groups)
                self.lite_indices, self.lite_len = list(range(begin, length)), 0
                if row == self.batch - 1:
                    self._A_K.current_len = self.KVcpu.current_len = length

        def _append(self, key_states, value_states):
            """Upstream _append; the new Value also enters the GPU Value cache, and only Keys flush to the CPU pool."""
            groups = self.num_key_value_groups
            lite = slice(self.attn_topk + self.lite_len, self.attn_topk + self.lite_len + 1)
            self.Kgpu[:, :, lite] = module.repeat_kv(key_states, groups)
            self.Vgpu[:, :, lite] = module.repeat_kv(value_states, groups)
            self.value_cache[:, :, self.current_len:self.current_len + 1].copy_(value_states)
            self.lite_indices[self.lite_len] = self.current_len
            self.lite_len += 1
            self.current_len += 1
            if self.lite_len >= self.num_lite_tokens:
                self.KVcpu.append(self.Klite)
                self.lite_len = 0

        def decode(self, query_states, key_states, value_states):
            """Upstream LightAttentionIndicesFactory.decode (long-context branch); missed Keys come from the CPU
            pool, missed Values from the GPU Value cache."""
            assert self.current_len >= self.gpu_capacity
            groups = self.num_key_value_groups
            with self._timed("append_new_token"):
                self._append(key_states, value_states)
                rep_key_states = module.repeat_kv(key_states, groups)
            with self._timed("factor_update"):
                B_Q, B_K, hat_q_t, hat_k_t = module.cast_lrqk_decode(
                    B_Q=self.B_Q, A_K=self._get_A_K_hitted_with_lite(), B_K=self.B_K, K__=self.Kgpu,
                    q=query_states, k=rep_key_states, lambda_dec_q=self.weights[2], lambda_dec_k=self.weights[3],
                    max_iter=self.max_iter[1], tol=self.tol[1])
                self.sync_cache(B_Q, hat_k_t, B_K)
            with self._timed("proxy_scores_and_topk"):
                hit_indices = self.compute_hit_indices(
                    query_states=hat_q_t, key_states=self.A_K.narrow(self.dim, 0, self.lite_indices[self.lite_len] - 1))
            with self._timed("hit_miss_bookkeeping"):
                hit_indices, dst_mask = self.only_indices_in_cpu(self.hit_indices, hit_indices)
                self.hit_indices = hit_indices
            bsz, kvheads = key_states.shape[:2]
            with self._timed("key_miss_fetch_cpu_gather_h2d"):
                out = module.take_along_dim_with_mask_python(
                    self.KVcpu.data, hit_indices.view(bsz, kvheads, groups, -1), dst_mask.view(bsz, kvheads, groups, -1))
                dstK = self.Kgpu.narrow(self.dim, 0, self.attn_topk)
                if out.numel() > 0:
                    dstK[dst_mask] = out
            with self._timed("value_miss_gather_gpu"):
                rows, heads, slots = dst_mask.nonzero(as_tuple=True)
                tokens = hit_indices[rows, heads, slots, 0]
                self.Vgpu[rows, heads, slots] = self.value_cache[rows, heads // groups, tokens]
            self.missed_slots = self.missed_slots + dst_mask.sum()  # fetched Keys (host->GPU), accumulated on GPU
            return self.Kgpu, self.Vgpu

    return TPFactory


class LRQKAttention(torch.nn.Module):
    """Attention whose decode support comes from the LRQK factory on local heads."""

    def __init__(self, base, factory_class, *, capacity, decode):
        super().__init__()
        self.base, self.factory_class, self.capacity, self.decode_attention = base, factory_class, capacity, decode
        self.layer_idx, self.batch = base.layer_idx, base.batch
        self.prefill_row, self.length = 0, 0
        self.timers = None  # breakdown mode: synchronized component timers (passed to the factory)
        self.reset()

    def reset(self):
        self.prefill_row, self.length = 0, 0
        self.factory = self.factory_class(batch=self.batch, query_heads=self.base.query_heads,
                                          kv_heads=self.base.kv_heads, value_dim=self.base.value_dim,
                                          capacity=self.capacity, head_start=self.base.kv_start * 4,
                                          device=torch.device("cuda"))
        self.factory._first_layer = self.layer_idx == 0

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        from benchmarks.system.v2q8_tp_baselines import flash_prefill
        from basisserve.kernels.compressed_v_decode_attention import compressed_v_decode_attention_triton

        self.factory.timers = self.timers
        query, _, key, value, _, _ = self.base.projections(hidden_states, position_embeddings)
        if hidden_states.shape[1] > 1:
            assert hidden_states.shape[0] == 1
            self.factory.prefill_row(self.prefill_row, query, key, value)
            with self.factory._timed("prefill_attention"):
                attention = flash_prefill(query, key, value, self.base.scaling)
            self.length = hidden_states.shape[1]
        else:
            key_gpu, value_gpu = self.factory.decode(query, key, value)
            with self.factory._timed("decode_attention"):
                if self.base.value_dim == key_gpu.shape[-1]:
                    # Upstream: flash_attn_with_kvcache over the hit cache (one KV head per query head).
                    attention = self.decode_attention(query, key_gpu, value_gpu, key_gpu.shape[2],
                                                      self.base.scaling)[:, :, None]
                else:
                    attention = compressed_v_decode_attention_triton(query, key_gpu, value_gpu,
                                                                     scale=self.base.scaling)
            self.length += 1
        return self.base.project(attention), None


def install(model, *, batch, capacity, rank, world_size, chunked, value_checkpoint=None):
    """Replace every attention layer; the base TP module supplies local projections and the output collective.
    value_checkpoint: a compressed-Value checkpoint (e.g. V96), or None for the original V128."""
    from basisserve.core.shadowkv_tp8_attention import FlashDecode
    from benchmarks.system.v2q8_basis import KV_HEADS, compressed_value_weights
    from benchmarks.system.v2q8_tp_baselines import TPAttention

    factory_class = make_factory_class()
    decode = FlashDecode(batch, torch.device("cuda"))
    kv_count = KV_HEADS // world_size
    layers = []
    for index, layer in enumerate(model.model.layers):
        weights = None if value_checkpoint is None else compressed_value_weights(
            layer.self_attn, Path(value_checkpoint), index, rank * kv_count, kv_count)
        base = TPAttention(layer.self_attn, value_weights=weights, batch=batch, capacity=capacity, rank=rank,
                           world=world_size)
        replacement = LRQKAttention(base, factory_class, capacity=capacity, decode=decode)
        layer.self_attn = replacement
        layer.mlp = chunked(layer.mlp, chunk_size=1024)
        layer.input_layernorm = chunked(layer.input_layernorm, chunk_size=2048)
        layer.post_attention_layernorm = chunked(layer.post_attention_layernorm, chunk_size=2048)
        layers.append(replacement)
    model.model.norm = chunked(model.model.norm, chunk_size=2048)
    return layers
