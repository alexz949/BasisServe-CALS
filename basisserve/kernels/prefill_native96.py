"""Experimental Hopper WGMMA paged attention; not wired into serving.

The 128-wide option is an experimental control with exactly the same algorithm.
Both read V96; only the PV accumulator and instruction width differ.
"""

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import TensorCoreIntrinEmitter
from tilelang.layout import make_full_bank_swizzled_layout, make_half_bank_swizzled_layout


class FullWidthBF16Emitter(TensorCoreIntrinEmitter):
    """Select one N96/N128 atom instead of TileLang's gcd(N, 256) rule."""

    def _initialize_wgmma_prefix(self, n_dim=16):
        assert self.warp_col_tiles in (96, 128)
        assert str(self.a_dtype) == str(self.b_dtype) == "bfloat16"
        self.wgmma_inst_m = 64
        self.wgmma_inst_n = self.warp_col_tiles
        self.wgmma_prefix = f"m64n{self.wgmma_inst_n}k16"


@tilelang.jit(target={"kind": "cuda", "arch": "sm_90a"})
def paged_prefill(batch, tokens, pages, table_width, heads, kv_heads,
                  q_stride, k_strides, v_strides, max_query, width=96):
    group = heads // kv_heads
    block_m, block_n, page = 64, 128, 16
    block_q = block_m // group
    scale = 128 ** -0.5 * 1.4426950408889634
    pv = FullWidthBF16Emitter(
        a_dtype="bfloat16", b_dtype="bfloat16", accum_dtype="float32",
        block_row_warps=4, block_col_warps=1, warp_row_tiles=16,
        warp_col_tiles=width, chunk=block_n)

    def pv_layouts(probs, values, accumulator):
        swizzle = (make_half_bank_swizzled_layout(values) if width == 96
                   else make_full_bank_swizzled_layout(values))
        pv._assign_b_shared_layout(swizzle)
        return {probs: pv.make_mma_load_layout(probs, matrix="A"),
                values: swizzle, accumulator: pv.make_mma_store_layout(accumulator)}

    @T.prim_func
    def main(
        Q: T.StridedTensor((tokens, heads, 128), (q_stride, 128, 1), "bfloat16"),
        K: T.StridedTensor((pages, page, kv_heads, 128), k_strides, "bfloat16"),
        V: T.StridedTensor((pages, page, kv_heads, 96), v_strides, "bfloat16"),
        O: T.Tensor((tokens, heads, 96), "bfloat16"),
        CU: T.Tensor((batch + 1,), "int32"),
        LEN: T.Tensor((batch,), "int32"),
        TABLE: T.Tensor((batch, table_width), "int32"),
    ):
        with T.Kernel(T.ceildiv(max_query, block_q), kv_heads, batch,
                      threads=128) as (bx, head, seq):
            qs = T.alloc_shared((block_m, 128), "bfloat16")
            ks = T.alloc_shared((block_n, 128), "bfloat16")
            vs = T.alloc_shared((block_n, width), "bfloat16")
            scores = T.alloc_fragment((block_m, block_n), "float32")
            probs = T.alloc_fragment((block_m, block_n), "bfloat16")
            acc = T.alloc_fragment((block_m, width), "float32")
            T.annotate_layout(pv_layouts(probs, vs, acc))
            maximum = T.alloc_fragment((block_m,), "float32")
            previous = T.alloc_fragment((block_m,), "float32")
            alpha = T.alloc_fragment((block_m,), "float32")
            denom = T.alloc_fragment((block_m,), "float32")
            rowsum = T.alloc_fragment((block_m,), "float32")
            start = CU[seq]
            qlen = CU[seq + 1] - start
            context = LEN[seq] - qlen
            prefix = T.min(LEN[seq], context + (bx + 1) * block_q)
            if bx * block_q < qlen:
                for i, d in T.Parallel(block_m, 128):
                    qs[i, d] = T.if_then_else(bx * block_q + i // group < qlen,
                        Q[start + bx * block_q + i // group, head * group + i % group, d], 0)
                T.fill(acc, 0)
                T.fill(denom, 0)
                T.fill(maximum, -T.infinity("float32"))
                for tile in T.Pipelined(T.ceildiv(prefix, block_n), num_stages=2):
                    for j, d in T.Parallel(block_n, 128):
                        pos = tile * block_n + j
                        if pos < prefix:
                            ks[j, d] = K[TABLE[seq, pos // page], pos % page, head, d]
                        else:
                            ks[j, d] = 0
                    for j, d in T.Parallel(block_n, width):
                        pos = tile * block_n + j
                        if pos < prefix and d < 96:
                            vs[j, d] = V[TABLE[seq, pos // page], pos % page, head, d]
                        else:
                            vs[j, d] = 0
                    T.wgmma_gemm(qs, ks, scores, transpose_B=True,
                                 clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(0)
                    for i, j in T.Parallel(block_m, block_n):
                        scores[i, j] = T.if_then_else(
                            tile * block_n + j < prefix and
                            tile * block_n + j <= context + bx * block_q + i // group,
                            scores[i, j] * scale, -T.infinity("float32"))
                    T.copy(maximum, previous)
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        alpha[i] = T.exp2(previous[i] - maximum[i])
                    for i, j in T.Parallel(block_m, block_n):
                        scores[i, j] = T.exp2(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, rowsum, dim=1)
                    for i in T.Parallel(block_m):
                        denom[i] = denom[i] * alpha[i] + rowsum[i]
                    for i, d in T.Parallel(block_m, width):
                        acc[i, d] *= alpha[i]
                    T.copy(scores, probs)
                    pv.wgmma(probs[:, :], vs[:, :], acc[:, :])
                for i, d in T.Parallel(block_m, width):
                    if bx * block_q + i // group < qlen and d < 96:
                        O[start + bx * block_q + i // group, head * group + i % group, d] = acc[i, d] / denom[i]

    return main


def compile_prefill(data, max_query, width):
    q, k, v = (data[key] for key in ("q", "k", "v"))
    assert width in (96, 128) and v.shape[-1] == 96
    assert q.shape[1] // k.shape[2] in (4, 8)
    return paged_prefill(len(data["seqused_k"]), q.shape[0], k.shape[0],
                         data["block_table"].shape[1], q.shape[1], k.shape[2],
                         q.stride(0), k.stride(), v.stride(), max_query, width)


def arguments(data):
    return [data[key] for key in ("q", "k", "v", "out", "cu_seqlens_q",
                                  "seqused_k", "block_table")]
