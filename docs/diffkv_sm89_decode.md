# SM89 TP8 decode attention

The C1 vLLM path has dedicated Triton decode specializations for four or eight
local query heads, one local KV head, Q/K width 128, and V width 64 or 96.
`NUM_HEADS` and `VALUE_DIM` are compile-time constants, so H8 is a distinct
SM89 specialization rather than an upstream fallback.

## Launch policy

The attention kernel launches one program per sequence and KV segment. The
default policy uses rank- and head-aware configurations from isolated L40S
sweeps. V96 switches to non-split above batch 32. H8/V64 retains eight segments
at batch 64 and switches to non-split above it:

| Geometry | B1--8 | B32 | B64 | B128+ |
| --- | ---: | ---: | ---: | ---: |
| H4/V96 | 16 | 8 | 1 | 1 |
| H8/V64 | 16 | 16 | 8 | 1 |
| H8/V96 | 16 | 16 | 1 | 1 |

A one-segment launch writes the final result directly and therefore has no
segment-reduction launch. Multi-segment launches use a head-count-specialized
reduction grid.

Times include segment reduction when present, but exclude KV-cache update,
AllGather, and the decoder GEMM. Selected-kernel speedups versus upstream at
context 4096 are:

| Geometry | B1 | B8 | B32 | B64 | B128 | B256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H4/V96 | 1.90x | 2.00x | 1.46x | 1.02x | 1.05x | 1.01x |
| H8/V64 | 2.05x | 1.97x | 1.69x | 1.10x | 1.05x | 1.02x |
| H8/V96 | 1.78x | 1.84x | 1.44x | 1.03x | 1.07x | 1.01x |

## AllGather output layout

For four- and eight-head decode, the attention output is a strided view of the prepared
feature-major NCCL source slot. Both the main kernel and segment reduction can
write that layout directly. This removes the previous token-major-to-feature-
major `copy_` before AllGather. Prefill retains the token-major output contract.

KV-cache update remains a separate kernel. It is sequenced in the same custom
operator as attention and the C1 output boundary, but is not fused into the
attention kernel.

## Validation status

GPU tests cover page sizes 16 and 32, empty sequences, segment counts 1 through
16, non-contiguous feature-major output, and CUDA Graph replay with changing
queries and values for H4/H8 and V64/V96. The isolated timing sweep is in
`evaluation/benchmark_diffkv_decode.py`; the raw full sweeps and selector smokes
are under `results/kernel_tuning_sm89/`. The complete kernel suite reports 76
passing tests. A full TP8 serving benchmark is still required to measure the
end-to-end layer and TPOT effect.
