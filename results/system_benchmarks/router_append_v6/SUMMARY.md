# SM89 register-router and mapped-append benchmark

Single NVIDIA L40S (SM89), `basis` environment, TP1-equivalent B1/GQA4
operator shapes. K128, Dense V128, Page32, B16R16, hard B2048 consisting of
62 routed historical pages and recent 64 tokens. Timings are the median of
three measurements; each measurement uses 20 warmups and 200 iterations.

The baseline and optimized router are compiled from the same source. The
baseline is built with `BASIS_DISABLE_REGISTER_ROUTER=1` and uses the legacy
shared-accumulator implementation. The optimized arm consumes MMA
accumulators from registers. This is not a change from FP32 routing state to
lower precision: both arms retain the same BF16 inputs and rounding points.

## Router

| Context | Legacy ms/layer | Register ms/layer | Reduction | 32-layer arithmetic saving |
|---:|---:|---:|---:|---:|
| 16K | 0.07265 | 0.05838 | 19.64% | 0.457 ms |
| 32K | 0.14012 | 0.10771 | 23.13% | 1.037 ms |
| 64K | 0.28066 | 0.20088 | 28.42% | 2.553 ms |
| 128K | 0.58011 | 0.39210 | 32.41% | 6.016 ms |

Selected page IDs were identical at every context. Maximum page-score
absolute difference was `0.0001111031` through 64K and `0.0001752377` at
128K.

## Full sparse-attention operator

Each sparse measurement includes routing, page selection, support packing,
exact selected QK, softmax and V128 reduction. Local uses GPU-resident exact
K; offload reads exact selected K directly from CUDA-mapped host memory.
The dense rows are full-support controls using the same Page32 attention
kernel over every page; they are not FlashAttention or an external serving
system baseline.

| Context | Dense local | Dense offload | Sparse local legacy → optimized | Sparse offload legacy → optimized | Dense-offload / optimized sparse-offload |
|---:|---:|---:|---:|---:|---:|
| 16K | 0.21696 | 1.26585 | 0.12737 → 0.11163 ms | 0.26336 → 0.24678 ms | 5.13× |
| 32K | 0.61942 | 2.51535 | 0.19660 → 0.16122 ms | 0.33170 → 0.29633 ms | 8.49× |
| 64K | 1.22302 | 5.00817 | 0.33145 → 0.25143 ms | 0.46844 → 0.38751 ms | 12.92× |
| 128K | 2.43038 | 9.99087 | 0.65312 → 0.46292 ms | 0.75951 → 0.57758 ms | 17.30× |

All values are CUDA milliseconds per layer. The optimized sparse-local
pipeline is 12.36%, 18.00%, 24.14% and 29.12% faster than the legacy sparse
pipeline across 16K--128K. The optimized sparse-offload pipeline is 6.29%,
10.67%, 17.28% and 23.95% faster. Matched local and offload outputs were
bitwise identical.

Arithmetic sums over 32 sequential attention layers are:

| Context | Optimized sparse local | Optimized sparse offload | Offload saving from router change |
|---:|---:|---:|---:|
| 16K | 3.57 ms | 7.90 ms | 0.53 ms |
| 32K | 5.16 ms | 9.48 ms | 1.13 ms |
| 64K | 8.05 ms | 12.40 ms | 2.59 ms |
| 128K | 14.81 ms | 18.48 ms | 5.82 ms |

These 32-layer values are arithmetic projections, not measured full-model
latencies. They exclude QKV projections, output projection, MLP, layer norms,
sampling and other model/runtime overhead.

## Decode append

The baseline runs the fused V/Base/Residual append followed by eight small
D2H exact-K copies. The optimized kernel writes exact K to mapped host memory
in the same append launch.

| Baseline | Optimized | Reduction | Speedup | 32-layer arithmetic saving |
|---:|---:|---:|---:|---:|
| 18.53 us/layer | 6.05 us/layer | 67.35% | 3.06× | 0.399 ms |

The mapped-host K result was bitwise exact. Combining the measured
sparse-offload and append savings gives arithmetic savings of approximately
0.93, 1.53, 2.99 and 6.22 ms per 32-layer token at 16K, 32K, 64K and 128K.
These components can interact in a full runtime and therefore are not an
end-to-end speedup claim.

## Command

```bash
PATH=/workspace/miniforge3/envs/basis/bin:$PATH \
CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 MAX_JOBS=4 \
/workspace/miniforge3/envs/basis/bin/python \
  -m benchmarks.system.bench_router_append_v6 \
  --lengths 16384 32768 65536 131072 \
  --warmup 20 --iterations 200 \
  --output results/system_benchmarks/router_append_v6
```

Raw results are in `benchmark.json`; execution messages are in `run.log`.
The older full-model runner could not be reproduced on this machine because
its pinned ShadowKV checkout and FlashAttention dependency are absent. No
full-model latency is reported from this run.
