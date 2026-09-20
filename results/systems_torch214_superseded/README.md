# BasisKV Section 5 systems results

Status: **PHASE 0 and PHASE 1 complete. PHASE 2-8 not benchmarked.**

Host: 8x NVIDIA L40S, driver 580.178.04, CUDA 13.0, no NVLink. GPU0-3 on NUMA node 0 and
GPU4-7 on NUMA node 1; pairs are `PIX`, every cross-group link is `SYS`. TP8 therefore
crosses the host-mediated path, while the project's earlier TP4 results fit inside one NUMA
node. **TP8 numbers here must not be compared against those TP4 numbers without stating
this topology difference.**

Software: torch 2.14.0+cu130, NCCL 2.30.7, nvcc 12.8, transformers 5.17.0, conda env `basis`.
The recorded historical results used torch 2.6.0+cu124, so this suite is not toolchain-matched
to them. `basisserve/kernels/feature_ragged_allgather.py` was changed from `-std=c++17` to
`-std=c++20` because torch 2.14's ATen requires C++20.

Model: `Qwen/Qwen3-8B-Base` @ `49e3418fbbbc`. C1 checkpoint: `Q3-8B-C1U-R64`
(`c1-uniform`, `uniform_per_layer_per_head`, rank 64 on every layer and head, decoder-refitted
endpoint after encoder sweep 6). Under TP8 the layout is 4 local query heads and
**1 local physical KV head per rank, sharded with replication factor 1**.

## TP8 KV ownership audit

`AttentionTPLayout(4096, 32, 8, tp_size=8)` reports `local_kv_heads=1`,
`kv_partition_mode=sharded`, `kv_replication_factor=1`. The invariant holds and no arm
replicates the full KV cache per rank.

The serving runtime is a separate matter: `basisserve/core/llama_tp4_c1_system.py` and
`basisserve/core/llama_tp4_k_offload.py` hardcode 2 local KV heads, 8 local query heads and
4 AllGather sources, so **the end-to-end serving path is TP4-only today**. PHASE 5 requires
generalizing those two files. The collective microbenchmark in this phase derives its geometry
from the process group and does not share that limitation.

## 1. NCCL baseline at TP8

Both collectives are latency-bound below roughly 32 KB per rank: AllReduce p50 sits at
34-40 µs from 1 KB to 64 KB. Every payload used by the decode arms below falls inside that
flat region, so **byte reduction alone cannot buy latency in this regime.**

| payload B/rank | AllReduce p50 | AllGather p50 | AR eff GB/s | AG eff GB/s |
|---:|---:|---:|---:|---:|
| 1,024 | 38.9 µs | 33.2 µs | 0.05 | 0.22 |
| 2,048 | 34.1 µs | 33.7 µs | 0.11 | 0.43 |
| 4,096 | 35.6 µs | 36.8 µs | 0.20 | 0.78 |
| 8,192 | 38.4 µs | 38.0 µs | 0.37 | 1.51 |
| 16,384 | 38.2 µs | 37.2 µs | 0.75 | 3.08 |
| 32,768 | 37.6 µs | 45.9 µs | 1.53 | 5.00 |
| 65,536 | 40.4 µs | 77.8 µs | 2.84 | 5.90 |
| 131,072 | 49.2 µs | 107.7 µs | 4.66 | 8.52 |
| 262,144 | 81.3 µs | 152.0 µs | 5.64 | 12.07 |
| 524,288 | 145.6 µs | 239.0 µs | 6.30 | 15.36 |
| 1,048,576 | 198.0 µs | 439.0 µs | 9.27 | 16.72 |

## 2. Attention-output block, complete (PHASE 1)

Five arms, Qwen3-8B layers 3/18/33, 50 warmup and 200 timed iterations, 3 repeats, CUDA events
with per-iteration maximum across ranks. Values are medians over layers and repeats.

Wire accounting per token and rank, ring-equivalent:

| arm | collective | width | ring bytes/token/rank |
|---|---|---:|---:|
| `dense_ar` | AllReduce | 4096 | 14336 |
| `lr_ar_wire` | AllReduce | 1024 | **3584** |
| `lr_ar_cap` | AllReduce | 2048 | 7168 |
| `c1_ar` | AllReduce | 2048 | 7168 |
| `basiskv_ag` | AllGather | 256 | **3584** |

### Complete block latency

| batch | Dense-AR | LR-AR wire | LR-AR cap | C1-AR | BasisKV-AG | AG vs Dense | AG vs LR-wire |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74.6 µs | 71.1 µs | 71.5 µs | 86.1 µs | **44.3 µs** | **1.686x** | 1.607x |
| 2 | 77.0 µs | 69.5 µs | 68.9 µs | 94.0 µs | **40.0 µs** | **1.926x** | 1.738x |
| 4 | 77.5 µs | 69.4 µs | 68.0 µs | 93.1 µs | **37.9 µs** | **2.046x** | 1.833x |
| 8 | 76.1 µs | 69.4 µs | 68.3 µs | 89.7 µs | **39.2 µs** | **1.940x** | 1.770x |
| 16 | 77.8 µs | 67.9 µs | 67.8 µs | 88.7 µs | **43.1 µs** | **1.804x** | 1.575x |
| 32 | 90.5 µs | 68.8 µs | 71.0 µs | 89.4 µs | **52.4 µs** | **1.729x** | 1.314x |

### Collective stage in isolation

| batch | Dense-AR | LR-AR wire | LR-AR cap | C1-AR | BasisKV-AG |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.6 µs | 36.5 µs | 36.9 µs | 36.5 µs | 16.9 µs |
| 2 | 36.5 µs | 36.4 µs | 37.4 µs | 36.6 µs | 16.9 µs |
| 4 | 36.1 µs | 36.9 µs | 36.2 µs | 36.6 µs | 17.9 µs |
| 8 | 36.4 µs | 36.5 µs | 36.6 µs | 37.0 µs | 19.5 µs |
| 16 | 49.4 µs | 36.6 µs | 37.1 µs | 36.7 µs | 20.9 µs |
| 32 | 81.7 µs | 36.9 µs | 49.4 µs | 49.5 µs | 29.6 µs |

## What the decomposition shows

1. **The entire gain is in the collective.** At batch 1 the collective drops 36.6 to 16.9 µs
   while projection is unchanged (13.1 vs 12.5 µs) and reconstruction gets *worse*
   (10.0 to 27.3 µs), because C1 pays a replicated `[2048, 4096]` decoder GEMM where dense
   row-parallel needs no reconstruction at all.
2. **Equal ring bytes still favour AllGather.** `lr_ar_wire` and `basiskv_ag` move an identical
   3584 ring bytes per token and rank, yet their collectives are 36.5 vs 16.9 µs. The advantage
   is the collective boundary and the prepared in-place plan, **not** the byte count.
3. **Same function, different boundary.** `c1_ar` and `basiskv_ag` compute a bitwise-identical
   output; only the collective differs. 86.1 vs 44.3 µs at batch 1.
4. Dense AllReduce leaves the latency floor at batch 16 (49.4 µs) and 32 (81.7 µs); BasisKV-AG
   is still at 29.6 µs at batch 32, so the absolute gap widens with rows.

Attribution caveat: `basiskv_ag` uses the prepared uniform in-place NCCL plan
(`gather_inplace_fast`), whereas the section 1 baseline uses `dist.all_gather_into_tensor`.
Part of the 2.16x collective ratio is the prepared plan removing per-call planning, offset
construction and tensor slicing, not the AllGather-versus-AllReduce choice alone. Separating
those two contributions needs a prepared-plan-versus-torch-API control that was not run.

## Correctness gates

Every timed configuration ran these checks first; any failure aborts before timing.

- `c1_ar` and `basiskv_ag` coordinates bitwise equal, and outputs bitwise equal.
- Local BF16 encoding against the same contraction accumulated in FP32 (rtol 0.01, atol 0.003).
- Every AllReduce arm against an explicit `all_gather`-and-sum FP32 reference, bounded by the
  BF16 summation error `2(P-1)u*sum|partial|` with `u = 2**-8`. Worst observed normalized
  deviation 0.24, so the bound is valid and tight to roughly a factor of four.
- No OOM at any batch in 1..32.

## Answers to the section 16 questions

1. **How much TP communication is removed?** Dense row-parallel moves 14336 ring bytes per
   token and rank; BasisKV-AG moves 3584. That is a **75.0% reduction**, exactly the ideal
   4096 to 256 width ratio, verified against the per-row formula in the CSV.
2. **How much faster is the TP8 attention-output path?** **1.69x to 2.05x** on the complete
   block versus Dense-AR, and 1.31x to 1.83x versus a wire-matched shared-basis LR-AllReduce.
   The collective stage alone is 2.16x at batch 1.
3-9. **Not benchmarked.** Questions 3-6 need PHASE 2-5, question 7 needs the routing arms,
   question 8 needs PHASE 7, question 9 needs GH200 hardware. The `ours_b16r16` routing factor
   bank is absent from both this repository and the Hugging Face repository, so arms C, D and E
   (Routing-only, Full BasisKV local-K, Full BasisKV host-K) could not be constructed.

## Claim boundaries respected here

This phase does not claim that communication is the dominant inference bottleneck, that these
ratios survive into end-to-end decode, or that they transfer to another topology. It is a
block-level microbenchmark on one host, and the reconstruction penalty in point 1 above is a
real cost that end-to-end numbers must absorb.

## Files

| file | contents |
|---|---|
| `hardware.json`, `hardware.txt`, `topology.txt` | GPU inventory, NUMA placement, `nvidia-smi topo -m`, PCIe link state |
| `software.json` | versions, toolchain, env vars, source hashes |
| `nccl_baseline.json` | AllReduce/AllGather latency-bandwidth sweep and pairwise send/receive matrix |
| `tp8_collective.csv` | one row per arm, batch, repeat and stage with full reproduction metadata |
| `tp8_collective.json` | same run with per-rank records, correctness audit and timing protocol |
| `manifest.json` | phase coverage and file hashes |
