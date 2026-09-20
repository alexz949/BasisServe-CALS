# Qwen3-32B TP8 C1 serving

> Benchmark status: the E2E numbers linked below were produced by the original
> unified Triton DiffKV kernel. They remain valid historical measurements, but
> are not a fully optimized C1 result. The current tree contains a separate
> SM89 prefill specialization; its matched dense/C1 E2E rerun is pending.

`BasisServeQwen3_32BTP8C1ForCausalLM` reuses the prepared output boundary and
paged K128/V64 attention of the [8B path](qwen3_8b_vllm_tp8_c1.md), with
validated Qwen3-32B geometry: hidden size 5120, 64 query heads, eight physical
KV heads, head dimension 128, and 64 layers. The older TP4 path is unchanged.

Each TP8 rank owns eight query heads and one KV head. The output boundary
packs 512 local coordinates per token, AllGathers 4096 global coordinates,
then executes one BF16 GEMM with a `[4096, 5120]` decoder. It does not split
the decoder into source-local GEMMs. Dense local O projection is
`[1024, 5120]`; C1 therefore has four times the decoder matrix elements per
rank. This additional work must be weighed against communication savings.

The authenticated factor bank is `qwen3-32b/c1/factor-banks/R64-S6`. Its
historical format label contains `v96`, but the manifest and every loaded
tensor are validated as rank 64. Manifest SHA256:
`26bafc0674362208b05d56b16f8b5a74a22756738cf355af39bf80b859195b3f`.

## Validation

Environment: `basis`, PyTorch `2.13.0+cu130`, vLLM `0.29.0`, eight NVIDIA
L40S GPUs connected over PCIe. No Slurm is available.

The eager and CUDA Graph engine smokes passed on 2026-09-20 with identical
greedy output tokens: request batches 1 and 3, up to 512 prompt tokens,
eight output tokens, a 256-token scheduler budget, and different request
lengths. All eight workers loaded all 64 folded V layers; graph mode
recorded 384 boundary capture calls per worker, eager mode zero. These
checks validate execution consistency, not quality relative to dense.
Commands and worker statistics are retained in `/tmp/c1_32b/eager.json`
and `/tmp/c1_32b/graph.json`; matching `.log` files retain runtime warnings.

After the paired benchmark, the 8B CUDA Graph regression smoke also passed
with output tokens identical to the pre-extension reference. All eight
workers reported 36 loaded layers and 216 capture calls. The exact command
and statistics are in `/tmp/c1_32b/8b_regression.json`, with its matching log.

CPU validation command (five tests passed):

```bash
OMP_NUM_THREADS=1 /workspace/miniforge3/envs/basis/bin/python -m pytest -q \
  tests/test_qwen3_8b_vllm_c1.py tests/test_vllm_fixed_cohort_metrics.py
```

## Benchmark protocol and interpretation

The shared driver is `evaluation/benchmark_vllm_qwen3_8b_c1.py`; it selects
the architecture from the authenticated factor geometry. The paired sweep
uses cohorts 1/2/4/8/16/32/64/128/256, 4096 prompt tokens, exactly 128 output
tokens, one warmup and three measured runs. Both arms use memory utilization
0.8, an 8192-token scheduler budget, synchronous scheduling, no prefix
caching, compilation mode NONE, and FULL_DECODE_ONLY CUDA Graphs.

Batch means requests submitted together, not a fixed execution batch.
Continuous batching and chunked prefill remain active. Per-request TPOT
includes scheduling interleaving; E2E throughput includes prefill.
Separate rank-zero profiles cover batches 1/32/256 after timed measurements.
Profiled kernel-duration sums are not wall time or an isolated measurement
of network transfer.

At startup, dense reported 875,584 KV token slots (207.29 complete
4224-token sequences), versus C1's 1,089,440 slots (257.92 sequences).
The C1 decoder consumes more weight memory, but K128/V64 requires 25% fewer
KV bytes per cached token than K128/V128. Batch 256 must therefore be
interpreted as a capacity-sensitive comparison, not pure kernel speedup.

See [the paired report](../results/vllm_32b_tp8/summary.md) for exact commands,
all measured medians, preemption counts, profiles, and caveats. Raw JSON,
CSV, traces, and logs are in `results/vllm_32b_tp8/`. Neither these timing
tests nor the graph smoke constitute a model-quality evaluation.

The full sweep completed on 2026-09-20: both arms exited zero, all 54
measured runs and six profiles passed validation. At batch 128 the E2E
median falls from 102.756 to 85.179 seconds (1.206x), with no preemptions.
At batch 256 it falls from 215.228 to 166.899 seconds (1.290x); dense has
one preemption in each measured run, C1 zero. Batch 1 and 2 E2E regress;
batch 1/2/4 TPOT regress. The runs emit non-fatal backend-availability,
warmup JIT, trace-export wait, and process-cleanup warnings retained in logs.
GPU allocations were released after shutdown.

## Why the gain differs from 8B

The per-rank C1 source width doubles from 256 to 512, while the dense
AllReduce vector grows only from 4096 to 5120. The decoder matrix grows
2.5 times per layer, and the unchanged MLP matrix products grow about
2.60 times per layer. Layer count alone does not explain the speedup ratio.

Separate batch-32 profiles give the following summed rank-zero GPU kernel
seconds (not E2E seconds or isolated transfer time):

| Model / arm | AllReduce | AllGather | GEMM/GEMV | Attention | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8B dense | 7.876 | 0.066 | 1.497 | 0.606 | 0.513 |
| 8B C1 | 4.015 | 1.114 | 1.814 | 0.602 | 0.544 |
| 32B dense | 17.501 | 0.066 | 6.421 | 1.329 | 1.315 |
| 32B C1 | 8.923 | 3.673 | 7.907 | 1.432 | 1.350 |

These profiles support a smaller relative communication saving and a larger
relative added GEMM cost on 32B. NCCL durations can include waiting, and
these categories do not constitute a controlled causal ablation.

For 32B batch-one decode graph replays, mean per-layer principal O/decoder
kernel time rises from 16.72 to 64.78 microseconds, while output collective
time falls from 37.26 to 13.43 microseconds. Attention rises from 13.44 to
19.15 microseconds; the C1 pre-AllGather copy takes 0.91 microseconds.
This identifies small-batch decoder execution as a higher optimization
priority than source-copy fusion, without claiming a measured bandwidth
bottleneck or changing the one-GEMM design.
