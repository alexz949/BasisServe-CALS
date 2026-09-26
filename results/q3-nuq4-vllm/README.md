# Qwen3-8B-Base TP8 NUQ4 Serving

## Scope

Same adaptive R64/R96 checkpoints, NUQ4 codebooks and static decoder input
scales as the frozen Qwen PPL runs. Full-context attention, BF16 folded
encoder, actual packed NUQ4 K/V with BF16 outliers, A8 E4M3 transport and
W8A8 decoder GEMM. No sparse routing, MLP changes, new fitting or SHA256.

K is quantized after k_norm and before RoPE, then restored and rotated inside
attention. V dynamic bounds are computed across all active KV heads with a
real BF16 AllGather; its cost is included, not replaced with per-shard bounds.
Attention-output latent uses a second, uint8 AllGather. Receive transpose keeps
FP8 codes and applies scales through the FP8 GEMM, without a BF16 latent arena.

## Formal Results

Completed on 2026-09-25: **54/54 new measured trials**, three repeats at each
of nine batches for both R64 and R96. All trials passed the eight-rank layer
and exception checks, with zero recorded preemptions and no OOM. Dense was
not rerun; its earlier matched 27-trial reference is reused as requested.

Full results, TTFT, TPOT, throughput, commands and raw-data links:
[formal/SUMMARY.md](formal/SUMMARY.md), [formal/summary.csv](formal/summary.csv).

| Batch | Dense request s | R64 request s | R64 speedup | R96 request s | R96 speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.106 | 1.406 | 0.787x | 1.436 | 0.770x |
| 128 | 41.128 | 40.586 | 1.013x | 42.978 | 0.957x |
| 256 | 83.668 | 79.629 | 1.051x | 84.922 | 0.985x |

These are three-repeat medians of full request-cohort wall time. At B256,
output throughput is 391.64 / 411.51 / 385.86 tokens/s for Dense / R64 / R96.
R64 is slower at B1-B32, approximately tied at B64, and modestly faster at
B128-B256. R96 is slower at every tested point. This implementation does not
establish a broad speed advantage over Dense or an optimal NUQ4 kernel.
Small differences must also be read as cross-date comparisons to historical
Dense, not contemporaneous paired measurements.

Both benchmark processes exited successfully and released all GPUs. Serving
kernels, model adapter, boundary and runner match the frozen `formal/source`
copies by direct byte comparison, without checksums. Regression coverage is
42 kernel/integration tests plus five summary tests. No new full-model PPL
was run for this serving backend.

Expected warnings: SM89/eight-PCIe-GPU custom AllReduce variants are unavailable,
so the engine uses PyNccl; Inductor is disabled by the benchmark protocol.
The logged `_v_stats`, `_pack` and `_unpack` JIT warnings occur inside the
first warmup, before measured repeats. The earlier interrupted preflight is
retained in its append-only logs and is excluded from formal results.

## Completed Smoke Checks

Environment: `basis`, 8 x L40S. These short checks preceded the formal grid.

| Path | Prompt | Outputs | Batches | Result |
|---|---:|---:|---|---|
| R64 eager, packed cache | 128 | 4 | 1, 2 | passed |
| R64 graph, unified attention | 4096 | 4 | 1, 2 | passed |
| R96 graph, unified attention | 4096 | 4 | 1, 2 | passed |
| R64 graph, split-K decode | 4096 | 4 | 1, 2 | passed |
| R96 graph, split-K decode | 4096 | 4 | 1, 2 | passed |
| Dense native FlashAttention | 4096 | 4 | 1, 2 | passed |

Every successful NUQ4 run reports all 36 loaded value projections and no
exception overflow on any of the eight ranks. Graph model smokes report
144 capture-time boundary calls per rank. The local suite passes 41 tests;
the distributed boundary independently passes 36 changing-input cases with
exact agreement for global statistics and A8/W8 decoder outputs.

These short checks use a **20% GPU memory budget**, one warmup and one measured
repeat. Llama calibration runs concurrently. Timing JSON is retained, but
these are not publication speedups. Raw files distinguish `graph_4096.json`
(initial unified kernel) from `graph_splitk_4096.json` (current decode path).
The one-SM small-batch decode bottleneck is removed by split-K; this does not
establish that the new backend is optimal, particularly its prefill path.

## Exception Capacity

The initial eight-exception-per-token page reserve overflowed during R64 B1
warmup. That run was rejected, not silently accepted; the failure remains in
`smoke_r64.log` and the append-only journal. Serving now uniformly reserves
16 BF16 exceptions/token for each K and V page, pooled across 16 tokens.
The quantizer and outlier inclusion rules are unchanged. Future overflows
still invalidate a trial; there is no clipping or hidden dense fallback.

Including bitmap, offsets, scale fields, page header and reserved exceptions:

| Local V width | Packed K+V bytes/token/layer/rank | BF16 K+V | Ratio |
|---:|---:|---:|---:|
| 64 | 210 | 384 | 1.83x |
| 96 | 230 | 448 | 1.95x |

These are storage-layout calculations, excluding model weights and temporary
workspaces. Actual adaptive widths vary by layer. vLLM allocation statistics
record both logical page bytes and physical page strides; the completed
smokes used exact layer-specific strides without hidden page padding.
vLLM's public cache setting remains `auto`, but this backend supplies a
uint8 packed cache specification; it is not BF16 cache simulation.

At the formal 80% memory budget, the engine logs report the following logical
KV pool capacities. Dense is the reused 2026-09-20 run; NUQ4 values come from
this run's initialization logs. These are allocated token capacities, not a
measured largest successful batch or guaranteed serving concurrency.

| Path | Engine-reported KV pool tokens | Source log |
|---|---:|---|
| Dense | 1,915,328 | [dense.log](../vllm_tp8/dense.log) |
| NUQ4 R64 | 4,600,976 | [formal_r64.log](formal_r64.log) |
| NUQ4 R96 | 4,186,704 | [formal_r96.log](formal_r96.log) |

## Commands and Logs

Working directory: `/workspace/BasisServe-CALS`. Slurm is unconfigured, so
smoke jobs ran directly. Source snapshots are in `smoke/source_unified` and
`smoke/source_splitk`. Kernel test logs are in `../q3-nuq4-cache/`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/benchmark_vllm_qwen3_nuq4.py --phase smoke --rank 64 --prefill-tokens 4096 --output results/q3-nuq4-vllm >> results/q3-nuq4-vllm/smoke_r64.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/benchmark_vllm_qwen3_nuq4.py --phase smoke --rank 96 --prefill-tokens 4096 --output results/q3-nuq4-vllm >> results/q3-nuq4-vllm/smoke_r96.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/benchmark_vllm_qwen3_nuq4.py --phase smoke --arm dense --prefill-tokens 4096 --output results/q3-nuq4-vllm >> results/q3-nuq4-vllm/smoke_dense.log 2>&1
```

## Completed Formal Grid

The user approved the formal grid, then explicitly cancelled rerunning Dense.
R64 and R96 completed with `--phase formal` and separate
`formal_{r64,r96}.log` files after both long-output checks passed. Calibration
jobs finished before timing. Formal configuration: 80% memory budget, 4096 prompt
tokens, 128 output tokens (127 decode forwards), scheduler budget 8192,
batches 1/2/4/8/16/32/64/128/256, one warmup and three measured repeats per point.
That is 54 new measured trials plus 18 warmups across R64/R96.

Executed formal commands (`basis`, direct local execution):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/benchmark_vllm_qwen3_nuq4.py --phase formal --prefill-tokens 4096 --output results/q3-nuq4-vllm --rank 64 > results/q3-nuq4-vllm/formal_r64.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/benchmark_vllm_qwen3_nuq4.py --phase formal --prefill-tokens 4096 --output results/q3-nuq4-vllm --rank 96 > results/q3-nuq4-vllm/formal_r96.log 2>&1
```

### Reused Dense Reference

The existing complete [Dense JSON](../vllm_tp8/dense.json) has all nine batches
and three measured repeats, dated 2026-09-20. It uses the same Qwen3-8B-Base
snapshot, eight L40S GPUs, BF16, torch 2.13.0+cu130, vLLM 0.29.0, 4096 input
and 128 output tokens, 8192 scheduling budget, 80% memory budget, paused
cohort admission and full decode CUDA Graphs. Its [log](../vllm_tp8/dense.log)
identifies FlashAttention 2. It is reused in place, not copied or modified.
The old file does not record preemption counts or the resolved default cache
block size; missing preemptions are reported as unrecorded, never zero.
This is a cross-run comparison, not a claim of simultaneous measurement.

### Long-Output Checks

`--phase preflight` checks B1 and B256 with 128 output tokens and the formal
80% memory and graph settings, one warmup plus one measured repeat each.
Both R64 and R96 passed; see [preflight/SUMMARY.md](preflight/SUMMARY.md).
Logs are `preflight_r64.log` and `preflight_r96.log`; source snapshot is
`preflight/source`. No Dense preflight will run following the user's update.

The first R64 B256 warmup was deliberately interrupted after finding that
the prefill sequence count was a Triton compile-time constant. Continuous
batching changed this value repeatedly, causing recompilation stalls.
Making it a non-specialized runtime parameter preserves full-context
attention and quantization, and passed 47 tests (14 upstream deprecation
warnings). The append-only log/journal retains the interrupted attempt.
Only completed JSON files, not partial journal trials, enter the summary.

The revised summary reads the existing Dense file explicitly:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/summarize_qwen3_nuq4.py --root results/q3-nuq4-vllm/formal --dense results/vllm_tp8/dense.json
```

Wall time and aggregate output throughput include prefill. TTFT is from
queued time and includes paused admission; TPOT spans first to last token
divided by 127 and can include scheduling interleaving. Do not label it an
isolated attention-kernel latency. The formal comparison is in the linked summary.

Batch denotes the admitted request cohort, not a constant active decode batch.
This grid measures the combined C1 + NUQ4 cache + A8/W8 decoder path against
Dense, not the isolated contribution of A8 or cache quantization. No matched
BF16 C1 ablation is rerun here.

Nothing from this integration has been committed or uploaded.
