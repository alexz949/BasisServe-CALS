# Qwen3-8B-Base TP8 NUQ4 Optimization

Status: completed; measured implementation frozen in `staged/source/`.
Environment: `basis`, 8 x L40S (SM89).

## Fixed Protocol

- B128 request cohort, 4096 input + 128 output tokens (127 decode forwards).
- Scheduler budget 8192, maximum sequences 256, GPU memory utilization 0.8.
- One warmup followed by three measured requests, serial TP8 execution.
- Same adaptive R64/R96 factors, frozen NUQ4 codebooks and A8 scales.
- Actual packed K/V with BF16 outliers, BF16 encoder, A8 transport, W8A8 decoder.
- No Dense rerun, SHA256, MLP changes, sparse routing, or quantizer refitting.
- Tests and full-length preflight precede each accepted formal candidate.

## Full Request Results

| Implementation | Rank | Request seconds (three repeats) | Median s | Historical Dense / median |
|---|---:|---|---:|---:|
| Historical Dense | - | See original JSON | 41.128 | 1.000x |
| Original NUQ4 | 64 | See original JSON | 40.586 | 1.013x |
| Original NUQ4 | 96 | See original JSON | 42.978 | 0.957x |
| Larger tiles + split-K | 64 | 32.403, 32.425, 32.446 | 32.425 | 1.268x |
| Mixed-phase separation + smaller decode tiles | 64 | 31.224, 31.252, 31.275 | 31.252 | 1.316x |
| Bounded prefill staging + packed split-K decode | 64 | 30.408, 30.423, 30.445 | 30.423 | 1.352x |
| Bounded prefill staging + packed split-K decode | 96 | 32.592, 32.610, 32.649 | 32.610 | 1.261x |

The tile candidate's R64 preflight was 32.392 s. All accepted new trials
have zero preemptions and no outlier overflow on any rank. Candidate regression:
53 passed, 14 upstream deprecation warnings. The mixed-phase candidate passed
57 tests and its preflight was 31.221 s. R96 was not tested with these
intermediate candidates; its final staged result is reported separately above.

Dense is reused from [the original result](../vllm_tp8/dense.json), not a
contemporaneous paired run. The earlier BF16 C1 reference was approximately
30.351 s (1.355x), but used a different, uniform R64 checkpoint. It is a
performance-level target, not a matched quantization ablation.

The optimized R64 result is within 0.24% of that BF16 reference's request
latency, recovering the requested performance level. R96 reaches 1.261x,
not 1.35x. These conclusions apply to the measured B128 configuration, not
the whole batch sweep, every model, or a claim of globally optimal kernels.

## Changes and Diagnostics

The accepted tile candidate preserves the cache implementation byte-for-byte
relative to the original frozen serving source. Only attention scheduling changes:
prefill BM128, BN128 for V widths up to 64 and BN64 otherwise, eight warps,
one pipeline stage. Decode keeps BN32/four warps, uses one pipeline stage,
and targets 512 sequence/split blocks (at most 32 splits per sequence).

Original rank-zero full-request profile: packed prefill/mixed attention 8.296 s,
packed pure decode 4.800 s, V statistics 0.383 s. These are summed kernel
durations in a separate profiled request, not an additive wall-time breakdown.
The separate non-profiled baseline request was 40.502 s.

Two alternative outlier-index implementations were tested and rejected:

- Byte popcount plus gather: numerically correct after small-width compiler
  padding, but increased shared-memory/layout costs and was slower.
- Direct word popcount: numerically correct with one pipeline stage, but no
  consistent speed benefit. Default three-stage attention exceeded shared
  memory on some widths. The failure logs and source snapshots are retained.

Neither rejected bitmap implementation is used in the tile formal results.
The new exhaustive bitmap-pattern regression remains in the test suite.

The mixed-phase candidate dispatches single-token sequences to split-K decode
even when they share a scheduled batch with prefill. GPU query-length predicates
preserve interleaved sequence order without GPU-to-CPU readback. Prefill uses
BM256/BN32/16 warps; decode uses BN16/eight warps and the same split budget.
Its separate rank-zero profile reduced summed attention kernel time from
13.096 s to 3.852 s. AllReduce remained approximately 15.44 s and AllGather
approximately 3.51 s; these sums are not an additive request breakdown.

Additional rejected probes are retained: embedding bitmap prefix counts in
metadata did not improve on the unchanged cache layout, and a four-head SIMT
decode kernel was slower than tensor-core decode. Neither is in serving code.

## Bounded Prefill Staging Candidate

R64 preflight was 30.361 s; R96 preflight was 32.715 s. Both passed, followed
by three formal repeats each. R64 formal median is 30.423 s / 538.54 output
tokens per second including prefill, a 25.04% latency reduction versus original
NUQ4. R96 is 32.610 s / 502.42 output tokens per second, a 24.12% reduction.
These are whole-cohort completion times, not single-request isolated latency.
All 128 requests in each measured cohort produced 128 tokens; all trials had
zero preemptions and all eight ranks reported no outlier overflow.

Persistent KV remains packed NUQ4 plus BF16 outliers. Only prefill reconstructs
quantized pages into a bounded, layer-reused BF16 workspace and runs the existing
DiffKV prefill kernel. Dequantization, RoPE, query copies and output scatter are
included in timing. Decode continues to read packed pages directly. The
workspace is 20 MiB per GPU at this scheduler budget, allocated before cache
profiling and recorded in worker statistics. It is not a full BF16 cache.

Scheduling uses existing CPU query offsets and upper length bounds; actual GPU
lengths control attention. Multiple prefill sequences are chunked to respect
workspace capacity. Regression covers adaptive widths 32 through 128, shuffled
physical pages, interleaved decode/empty sequences, history prefixes, upper
length bounds and CUDA Graph replay: 64 passed, 14 upstream warnings.

Standalone complete attention microbenchmarks (not full-model speedups):

| Width | Shape | Direct packed ms | Staged ms |
|---|---|---:|---:|
| 64 | Prefill | 0.689 | 0.259 |
| 64 | Mixed | 1.196 | 0.698 |
| 96 | Prefill | 0.668 | 0.270 |
| 96 | Mixed | 1.138 | 0.704 |

Maximum absolute output difference versus the packed reference was at most
0.00390625 in these fixtures. This is numerical kernel validation, not PPL or
generated-token equivalence.

Final read-only audit checked all six formal trial records and all 22 archived
source files against the working implementation by direct byte comparison,
without hashing. The cache encoder is also byte-identical to the original
NUQ4 benchmark source. No Dense rerun, quantization refit, MLP edit, or PPL
rerun was performed. JIT warnings occurred during warmup; upstream deprecation
warnings and worker shutdown messages are retained in the logs.

## Artifacts and Commands

- [Original NUQ4 summary](../q3-nuq4-vllm/formal/SUMMARY.md)
- [Original profile](base/preflight/r64/profiles/b128.kernels.csv)
- [Tile R64 preflight](tiles/preflight/r64/graph_splitk_4096.json)
- [Tile R64 formal](tiles/formal/r64/graph_splitk_4096.json)
- [Frozen candidate source](tiles/source/)
- [Regression log](tiles/tests.log)
- [Mixed R64 formal](mixed/formal/r64/graph_splitk_4096.json)
- [Mixed profile](mixed/preflight/r64/profiles/b128.kernels.csv)
- [Mixed frozen source](mixed/source/)
- [Staging microbenchmark](staging.json)
- [Staging regression log](staged/tests.log)
- [Staging frozen source](staged/source/)
- [Staged R64 preflight](staged/preflight/r64/graph_splitk_4096.json)
- [Staged R64 formal](staged/formal/r64/graph_splitk_4096.json)
- [Staged R96 preflight](staged/preflight/r96/graph_splitk_4096.json)
- [Staged R96 formal](staged/formal/r96/graph_splitk_4096.json)
- [Final formal summary, including TTFT and TPOT](staged/formal/SUMMARY.md)
- [Final formal CSV](staged/formal/summary.csv)
- [Final audit](staged/audit.log)
- Kernel sweeps: `tiles.json`, `decode_tiles.json`, `bitmap_tiles.json`,
  `word_tiles.json`; each has logs and append-only JSONL measurements.

Formal command (R96 substitutes `--rank 96`; same protocol for later candidates):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/benchmark_vllm_qwen3_nuq4.py --phase formal --rank 64 --batch-sizes 128 --prefill-tokens 4096 --output results/q3-nuq4-opt/staged > results/q3-nuq4-opt/staged/formal_r64.log 2>&1
```

Preflight uses the same command with `--phase preflight` and a separate log.
Earlier candidates use their own `tiles` or `mixed` output directory instead.

Regression command:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q tests/test_nuq4_attention.py tests/test_nuq4_prefill.py tests/test_nuq4_cache.py tests/test_vllm_nuq4.py tests/test_qwen3_nuq4_artifacts.py tests/test_qwen3_nuq4_summary.py
```

Summary command:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/summarize_qwen3_nuq4.py --root results/q3-nuq4-opt/staged/formal --dense results/vllm_tp8/dense.json
```

No GitHub or Hugging Face upload has been performed for this optimization run.
