# L40S DiffKV prefill specialization

The C1 serving path now uses a dedicated causal paged-prefill Triton kernel for
QK128 with V64 or V96, one local KV head and four/eight local query heads. The implementation
is adapted from vLLM's DiffKV attention algorithm; this is not a port of FA3 or a
claim of globally optimal performance. It retains FP32 online-softmax state and
accumulators, BF16 tensor-core products, the packed `128 + V-rank` KV layout, and
the existing single output-decoder GEMM.

`basisserve/vllm/diffkv_attention.py` dispatches batches with `max_query_len > 1`
to the new 2D kernel, including mixed prefill/decode batches. Pure decode uses
the dedicated SM89 split-KV path for four or eight local query heads. No installed
vLLM files are modified; backend identity and cache metadata remain compatible
with its DiffKV cache specification.

Production dispatch uses the complete independent-kernel sweep rather than one
global tile. Short/ragged prefill (`max_query_len <= 128`) uses `BLOCK_M=32` and
KV tile 128. For 8B TP8 (four local query heads), single-sequence long prefill
uses `[128,64,4 warps,3 stages]` and multi-sequence work uses `[64,64,4,3]`.
For 32B TP8 (eight local query heads), V64 long prefill uses `[128,128,8,3]`.
V96 uses `[32,128,8,2]` for short/ragged work, `[128,128,8,2]` for ordinary
long or multi-sequence work, and `[128,32,4,3]` for a single sequence whose KV
length is at least 16K. These choices are shared by the four- and eight-head
V96 specializations.
The dispatch inputs are Python metadata already available before launch, so it
does not read GPU tensors or synchronize during CUDA Graph capture. `BLOCK_M`
counts `(query token, GQA head)` pairs. The page-table gather itself is masked
at sequence tails. Q and output token strides need not be contiguous.

## Existing 32K trace diagnosis

Source: `results/vllm_8b_tp8_seqlen/profiles/{dense,c1}_s32640.trace.json.gz`.
These are rank-zero summed GPU kernel durations, **not E2E wall time**.
The original run uses decode-only CUDA Graphs, so graph ID separates pure decode
from prefill/mixed launches.

| Attention phase | Dense FA2 seconds | Old C1 seconds |
| --- | ---: | ---: |
| Prefill/mixed, 4,608 main kernel launches | 10.4936 | 15.2056 |
| Pure decode, main kernel plus reduction | 1.7753 | 1.3778 |

The excess is in prefill, while decode already benefits from compact V. A 25%
smaller KV cache is not automatically 25% lower latency: QK still has dimension
128, and tensor-core utilization, repeated KV reads and launch geometry matter.

## Validation and preliminary microbenchmark

Environment: `basis`, NVIDIA L40S / SM89, BF16, vLLM 0.29.0. Single-layer timings
use CUDA Graph replay; data preparation is excluded. FA2 requires equal Q/K/V
dimensions, so its reference uses contiguous K128 and V padded with zeros to
128. The first V-rank output channels are a mathematical reference for compact V.
This FA2 comparison executes the dense-width attention kernel, not a native
DiffKV FA2 kernel. Cache working sets, batch composition and timing conditions
differ from a full model; these numbers must not be reported as E2E speedups.

The dedicated kernel's serial GPU-0 quick test is in
`results/diffkv_sm89/prefill_h4_gpu0.json`. At the original fixed configuration
`[64,64,4,2]`:

| Workload | Old unified ms | New prefill ms | Dense-width FA2 ms |
| --- | ---: | ---: | ---: |
| Q=8192, no prefix | 0.858 | 0.451 | 0.429 |
| Q=8192, prefix=24448 | 5.557 | 2.429 | 2.571 |
| Q=8161 with long prefix + 31 decode rows | 5.933 | 2.847 | 4.001 |

Relative L2 error against FA2 is below 0.0032 in these cases. Independent FP32
reference tests cover local head counts 4/8, page sizes 16/32, ragged lengths,
empty query sequences, randomized page mappings, noncontiguous Q/output, output
canaries, and graph replay after changing Q/V. CUDA memcheck reports zero errors.

The V96 sweeps are in `results/kernel_tuning_sm89/prefill_h4_v96_full.json`
and `prefill_h8_v96_full.json`. Against upstream unified DiffKV, the selected
H4 configurations are 1.25x, 2.00x, 2.60x and 1.91x faster for ragged, 8K,
32K and mixed cases. H8 is 1.28x, 2.28x, 2.63x and 2.43x faster respectively.

`quick_h4`, `quick_h8` and `tune_h4` are earlier exploratory results using a
specialized launcher around the upstream kernel. Files prefixed `prefill_` test
the independent kernel. They are not interchangeable measurements. The final
independent sweeps are `tune_prefill_h4.json` and `tune_prefill_h8.json`: each
covers all 36 combinations of BLOCK_M 32/64/128, KV tile 32/64/128, warps 4/8,
and stages 2/3 over ragged, 8K, 32K and mixed cases. The selected configurations
beat the old unified kernel in every corresponding production class. The full
Qwen3-8B 4K+128 batch sweep in `results/vllm_tp8_sm89` reports matched E2E
speedups from 1.012x at batch one through 1.430x at batch 256, with zero scheduler
preemptions. Relative to the prior C1 run, the new kernel improves C1 E2E by
about 1--2% in this decode-heavy 4K protocol. The long sequence-length sweep is
the primary test of whether the old long-prefill attention gap is removed.

## Reproduction

Run from the repository root. Activate `basis` (or include its `bin` in `PATH`,
which is required for the existing prepared-NCCL C++ extension to find Ninja):

```bash
export PATH=/workspace/miniforge3/envs/basis/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m pytest -q tests/test_diffkv_prefill.py tests/test_qwen3_8b_vllm_c1.py tests/test_vllm_fixed_cohort_metrics.py
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest -q tests/test_diffkv_prefill.py
CUDA_VISIBLE_DEVICES=0 python -m evaluation.benchmark_diffkv_prefill --value-rank 96 --output results/kernel_tuning_sm89/prefill_h4_v96_full.json
CUDA_VISIBLE_DEVICES=1 python -m evaluation.benchmark_diffkv_prefill --heads 8 --value-rank 96 --output results/kernel_tuning_sm89/prefill_h8_v96_full.json
```

Use `--quick` only for a four-configuration smoke. `--heads 8` tests the 32B TP8
head geometry; it is not by itself a full 32B model benchmark.

The TP8 smoke **passed** with 32640 prefill tokens, eight output tokens, batch
sizes one/three, an 8192-token scheduling budget and decode CUDA Graphs. All eight
workers report `C1DiffKVImpl` and nonzero graph capture counts. This verifies
serving integration, not model quality or full B=32 performance. Its exact
command is recorded in `results/diffkv_sm89/tp8_graph.json`; logs are in the
adjacent `.log`. The first launch failed before
attention initialization because the shell PATH did not include the already
installed Ninja binary. The corrected invocation includes the environment's
`bin` and retains the initial failure log. Non-fatal PCIe custom-allreduce and
process-cleanup warnings are also retained in the log.
