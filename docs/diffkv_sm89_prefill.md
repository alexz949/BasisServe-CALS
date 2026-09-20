# L40S DiffKV prefill specialization

The C1 serving path now uses a dedicated causal paged-prefill Triton kernel for
QK128/V64, one local KV head and four/eight local query heads. The implementation
is adapted from vLLM's DiffKV attention algorithm; this is not a port of FA3 or a
claim of globally optimal performance. It retains FP32 online-softmax state and
accumulators, BF16 tensor-core products, the packed 192-channel KV layout, and
the existing single output-decoder GEMM.

`basisserve/vllm/diffkv_attention.py` dispatches batches with `max_query_len > 1`
to the new 2D kernel, including mixed prefill/decode batches. Pure decode retains
the existing upstream split-KV path. This preserves decode CUDA Graph behavior
and avoids spending effort on the already-small segment reduction. No installed
vLLM files are modified; backend identity and cache metadata remain compatible
with its DiffKV cache specification.

The initial production configuration is `BLOCK_M=64`, KV tile 64, four warps,
two pipeline stages. `BLOCK_M` includes grouped query heads: for 8B TP8 this is
16 query tokens per CTA, versus four with the old `BLOCK_M=16`. Tuning is offline,
not performed during CUDA Graph capture. The page-table gather itself is masked
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
128. The first 64 output channels are a mathematical reference for compact V64.
This FA2 comparison executes the dense-width attention kernel, not a native
DiffKV FA2 kernel. Cache working sets, batch composition and timing conditions
differ from a full model; these numbers must not be reported as E2E speedups.

The dedicated kernel's serial GPU-0 quick test is in
`results/diffkv_sm89/prefill_h4_gpu0.json`. At configuration `[64,64,4,2]`:

| Workload | Old unified ms | New prefill ms | Dense-width FA2 ms |
| --- | ---: | ---: | ---: |
| Q=8192, no prefix | 0.858 | 0.451 | 0.429 |
| Q=8192, prefix=24448 | 5.557 | 2.429 | 2.571 |
| Q=8161 with long prefix + 31 decode rows | 5.933 | 2.847 | 4.001 |

Relative L2 error against FA2 is below 0.0032 in these cases. Independent FP32
reference tests cover local head counts 4/8, page sizes 16/32, ragged lengths,
empty query sequences, randomized page mappings, noncontiguous Q/output, output
canaries, and graph replay after changing Q/V. CUDA memcheck reports zero errors.

`quick_h4`, `quick_h8` and `tune_h4` are earlier exploratory results using a
specialized launcher around the upstream kernel. Files prefixed `prefill_` test
the independent kernel. They are not interchangeable measurements. The final
full B=32 sequence-length sweep remains pending explicit launch confirmation.
After the TP8 smoke completed, a separate Llama-3.1 Fisher collection job occupied
all eight GPUs. No further timing runs were launched under contention. The
36-configuration upstream-launcher sweep covered BLOCK_M 32/64/128, KV tile
32/64/128, warps 4/8, stages 2/3. The independent kernel has only had the four
quick configurations rechecked so far; its full configuration sweep and E2E
measurement remain outstanding. In particular, no claim is made yet that the
approximately 4.3-second full-cohort attention gap has disappeared.

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
CUDA_VISIBLE_DEVICES=0 python -m evaluation.benchmark_diffkv_prefill --quick --output results/diffkv_sm89/prefill_h4_gpu0.json
```

Omit `--quick` for the complete tile/warps/stages sweep. Use `--heads 8` to test
the 32B TP8 head geometry; this is not a full 32B model benchmark.

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
