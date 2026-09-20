# Qwen3-8B TP8 C1 serving

> Benchmark status: the E2E numbers linked below were produced by the original
> unified Triton DiffKV kernel. They remain valid historical measurements, but
> are not a fully optimized C1 result. The current tree contains a separate
> SM89 prefill specialization; its matched dense/C1 E2E rerun is pending.

The `BasisServeQwen3_8BC1ForCausalLM` vLLM architecture folds each rank's
physical KV-head encoder into its V projection. It uses a dedicated SM89
prefill kernel and the vLLM Triton DiffKV decode path for paged K128/V64
attention, then the repository's
prepared `uniform_nccl` collective and one BF16 decoder GEMM per layer.

The TP8 layout is four query heads per rank, 64 coordinates per query head,
256 local coordinates, and 2048 gathered coordinates. The receive arena is
feature-major `[2048, scheduled_tokens]`; decoding is `arena.T @ decoder`
with decoder shape `[2048, 4096]`.

The native attention backend produces token-major output. This version has
one source-local transpose/copy before AllGather. It does not use the
repository's handwritten contiguous-cache CUDA attention kernel, and does
not yet write attention output directly into the collective source slot.

## Runtime contract

- Tested environment: `basis`, PyTorch `2.13.0+cu130`, vLLM `0.29.0`, eight L40S.
- Qwen3-8B-Base geometry, TP8, PP1, DP1, BF16 model and KV cache, uniform R64.
- The factor bank uses `basisserve.qwen3_8b.gqa_c1_joint.v1` and an authenticated
  `results.json`. Each layer's safetensors SHA256 is checked before loading.
- Supported execution configuration: compilation mode `NONE`, with eager
  execution or `FULL_DECODE_ONLY` CUDA Graphs. Torch.compile/Inductor is not
  part of the validation.
- Chunked prefill uses specialized paged DiffKV attention. Graphs cover decode;
  prefill and mixed batches execute eagerly.
- One communicator is shared across sequential layers. Prepared plans are
  keyed by scheduled token count and CUDA stream, and must be warmed on the
  capture stream. The GEMM result owns its storage; returned results do not
  alias the reusable collective arena.
- Quantization, LoRA, speculative decoding, and concurrent microbatch streams
  are outside this version's tested scope.

## Reproduce validation

From the repository root, use the `basis` environment:

```bash
export PATH=/workspace/miniforge3/envs/basis/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p /tmp/c1_vllm

python -m pytest -q tests/test_qwen3_8b_vllm_c1.py
torchrun --standalone --nproc-per-node=8 \
  tests/distributed_vllm_c1_output_smoke.py 2>&1 | tee /tmp/c1_vllm/boundary.log

C1_MODEL=/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4
C1_FACTORS=/workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-8b/c1/factor-banks/R64-S6
python evaluation/smoke_vllm_qwen3_8b_c1.py \
  --model "$C1_MODEL" --factor-dir "$C1_FACTORS" \
  --execution-mode eager --output-json /tmp/c1_vllm/eager.json \
  2>&1 | tee /tmp/c1_vllm/eager.log
python evaluation/smoke_vllm_qwen3_8b_c1.py \
  --model "$C1_MODEL" --factor-dir "$C1_FACTORS" \
  --execution-mode cuda_graph --reference-json /tmp/c1_vllm/eager.json \
  --output-json /tmp/c1_vllm/graph.json 2>&1 | tee /tmp/c1_vllm/graph.log
```

The distributed boundary smoke changes input data on every replay, tests
batch sizes 1/3/8, and checks two consecutive decoders sharing one arena.
It compares against the same feature-major BF16 GEMM layout. A row-major
cuBLAS reference can choose a different reduction and is not bitwise equal.

The engine smoke uses up to 512 prompt tokens, eight generated tokens,
256 scheduled tokens per step, and request batches 1/3 with different prompt
lengths. It checks all 36 folded V layers on every rank, records collective
capture calls, and compares greedy tokens between eager and graph execution.
Its elapsed times include warmup effects and are not E2E benchmark results.

Validation completed on 2026-09-20: both CPU tests passed; the distributed
boundary passed 36 changing-input replays; eager and graph engine smokes
passed with identical greedy output tokens. All eight workers reported
36 loaded V layers. Each worker recorded 216 C1 boundary capture calls in
graph mode and zero in eager mode (the count includes graph profiling).

The first engine attempt exposed the removed vLLM `skip_prefixes` loader
argument; the integration now filters replaced O-projection weights before
using the current loader. The first distributed check used a row-major
GEMM reference and failed bitwise comparison; the corrected same-layout
reference passes without relaxing tolerance. Logs retain both attempts.

Observed non-fatal warnings: PCIe TP8 disables vLLM's unsupported custom
all-reduce variants; MLP all-reduces use PyNccl. Initial Triton compilation
caused warmup/JIT warnings. Both engine tests exited with status zero; vLLM
reported graceful worker exits followed by forced cleanup of the EngineCore
process during its automatic shutdown.

## Partial decoder cost

```bash
python benchmarks/bench_source_partial_decoder.py \
  --batches 1,8,256 --warmup 3 --iterations 20 \
  --output-json /tmp/c1_vllm/decoder.json 2>&1 | tee /tmp/c1_vllm/decoder.log
```

2026-09-20 L40S smoke, resident inputs, CUDA Graph p50 in microseconds:

| Batch | One GEMM | Two groups | Four groups | Eight groups |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.47 | 12.38 | 16.86 | 28.42 |
| 8 | 17.17 | 19.71 | 27.71 | 34.70 |
| 256 | 24.83 | 28.21 | 34.77 | 47.49 |

Splitting into eight source GEMMs costs 1.91–2.48 times the decoder latency
at these points. Each partial `addmm_` also rounds into a BF16 output:
relative L2 error against the FP32 reference rises from about 0.00166 to
0.00334–0.00357. This experiment measures the cost that overlap would need
to recover; it does not measure overlap or rule out a fused device kernel.
The serving path keeps one decoder GEMM.

Prepared NCCL exposes whole-collective completion, not per-source arrival
notifications. Per-source overlap would require an additional communication
protocol and is not implemented here.

## Full benchmark protocol

The intended sweep is concurrency 1/2/4/8/16/32/64/128/256, 4096 input tokens
and exactly 128 generated tokens, TP8. Fix the scheduler token budget, prefix
caching, chunked prefill, graph sizes, dtype and memory utilization across
dense and C1 arms. Report TTFT, per-request TPOT, output throughput and wall
latency separately. With interleaved prefill/decode, subtracting the last
first-token timestamp from the last completion timestamp does not measure
pure decode throughput for the whole cohort.

The full sweep completed on 2026-09-20 in the `basis` environment on eight
L40S GPUs, without Slurm. Each arm used one full-length warmup and three
measured runs per batch, an 8192-token scheduler budget, memory utilization
0.8, synchronous scheduling, and decode-only CUDA Graphs. Profiling was
performed separately after each timed sweep, on rank zero for batches
1/32/256. The driver is `evaluation/benchmark_vllm_qwen3_8b_c1.py`; use
`evaluation/summarize_vllm_qwen3_8b_c1.py` to regenerate the summary.

See [the benchmark report](../results/vllm_tp8/summary.md) for commands,
all nine batch sizes, metric definitions, caveats and profile attribution.
C1 E2E speedup ranges from approximately 1.01x at batch one to 1.42x at
batch 256, in this matched compilation-mode-NONE configuration. Batch-one
and batch-two TPOT regress, so these results do not establish universally
faster decode or an optimal kernel. The profiles prioritize small-batch
decoder and paged DiffKV attention work over the source-local copy.

The benchmark metric test plus the two C1 algebra tests pass (three tests).
Both benchmark processes exit zero and validate exact output lengths,
non-corrupted request statistics, and all C1 layers/capture counters.
vLLM shutdown emits EngineCore/worker termination and Python resource-tracker
warnings; logs preserve these. GPU allocations are released after shutdown.
