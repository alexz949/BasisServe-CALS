# Uniform feature-major AllGather optimization

This implementation keeps the current C1 wire contract unchanged:

```text
one common source width per layer
source-major feature-major arena [TP * local_width, tokens]
exact AllGather semantics on every rank
compressed-V attention may write directly into its local source slot
```

It does **not** introduce per-source ranks, decoder sharding, a different
compression ratio, or an approximate collective.

## Implemented stages

### 1. Prepared uniform NCCL (`uniform_nccl`)

`FeatureRaggedCommunicator.prepare_uniform(...)` creates a stream-bound plan
outside the hot path. The plan permanently stores:

- local and total wire width;
- dtype and element count;
- the final receive arena and this rank's in-place source slice;
- NCCL datatype and CUDA stream.

The serving hot call uses `gather_inplace_fast()` and is therefore just an in-place `ncclAllGather`; it no longer builds
ragged offsets, scans source widths, looks up workspaces, or constructs tensor
slices on every layer invocation.

The checked API verifies device/stream invariants; the Qwen serving path uses the explicit fast API after configuration. This is the safe first deployment backend and is the default for the modified
Qwen3-8B TP4 decode benchmark.

### 2. CUDA Graph measurement (`uniform_nccl_graph`)

`benchmarks/bench_uniform_allgather.py` can capture and replay the prepared NCCL
call. This separates host/Python/C++ launch overhead from the NCCL device-kernel
floor without changing the model runtime by default.

### 3. Experimental single-node CUDA-IPC collectives (`uniform_ipc`)

A symmetric double-buffer arena is allocated with `cudaMalloc`, exported with
CUDA IPC handles, and mapped by every TP process. Four exact fixed-TP kernels
are available:

- `fanout`: one or more independent CTAs load disjoint local byte ranges once and replicate stores to every peer;
- `fanout_warp`: one warp per remote peer, trading repeated local reads for parallel peer issue;
- `recursive_doubling`: `log2(TP)` exchange phases for TP2/TP4/TP8;
- `ring`: `TP-1` forwarded-block phases for larger messages, with 1/2/4/8 independent chunk channels;
- `auto`: conservative payload thresholds choose `fanout`, `recursive_doubling`, or `ring`; `fanout_warp` remains an explicit autotuning candidate.

All kernels use 16-byte vector copies when alignment permits. `--ipc-channels`
may explicitly select 1/2/4/8 CTAs for `fanout` or `ring`; `0` uses the
algorithm default. Ring auto-channels use one CTA through 128 KiB/source, four
through 512 KiB/source, and eight above 512 KiB/source. Per-source,
per-channel, per-phase 64-bit epochs are stored after the arena. Remote data is published
before a system-scope release store; receivers poll the corresponding local
flag with a system-scope acquire load. No remote peer atomics are used.

The initial automatic thresholds are intentionally only autotuning seeds:

```text
TP4: fanout <= 16 KiB/source, recursive doubling <= 128 KiB/source, else ring
TP8: fanout <=  8 KiB/source, recursive doubling <= 128 KiB/source, else ring
```

They must be replaced by results from the target L40S topology.

## Current safety contract

`uniform_ipc` is an opt-in experimental backend with the following constraints:

- single-node TP2, TP4, or TP8 with one process per GPU;
- all ranks call collectives in exactly the same order;
- one common source width per layer;
- one CUDA stream per prepared plan;
- Qwen3-8B and Qwen3-32B integration currently enables it only for decode-only
  `max_forward_tokens=1`;
- host-managed epochs make this v1 backend non-capturable by CUDA Graphs;
- `prepare_ipc()` is collective and invalidates plans belonging to an older
  arena generation;
- IPC reconfiguration and shutdown use a two-phase lifetime protocol: every
  rank closes imported peer mappings, crosses a process-group barrier, and only
  then frees or replaces its exported local allocation;
- logical rank order is currently used directly by recursive doubling and ring, so topology-aware rank permutations are not yet implemented;
- the CUDA-IPC/P2P topology and system-scope flag ordering must pass the supplied
  long-running smoke test on the deployment machines before serving use.

`uniform_nccl` remains the fallback and recommended first benchmark target.

## AllGather-only benchmark

Qwen3-8B TP4 with V rank 64 has eight local query heads, hence
`local_width = 8 * 64 = 512`. At BF16 and one decode token this is a 1 KiB block
per source:

```bash
torchrun --standalone --nproc-per-node=4 \
  benchmarks/bench_uniform_allgather.py \
  --local-width 512 --tokens 1 --dtype bfloat16 \
  --backends feature_direct,uniform_nccl,uniform_nccl_graph,uniform_ipc \
  --ipc-algorithm auto \
  --warmup 100 --iters 2000 \
  --output-json results/uniform_ag_tp4_v64_b1.json
```

Test one custom algorithm explicitly:

```bash
torchrun --standalone --nproc-per-node=4 \
  benchmarks/bench_uniform_allgather.py \
  --local-width 768 --tokens 1 --dtype bfloat16 \
  --backends uniform_nccl,uniform_ipc \
  --ipc-algorithm fanout \
  --ipc-channels 1 \
  --warmup 100 --iters 5000
```

The benchmark checks the exact gathered arena before reporting p50/p90/p95/p99
latencies. `bytes_sent_per_rank` is `(TP-1) * local_block_bytes` for every
backend, so the comparison does not hide a wire-volume change.

## Long-running ordering smoke test

Run each custom algorithm with changing producer data and many alternating-slot
epochs:

```bash
for algorithm in fanout fanout_warp recursive_doubling ring; do
  torchrun --standalone --nproc-per-node=4 \
    tests/distributed_uniform_allgather_smoke.py \
    --backend uniform_ipc --ipc-algorithm "$algorithm" \
    --ipc-channels 1 \
    --local-width 512 --tokens 1 --dtype uint8 \
    --iterations 20000 --check-every 1
done
```

Also run widths/ranks and token counts representative of the real model, not
only the 1 KiB point.

## Qwen3-8B TP4 decode integration

Prepared NCCL:

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_qwen3_8b_tp4_decode.py \
  ... \
  --arm c1_uniform_r64 \
  --c1-decode-attention cuda \
  --c1-allgather-backend uniform_nccl
```

Experimental direct fanout:

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_qwen3_8b_tp4_decode.py \
  ... \
  --arm c1_uniform_r64 \
  --c1-decode-attention cuda \
  --c1-allgather-backend uniform_ipc \
  --c1-ipc-algorithm fanout \
  --c1-ipc-channels 1
```

The CUDA attention output still writes directly into the current local
feature-major slot. Only the collective implementation changes.

## Qwen3-32B TP4 uniform-V64 integration

Qwen3-32B has two local KV sources and sixteen local query heads under TP4.
For uniform rank 64, the serving decode path now views the packed compact cache
as `[batch, 2, sequence, 64]` without copying it and launches one CUDA kernel
over all sixteen query heads. The handwritten kernel accepts the runtime GQA
ratio `16 / 2 = 8` and writes its result directly into the collective slot:

```text
[B, 16, 1, 128] query
[B,  2, S, 128] key
[B,  2, S,  64] compact value
        -> one CUDA decode launch
[16 * 64, B] feature-major local slot
        -> prepared AllGather
[4 * 16 * 64, B] global arena
```

This removes the previous two per-source decode launches, `torch.cat`, and
token-major-to-feature-major pack from the incremental decode path. At batch
64 the local BF16 block is `1024 * 64 * 2 = 128 KiB` per rank, so it must be
benchmarked separately from Qwen3-8B's 1 KiB batch-one point.

Prepared NCCL end-to-end serving:

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_qwen3_32b_tp4_prefill.py \
  --arm c1_ragged \
  --model /path/to/Qwen3-32B \
  --factor-dir /path/to/uniform-v64-factors \
  --c1-decode-attention cuda \
  --c1-allgather-backend uniform_nccl \
  --batch-size 64 --prompt-length 1024 --output-tokens 128 \
  --output-json results/qwen3_32b_uniform_v64_cuda_nccl.json
```

The C1 prefill path remains a separate compute-bound measurement. Uniform
local ranks use one two-source Triton prefill launch, but still execute the C1
global decoder. A dense-prefill/compact-cache decode hybrid is intentionally
left as a separate follow-up rather than being mixed into this decode-kernel
experiment.

## Recommended optimization order on L40S

1. Compare `feature_direct` and `uniform_nccl` to quantify dynamic hot-path
   overhead.
2. Compare `uniform_nccl` and `uniform_nccl_graph` to isolate host launch cost.
3. Run `fanout`, `fanout_warp`, `recursive_doubling`, and `ring` for every real payload size. For `fanout` and `ring`, sweep 1/2/4/8 channels (`SWEEP_CHANNELS=1` in the supplied script); do not trust the initial `auto` thresholds yet.
4. Use Nsight Systems to measure producer-finish to collective-start gaps and
   collective completion latency.
5. Stress-test changing data for at least tens of thousands of epochs.
6. Promote only topology-specific winning thresholds into `auto`.

## Validation status of this patch

The code was validated in a CPU-only artifact environment as follows:

- changed Python files compile successfully;
- the pure-Python TP2/TP4/TP8 recursive-doubling and ring simulations pass;
- the host C++ translation unit passes a syntax-only compile against the installed PyTorch and NCCL headers;
- the extracted SM89 device-kernel body passes a Clang CUDA device syntax compile;
- the complete CUDA translation unit was **not** built with NVCC, and the distributed runtime was not executed, because this environment has no CUDA toolkit or visible GPU.

Therefore `uniform_nccl` is the low-risk first target. Treat `uniform_ipc` as an experimental implementation until it builds and passes the long-running smoke test on the L40S node.

## Deliberately deferred final stage

The current patch does not yet duplicate compact attention coordinates directly
from the attention epilogue registers into every peer arena. That requires a
cross-kernel completion protocol because multiple attention CTAs jointly produce
one source block. The present IPC arena, peer-pointer table, epoch layout, and
smoke benchmark are designed as the prerequisite for that next step. Keeping it
separate lets the custom collective be validated before coupling failures to the
attention kernel.
