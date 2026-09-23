# TP8 Full-Scan Routing

Date: 2026-09-23 UTC. Worktree: `/workspace/BasisServe-CALS-opt`.
Environment: conda `basis`, PyTorch `2.13.0+cu130`, eight NVIDIA L40S GPUs.

## Method Boundary

The active Llama-3.1-8B TP8 Basis-joint path no longer uses two-stage
coarse screening. It scores every historical Page32 with the B16R16 router.
There is no 512-page candidate limit and no optional two-stage fallback.
Coarse-only min/max and ring metadata are neither allocated nor updated.

Page masses are normalized over all non-sink historical pages per query
head, then reduced by maximum across the four query heads sharing one KV
head. The selected support remains sink page 0 plus 61 highest-scoring
pages, sorted by page ID, and the most recent 64 tokens: 2048 support slots.
Invalid positions in a partial historical page are masked.

The fused selector dispatches sorting capacity by page count, up to 4096
pages. These capacity buckets do not prune candidates. They cover the
model's 131072-token context limit, whose historical region has at most
4094 Page32 pages after excluding the recent 64 tokens.

This removal changes the routing algorithm relative to the earlier
two-stage experiment. It is not a numerically equivalent speed-only
optimization. New quality and performance measurements are required.

## Retained Execution Optimizations

- Fused decode append: V/base/residual encoding and mapped-host K updates.
- Fused final page selection, token-index construction, and slot planning.
- Persistent GPU K slots and fetching only missing selected keys.
- Fused QKV projection, V96 representation, stride-correct feature-major
  slot-attention output, and the existing TP8 NCCL communication path.
- Shared rank-0-first extension compilation and preallocated workspaces.

Dense execution is unchanged. Independent historical TP1 two-stage
experiments and their `select_and_pack` API remain archived in the repo;
the current TP8 path does not call them. In particular, preserving those
artifacts does not introduce a two-stage fallback in TP8.

## Verification

Run from the worktree above. Unit/kernel regression command:

```bash
CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python -m pytest tests/test_tp8_full_scan_routing.py \
  tests/test_slot_indexed_attention.py tests/test_uniform_allgather_timing.py -q
```

Result: 26 tests passed in 7.97 seconds. Full-scan selection tests cover
62 through 4096 pages, batch 8, tied scores, partial pages, noncontiguous
score storage, slot reuse/eviction, and high-scoring pages beyond position
512. Router scores match the BF16 mathematical reference through 130001
historical tokens (rtol/atol 0.003).

The first run passed 25 tests and failed a test's hit-count expectation:
partially filled support can retain extra valid cached tokens from earlier
selections. The reference now counts actual resident tokens, not only the
immediately preceding support. Key values and selected IDs were already
correct. Both runs remain in the test log.

TP8 integration smoke command, not a formal performance benchmark:

```bash
CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_llama31_8b_tp8_decode_grid.py \
  --arms basis_joint --contexts 4096 65536 --batches 1 8 --cohorts 0 \
  --conditioning-steps 4 --measure-steps 8 --tag full_scan_smoke \
  --output-root results/system_benchmarks/tp8_full_scan/smoke
```

Logs and new results live under `results/system_benchmarks/tp8_full_scan/`.
All four trials completed successfully: 4096/65536 prompt tokens crossed
with batches 1/8. All 32 rank JSONs report completion and the new routing
mode; generated token IDs agree across the eight ranks in each trial.
Every sequence has 13 generated tokens (one prefill prediction, four
conditioning steps, eight measured steps). All recorded step times are
finite and positive. This is an integration check, not a quality metric.
See the [smoke summary](../results/system_benchmarks/tp8_full_scan/SUMMARY.md).

New results identify routing as `full_scan_b16r16_persistent_slots`.
The grid launcher rejects manifests with a different routing mode to avoid
silently reusing completed two-stage trials.

Environment warnings: the container denies `set_mempolicy` with
`Operation not permitted`, so strict NUMA host-memory placement is not
guaranteed. CPU affinity is still applied. NCCL also reports that barriers
infer the device from the current context/rank. The run uses one rank per
GPU with matching local-rank device selection.

Historical measurements in `tp8_path_opt/` describe the removed two-stage
path. Their JSONs and frozen source patch are preserved without changes.
Neither their reported speedups nor short smoke timings establish the
performance or quality of the full-scan implementation.
