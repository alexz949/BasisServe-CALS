# Quest Upstream CUDA GQA Patch

## Scope

Target: Llama-3.1-8B-Instruct, TP8, B1, four query heads and one KV head per GPU.
Upstream Quest commit: `01c1623bf9395009520874e989e29f683203b357`.
This patch is distinct from the earlier local Triton reimplementation.
The later six-trial formal comparison is in the
[paired summary](../paired/RESULTS_SUMMARY.md), which also links the fixed HF
archive containing all raw logs, smoke records and source snapshots below.

The decode path uses the upstream CUDA KV append/metadata update, min/max page
estimate, RAFT radix top-k, paged attention, split-KV planning and merge kernels.
It does not call PyTorch top-k or the Basis selected-token attention kernel.
Prefill retains the matched benchmark's Flash SDPA and Llama-3.1 RoPE.

Protocol: full uncompressed GPU K128/V128, page 16, 127 historical pages selected
independently per query head plus the newest page, first two layers Dense.
BF16 is added to upstream dtype dispatch; the original FP16 path remains tested.
The per-query-head budget is not matched physical retrieval to Basis's shared
per-KV-group selection. No KV4, A8, MLP change, or quality evaluation is included.

## CUDA Changes

- Separate query-head page-list indexing from physical KV-head addressing.
- Selected attention uses one query head per tile because different Q heads may
  select different pages. KV storage remains shared without replication.
- Work estimation, split planning and launch grids count query-head work units.
- Permit GQA in the decode handler and derive metadata head count from KV data.
- Replace the fixed 32-row top-k template with the actual runtime row count.
- Enable BF16 dispatch in addition to FP16.
- Add BF16 min/max vector reduction and type-correct extrema initialization.
- Supply the missing BF16 vector-load storage trait for the pinned RAFT version;
  radix top-k still operates on the original BF16 scores, without an FP16 cast.

The bridge reuses the decode plan while the selected page budget is fixed;
initial planning occurs in prefill. It allocates one upstream paged KV cache per
sparse layer, not a dense cache plus a duplicate paged cache. Each TP rank still
uses the original row-parallel output projection/all-reduce.

`quest_gqa.patch` contains the upstream source edits. Apply it to the pinned
Quest revision; the dependency submodules and header-only dependencies listed
below must also be present. Python integration and standalone bindings live in
`basisserve/kernels/quest_native.py`, `basisserve/core/quest_native_tp8.py` and
`basisserve/kernels/csrc/quest_native_bindings.cu` in the optimization worktree.

## Build

Working directory: `/workspace/BasisServe-CALS-opt`. Environment: `basis`.
No pip installation or downgrade of the shared environment was performed.
The Torch extension builds the four upstream CUDA translation units needed for
append/metadata, scoring, selection and decode, with standalone bindings for SM89.
Unused upstream prefill attention, RoPE and RMSNorm are not linked; the matched
model framework already provides those operations.
Pinned header dependencies, under `/workspace/Quest/kernels/3rdparty`:

| Dependency | Version/commit |
| --- | --- |
| FlashInfer, upstream submodule | `9f49803b1db0a40ea0019ad98b8bb5d4f1593c77` |
| RAFT, upstream submodule | `1e4961e2354afba116e3479c5ec9041937b9922e` |
| RMM | `v24.06.00`, `d889275f7e127ccbc9d2e5547086509a7faa84ea` |
| spdlog | `v1.13.0`, `7c02e204c92545f869e2f04edaab1f19fe8b19fd` |
| NVTX | `v3.1.0`, `e170594ac7cf1dac584da473d4ca9301087090c1` |
| fmt | `10.2.1`, `e69e5f977d458f2650bb346dadf2ad30c5320281` |

The standalone build enables CUDA half operators, header-only external fmt,
and the experimental CUDA memory-resource API required by RMM. Initial build
failures for missing NVTX and fmt/half-operator flags are retained in `build.log`.
No SHA256 checks were performed.

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -c 'from basisserve.kernels.quest_native import load_quest_native; load_quest_native()' >> results/system_benchmarks/quest_tp8/native/build.log 2>&1
```

## Validation Status

The extension built successfully. Native operator tests: **8 passed in 2.42s**.
Coverage includes GQA 4:1 and 8:1, multiple KV heads, FP16 and BF16, the original
32-head MHA layout, 64K, partial/new pages, negative/tied top-k inputs, cache
contents, min/max metadata, per-query selections and selected attention output.

An initial FP32 reference failed on one of 32,760 BF16 scores. The page metadata
was bitwise correct; FP64 gave `178.50001192092896`, correctly rounded by native
CUDA to BF16 `179`. PyTorch's FP32 sum rounded first to `178.5`, then to BF16 `178`.
The corrected test uses an FP64 mathematical reference and permits one final
FP16/BF16 ULP for reduction-order rounding, while checking cache and metadata
exactly. See `rounding.log`; the initial failure is retained in `test.log`.

Tests use eager execution on the default CUDA stream. CUDA Graph capture and
non-default streams have not been validated. No model-quality claim is made.

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest tests/test_quest_native.py -q >> results/system_benchmarks/quest_tp8/native/test.log 2>&1
```

## TP8 Smoke Results

Both trials completed on all eight ranks with finite logits and no OOM, using
the `basis` environment and 8 x L40S. Slurm remains unavailable; smoke ran locally.
The only model-run warning was NCCL's informational `barrier()` device inference.

| Context / B1 | Conditioning forwards | Measured forwards | Mean ms/step | Median ms/step | Wall tokens/s | Max-rank prefill peak allocated bytes | Rank-0 decode peak allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2 | 8 | 25.9578 | 25.8239 | 38.8476 | 3178714112 | 3009676800 |
| 65536 | 4 | 8 | 26.8713 | 26.0969 | 37.6897 | 5802550272 | 4084326912 |

The previous Triton port's short-run means were 34.4634 and 34.1829 ms/step.
The native CUDA patch is about 24.68% and 21.39% lower latency, respectively, in
these separate smoke runs. These are not formal repeated estimates. In particular,
the 64K mean/median difference shows why eight measured steps are insufficient
for a publication comparison. Do not report the previous 1.43x Basis/Quest ratio
as a comparison against the native CUDA path.

All smoke uses the same calibration-bank first-row prompt as the previous port.
At the smoke stage, formal cohorts and Dense/Basis were not rerun. The subsequent
Quest/Basis paired run is documented separately above. This is the official CUDA **operator path with a local patch and
TP8 integration**, not an unmodified upstream end-to-end application. First-use
prefill overhead is not a formal request-latency measurement. Allocated memory
is PyTorch allocator accounting, not total process/device memory.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m torch.distributed.run --standalone --nproc_per_node=8 benchmarks/system/bench_llama31_8b_tp8_combined.py --arm quest_native --length 4096 --batch 1 --conditioning-steps 2 --measure-steps 8 --tag smoke --output-root results/system_benchmarks/quest_tp8/native > results/system_benchmarks/quest_tp8/native/smoke4k.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m torch.distributed.run --standalone --nproc_per_node=8 benchmarks/system/bench_llama31_8b_tp8_combined.py --arm quest_native --length 65536 --batch 1 --conditioning-steps 4 --measure-steps 8 --tag smoke --output-root results/system_benchmarks/quest_tp8/native > results/system_benchmarks/quest_tp8/native/smoke64k.log 2>&1
```

The earlier Triton formal-grid proposal was not used. The subsequent native
formal grid explicitly used `--arms quest_native basis_joint`. All build,
diagnostic and run logs are retained in the linked HF archive; source snapshots
are under its `native/source/` and `paired/source/` directories.
