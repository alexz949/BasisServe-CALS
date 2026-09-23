# ShadowKV TP8: Formal Request Benchmark

Status: the three-arm request and independent steady-decode grid completed. Successful formal trials: 108/108; failed formal trials: 0. All eight rank records for every completed trial passed the summary validator. The 4K tests below remain correctness preflight only.

## Formal Results

The grid used 65536 and 130048 prompt tokens, batches 1/4/8, three frozen real-text cohorts, and both request and steady modes. The figure at `../plots/shadowkv_request_breakdown.png` uses the cohort with median total request time for each arm/workload; the separate steady-decode table uses three-cohort medians. Request time includes 128 total output tokens (one prefill prediction plus 127 decode steps), not 128 decode steps after prefill.

| Context | Batch | BasisKV request | ShadowKV request | ShadowKV online construction | ShadowKV SVD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64K | 1 | 7.630 s | 19.480 s | 9.866 s | 3.739 s |
| 64K | 8 | 43.392 s | 89.662 s | 39.020 s | 29.863 s |
| ~128K | 1 | 13.562 s | 28.616 s | 12.417 s | 5.932 s |
| ~128K | 8 | 95.624 s | 167.939 s | 59.188 s | 47.724 s |

At ~128K/B8, the ShadowKV request had a median max-rank peak allocated GPU memory of 39.764 GiB and a decode-ready allocated memory of 14.055 GiB. Its pinned CPU V capacity was 63.5 GiB across eight ranks. Those are different memory definitions; none is a measured STAR-KV capacity result. No ShadowKV, BasisKV, or Dense trial failed or OOMed in this grid.

Exact formal command, run from `/workspace/BasisServe-CALS-opt` in `basis` (the log retains every per-trial command):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_llama31_8b_tp8_request_grid.py \
  --contexts 65536 130048 --batches 1 4 8 --cohorts 0 1 2 \
  --arms dense basis_joint shadowkv --modes request steady \
  --output-root results/system_benchmarks/tp8_external_baselines/shadowkv \
  > results/system_benchmarks/tp8_external_baselines/shadowkv/formal_grid.log 2>&1
```

`summary.csv` contains all 108 validated trials; `request_breakdown.csv`, `decode_summary.csv`, and `memory_summary.csv` contain complete three-cohort groups. `failures.csv` has a header and no failure rows. `SUMMARY.md` records the model and source revisions, software versions, and timing boundaries. The Matplotlib summary command and its output are retained in `formal_summary.log`.

The [complete raw ShadowKV artifact tree](https://huggingface.co/alexz949/BasisServe-CALS/tree/26b66c9ad858cda646735ef28e1a6b6197e580fd/results/system_benchmarks/tp8_external_baselines/shadowkv) is pinned at HF revision `26b66c9ad858cda646735ef28e1a6b6197e580fd`.

## Fixed Algorithm

- Upstream: https://github.com/ByteDance-Seed/ShadowKV
- Local upstream checkout: `/workspace/BasisServe-CALS/external/ShadowKV`
- Upstream commit: `e51904cdeab7d4d34013370f09f2cf5fcd655e15`.
- Concatenate all eight pre-RoPE local KV heads on owner `layer % 8`; apply the official FP32 `torch.svd` decomposition and truncate to rank 160.
- Decompose batch items sequentially. Broadcast the shared BF16 U and send each rank only its local SV shard. Free global K and full-SVD temporaries after construction.
- Reuse the unchanged official CPU cache methods for landmarks, retrieval, hit reuse, reconstructed misses, and CPU V transfers.
- Preserve exact initial selected K, outlier chunks, local prompt tokens, and the upstream multiple-of-eight chunk alignment.
- Increase selected-buffer capacity only to accommodate the actual local prompt tail and the requested decode horizon.
- Use upstream FlashAttention KV-cache decode, not the repository's quality-evaluation Triton replacement.
- Prefill projection/query chunking and decoder-layer/MLP execution reuse the frozen TP8 infrastructure. No new MLP or Dense optimization is planned.

## Correctness Validation

1. All owner ranks, B1 and B2: identical replicated U; correct local SV shards; reconstruction agreement with unchanged upstream `get_svd`.
2. Local-head native cache versus full-head official reference: initial selections, subsequent selected chunks, reconstructed K, and exact CPU V transfers.
3. Short real-text model-level comparison before formal runs, including generated tokens and selection agreement.
4. No global SVD or K gather during steady-state decode.

All three standalone GPU tests passed. Factor reconstruction relative errors were zero for all ten request cases. Initial selections, subsequent selected sets, CPU V transfers, and canonicalized selected K matched the official reference (all sixteen selected-K comparisons had zero relative error). Random tensors in the first two tests are correctness inputs, not benchmark prompts or baseline checkpoint replacements.

The 4096-token, B1 real-text model smoke produced nine tokens (prefill prediction plus eight decode steps), all identical to the existing single-GPU official-cache adapter. Maximum logits relative L2 error was 0.01689821; minimum selected-set overlap was 0.96875, and mean overlap was 0.98939345. Each TP rank performed four prefill SVDs, and no SVD during decode. This passed the preselected thresholds: exact generated IDs, logits relative error below 0.02, and selection overlap at least 0.95. It is a short correctness check, not a full quality evaluation or speed benchmark. Test-only selection/logit all-gathers are not part of the production adapter.

Commands, run from `/workspace/BasisServe-CALS-opt` in the `basis` environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
torchrun --standalone --nproc-per-node=8 tests/distributed_shadowkv_tp8_factors_smoke.py \
  --upstream /workspace/BasisServe-CALS/external/ShadowKV \
  --output results/system_benchmarks/tp8_external_baselines/shadowkv/smoke \
  > results/system_benchmarks/tp8_external_baselines/shadowkv/factors_smoke.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
torchrun --standalone --nproc-per-node=8 tests/distributed_shadowkv_tp8_cache_smoke.py \
  --upstream /workspace/BasisServe-CALS/external/ShadowKV \
  --output results/system_benchmarks/tp8_external_baselines/shadowkv/smoke \
  > results/system_benchmarks/tp8_external_baselines/shadowkv/cache_smoke.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
torchrun --standalone --nproc-per-node=8 tests/distributed_shadowkv_tp8_model_smoke.py \
  --upstream /workspace/BasisServe-CALS/external/ShadowKV \
  --output results/system_benchmarks/tp8_external_baselines/shadowkv/smoke \
  --decode-steps 8 \
  > results/system_benchmarks/tp8_external_baselines/shadowkv/model_smoke.log 2>&1
```

Per-rank records are in `smoke/factors_rank*.json` and `smoke/cache_rank*.json`. The model comparison is in `smoke/model_reference.json`, with saved logits/selections in `smoke/model_reference.safetensors`. These are correctness measurements, not formal performance results.

## Formal Measurement Protocol

- Contexts 65536 and 130048; batches 1, 4, 8; three frozen real-text cohorts.
- Request window: 128 output tokens in total, including the prefill prediction and 127 subsequent decode steps; greedy, ignore EOS.
- Separate steady-state window: 16 conditioning and 128 measured decode steps from a fresh request.
- Record synchronized whole-request latency and every construction component. Construction is interleaved with layer prefill; an inclusive prefill span must not be added to SVD time again.
- Preserve every failure with its phase, command, log, and rank details.
- A rank log interrupted mid-JSON line keeps its last complete stage; `failures.csv` names truncated rank logs instead of dropping the OOM record.
- Formal runs followed correctness checks and explicit confirmation of their exact command.

### Confirmed Timing Definition

The user confirmed that BasisKV's ordinary projections, routing encoding, metadata updates, and cache writes belong to prefill and remain fully included in request latency. Dense uses the same ordinary-prefill category.

The separate construction category denotes ShadowKV-specific prompt-dependent global-K gather, SVD, factor redistribution, and related preparation, including the transfer that makes its representation decode-ready. BasisKV has zero cost in this narrowly defined category because it does not perform this construction, not because its encoding or cache population is free. The figure caption must state this distinction.

ShadowKV construction is interleaved with layer prefill. Report nonoverlapping components: ordinary prefill plus ShadowKV-specific construction reaches the first-output/decode-ready boundary; subsequent decode ends at the final output. Do not add an inclusive prefill span to the same construction cost again. Record the actual first-output timestamp separately if the first prediction is available before preparation is complete.

The request produces 128 tokens total (one prefill prediction plus 127 decode steps). A separate steady-state experiment measures 128 decode steps after 16 conditioning steps. These counts and timing windows must not be conflated or labeled as the same 128-step decode measurement.

`benchmarks/system/tp8_request_timing.py` validates and aggregates ordered, nonoverlapping phases from eight ranks on one host's common monotonic clock. Its input contract requires CUDA-synchronized boundaries. It uses the latest start and latest end for each distributed phase, rather than summing maximum local durations: otherwise a nonowner's wait for SVD inside redistribution can double-count the owner's SVD. The request spans the earliest synchronized start to the latest final output. Synchronization/instrumentation overhead stays included and must be disclosed; these are not unsynchronized deployment timings.

For an additive figure, use `prefill_ms + representation_build_ms + decode_after_representation_ready_ms`. The separately retained `decode_after_first_token_ms` can overlap post-first-token preparation and is not the additive decode component. The helper is connected to `benchmarks/system/bench_llama31_8b_tp8_request.py` and checked against all eight rank records.

## Runner Preflight

- The 4K/B1 request trial completed for ShadowKV, Dense, and BasisKV with 128 total output tokens. All three wrote eight rank JSON files and one replica JSON file. Dense and BasisKV generated IDs matched the preceding frozen runner for all 128 tokens.
- The independent 4K/B1 ShadowKV steady trial completed 16 conditioning steps and 128 measured steps across eight ranks.
- The independent 4K/B8 ShadowKV steady trial also completed 16 conditioning steps and 128 measured steps across eight ranks. The raw-trial validator confirmed 145 generated tokens per request and eight consistent rank records; no warning, error, or OOM appeared in its launcher log.
- Independent 4K/B1 Dense and BasisKV steady trials each completed 16 conditioning steps and 128 measured steps across eight ranks. The first 128 generated tokens matched their respective request smoke trials exactly; the raw-trial validator and `smoke/steady_validation.log` retain the checks. Their means were 30.0732 and 24.2105 ms/step, respectively, as preflight observations only. Neither launcher log contained a warning, error, or OOM.
- The 4K/B8 ShadowKV state check in `smoke/memory_validation.log` verified all eight ranks against the actual BF16 tensor shapes: U 335,544,320 bytes/rank, SV 10,485,760 bytes/rank, pinned CPU V 268,435,456 bytes/rank (2 GiB across ranks). These are active tensor capacities, not total process GPU memory or formal 64K/128K results.
- The 4K/B8 ShadowKV request trial also completed 128 total output tokens across all eight ranks. It produced 833 ordered construction phases (32 layers x (two preparation phases plus three per-request phases x eight requests), then H2D); the additive parts matched total request time.
- The summary validator computes the expected construction phases from batch size; it validated the existing B1 (161) and B8 (833) raw trials. Its output is in `smoke/summary_validation.log`.
- The ShadowKV request contained 161 ordered phase records: 32 layers x (two preparation phases plus gather/SVD/redistribution), then post-prefill H2D preparation. The three additive components equaled total request time in the raw replica JSON.
- These 4K runner numbers are preflight only and do not enter the completed 64K/128K formal grid or paper figure.
- Run commands and logs are retained under `smoke/*_launcher.log`; rank JSON and replica JSON are under `smoke/{request_4k_b1,dense_request_4k_b1,basis_request_4k_b1,steady_4k_b1,request_4k_b8,steady_4k_b8,dense_steady_4k_b1,basis_steady_4k_b1}/`.

The 4K/B8 steady preflight used `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9` with `/workspace/miniforge3/bin/conda run --no-capture-output -n basis torchrun --standalone --nproc-per-node=8 benchmarks/system/bench_llama31_8b_tp8_request.py --arm shadowkv --mode steady --length 4096 --batch 8 --cohort 0 --tokens /workspace/runs/l31-cal128/tp8-benchmark-prompts/p4096_c0.safetensors --prompt-manifest /workspace/runs/l31-cal128/tp8-benchmark-prompts/p4096_c0.json --output-dir results/system_benchmarks/tp8_external_baselines/shadowkv/smoke/steady_4k_b8`. Its measured 128-step mean was 33.2811 ms/step; this is a smoke-test observation only and must not enter the formal latency table.

The runner warms the actual model path using a separate 4096-token request at the same batch, then loads a fresh model and cache for the measured request. Model loading and backend compilation occur before `t0`. Every rank records BF16 eager execution with TF32 disabled, the same CPU affinity policy and frozen prompt cohorts, GPU allocator snapshots, NVML process memory, and exact active cache-state sizes. ShadowKV records owner-phase allocator peaks and pinned CPU V capacity. The request records one prefill prediction plus 127 decode steps; steady decode has a fresh prefill, 16 conditioning steps, and 128 measured steps. The first-token timestamp is retained separately from representation-ready so H2D preparation is counted exactly once.

`freeze/source.tar.gz` archives the new runner and adapter files listed in `freeze/manifest.json`. The preceding frozen grid's source archive covers reused TP8 implementation files. The official ShadowKV checkout is clean at commit `e51904cdeab7d4d34013370f09f2cf5fcd655e15`.

The summary/plot step uses Matplotlib 3.10.7, installed in `basis` after the TP8 preflight. Torch, CUDA runtime, NumPy, and FlashAttention versions were rechecked and unchanged. Installation output is in `plot_dependency.log`.

Thirteen CPU-only timing-accounting tests, three batch-dependent summary-contract tests, and one truncated-OOM-log test passed. Their constructed timestamps and phase lists are test fixtures, not latency measurements:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python -m pytest -q tests/test_tp8_request_timing.py \
  tests/test_tp8_shadowkv_grid.py \
  tests/test_tp8_shadowkv_summary.py \
  > results/system_benchmarks/tp8_external_baselines/shadowkv/timing_tests.log 2>&1
```

## Dependency Preparation

The `basis` environment initially had no `flash_attn` package. The official release has no PyTorch 2.13 Python 3.12 wheel; unchanged upstream source was downloaded and built locally after the preceding frozen benchmark completed.

Source release: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1

```bash
CUDA_VISIBLE_DEVICES='' FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python -m pip download --no-deps --no-build-isolation --no-binary=:all: \
  --dest /workspace/.cache/tp8-baseline-deps flash-attn==2.8.3.post1
```

Downloaded source: `/workspace/.cache/tp8-baseline-deps/flash_attn-2.8.3.post1.tar.gz`.

The first source build failed with incompatible compiler/toolkit headers: the existing environment had nvcc 13.4 but CUDA runtime headers 13.0. The failure is retained in `dependency_build.log`. No compatibility checks or baseline source were patched.

A matching build-only CUDA toolchain was installed under `/workspace/.cache/tp8-cuda13` using `pip install --no-deps --target`: `nvidia-cuda-nvcc==13.0.88`, `nvidia-cuda-crt==13.0.88`, `nvidia-nvvm==13.0.88`, `nvidia-cuda-runtime==13.0.96`, and `nvidia-cuda-cccl==13.0.85`. Its `lib/libcudart.so` symlink points to `libcudart.so.13`. See `toolchain.log`. This does not replace the environment's Torch or CUDA runtime packages.

The second build completed successfully. `flash_attn==2.8.3.post1` installed and imported successfully, and the model smoke exercised `flash_attn_with_kvcache` on all eight GPUs. Torch remained `2.13.0+cu130`, with CUDA runtime version `13.0`:

```bash
CUDA_VISIBLE_DEVICES='' CUDA_HOME=/workspace/.cache/tp8-cuda13/nvidia/cu13 \
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=2 NVCC_THREADS=2 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python -m pip install --no-deps --no-build-isolation \
  /workspace/.cache/tp8-baseline-deps/flash_attn-2.8.3.post1.tar.gz \
  >> results/system_benchmarks/tp8_external_baselines/shadowkv/dependency_build.log 2>&1
```

SM80 code is compatible with the L40S SM89 GPUs. Compilation uses two jobs, and no dependency upgrades are requested. The preceding frozen benchmark had already completed before this installation began.
