# LRQK at TP1, 64K, after the prefill offload stream-order fix

2026-11-11, one L40S on node6 (Slurm jobs 240498, 240499, 240501; diagnostic breakdown 240507; trial 240508). This note reports
the correctness checks and the 12 formal runs of the two LRQK placements with the fix `be7c8a2` ("Fix LRQK prefill offload stream
ordering"). `SUMMARY.md` in this directory is the generated version of the same tables.

## Bottom line

- **All 12 formal runs completed.** No out-of-memory, no failed run, no rerun of a formal run.
- **The fix is in effect and both placements compute the same thing.** The stream-order check passes, and the
  two placements produce identical tokens and log-probabilities, at 8K (teacher forced) and at 64K (free
  running, all three prompts).
- **Steady decode is about 0.31 to 0.33 s per token for both placements.** Offloading K only is slightly faster
  than offloading K and V (p50 314 ms against 333 ms) and uses 4 GiB more GPU memory.
- **These are absolute timings on this host, not a reproduction of the earlier L40S runs.** The GPU model
  matches; the CPU, driver and Transformers version do not (see "Limits").

## Setup

| | |
|---|---|
| Model | Llama-3.1-8B-Instruct, BF16, original V128 and Wo, revision `0e9e39f2` |
| Workload | TP1, batch 1, 65536 prompt tokens; request = 128 generated tokens; steady = 16 conditioning + 128 measured steps |
| Prompts | Row 0 of cohorts 0, 1 and 2 of the TP4 64K inputs: three different prompts, not three repetitions |
| LRQK | rank 32, top-k 2048, 64 lite tokens, iterations (2, 2), tolerance (0.01, 0.01); upstream `tenghuilee/LRQK@caf16293`, unmodified |
| Arms | `lrqk_upstream_koff_v128`: exact K in pinned host memory, V on GPU. `lrqk_upstream_kvoff_v128`: K and V in pinned host memory |
| GPU | NVIDIA L40S, physical GPU 6 of node6 (PCI `D5:00.0`), driver 550.54.15, no uncorrected ECC errors, no other process on it |
| CPU | Xeon Gold 6338; 24 threads on 12 physical cores (both hyperthreads) of the GPU's NUMA node, `OMP_NUM_THREADS=24` |
| Software | conda env `basis` (vLLM environment), Python 3.11.16, PyTorch 2.13.0+cu129, Transformers 5.17.0, NCCL 2.29.7 |
| Settings in force | PyTorch default CUDA allocator; `torch.compile` on, as upstream; float32 matmul precision `high` and `cuda.matmul.allow_tf32 = True`, which upstream LRQK sets at import |
| Repository | `be7c8a2` plus uncommitted entry-point changes (diff sha256 `236dac4a93df`) |

Each run is a fresh process: same-shape warmup request, fresh state, then the measurement. No component timers
are active. The two arms run one after the other, never together, and alternate which goes first.

## Correctness

- **Stream order.** `check_lrqk_offload_order.py`, judged from its `result.json`: with the fix, the copies of K, V
  and A_K have 0 wrong elements. In the same probe the unordered upstream copy gets all 17,301,504 elements of
  K wrong. The script keeps the source tensors alive, so it isolates the producer-to-copy ordering; it is not a
  full allocator-lifetime regression test.
- **Version actually loaded.** Every run records the files it imported and the class hierarchy of its prefill
  cache. In all runs `lrqk_upstream.py`, `lrqk_attention.py` and `bench_v2q8_tp.py` match the reviewed hashes,
  and the prefill class derives from `OrderedOffloadPrefill`.
- **Smoke.** Both arms complete at 8192 and at 65536 tokens with the expected output length and finite
  log-probabilities.
- **Placement consistency, 8192 tokens, teacher forced.** The fixed K-only arm decodes 128 greedy tokens; the
  fixed K+V arm is fed exactly those tokens.

  | Cohort | K+V argmax equals the K-only token | Max abs log-probability difference |
  |---|---|---|
  | 0 | 128 / 128 | 0.0 |
  | 1 | 128 / 128 | 0.0 |
  | 2 | 128 / 128 | 0.0 |

  This matches the earlier TP8 check (top-1 100 %, maximum difference 0). No threshold was relaxed.
- **Placement consistency, 65536 tokens, free running.** In the formal runs the two arms generate identical
  tokens for every cohort, in both request and steady mode, with identical log-probabilities.

These checks cover the implementation and the placement. They say nothing about quality against Dense.

## Results

Median of the three cohorts; the range is in brackets.

| | K offload, V on GPU | K and V offload |
|---|---|---|
| Decode-ready | 15.73 s (15.70 to 15.76) | 18.07 s (17.98 to 18.36) |
| Request end to end, 128 tokens | 54.23 s (54.09 to 55.87) | 65.45 s (64.84 to 66.48) |
| Request: mean decode step | 302.9 ms (302.2 to 316.1) | 370.8 ms (369.0 to 381.1) |
| Steady: mean step | 325.5 ms (314.0 to 330.0) | 342.0 ms (337.3 to 350.4) |
| Steady: p50 step | 314.4 ms (312.2 to 320.3) | 332.5 ms (326.5 to 337.2) |
| Steady: p95 step | 390.1 ms (372.7 to 393.3) | 401.3 ms (400.1 to 414.6) |
| Steady: tokens per second | 3.07 (3.03 to 3.18) | 2.92 (2.85 to 2.96) |
| GPU resident, allocated | 26.04 GiB | 22.03 GiB |
| Prefill peak, allocated / reserved | 26.20 / 37.51 GiB | 22.31 / 31.08 GiB |
| Decode peak, allocated / reserved | 26.20 / 30.39 GiB | 22.19 / 22.44 GiB |
| Pinned host memory for offloaded state | 6.00 GiB | 12.00 GiB |
| Process RSS | 19.99 GiB | 27.93 GiB |

The memory figures are the same in all three cohorts. The request step time is the mean over the 127 decode
steps of a request; the steady figures come from the separate steady runs. They are different metrics.

Per cohort:

| Arm | Cohort | Decode-ready s | E2E s | Request step ms | Steady mean / p50 / p95 ms |
|---|---|---|---|---|---|
| K offload | 0 | 15.70 | 54.09 | 302.2 | 330.0 / 314.4 / 390.1 |
| K offload | 1 | 15.73 | 55.87 | 316.1 | 325.5 / 320.3 / 393.3 |
| K offload | 2 | 15.76 | 54.23 | 302.9 | 314.0 / 312.2 / 372.7 |
| K and V offload | 0 | 18.07 | 66.48 | 381.1 | 342.0 / 332.5 / 401.3 |
| K and V offload | 1 | 18.36 | 65.45 | 370.8 | 337.3 / 326.5 / 400.1 |
| K and V offload | 2 | 17.98 | 64.84 | 369.0 | 350.4 / 337.2 / 414.6 |

## Observations

- **K-only is faster in every metric and every cohort.** Prefill is 2.3 s shorter, the steady p50 is 18 ms (5 %)
  lower, and a 128-token request finishes 11 s sooner.
- **K+V trades GPU memory for host memory.** It holds 4.0 GiB less on the GPU and peaks 6.4 GiB lower in
  reserved memory during prefill. It pins 6 GiB more host memory and its RSS is 8 GiB higher.
- **The request and steady step times disagree in opposite directions for the two arms.** K-only: 303 ms in a
  request, 326 ms steady. K+V: 371 ms in a request, 342 ms steady. The cause is not established; the breakdown below shows where
  the two arms differ, not why the two modes do.
- **Two outlier steps, K-only steady only.** Step 112 of cohort 0 took 1225 ms and step 85 of cohort 2 took
  1220 ms. Every other step of all six steady runs is at most 450 ms. Both values are kept, which is why the
  K-only mean exceeds its p50; without them the means would be 322.9 ms and 306.9 ms. The cause is unknown. K+V
  shows no such step.

## Breakdown (diagnostic, not a latency result)

One instrumented run per arm on cohort 0 (`breakdown/`, job 240507). Every component is timed with a synchronized
wall clock, which serializes the work, so the totals differ from the uninstrumented step times above (K-only
322 ms against 314 ms steady p50; K+V 418 ms against 333 ms). Use it for where the time goes.

Decode, ms per step, summed over the 32 layers:

| Component | K offload | K and V offload |
|---|---|---|
| Factor update (LRQK decode solver) | 134.1 | 128.4 |
| `torch.cuda.empty_cache()`, once per layer | 48.3 | 126.0 |
| Fetching missed entries from host | 35.2 (K: indices 11.6, CPU gather + H2D 21.8; V: GPU gather 1.8) | 48.5 (K and V: CPU gather + H2D) |
| Hit/miss bookkeeping | 31.6 | 35.3 |
| Proxy scores and top-k | 12.9 | 12.9 |
| Attention (FlashAttention) | 6.1 | 6.3 |
| Rest of the LRQK decode call | 19.5 | 28.4 |
| Rest of the model (projections, MLP, norms, lm_head) | 33.9 | 32.2 |
| **Forward total, instrumented** | **321.7** | **418.2** |

Prefill, seconds per request:

| Component | K offload | K and V offload |
|---|---|---|
| Attention (FlashAttention) | 5.33 | 5.40 |
| Rest of the model | 6.15 | 6.19 |
| Storing K (and V) to the host pool | 2.55 | 4.85 |
| ALS factor fit | 1.52 | 1.52 |
| Rest of the LRQK prefill call | 0.41 | 0.41 |
| **Forward total, instrumented** | **15.97** | **18.38** |

- **Attention is 2 % of a decode step.** The factor update is the largest item in both arms, about 130 ms.
- **The arms differ mainly in `empty_cache()` and in the miss fetch.** `empty_cache()` costs 78 ms more per step
  with K and V offloaded, and fetching K and V through the CPU gather costs 13 ms more than fetching K alone.
- **The prefill difference is the host store.** Writing V as well as K to the host pool adds 2.3 s, which matches
  the 2.3 s difference in decode-ready time.
- The breakdown sums each component over a run, so it does not locate the two 1.2 s outlier steps.

## Trial: upstream's per-layer `empty_cache()` in decode turned off (not the baseline)

Upstream calls `torch.cuda.empty_cache()` after every layer's decode update, 32 times per step
(`lrqk_attention.py`, `DynamicLRQKCache.update`). Two trial arms, `lrqk_upstream_koff_noempty_v128` and
`lrqk_upstream_kvoff_noempty_v128`, make that call a no-op while a decode-phase update runs. Upstream's source is
not edited and prefill is untouched. One run per cell, cohort 0 only (`trial_noempty/`, job 240508); the baseline
column is the cohort-0 formal run above.

| Cohort 0 | K offload, baseline | K offload, trial | K and V offload, baseline | K and V offload, trial |
|---|---|---|---|---|
| Decode-ready | 15.70 s | 15.68 s | 18.07 s | 18.45 s |
| Request end to end | 54.09 s | 45.49 s | 66.48 s | 49.30 s |
| Request: mean decode step | 302.2 ms | 234.7 ms | 381.1 ms | 242.9 ms |
| Steady: mean step | 330.0 ms | 231.0 ms | 342.0 ms | 243.9 ms |
| Steady: p50 step | 314.4 ms | 230.2 ms | 332.5 ms | 242.1 ms |
| Steady: p95 step | 390.1 ms | 238.3 ms | 401.3 ms | 255.6 ms |
| Steady: slowest step | 1224.6 ms | 248.4 ms | 424.0 ms | 295.6 ms |
| GPU resident, allocated | 26.04 GiB | 26.04 GiB | 22.03 GiB | 22.03 GiB |
| GPU reserved during decode | 30.26 GiB | 39.08 GiB | 22.26 GiB | 36.64 GiB |
| Prefill peak reserved | 37.51 GiB | 39.08 GiB | 31.08 GiB | 36.64 GiB |

- **Same output.** The trial arms generate the same tokens as the baseline at 8K and at 64K, in request and
  steady mode, with identical log-probabilities.
- **Steady p50 drops by 27 % in both arms** (84 ms and 90 ms per step), and the request and steady step times now
  agree for each arm.
- **The cost is reserved GPU memory.** Allocated memory is unchanged, but the allocator no longer returns its
  cache to the driver: reserved memory stays at 39.1 GiB (K-only) and 36.6 GiB (K+V) through decode, against
  30.3 GiB and 22.3 GiB. That leaves about 4.5 GiB of headroom on a 44.5 GiB L40S at 64K.
- **No 1.2 s step in the K-only trial run.** This is one run; the baseline showed such a step in two of its three
  cohorts.
- These are single runs on one prompt. The baseline in this report remains upstream as is.

## Warnings and interference

- **Warnings.** Each run logs 5 or 6 warning lines: `torch._dynamo` reaching its recompile limit of 8 on one
  function, autotune messages, and deprecation notices. No run reported an out-of-memory condition.
- **CPU.** The cluster does not confine jobs to their CPUs, and six GPU jobs of another user ran on the other
  GPUs of node6. Other processes used the 24 pinned CPUs for less than 0.1 logical CPU on average and for at most
  4.3 in a single 5 s sample. LRQK itself kept about 10 of the 24 busy on average. `load.jsonl` in each run
  directory has the samples.
- **Earlier attempt.** The first smoke job (240497) failed in its first benchmark run, after the measurement, in
  the newly added code that records imported source hashes (a generated module names a file that does not
  exist). That code was fixed and the smoke was rerun with the same settings as job 240498. The log is
  `logs/smoke_240497.log`.

## Limits

- **Not a like-for-like repeat of the earlier TP1/64K runs.** Those ran on a Xeon Platinum 8558 with driver 570
  and Transformers 5.22.2, before the fix, and with known CPU contention. They reported about 369 ms per step
  for K+V and 430 ms for K-only, the opposite ordering to this run. The difference cannot be attributed to the
  stream fix.
- **No matched comparison with BasisKV or ShadowKV.** They were not rerun here. Putting these numbers next to
  theirs compares different hosts and sessions.
- **One host, three prompts, one run per cell.** The ranges above are across prompts, not repeated measurements
  of one prompt.

## Reproduce

```bash
PY=/home/yl1139/basis_artifacts/envs/vllm/basis/bin/python
MODEL=~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CUDA=/home/yl1139/basis_artifacts/cuda/12.9
# upstream LRQK at caf16293 in external/LRQK; inputs in results/rebuttal/v2q8_l40s_sparse/tp4_64k/inputs/
$PY benchmarks/system/run_lrqk_tp1_retest.py --model $MODEL --cuda-home $CUDA --phases build,check,smoke8k,consistency,probe64k
$PY benchmarks/system/run_lrqk_tp1_retest.py --model $MODEL --cuda-home $CUDA --phases formal --cohorts 0
$PY benchmarks/system/run_lrqk_tp1_retest.py --model $MODEL --cuda-home $CUDA --phases formal --cohorts 1,2
python3 benchmarks/system/summarize_lrqk_tp1_retest.py --root results/rebuttal/v2q8_l40s_sparse/tp1_lrqk_retest
```

Each benchmark process is launched as
`python -m torch.distributed.run --nproc-per-node=1 benchmarks/system/bench_v2q8_tp.py --arm <arm> --mode <mode>
--length 65536 --batch 1 --cohort <c> --inputs <prompts> --model <snapshot> --output <dir>`; the exact command and
environment of every run are in its `run.json`.

## Files

| Path | Content |
|---|---|
| `SUMMARY.md`, `runs.csv`, `summary.csv` | Generated tables: one row per run, and per-metric medians and ranges |
| `correctness-summary.json` | Source review, stream-order check, smoke rows, placement consistency |
| `manifest.json` | Host, GPU, CPU pinning, software, model and input hashes, source hashes, per invocation |
| `formal/<arm>_<mode>_b1_t65536_c<cohort>/` | `result.json`, `rank_0.json`, `run.log`, `run.json`, `load.jsonl` |
| `consistency/`, `smoke/`, `checks/` | Correctness and smoke runs; their timings are not results |
| `breakdown/` | The two instrumented diagnostic runs (cohort 0); not latency results |
| `trial_noempty/` | The trial arms without the decode `empty_cache()`: 8K smoke and cohort 0, with their own `SUMMARY.md` |
| `logs/` | Slurm logs of the three jobs and of the failed first attempt |
