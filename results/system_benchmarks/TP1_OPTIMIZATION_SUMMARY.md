# TP1 Sparse Decode: Latest Optimization Results

## Main Findings

The latest completed experiment changes the full-scan router from 8 warps to
16 warps per block. It does not introduce a coarse shortlist, change the
scoring algorithm, or modify MLP. All eligible historical pages remain scored.
The candidate is isolated; it has not replaced the production kernel.

Environment: `basis`, one NVIDIA L40S, Llama-3.1-8B-Instruct, TP1/B1, BF16,
TF32 off, Dense V128, B16R16/Page32 routing, 2,048-token attention support.
K-offload keeps historical exact K in pinned CPU memory, Dense V on GPU,
and reuses persistent GPU K slots. GPU-local keeps both K and V on GPU.

## Paired Full-Model Results

CUDA-event pooled median latency, ms/token. Each row includes two repeats
per arm, executed baseline/candidate/candidate/baseline in fresh processes.
Each trial has 16 warmup and 128 measured teacher-forced decode steps.

| Storage | Context | 8-warp baseline | 16-warp candidate | Latency reduction |
|---|---:|---:|---:|---:|
| GPU-local | 65,536 | 32.163 | 31.800 | 1.13% |
| GPU-local | 131,072 | 38.448 | 37.877 | 1.49% |
| K-offload | 65,536 | 34.232 | 33.871 | 1.05% |
| K-offload | 131,072 | 40.615 | 40.031 | 1.44% |

All 16 paired trials completed. Each has finite logits, 32/32 exact
first-warmup router-score audits, and 32/32 attention validations. All 144
recorded argmax tokens match the baseline within each storage/context group.
These checks are not a downstream quality evaluation or proof of bitwise
equality of all logits. Synthetic router screening passed 90 exact-score and
selected-page checks, in addition to 72 earlier smoke checks.

K-offload peak PyTorch allocated GPU memory is unchanged between arms:
24.033 GiB at 64K and 32.939 GiB at 128K. Last-step cache hit rates differ by
less than 0.03 percentage points; cache behavior is not claimed bitwise equal.
See [local details](tp1_router_config/SUMMARY.md) and
[offload details](tp1_router_config/offload/SUMMARY.md).

## Dense-Local Reference

Dense-local numbers come from the preceding
[GPU-local experiment](tp1_sparse_local/SUMMARY.md), not a new matched run.
The speedups below are cross-run references, not additional paired results.

| Context | Dense-local ms/token | Sparse local, 16 warps | Speedup | Sparse K-offload, 16 warps | Speedup |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 35.952 | 31.800 | 1.131x | 33.871 | 1.061x |
| 131,072 | 47.455 | 37.877 | 1.253x | 40.031 | 1.185x |

These are full-model steady decode timings, not isolated attention latency,
TTFT, or complete request E2E. They use one frozen teacher-forced prompt,
not the final three-cohort greedy protocol. Greedy argmax selection is outside
the timed interval. No new Dense or downstream quality evaluation was run.

## Other Outcomes

- K-slot refresh: warp-aggregated planning and a 128-thread fetch candidate
  passed 180 stateful validation checks in each of smoke and micro screening,
  but gave no useful overall improvement. Neither was adopted; no full-model
  trial was launched for them. See [details](tp1_slot_refresh/SUMMARY.md).
- Hardware-counter profiling was blocked by `ERR_NVGPUCTRPERM`.
  No achieved SM utilization or hardware bottleneck is established.
- The 1,024-candidate two-stage experiment has not run. There is no measured
  speed, recall, or quality result for it. It is excluded from the tables.
- Earlier two-stage 512-page results are algorithmically different and are
  excluded. Their approximately 2.8x figure referred to an attention block,
  not full-model speedup or the current full-scan path.

## TP8 Routing Clarification

There were two different 144-trial TP8 grids:

| Result directory | BasisKV routing | Two-stage coarse shortlist? |
|---|---|---|
| `llama31_8b_tp8_combined` | `two_stage_512_persistent_slots` | Yes |
| `llama31_8b_tp8_full_scan` | `full_scan_b16r16_persistent_slots` | No |

The final frozen TP8 BasisKV Joint + ALS results are the second grid:
132 successful trials and 12 prefill OOMs. BasisKV scans all eligible pages
with B16R16 scores before final sparse support selection. ALS-full uses full
attention, not this two-stage shortlist. No coarse shortlist does not mean
BasisKV attends to every token: its final support remains sparse.
See the [published TP8 summary](https://github.com/alexz949/BasisServe-CALS/blob/main/results/system_benchmarks/llama31_8b_tp8_full_scan/RESULTS_SUMMARY.md).
Do not combine the two TP8 versions into a pure kernel-only speedup claim.

## Commands

Executed from the repository root in the `basis` environment, directly on
GPU 0 under the user's authorization. These commands document completed runs;
preparing this summary did not launch new experiments.

```bash
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9

/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_router_config.py --phase paired --config w16u1 \
  > results/system_benchmarks/tp1_router_config/paired.log 2>&1

/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_router_config.py --phase paired \
  --config w16u1 --basis-storage offload \
  > results/system_benchmarks/tp1_router_config/offload/paired.log 2>&1

/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_slot_refresh.py --phase micro --cuda-graph \
  > results/system_benchmarks/tp1_slot_refresh/micro.log 2>&1
```

The detailed summaries preserve per-repeat metrics, validation scope, and
artifact locations. Full raw logs, frozen inputs, and generated binaries are
not included in the proposed small GitHub results upload.
