# Llama-3.1-8B TP8: Quest CUDA vs BasisKV

## Result

Completed on 2026-09-26: **6/6 successful trials, 48/48 complete rank records**,
with no OOM or non-finite-logit failure. Context 65,536, batch 1, TP8 on 8 x L40S.

| Method | Mean decode ms/step (sample SD) | Mean wall tokens/s (sample SD) |
| --- | ---: | ---: |
| Quest upstream CUDA + local GQA/BF16 patch | 25.8355 (0.1445) | 39.0870 (0.2359) |
| BasisKV Joint V96, full scan, historical K offload | 24.4464 (0.0719) | 41.1633 (0.1389) |

Basis is **1.0568x** faster by the ratio of the two mean step latencies,
or **5.377% lower latency**. Mean wall throughput is 5.312% higher.
These are full-model steady-decode results, not attention-only or end-to-end
request latency. The result does not support the earlier 1.43x comparison made
using the slower Triton Quest port and historical Basis measurements.

| Cohort | Prompt sample | Quest ms/step | Basis ms/step | Quest / Basis |
| --- | --- | ---: | ---: | ---: |
| 0 | sample_000 | 25.844749 | 24.455758 | 1.056796x |
| 1 | sample_023 | 25.975217 | 24.513101 | 1.059646x |
| 2 | sample_046 | 25.686643 | 24.370226 | 1.054017x |

## Measurement

- Model: Llama-3.1-8B-Instruct, BF16; conda environment `basis`.
- Three distinct frozen prompt cohorts, identical prompt inputs within each pair.
- Fresh processes for each trial; fixed serial order Quest then Basis per cohort.
- Each trial prefills 65,536 tokens, runs 16 conditioning decode forwards, then
  measures 128 additional decode forwards. Context grows during decoding.
- Each step uses the maximum CUDA-event interval across TP ranks. The table
  reports the arithmetic mean of three trial means and their sample SD, n=3.
  The eight ranks are not independent repetitions.
- Throughput is 128 divided by the maximum measured-loop wall time across ranks,
  averaged over trials. It is not the reciprocal of mean rank-max event latency.
- The measured loop includes full model forward, token selection, finite-logit
  bookkeeping and token recording. Component profiling and tracing are disabled.
- No Dense rerun, output-token equivalence test or quality evaluation was run.

## Protocol Differences

Quest uses the upstream CUDA append, min/max estimate, RAFT radix top-k and paged
attention kernels with a local GQA/BF16 patch. Each rank has four query heads and
one physical KV head. It retains full uncompressed GPU K128/V128, page size 16,
127 historical pages plus the newest page per query head, and Dense attention
for the first two layers. The nominal budget is 2,048 tokens per query head;
different query heads can select different pages. The newest page can be partial.

Basis uses V96, full-scan B16R16 routing, page size 32, 62 selected historical
pages plus 64 recent tokens: 2,048 physical tokens shared by the KV group.
Historical exact K is in pinned host memory with persistent GPU slots; V96 is
GPU resident. There is no two-stage shortlist. Its latent communication and
value decoder are included in full-model timing.

Thus this is a comparison of two configured model execution paths, not an
equal-quality or identical-physical-support routing-kernel ablation. Quest is
the upstream CUDA operator path with a local patch and TP8 integration, not an
unmodified upstream end-to-end application. See
[native patch documentation](../native/README.md) and
[upstream patch](../native/quest_gqa.patch).

## Memory and Warnings

Across all three cohorts, maximum-rank decode peak allocated memory is
**3.80585 GiB for Quest** and **4.16538 GiB for Basis**. Basis additionally records
538,050,560 bytes of persistent host exact-K storage per rank. These are allocator
and explicit-buffer metrics, not total process/device memory. K offload does not
imply lower total allocated GPU memory at this B1 point; weights, factors,
communication buffers and workspaces also contribute. No capacity claim follows.

Both methods log `set_mempolicy: Operation not permitted`; host NUMA placement is
unverified. NCCL also warns that `barrier()` infers the current device. Neither
warning prevented completion. No SHA256 was computed or checked; prompt matching
was verified using file paths, cohort fields and sample IDs.

## Reproduction

Working directory: `/workspace/BasisServe-CALS-opt`. Slurm is unavailable on this
machine; the user approved direct serial execution. Basis first passed a 64K/B1
smoke with 4 conditioning and 8 measured steps on all eight ranks (24.5879 ms).
Quest native operator tests and 4K/64K smoke are documented in `../native/`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_llama31_8b_tp8_decode_grid.py --arms quest_native basis_joint --contexts 65536 --batches 1 --cohorts 0 1 2 --conditioning-steps 16 --measure-steps 128 --tag paired --output-root results/system_benchmarks/quest_tp8/paired > results/system_benchmarks/quest_tp8/paired.log 2>&1
```

Raw data: `paired_c{cohort}_{arm}_p65536_b1_r{cohort}/rank{rank}.json`.
Exact child commands and completion status: `decode_grid_trials.json`.
Per-trial logs: `launcher_{arm}_p65536_b1_c{cohort}.log`; queue log: `../paired.log`.
Small tabular export: [trials.csv](trials.csv). Local integration and Basis hot-path
source snapshots are under `source/`; upstream patch sources are in `../native/`.

## Published Raw Data

The [HF archive](https://huggingface.co/alexz949/BasisServe-CALS/resolve/61958dd0106b32802a628c1596c54d95e6283a85/results/system_benchmarks/quest_tp8/raw.tar.gz)
contains all paired rank JSON, launcher logs, Basis smoke, native validation
logs and existing source snapshots. Its
[202-file inventory](https://huggingface.co/alexz949/BasisServe-CALS/blob/61958dd0106b32802a628c1596c54d95e6283a85/results/system_benchmarks/quest_tp8/raw_manifest.json)
lists repository-relative archive member paths. Raw files and source snapshots
are archived on HF rather than copied into this GitHub summary directory.
Archived Markdown preserves the pre-publication status; these GitHub summaries
include the subsequent publication links. All three archives were verified by
remote path and byte size without an explicit SHA256 validation.
