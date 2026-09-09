# Window-major residual Fisher statistics

## Implemented change

`build_multi_query_statistics` now iterates over documents before queries. For each document it transfers the captured K/V rows once, computes C1 Value codes once, and computes Base pre-K, RoPE, and the post-RoPE residual once per Base alternative. Each query still has its own causal prefix, exact-teacher softmax, page mass, and Page-Fisher Gram.

The production path no longer calls the single-query statistics builder repeatedly. The single-query builder remains in use elsewhere and serves as the independent oracle in tests and the benchmark.

Unchanged semantics:

- Query-position-major, document-minor output order and pairing.
- Equal weighting of window/query examples.
- Separate Q heads and GQA groups; no query averaging.
- Exact causal prefix, pinned-prefix exclusion before softmax, partial-page handling.
- Original teacher-energy and per-query reconstruction diagnostics.
- Page-Fisher kernel, BCD solver, residual rank, Base factors, and C1 factors.

Final CPU Gram storage is preallocated directly. There is no list of complete per-query Grams followed by concatenation into a second complete buffer. Per-query/group Fisher computation remains sequential; this change does not claim fused or fully batched QK/Fisher kernels.

The fitting protocol now records window-major execution. Existing checkpoints and their manifests were not overwritten or refitted. No new query-layout experiment or RULER run was launched for this optimization.

## Verification

Nineteen small CPU regression tests passed in `basis`. Tests cover comparison with separate single-query oracles, Fisher loss additivity, query ordering, causal and pinned masks, partial pages, widely spaced/unsorted Q positions, multiple Base alternatives, input immutability, and feature-computation call counts. For two documents and two Base alternatives, the builder computes Value codes twice and Base/RoPE four times, independent of Q count.

GPU smoke job `8300690` ran on one NVIDIA A100 80 GB in `gpu_small`, using two CPUs and the `basis` environment. It completed with exit 0:0 in 29 seconds. The existing Transformers RotaryEmbedding `device` deprecation warning was present; no numerical or runtime failure occurred.

The smoke used real 32768-token captures, Q32 terminal-8K positions, two fit windows, layers 0 and 33, and three repetitions per implementation. TF32 was enabled to match the fitting path. Run order alternated. Input materialization occurred before timing, and GPU synchronization bracketed each timed builder call.

| Layer | Previous query-major median | Window-major median | Statistics-only speedup | Gram max absolute difference |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.754685 s | 0.240988 s | 7.2812× | 0 |
| 33 | 1.749256 s | 0.240112 s | 7.2852× | 0 |

Both layers had zero relative Frobenius Gram difference. Query rows, teacher energy, reconstruction diagnostics, and Fisher loss passed comparison checks. The saved-factor Fisher losses were identical: 64.1807327271 for layer 0 and 537.5581054688 for layer 33.

These measurements cover only statistics construction. They exclude BCD and RULER, use A100 rather than L40S, and do not establish a 7.28× speedup for full fitting. No full 64/16-window refit timing was measured. Exact equality on these examples is not a promise of bitwise equality for every GEMM shape/backend.

## Command and artifacts

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. Executed through Slurm:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/benchmark_residual_statistics.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/mse_base_q32_r8 \
  --query-capture results/calibration/q32_terminal8k \
  --layers 0,33 --windows 2 --repeats 3 \
  --output-dir results/evaluation/fisher_window_major_smoke
```

- [Production builder](../evaluation/fit_qwen3_8b_q8_fisher_residual.py).
- [Regression tests](../tests/test_q8_fisher_residual.py).
- [Oracle/timing smoke](../evaluation/benchmark_residual_statistics.py).
- [Measured result](../results/evaluation/fisher_window_major_smoke/result.json).
- Logs: `logs/fisher-window-smoke-8300690.out` and `.err`.

The temporary submission file was removed after submission. No commit or push was performed.

## Query-position experiment context

Spreading Q within each existing window and adding more independent document windows are different changes. The builder now supports arbitrary distinct causal Q positions mathematically; capture/fitting protocol support for any newly chosen layout must still be set explicitly.

A prior full-context Q16 experiment exists: positions 2047, 4095, ..., 32767, with both Adam Base and residual refitted. Its RULER score was 79.1477%, versus 80.4356% for the terminal-query comparison. All 36 Base maps changed, so that experiment does not isolate residual query-position coverage.

It is distinct from holding the current closed-form Base fixed and changing only Q32 residual positions. The latter full-context experiment has not been run as part of this change. Earlier Q positions also expose shorter causal Key histories; moving Q out of terminal 8K changes both position coverage and the prefix-length distribution. With B2048, queries whose entire prefix fits within 2048 tokens do not require sparse page selection.

Prior record: [Uniform-32K Q16 Base and residual experiment](uniform32k_q16_base_fisher_protocol.md).
