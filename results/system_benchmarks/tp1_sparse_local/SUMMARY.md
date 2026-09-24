# Frozen TP1 full-scan decode timing

Llama-3.1-8B-Instruct, one L40S, TP1/B1, BF16, TF32 disabled. Dense uses full GPU-resident K/V. Basis uses Dense V128, full B16R16 Page32 routing, 2,048-token support and fused full-scan selection/packing. Basis keeps full K/V on GPU and directly indexes selected K/V; no K-slot refresh or host fetch.

Completed: 8/8; failed: 0.
Two fresh processes per method/context, 16 warmup + 128 measured steps, same physical GPU, Dense/Basis/Basis/Dense order. Component profiling is disabled. Outer CUDA events and synchronized wall timing remain enabled.

| Context | Method | CUDA p50 ms | CUDA mean ms | CUDA p95 ms | Wall tok/s | Repeat p50 ms |
|---:|:---|---:|---:|---:|---:|:---|
| 64K | dense | 35.952 | 35.956 | 35.984 | 27.80 | 35.950624;35.953985 |
| 64K | basis | 32.139 | 32.115 | 32.200 | 31.12 | 32.131920;32.159904 |
| 128K | dense | 47.455 | 47.446 | 47.517 | 21.07 | 47.493937;47.389055 |
| 128K | basis | 38.618 | 38.549 | 38.755 | 25.93 | 38.601856;38.683537 |

## Matched comparison

- 64K: Dense-local / Basis CUDA p50 = 1.119x.
- 128K: Dense-local / Basis CUDA p50 = 1.229x.

## Scope and provenance

These are steady decode measurements on one frozen teacher-forced prompt, not greedy generation or measured request E2E. Greedy selection is outside the timed interval. This does not complete the three-cohort final protocol. The long context is 131072 to match the earlier profile, not 130048.

Correctness flags and differences from the old full-scan output are in `summary.csv`. Finite logits and argmax agreement are not a downstream quality evaluation.

- Environment: `basis`, PyTorch 2.13.0+cu130, CUDA 13.0.
- Git commit: `bab891d3e108d4280eeee65e7bb6c07f86dea83a`; dirty status is in `worktree_status.txt`.
- Executed sources: `frozen_source/`; input copies: `frozen_inputs/`. Original input file sizes and modification times were checked before/after the run.
- Per-trial exact commands, raw results and logs: `raw/`. Failures are retained in `outcomes.json`.

```bash
/workspace/miniforge3/envs/basis/bin/python benchmarks/system/run_tp1_frozen_decode.py --contexts 65536 131072 --gpu 0 --basis-storage local --output-root results/system_benchmarks/tp1_sparse_local
```
