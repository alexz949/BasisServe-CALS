# TP8 Full-Scan Smoke Verification

Date: 2026-09-23 UTC. Environment: conda `basis`, PyTorch `2.13.0+cu130`,
8 x NVIDIA L40S. Worktree: `/workspace/BasisServe-CALS-opt`.
Routing mode: `full_scan_b16r16_persistent_slots`.

This run removes coarse screening and the 512-candidate limit from the
active TP8 path, while retaining fused cache append and final selection /
slot planning. It is not a formal timing or model-quality evaluation.

## Commands

```bash
CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python -m pytest tests/test_tp8_full_scan_routing.py \
  tests/test_slot_indexed_attention.py tests/test_uniform_allgather_timing.py -q

CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_llama31_8b_tp8_decode_grid.py \
  --arms basis_joint --contexts 4096 65536 --batches 1 8 --cohorts 0 \
  --conditioning-steps 4 --measure-steps 8 --tag full_scan_smoke \
  --output-root results/system_benchmarks/tp8_full_scan/smoke
```

The launcher records each exact torchrun command in
`smoke/decode_grid_trials.json`. Local logs: `tests.log`, `smoke.log`, and
`smoke/launcher_*.log` (log files are ignored by the repository).

## Results

Kernel/unit regressions: **26 passed in 7.97 seconds** on the final run.
The first run had 25 passes and one incorrect cache hit-count test
expectation, corrected to include all still-resident tokens. Both runs
are preserved in `tests.log`; no kernel change was needed for that failure.

Integration: **4/4 trials completed, 32/32 rank JSONs validated**.
Across ranks, generated token sequences agree exactly and have 13 tokens
each. Every result records the new routing mode and eight finite,
positive measured step times.

Short-run timing diagnostics only, with four conditioning steps and eight
measured steps per trial; do not report these as paper benchmark results:

| Prompt tokens | Batch | Mean decode step (ms) | Wall throughput (tokens/s) |
|---:|---:|---:|---:|
| 4096 | 1 | 24.6083 | 41.09 |
| 4096 | 8 | 25.8149 | 311.84 |
| 65536 | 1 | 24.7184 | 40.70 |
| 65536 | 8 | 24.0348 | 333.47 |

Warnings: container restrictions deny NUMA `set_mempolicy`; strict host
memory placement is not guaranteed. NCCL barriers warn about inferred
device selection. No integration trial failed. No quality evaluation or
new formal performance grid was run. Old `tp8_path_opt` measurements
belong to the removed two-stage algorithm and were not overwritten.

Implementation details: [full-scan routing](../../../docs/tp8_full_scan_routing.md).
