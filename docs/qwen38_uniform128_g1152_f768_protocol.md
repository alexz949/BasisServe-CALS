# Qwen3.8-27B: prioritize Uniform V128

The user requested Uniform first after launching the Two-sided rank-bank pipeline. The prior driver and its three rank32 fitting workers were stopped before any candidate completed. All completed capture/teacher artifacts and original logs were preserved. The original rank-bank pipeline is no longer running automatically.

Use all 16 full-attention layers at exactly rank128, followed by frozen-V recapture and source-private Wo fitting at GDN1152 / Full768. This requires 16 candidate fits instead of 160 and skips Two-sided allocation and its 33 terminal-KL profiles. Same pinned model, calibration splits, gate checks, six V encoder sweeps, PCG cap200, separable preconditioning, BF16 exports, and FP64 Wo fitting as described in `docs/qwen38_v128_g1152_f768_protocol.md`.

Model weights load from the standard HF cache. Direct execution uses `lowrank` and the already audited project-local Transformers dependency. GPU2/5/6 fit layer shards with two OMP/MKL threads each. GPU6 performs frozen-V Wo capture with FP64 Gram accumulation on CPU. CPU assembles banks.

```bash
bash scripts/run_qwen38_uniform128_g1152_f768.sh \
  >> results/q38_hybrid/logs/uniform128_pipeline.log 2>&1
```

The script and driver log contain every exact Python command. V shard layer lists are `3,15,27,39,51,63`, `7,19,31,43,55`, and `11,23,35,47,59`, all with `--ranks 128 --chunk-rows 2048 --linear-max-iter 200 --encoder-preconditioner separable`.

- Shared native capture: `results/q38_hybrid/capture/`.
- Rank128 factors: `results/q38_hybrid/factors/lXX_r128.pt` (reusable for a later Two-sided run).
- Uniform bank: `results/q38_hybrid/banks/c1_uniform_v128.pt`.
- Uniform-specific moments: `results/q38_hybrid/wo_uniform128_moments/`.
- Uniform-specific Wo bank: `results/q38_hybrid/wo_uniform128_g1152_f768/wo_bank.pt`.
- Logs: `results/q38_hybrid/logs/uniform128_*.log`.

No downstream benchmark is launched by this script. This is a launch protocol, not a statement that fitting or convergence is complete. Script syntax and `git diff --check` passed before launch; the earlier 19 tests and real-checkpoint smoke remain applicable because this switch changes orchestration, rank selection and output paths only.
