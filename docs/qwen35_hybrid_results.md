# Qwen3.5-9B Hybrid C1 results

Completed: six C1 V-only banks, six frozen-V + Wo combinations, nine PaLU banks, and dense baseline. All 22 configurations have both complete PPL evaluations. Values are WikiText-2 / C4 PPL; lower is better.

## Protocol

- Calibration: same C4 fit token IDs for C1 and PaLU, 256 × 2048. ALS held-out: 64 × 2048. KL profile: 128 × 2048; independent confirmation: 16 × 2048.
- Full WikiText-2 raw test: 146 windows, 297,047 scored tokens including the final short block. Independent C4 validation: 128 × 2048, 262,016 scored tokens. Context resets at block boundaries; no sliding overlap.
- Six encoder sweeps and decoder refits. V work FP32, Wo work FP64, BF16 factors. K and GDN recurrent-state representation remain native.
- Wo: independently recaptured frozen-V trajectories; all 8 full-attention and 24 GDN layers; TP4, 1024 → 512 per source, total gathered width 2048.
- Environment: lowrank. Local A100 processes, two CPU threads per process, no Slurm. Final two-sided pipelines used GPU 0 for V64, GPU 2 for V80, GPU 5 for V96.
- Model revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a. Config, original weight shards, windows and factor hashes are recorded in final_summary.json.

Dense baseline: **8.6511 / 11.3291**.

## C1 results

| V rank | Uniform V | Uniform V + Wo | Two-sided V | Two-sided V + Wo |
|---|---|---|---|---|
| 64 | 9.5704 / 12.3024 | 9.6221 / 12.8608 | 8.5679 / 12.1253 | 8.9572 / 12.6917 |
| 80 | 9.1565 / 12.0088 | 9.0877 / 12.5339 | 8.5129 / 11.8779 | 8.7167 / 12.4060 |
| 96 | 8.7765 / 11.8161 | 8.7044 / 12.3182 | 8.5274 / 11.7284 | 8.5444 / 12.2326 |

## PaLU results

Official Fisher allocation and block32 rounding are retained. Fisher uses the repository official-PaLU double-label-shift allocation loss; PPL uses ordinary one-token-shift NLL.

| Nominal V rank | MLRD | GLRD2 | GLRD4 |
|---|---|---|---|
| 64 | 10.7685 / 13.1368 | 9.3953 / 12.3951 | 8.7214 / 11.8871 |
| 80 | 9.4295 / 12.5137 | 9.0635 / 12.0662 | 8.8168 / 11.7287 |
| 96 | 9.3369 / 12.3077 | 8.9543 / 11.8963 | 8.6507 / 11.6183 |

| Bank | Realized V retention |
|---|---|
| palu_mlrd_v64 | 25.0000% |
| palu_mlrd_v80 | 31.2500% |
| palu_mlrd_v96 | 35.9375% |
| palu_glrd2_v64 | 25.0000% |
| palu_glrd2_v80 | 32.0312% |
| palu_glrd2_v96 | 38.2812% |
| palu_glrd4_v64 | 25.0000% |
| palu_glrd4_v80 | 31.2500% |
| palu_glrd4_v96 | 37.5000% |

## Two-sided allocation

Layer order: 3, 7, 11, 15, 19, 23, 27, 31. Independent KL confirmation does not reselect schedules.

| Anchor | Layer ranks | KL difference (two-sided minus uniform) | Wo PPL change (Wiki / C4) |
|---|---|---|---|
| 64 | 32, 48, 32, 48, 96, 48, 128, 80 | -0.010785 | +4.544% / +4.671% |
| 80 | 48, 48, 48, 64, 128, 96, 128, 80 | -0.008200 | +2.393% / +4.446% |
| 96 | 64, 80, 64, 80, 128, 128, 128, 96 | -0.005939 | +0.199% / +4.299% |

## Numerical findings and limits

- All 56 V candidates have six encoder and seven decoder history entries. Of 728 blocks, 279 missed the requested residual tolerance; 272 hit the iteration cap. Maximum true relative residual: 0.886078. These are finite-iteration damped ALS results, not exact normal-equation solutions.
- Initial Wo FP32 failed the existing SPD guard. All final Wo banks use FP64 with unchanged damping and six sweeps; failed logs remain preserved. Final encoder maximum residuals are recorded in six wo_*_audit.json files.
- C1 uses no optimizer/backward. PaLU Fisher uses backward for allocation only.
- Local error reports retain the cross term on common native-dense inputs. These differ from frozen-V recaptured Wo fit statistics and whole-model PPL.
- Runtime is a single-process Private AllGather equivalent. No Qwen3.5 multi-GPU latency or throughput speedup was measured.
- Theoretical BF16 TP4 ring output bytes/token/rank: dense AllReduce 12,288 → compressed AllGather 3,072, a 75% reduction; 50% versus uncompressed AllGather. Only output collective traffic is counted.
- V cache retention: 25% / 31.25% / 37.5% at rank 64 / 80 / 96. Total token-growing K+V retention: 62.5% / 65.625% / 68.75%. GDN state and model weights are outside this calculation.
- One calibration draw and one evaluation sample were used. Better WikiText PPL than dense in some arms is an observation, not evidence of general capability gains.

## Artifacts and execution

- Full machine-readable results, actual evaluation commands, factor hashes, 56 candidate diagnostics, 51 KL profiles and 3 confirmations: results/q35_hybrid/final_summary.json.
- V and PaLU banks: results/q35_hybrid/banks/.
- Wo banks: results/q35_hybrid/wo_{uniform,twosided}_v{64,80,96}/wo_bank.pt. Load each together with its corresponding V bank and original model.
- Local error reports: results/q35_hybrid/c1_{uniform,twosided}_v{64,80,96}_local_errors.json.
- Logs: results/q35_hybrid/logs/. Original failures and resource-driven restart provenance remain available.
- Staged commands and implementation details: docs/qwen35_gated_v_then_private_ag.md.

The following commands show the final V80 pipeline; V64/V96 substitute their corresponding paths and physical GPU. Actual stdout/stderr were redirected to corresponding logs. Existing outputs are exclusive-created and must not be overwritten.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export PYTHONPATH=results/q35_hybrid/deps:.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=2
python -u -m evaluation.qwen35_hybrid_wo capture --bank results/q35_hybrid/banks/c1_twosided_v80.pt --output results/q35_hybrid/wo_twosided_v80_moments
python -u -m evaluation.qwen35_hybrid_wo fit --bank results/q35_hybrid/banks/c1_twosided_v80.pt --moments results/q35_hybrid/wo_twosided_v80_moments --output results/q35_hybrid/wo_twosided_v80 --tp-size 4 --work-dtype float64
python -u -m evaluation.run_qwen35_hybrid evaluate --bank results/q35_hybrid/banks/c1_twosided_v80.pt --wo-bank results/q35_hybrid/wo_twosided_v80/wo_bank.pt --output results/q35_hybrid/c1_twosided_v80_wo_ppl.json
python -u -m evaluation.qwen35_hybrid_local_errors --bank results/q35_hybrid/banks/c1_twosided_v80.pt --wo-bank results/q35_hybrid/wo_twosided_v80/wo_bank.pt --output results/q35_hybrid/c1_twosided_v80_local_errors.json
python -m evaluation.summarize_qwen35_hybrid --output results/q35_hybrid/final_summary.json
```

No Git commit, push or upload has been performed.
