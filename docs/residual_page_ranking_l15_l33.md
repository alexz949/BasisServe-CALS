# Page-boundary repair: layers 15 and 33, all GQA groups

## Protocol and execution

User requested the previously queued A100 expansion be run on L40S. Job 8300760 array 0–15, concurrency four, completed successfully on lovelace; every element exited 0:0 in 10–12 seconds including startup. One L40S, two CPUs and 32 GiB host RAM per worker; basis environment. No original checkpoints or prior A100 smoke outputs were overwritten. The temporary submission script was removed. Logs: `logs/page-rank-l40s-8300760_{0..15}.out` and `.err`.

Only layer/group coverage changed: layers 15 and 33, eight groups per layer. Each group retains fit windows 0/1, diagnostic windows 64/65, four queries at positions 26623/28671/30719/32767, closed-form Base16, initial terminal-Q32 Fisher R8, Page32/B2048/pinned page0, two E/U alternating sweeps, margin 0.05, damping 0.01 and up to eight wrong-page pairs per query. Actual BF16 selector coverage and fixed-pair hinge loss gate fit updates; diagnostic data never control acceptance. This remains an offline dense-teacher cache diagnostic, not an all-layer bank or RULER evaluation.

Algorithm and derivative details are in [the original smoke report](residual_page_ranking_smoke.md). Five core tests passed before expansion. All sixteen factor artifact hashes and result identities were verified; all sixteen runs satisfy non-decreasing aggregate fit mass. Production-sidecar score equality is asserted within every run. Individual program wall times were about 1.0–1.8 seconds; peak allocated GPU memory was below 0.184 GiB. These are small cached-data runs, not full calibration timing estimates.

## Group-averaged teacher mass coverage

Each group has the same window/query/head count, so arithmetic means across eight groups are equally weighted means across these observations. The observation count does not provide independent-document confidence intervals: there are only two diagnostic windows.

| Layer | Fit before | Fit after | Diagnostic before | Diagnostic after | Diagnostic change, pp | Diagnostic groups improved / regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 93.3861% | 93.7735% | 93.2512% | 93.2088% | -0.0424 | 2 / 6 |
| 33 | 89.5770% | 90.3516% | 92.7950% | 92.8958% | 0.1008 | 5 / 3 |

Diagnostic non-sink coverage: layer 15, 89.4487% → 89.3617% (-0.0871 pp); layer 33, 92.5084% → 92.6142% (+0.1058 pp).

## Per-group changes

| Layer | Group | Fit full mass change, pp | Diagnostic full mass change, pp | Diagnostic non-sink change, pp |
| --- | --- | ---: | ---: | ---: |
| 15 | 0 | 0.514430 | -0.128305 | -0.128722 |
| 15 | 1 | 0.197792 | -0.022507 | -0.093406 |
| 15 | 2 | 0.746202 | -0.146508 | -0.149179 |
| 15 | 3 | 0.880444 | -0.038308 | -0.222695 |
| 15 | 4 | 0.253141 | 0.024831 | 0.051969 |
| 15 | 5 | 0.066054 | -0.028145 | -0.029939 |
| 15 | 6 | 0.128382 | -0.011885 | -0.017524 |
| 15 | 7 | 0.312698 | 0.011796 | -0.107074 |
| 33 | 0 | 0.220746 | 0.247985 | 0.264883 |
| 33 | 1 | 0.805259 | 0.062406 | 0.062937 |
| 33 | 2 | 0.845438 | -0.239855 | -0.247729 |
| 33 | 3 | 1.488167 | -0.424689 | -0.464362 |
| 33 | 4 | 1.070130 | 0.217509 | 0.218672 |
| 33 | 5 | 0.538683 | -0.391841 | -0.410253 |
| 33 | 6 | 0.062597 | 0.044006 | 0.048864 |
| 33 | 7 | 1.165861 | 1.291239 | 1.373565 |

## Scope of the finding

All groups improve on fit, but this is constrained by the acceptance gate and is not independent evidence of generalization. Diagnostic full mass improves in 7/16 groups and regresses in 9/16. Layer 33's positive average is strongly influenced by group 7 (+1.2912 pp). The initial single-group result therefore does not establish a broad improvement. This tiny-data expansion neither establishes task-accuracy benefit nor identifies the cause of diagnostic regressions.

## Artifacts and command

[Machine-readable summary](../results/evaluation/page_rank_l15_l33/summary.json). Each `results/evaluation/page_rank_l15_l33/l{layer}_g{group}` directory contains its detailed `result.json` and single-group `group_factors.safetensors`. These are not installed into a deployment bank. No RULER job or GitHub write occurred.

Working directory `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. In the command below `ranking_layer` is 15 or 33 and `ranking_group` is 0–7; task indices 0–7 map to layer 15 and 8–15 to layer 33.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/smoke_residual_page_ranking.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_q32_r8 --query-capture results/calibration/q32_terminal8k --layer "$ranking_layer" --group "$ranking_group" --output-dir "results/evaluation/page_rank_l15_l33/l${ranking_layer}_g${ranking_group}"
```

