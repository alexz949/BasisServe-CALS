# Qwen3-8B residual attention-mass recall

Frozen C1-V80 + Base16; Page32, B2048 including page0; uniform R8 versus the frozen average-R8 schedule.
C4 confirmation indices 64–79: 16 packed 32768-token windows, final 128 queries, all 36 layers and 32 query heads.
Shared full-attention C1 teacher Q/K/latent inputs; sparse outputs do not feed back into this diagnostic.

Mass probabilities use FP32 exact QK from BF16 cached activations. Routing uses the unchanged native BF16 selector.
Non-sink recall conditions on tokens outside page0. Unsupported rows are excluded, not treated as zero.
P01/P10/median pool query-head observations; they are not independent-window confidence intervals.

| Metric | Uniform mean | Adaptive mean | Uniform P01 | Adaptive P01 | Uniform P10 | Adaptive P10 |
|---|---:|---:|---:|---:|---:|---:|
| mass | 0.86151771 | 0.86234851 | 0.28393322 | 0.27299566 | 0.63811126 | 0.63838464 |
| non_sink | 0.83567803 | 0.83684502 | 0.18610248 | 0.18795598 | 0.58963907 | 0.58945721 |

## Per-layer means

| Layer | Adaptive rank | Uniform mass | Adaptive mass | Uniform non-sink | Adaptive non-sink |
|---|---:|---:|---:|---:|---:|
| 0 | 8 | 0.87135454 | 0.87135454 | 0.87134595 | 0.87134595 |
| 1 | 4 | 0.74679686 | 0.74065777 | 0.74662653 | 0.74048690 |
| 2 | 8 | 0.69170506 | 0.69170506 | 0.69092298 | 0.69092298 |
| 3 | 4 | 0.61307512 | 0.59745231 | 0.61100680 | 0.59524713 |
| 4 | 4 | 0.69998064 | 0.69471160 | 0.69828248 | 0.69296699 |
| 5 | 8 | 0.74036595 | 0.74036595 | 0.73923969 | 0.73923969 |
| 6 | 4 | 0.79195533 | 0.78654001 | 0.79150370 | 0.78607121 |
| 7 | 16 | 0.90066132 | 0.90701058 | 0.70787160 | 0.72430255 |
| 8 | 4 | 0.95260338 | 0.95079845 | 0.90381918 | 0.90046240 |
| 9 | 16 | 0.91722405 | 0.92382328 | 0.79227625 | 0.80650195 |
| 10 | 4 | 0.96116969 | 0.96014341 | 0.92452167 | 0.92270100 |
| 11 | 8 | 0.96850352 | 0.96850352 | 0.93497103 | 0.93497103 |
| 12 | 4 | 0.93991223 | 0.93721581 | 0.89752637 | 0.89372120 |
| 13 | 8 | 0.90010510 | 0.90010510 | 0.79978141 | 0.79978141 |
| 14 | 4 | 0.95524981 | 0.95333470 | 0.94002328 | 0.93773940 |
| 15 | 4 | 0.92642114 | 0.92221607 | 0.90602343 | 0.90069035 |
| 16 | 4 | 0.91215496 | 0.90785752 | 0.86934659 | 0.86332200 |
| 17 | 8 | 0.86133568 | 0.86133568 | 0.85148115 | 0.85148115 |
| 18 | 8 | 0.90050320 | 0.90050320 | 0.89770563 | 0.89770563 |
| 19 | 4 | 0.89384448 | 0.88569107 | 0.89089642 | 0.88254412 |
| 20 | 16 | 0.87848463 | 0.89965840 | 0.87688870 | 0.89840268 |
| 21 | 4 | 0.88247608 | 0.87369618 | 0.87832049 | 0.86870303 |
| 22 | 8 | 0.86587667 | 0.86587667 | 0.85881876 | 0.85881876 |
| 23 | 8 | 0.87937329 | 0.87937329 | 0.87585835 | 0.87585835 |
| 24 | 16 | 0.85804055 | 0.88106882 | 0.85500473 | 0.87850544 |
| 25 | 4 | 0.87014975 | 0.86204898 | 0.86352952 | 0.85513485 |
| 26 | 8 | 0.84939449 | 0.84939449 | 0.84047763 | 0.84047763 |
| 27 | 4 | 0.86031419 | 0.85268308 | 0.85396163 | 0.84542285 |
| 28 | 4 | 0.86298116 | 0.84913545 | 0.84688723 | 0.83161955 |
| 29 | 16 | 0.84910624 | 0.88235571 | 0.84383107 | 0.87840706 |
| 30 | 8 | 0.83703598 | 0.83703598 | 0.81832108 | 0.81832108 |
| 31 | 8 | 0.86974270 | 0.86974270 | 0.86180610 | 0.86180610 |
| 32 | 16 | 0.85313579 | 0.85821967 | 0.84035635 | 0.84503200 |
| 33 | 16 | 0.85940084 | 0.87645585 | 0.85255189 | 0.87032962 |
| 34 | 16 | 0.86165314 | 0.87493395 | 0.82969873 | 0.84652165 |
| 35 | 4 | 0.93255018 | 0.93154152 | 0.82292469 | 0.82085485 |

## Paired window means

Deltas are adaptive minus uniform; positive means higher retained mass.

| Window | Delta mass | Delta non-sink |
|---|---:|---:|
| 64 | +0.00233810 | +0.00278993 |
| 65 | +0.00127265 | +0.00173735 |
| 66 | +0.00076762 | +0.00116089 |
| 67 | +0.00220879 | +0.00256449 |
| 68 | +0.00009393 | +0.00045939 |
| 69 | +0.00252796 | +0.00307449 |
| 70 | -0.00018153 | +0.00016220 |
| 71 | +0.00082340 | +0.00129917 |
| 72 | +0.00025194 | +0.00051792 |
| 73 | +0.00134883 | +0.00163544 |
| 74 | +0.00107390 | +0.00134820 |
| 75 | +0.00037589 | +0.00050600 |
| 76 | +0.00017880 | +0.00049069 |
| 77 | -0.00036125 | -0.00012003 |
| 78 | -0.00030688 | -0.00013693 |
| 79 | +0.00088060 | +0.00118255 |

This is post-allocation analysis on previously inspected confirmation windows, not a new untouched test.
It does not measure full-sequence sparse trajectories, full PPL, RULER, offload memory or latency.
Environment: basis; formal GPU: NVIDIA L40S. Per-window JSON records commands, source hashes, raw-data hashes and timings.
Protocol and commands: [protocol](../../../docs/q8_residual_mass_recall_protocol.md).

## Rank-group changes and lower tail

| Adaptive rank group | Layers | Mean total-mass delta, pp | Mean non-sink delta, pp |
|---|---:|---:|---:|
| R8 to R4 | 16 | -0.599444 | -0.671951 |
| R8 unchanged | 12 | 0.000000 | 0.000000 |
| R8 to R16 | 8 | +1.572747 | +1.869046 |

Overall mean total mass increased by 0.083080 percentage points, and non-sink recall by 0.116699 points. Total mass improved in 13/16 window means; non-sink recall improved in 14/16. Paired-window standard errors of the mean deltas are 0.023122 and 0.025003 percentage points, respectively.

Total-mass P01 decreased from 28.393322% to 27.299566%, while non-sink P01 increased from 18.610248% to 18.795598%. Total-mass P10 increased from 63.811126% to 63.838464%; non-sink P10 decreased from 58.963907% to 58.945721%. Thus the small mean gain is not a uniform lower-tail improvement.

Window64 contributes 17.5893% of the net total-mass gain and 14.9420% of the net non-sink gain. Descriptively excluding that window leaves mean improvements of 0.073031 and 0.105879 percentage points, respectively. All main results include it; no schedule was adjusted from these diagnostics. Local shared-teacher recall does not by itself explain the earlier terminal KL change on different student trajectories.

## Completed execution and independent checks

| Stage | Slurm job | Elapsed | State / exit |
|---|---|---:|---|
| Smoke | 8300221 | 00:26 | COMPLETED / 0:0 |
| Four evaluation shards | 8300222_0 / 1 / 2 / 3 | 00:48 each | COMPLETED / 0:0 |
| CPU aggregation | 8300223 | 00:17 | COMPLETED / 0:0 |

The user explicitly approved L40S on `yangGrp` for consistency with the earlier KL run. All GPU jobs used `basis` and NVIDIA L40S. Each GPU worker requested two CPUs and 64 GiB host memory; aggregation requested two CPUs and 8 GiB. Logs are `logs/mass-smoke-8300221.{out,err}`, `logs/mass-eval-8300222_{0,1,2,3}.{out,err}` and `logs/mass-summary-8300223.{out,err}`. Temporary sbatch files were deleted after submission.

Smoke computation took 9.73 seconds and peaked at 26.86 GiB allocated GPU memory. Formal windows took 8.44–8.77 seconds each and peaked at 23.20 GiB. These timings include the diagnostic workflow, not an optimized sparse decode benchmark.

All five GPU native-selector checks passed: layer0 R8, layer1 R8/R4 and layer33 R8/R16. Materialized sidecars, page IDs and validity masks matched the actual native sparse attention forward bitwise. Teacher hidden states were unchanged by capture hooks, and every formal window reproduced the prior L40S teacher NLL exactly. Smoke and formal window64 observations matched bitwise. No NaN, OOM or failed GPU check/task was observed.

Independent post-run CPU verification checked the 16 window indices, source/protocol/schedule/raw-data hashes, finite observations, identical unchanged-R8 layers, and 2,359,296 observations per arm. NumPy recomputation matched all pooled means/quantiles, per-layer means and paired-window standard errors. The identity `mass = sink_mass + (1 - sink_mass) * non_sink_recall` held within maximum absolute error 7.1526e-7. Every query had valid non-sink support. The original factors, allocation and terminal KL files were not modified.
