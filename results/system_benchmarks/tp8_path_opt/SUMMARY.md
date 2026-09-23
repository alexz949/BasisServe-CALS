# Corrected TP8 Decode Pilot

Conda environment: `basis`. Jobs ran directly on 8 x L40S.

This is a single-cohort pilot, not the full three-cohort grid or a task-quality evaluation.

## Completion

```json
{
  "smoke": {
    "complete": 4,
    "failed": 0
  },
  "timing": {
    "complete": 24,
    "failed": 0
  },
  "profile": {
    "complete": 8,
    "failed": 0
  }
}
```

Every successful trial has eight rank JSON files, complete step arrays, and identical tokens/timings across ranks.

## Uninstrumented Timing

| Prompt | B | Arm | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | vs ALS-full | GPU GiB/rank | Host K GiB total |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | dense | 29.70 | 29.28 | 32.67 | 33.98 | 1.000x | 0.680x | 2.80 | 0.00 |
| 4096 | 1 | als_full | 20.19 | 19.91 | 22.74 | 49.90 | 1.471x | 1.000x | 3.74 | 0.00 |
| 4096 | 1 | basis_joint | 24.98 | 24.29 | 27.89 | 40.28 | 1.189x | 0.808x | 3.73 | 0.26 |
| 4096 | 8 | dense | 30.79 | 30.49 | 33.68 | 260.90 | 1.000x | 0.680x | 3.26 | 0.00 |
| 4096 | 8 | als_full | 20.93 | 20.42 | 24.40 | 383.82 | 1.471x | 1.000x | 4.22 | 0.00 |
| 4096 | 8 | basis_joint | 25.13 | 24.66 | 28.34 | 319.40 | 1.225x | 0.833x | 4.17 | 2.07 |
| 16384 | 1 | dense | 29.82 | 29.34 | 32.90 | 33.64 | 1.000x | 0.846x | 2.99 | 0.00 |
| 16384 | 1 | als_full | 25.23 | 24.98 | 27.18 | 39.64 | 1.182x | 1.000x | 3.90 | 0.00 |
| 16384 | 1 | basis_joint | 26.20 | 25.45 | 30.16 | 38.43 | 1.138x | 0.963x | 3.82 | 1.01 |
| 16384 | 8 | dense | 31.14 | 30.83 | 33.49 | 257.04 | 1.000x | 0.850x | 4.76 | 0.00 |
| 16384 | 8 | als_full | 26.47 | 26.46 | 26.73 | 302.24 | 1.176x | 1.000x | 5.53 | 0.00 |
| 16384 | 8 | basis_joint | 26.02 | 25.87 | 27.24 | 308.21 | 1.197x | 1.017x | 4.91 | 8.07 |
| 65536 | 1 | dense | 85.59 | 85.61 | 86.12 | 11.68 | 1.000x | 0.864x | 3.74 | 0.00 |
| 65536 | 1 | als_full | 73.95 | 73.91 | 74.48 | 13.52 | 1.157x | 1.000x | 4.57 | 0.00 |
| 65536 | 1 | basis_joint | 26.56 | 26.71 | 29.33 | 37.84 | 3.223x | 2.785x | 4.20 | 4.01 |
| 65536 | 8 | dense | 87.42 | 87.47 | 87.63 | 91.52 | 1.000x | 0.864x | 10.76 | 0.00 |
| 65536 | 8 | als_full | 75.51 | 75.52 | 75.74 | 105.94 | 1.158x | 1.000x | 10.80 | 0.00 |
| 65536 | 8 | basis_joint | 25.78 | 25.72 | 26.47 | 310.92 | 3.390x | 2.929x | 7.76 | 32.07 |
| 130048 | 1 | dense | 160.68 | 160.71 | 161.26 | 6.22 | 1.000x | 0.860x | 4.74 | 0.00 |
| 130048 | 1 | als_full | 138.13 | 138.12 | 138.52 | 7.24 | 1.163x | 1.000x | 5.46 | 0.00 |
| 130048 | 1 | basis_joint | 26.74 | 26.31 | 28.93 | 37.60 | 6.009x | 5.165x | 4.69 | 7.95 |
| 130048 | 8 | dense | 162.61 | 162.66 | 162.81 | 49.20 | 1.000x | 0.861x | 18.64 | 0.00 |
| 130048 | 8 | als_full | 140.05 | 140.05 | 140.38 | 57.12 | 1.161x | 1.000x | 17.76 | 0.00 |
| 130048 | 8 | basis_joint | 28.48 | 28.44 | 29.24 | 281.70 | 5.710x | 4.917x | 11.59 | 63.57 |

GPU GiB is the maximum per-rank allocated decode-resident memory, not prefill peak or total host memory.
Latency is the mean of per-step rank maxima; tokens/s uses maximum-rank wall time. These are not exact reciprocals.
The timed loop includes greedy token selection, finite-logit checks, and generated-token copies for all arms.

## Component Profiles

Profiles contain CUDA-event instrumentation. They are not comparable to uninstrumented latency.
`attention_total` contains attention subcomponents. Do not add it to those components; do not sum per-component rank maxima.
Collective intervals can include rank arrival skew and CPU launch gaps, not only transfer time.

| Prompt | B | Component | Rank 0 ms/step | Median rank ms/step | Max rank ms/step |
|---:|---:|---|---:|---:|---:|
| 4096 | 1 | attention_residual | 0.118 | 0.115 | 0.405 |
| 4096 | 1 | attention_total | 19.659 | 19.631 | 19.659 |
| 4096 | 1 | mlp_block | 10.677 | 10.698 | 10.733 |
| 4096 | 1 | output_allgather | 14.241 | 14.095 | 14.256 |
| 4096 | 1 | output_decoder | 1.386 | 1.383 | 1.976 |
| 4096 | 1 | pre_attention_norm | 0.650 | 0.663 | 2.613 |
| 4096 | 1 | qkv_projection_rope_append | 1.437 | 1.523 | 5.274 |
| 4096 | 1 | router_candidates | 0.072 | 0.073 | 0.225 |
| 4096 | 1 | router_fine | 0.543 | 0.547 | 0.738 |
| 4096 | 1 | router_postprocess | 0.775 | 0.773 | 0.950 |
| 4096 | 1 | sparse_attention | 0.569 | 0.635 | 2.151 |
| 4096 | 8 | attention_residual | 0.140 | 0.145 | 0.237 |
| 4096 | 8 | attention_total | 19.998 | 19.870 | 20.293 |
| 4096 | 8 | mlp_block | 11.215 | 11.259 | 11.591 |
| 4096 | 8 | output_allgather | 11.777 | 11.621 | 13.234 |
| 4096 | 8 | output_decoder | 1.376 | 1.400 | 1.592 |
| 4096 | 8 | pre_attention_norm | 0.944 | 0.977 | 1.649 |
| 4096 | 8 | qkv_projection_rope_append | 2.266 | 2.368 | 3.729 |
| 4096 | 8 | router_candidates | 0.111 | 0.112 | 0.154 |
| 4096 | 8 | router_fine | 0.833 | 0.834 | 0.884 |
| 4096 | 8 | router_postprocess | 1.580 | 1.545 | 1.584 |
| 4096 | 8 | sparse_attention | 1.102 | 1.102 | 1.298 |
| 16384 | 1 | attention_residual | 0.298 | 0.130 | 0.298 |
| 16384 | 1 | attention_total | 19.311 | 21.263 | 21.461 |
| 16384 | 1 | mlp_block | 10.132 | 10.441 | 10.507 |
| 16384 | 1 | output_allgather | 7.087 | 14.374 | 15.027 |
| 16384 | 1 | output_decoder | 1.808 | 1.425 | 1.808 |
| 16384 | 1 | pre_attention_norm | 1.977 | 0.776 | 1.977 |
| 16384 | 1 | qkv_projection_rope_append | 4.065 | 1.696 | 4.065 |
| 16384 | 1 | router_candidates | 0.613 | 0.384 | 0.613 |
| 16384 | 1 | router_coarse | 0.843 | 0.250 | 0.843 |
| 16384 | 1 | router_fine | 0.744 | 0.626 | 0.744 |
| 16384 | 1 | router_postprocess | 1.017 | 0.965 | 1.017 |
| 16384 | 1 | sparse_attention | 1.338 | 0.726 | 1.338 |
| 16384 | 8 | attention_residual | 0.120 | 0.120 | 0.200 |
| 16384 | 8 | attention_total | 22.630 | 22.491 | 22.630 |
| 16384 | 8 | mlp_block | 9.397 | 9.464 | 9.710 |
| 16384 | 8 | output_allgather | 11.788 | 11.387 | 12.035 |
| 16384 | 8 | output_decoder | 1.311 | 1.308 | 1.320 |
| 16384 | 8 | pre_attention_norm | 0.789 | 0.852 | 3.019 |
| 16384 | 8 | qkv_projection_rope_append | 1.943 | 2.154 | 6.231 |
| 16384 | 8 | router_candidates | 0.468 | 0.491 | 0.819 |
| 16384 | 8 | router_coarse | 0.492 | 0.525 | 1.323 |
| 16384 | 8 | router_fine | 2.630 | 2.652 | 2.911 |
| 16384 | 8 | router_postprocess | 2.164 | 2.116 | 2.266 |
| 16384 | 8 | sparse_attention | 0.873 | 0.859 | 0.873 |
| 65536 | 1 | attention_residual | 0.123 | 0.122 | 0.387 |
| 65536 | 1 | attention_total | 21.976 | 21.994 | 22.044 |
| 65536 | 1 | mlp_block | 10.740 | 10.739 | 10.803 |
| 65536 | 1 | output_allgather | 14.918 | 14.991 | 15.270 |
| 65536 | 1 | output_decoder | 1.412 | 1.413 | 1.975 |
| 65536 | 1 | pre_attention_norm | 0.713 | 0.703 | 2.558 |
| 65536 | 1 | qkv_projection_rope_append | 1.622 | 1.615 | 5.240 |
| 65536 | 1 | router_candidates | 0.576 | 0.574 | 0.922 |
| 65536 | 1 | router_coarse | 0.263 | 0.259 | 1.126 |
| 65536 | 1 | router_fine | 0.658 | 0.655 | 0.833 |
| 65536 | 1 | router_postprocess | 1.041 | 1.024 | 1.159 |
| 65536 | 1 | sparse_attention | 0.723 | 0.713 | 1.702 |
| 65536 | 8 | attention_residual | 0.128 | 0.122 | 0.131 |
| 65536 | 8 | attention_total | 22.375 | 22.708 | 23.460 |
| 65536 | 8 | mlp_block | 8.720 | 8.715 | 8.782 |
| 65536 | 8 | output_allgather | 7.856 | 9.399 | 12.045 |
| 65536 | 8 | output_decoder | 1.339 | 1.330 | 1.341 |
| 65536 | 8 | pre_attention_norm | 1.406 | 1.217 | 1.582 |
| 65536 | 8 | qkv_projection_rope_append | 3.573 | 2.883 | 4.239 |
| 65536 | 8 | router_candidates | 0.808 | 0.764 | 0.887 |
| 65536 | 8 | router_coarse | 1.038 | 0.943 | 1.228 |
| 65536 | 8 | router_fine | 2.944 | 2.838 | 2.985 |
| 65536 | 8 | router_postprocess | 2.676 | 2.608 | 2.774 |
| 65536 | 8 | sparse_attention | 0.860 | 0.856 | 0.867 |
| 130048 | 1 | attention_residual | 0.133 | 0.137 | 0.326 |
| 130048 | 1 | attention_total | 21.070 | 21.090 | 21.392 |
| 130048 | 1 | mlp_block | 10.526 | 10.534 | 10.600 |
| 130048 | 1 | output_allgather | 12.901 | 13.109 | 14.455 |
| 130048 | 1 | output_decoder | 1.426 | 1.441 | 1.893 |
| 130048 | 1 | pre_attention_norm | 0.897 | 0.869 | 2.348 |
| 130048 | 1 | qkv_projection_rope_append | 2.086 | 1.991 | 5.084 |
| 130048 | 1 | router_candidates | 0.644 | 0.645 | 0.950 |
| 130048 | 1 | router_coarse | 0.446 | 0.442 | 1.168 |
| 130048 | 1 | router_fine | 0.696 | 0.697 | 0.852 |
| 130048 | 1 | router_postprocess | 1.020 | 1.011 | 1.107 |
| 130048 | 1 | sparse_attention | 0.856 | 0.865 | 1.661 |
| 130048 | 8 | attention_residual | 0.125 | 0.125 | 0.217 |
| 130048 | 8 | attention_total | 25.506 | 25.292 | 25.507 |
| 130048 | 8 | mlp_block | 10.659 | 10.729 | 10.906 |
| 130048 | 8 | output_allgather | 14.404 | 13.453 | 14.585 |
| 130048 | 8 | output_decoder | 1.347 | 1.349 | 1.434 |
| 130048 | 8 | pre_attention_norm | 0.785 | 0.892 | 2.220 |
| 130048 | 8 | qkv_projection_rope_append | 1.738 | 2.134 | 5.153 |
| 130048 | 8 | router_candidates | 0.567 | 0.587 | 0.734 |
| 130048 | 8 | router_coarse | 0.992 | 1.077 | 1.693 |
| 130048 | 8 | router_fine | 2.561 | 2.632 | 3.042 |
| 130048 | 8 | router_postprocess | 2.409 | 2.402 | 2.585 |
| 130048 | 8 | sparse_attention | 0.799 | 0.824 | 0.865 |

## Collective Smoke Measurements

TP8 BF16, local width 384, 20 warmup calls and 100 measured calls per backend. Each completed backend passes exact output comparison.
The timing helper now executes graph replay and timing events on the same CUDA stream; the old side-stream measurement was not reliable.
These short standalone measurements do not reproduce model rank-arrival skew or its CPU affinity; they do not establish end-to-end backend speedups.

| B | Backend | Median us | Mean us | Status |
|---:|---|---:|---:|---|
| 1 | uniform_nccl | 18.16 | 19.50 | complete |
| 1 | uniform_nccl_graph | 18.35 | 18.78 | complete |
| 1 | uniform_ipc | 23.36 | 23.53 | complete |
| 8 | uniform_nccl | 35.38 | 37.75 | complete |
| 8 | uniform_nccl_graph | 27.71 | 30.60 | complete |
| 8 | uniform_ipc | 48.37 | 48.25 | complete |

## Validation Limits

- Finite logits are checked by the runner. Generated first-sequence samples are in `generated_samples.json`.
- `batch_consistency.csv` compares the same first prompt at B=1 and B=8; floating-point changes can alter greedy trajectories.
- Observed B=1/B=8 trajectory divergence in 8/12 arm/context pairs. Dense also diverges; teacher-forced comparisons are needed to diagnose this, not free-generation token agreement alone.
- Cross-rank agreement and finite logits do not establish task accuracy or numerical equivalence to Dense.
- Historical B>1 Basis-joint runs had an output-stride error and are not a correctness baseline.
- CPU affinity is set, but strict pinned-host NUMA placement is unavailable (`set_mempolicy: Operation not permitted`).
- No SHA256 validation was performed.

## Commands and Logs

Exact per-trial torchrun commands and return codes are in each phase's `decode_grid_trials.json`.
Top-level launcher logs are `smoke.log`, `timing.log`, and `profile.log`; each trial also has launcher and rank logs.
Timing uses 16 conditioning + 128 measured steps; profile uses 16 conditioning + 16 measured steps.
Source changes relative to the archived baseline are recorded in `source.patch`.
