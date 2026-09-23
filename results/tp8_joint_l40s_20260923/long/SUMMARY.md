# TP8 vLLM Joint-ALS + key routing

Conda: `lowrank`; vLLM 0.30.0; torch 2.13.0+cu130; FULL_DECODE_ONLY CUDA Graph.

Medians across three fixed prompt cohorts. A: complete request seconds, G=128. B: ms per steady full-batch decode step after 16 steps, measured over 128 intervals. B includes sampling and scheduler gaps; per-rank events and actual starting context lengths are in JSON.

| Test | P | B | Dense | ALS-full | Joint | Dense / Joint | ALS-full / Joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 65536 | 8 | 49.778 | 46.226 | 47.714 | 1.043 | 0.969 |
| A | 130048 | 8 | 109.644 | 113.073 | 114.740 | 0.956 | 0.985 |
| B | 65536 | 8 | 19.219 | 18.592 | 16.600 | 1.158 | 1.120 |
| B | 130048 | 8 | 30.748 | 29.046 | 17.480 | 1.759 | 1.662 |

Ratios above 1 mean a speedup. These results are separate from the custom eager TP8 protocol.
