# TP8 vLLM Joint-ALS + key routing

Conda: `lowrank`; vLLM 0.30.0; torch 2.13.0+cu130; FULL_DECODE_ONLY CUDA Graph.

Medians across three fixed prompt cohorts. A: complete request seconds, G=128. B: ms per steady full-batch decode step after 16 steps, measured over 128 intervals. B includes sampling and scheduler gaps; per-rank events and actual starting context lengths are in JSON.

| Test | P | B | Dense | ALS-full | Joint | Dense / Joint | ALS-full / Joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 4096 | 1 | 1.181 | 1.185 | 1.446 | 0.817 | 0.819 |
| A | 4096 | 8 | 3.636 | 3.140 | 3.953 | 0.920 | 0.794 |
| A | 16384 | 1 | 2.212 | 2.229 | 2.370 | 0.933 | 0.940 |
| A | 16384 | 8 | 12.052 | 10.146 | 11.194 | 1.077 | 0.906 |
| B | 4096 | 1 | 6.707 | 7.332 | 9.299 | 0.721 | 0.788 |
| B | 4096 | 8 | 7.990 | 8.416 | 13.864 | 0.576 | 0.607 |
| B | 16384 | 1 | 6.866 | 8.920 | 9.741 | 0.705 | 0.916 |
| B | 16384 | 8 | 10.251 | 10.511 | 15.280 | 0.671 | 0.688 |

Ratios above 1 mean a speedup. These results are separate from the custom eager TP8 protocol.
