# Page-granularity attention-mass sweep (no refit)

Script: q3_8b_128k/repo/evaluation/page_granularity_sweep.py. Support = sink 32 + recent 64 + K = B/P routed pages of size P chosen by (oracle) exact QK page scores or (router) the fixed B16R16 page router; page score = per-head logsumexp inside the page, softmax over pages, GQA max. Metric = exact softmax attention mass on the support (x100), mean over query heads / groups / queries / windows / layers. C4 held-out windows (validation_ids 32-47) truncated to 65536, 64 query positions per window (4096..65535).

```
== c4_64k_b16r16.json  windows=16 queries/window=64 layers=32
-- B=256
       P        1        2        4        8       16       32
  oracle    90.45    89.48    88.59    87.75    86.95    86.15
  router    89.51    88.58    87.74    86.97    86.25    85.54
oracle -L0    91.90    91.05    90.19    89.36    88.56    87.75   (excluding layer 0)
router -L0    90.94    90.12    89.32    88.56    87.84    87.13   (excluding layer 0)
     gap     0.94     0.90     0.85     0.78     0.70     0.60
  vs P32     3.97     3.04     2.19     1.43     0.70     0.00   (router, relative to P=32)
   P=1  router per-layer min/median/max  45.09/ 91.22/ 95.30   gap max  2.09 (layer 30)
   P=8  router per-layer min/median/max  37.63/ 88.92/ 94.38   gap max  1.88 (layer 30)
   P=32 router per-layer min/median/max  36.48/ 87.49/ 93.82   gap max  1.48 (layer 30)
-- B=2048
       P        1        2        4        8       16       32
  oracle    96.75    96.17    95.62    95.15    94.74    94.36
  router    96.37    95.79    95.25    94.79    94.40    94.05
oracle -L0    97.48    97.09    96.69    96.30    95.94    95.60   (excluding layer 0)
router -L0    97.09    96.70    96.31    95.94    95.60    95.28   (excluding layer 0)
     gap     0.38     0.38     0.37     0.35     0.33     0.31
  vs P32     2.32     1.73     1.20     0.74     0.35     0.00   (router, relative to P=32)
   P=1  router per-layer min/median/max  73.85/ 97.25/ 98.62   gap max  0.77 (layer 30)
   P=8  router per-layer min/median/max  59.26/ 96.22/ 97.59   gap max  0.82 (layer 30)
   P=32 router per-layer min/median/max  56.05/ 95.55/ 96.93   gap max  0.79 (layer 30)
```

Plot: llama31_8b_c4_64k_page_sweep.png

## Qwen3-8B post V96 (yarn4), same protocol
```
== qwen3_8b_post_c4_64k_b16r16.json  windows=16 queries/window=64 layers=36
-- B=256
       P        1        2        4        8       16       32
  oracle    87.50    85.61    83.53    81.20    78.67    75.80
  router    85.07    83.22    81.19    78.96    76.58    73.81
oracle -L0    87.90    86.05    83.99    81.68    79.15    76.28   (excluding layer 0)
router -L0    85.54    83.73    81.72    79.49    77.11    74.33   (excluding layer 0)
     gap     2.43     2.39     2.34     2.24     2.09     1.98
  vs P32    11.26     9.40     7.38     5.14     2.77     0.00   (router, relative to P=32)
   P=1  router per-layer min/median/max  63.99/ 86.43/ 95.30   gap max  5.10 (layer 0)
   P=8  router per-layer min/median/max  55.13/ 82.02/ 93.83   gap max  6.40 (layer 3)
   P=32 router per-layer min/median/max  42.63/ 77.50/ 92.95   gap max  4.52 (layer 3)
-- B=2048
       P        1        2        4        8       16       32
  oracle    95.89    94.86    93.75    92.61    91.39    89.97
  router    94.70    93.62    92.50    91.36    90.13    88.69
oracle -L0    96.07    95.07    94.00    92.87    91.66    90.24   (excluding layer 0)
router -L0    94.90    93.87    92.77    91.64    90.41    88.96   (excluding layer 0)
     gap     1.19     1.23     1.25     1.24     1.25     1.28
  vs P32     6.01     4.94     3.82     2.67     1.44     0.00   (router, relative to P=32)
   P=1  router per-layer min/median/max  83.91/ 95.52/ 98.16   gap max  2.25 (layer 2)
   P=8  router per-layer min/median/max  73.29/ 93.03/ 97.10   gap max  2.41 (layer 3)
   P=32 router per-layer min/median/max  69.37/ 91.05/ 96.32   gap max  4.73 (layer 3)
```

Plot: qwen3_8b_post_c4_64k_page_sweep.png

LongBench (ShadowKV-9, budget 256 + one sink page + recent 64) on Qwen3-8B with the same B16R16 bank: p32 43.15 / p16 44.15 / p8 45.08 (9-task mean); passage_retrieval_en 88.5 / 94.5 / 98.5; Full-K 46.20 / 100.
