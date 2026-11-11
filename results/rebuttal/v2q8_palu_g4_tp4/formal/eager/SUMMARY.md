# PaLU G-LRD4 TP4 grid: results/rebuttal/v2q8_palu_g4_tp4/formal/eager

Llama-3.1-8B-Instruct BF16, TP4, 4096-token prompts, 128 generated tokens, greedy. Medians over the repetitions inside one process after a warmup trial. Per-token bytes come from the deployed cache layout per rank (Keys are dense in every arm); pool capacity is the number of tokens the pre-allocated KV pool holds for that layout.

## decode

### batch 32

| Arm | Graph | Decode step p50 ms | Decode step mean ms | decode tok/s | steps each (p50 ms) | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 22.10 | 22.86 | 1378 | 22.2 / 21.9 / 22.1 | 32768 | 256 | 4.12 | 1026208 | 35.15 |
| `basis_v96` | eager | 24.81 | 27.22 | 1159 | 24.9 / 24.7 / 24.8 | 28672 | 192 | 3.61 | 1153088 | 35.14 |
| `basis_v64` | eager | 25.34 | 26.49 | 1191 | 25.0 / 25.3 / 25.4 | 24576 | 128 | 3.09 | 1356352 | 35.14 |
| `palu_g4_r96` | eager | 24.39 | 24.74 | 1274 | 24.5 / 24.4 / 24.2 | 40960 | 384 | 5.16 | 802656 | 35.21 |
| `palu_g4_r64` | eager | 24.97 | 26.99 | 1169 | 25.1 / 25.0 / 24.7 | 32896 | 258 | 4.14 | 1009920 | 35.22 |
| `palu_g4_sharded_r96` | eager | 37.67 | 39.47 | 803 | 37.6 / 37.7 / 40.3 | 28672 | 192 | 3.61 | 1148704 | 35.22 |
| `palu_g4_sharded_r64` | eager | 38.45 | 43.37 | 731 | 39.5 / 38.5 / 38.1 | 24640 | 129 | 3.10 | 1349184 | 35.21 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 66.22 | 3.000 | 0.00 |
| `palu_g4_sharded_r64` | 66.22 | 2.016 | 0.00 |

### batch 64

| Arm | Graph | Decode step p50 ms | Decode step mean ms | decode tok/s | steps each (p50 ms) | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 32.06 | 32.34 | 1931 | 32.2 / 32.1 / 31.7 | 32768 | 256 | 8.25 | 1025696 | 35.17 |
| `basis_v96` | eager | 30.01 | 30.50 | 2046 | 30.3 / 30.0 / 30.0 | 28672 | 192 | 7.22 | 1152416 | 35.16 |
| `basis_v64` | eager | 27.06 | 27.91 | 2257 | 27.7 / 26.7 / 27.1 | 24576 | 128 | 6.19 | 1355680 | 35.16 |
| `palu_g4_r96` | eager | 36.82 | 37.08 | 1690 | 37.0 / 36.8 / 36.6 | 40960 | 384 | 10.31 | 802176 | 35.22 |
| `palu_g4_r64` | eager | 33.00 | 33.87 | 1848 | 33.0 / 32.5 / 33.0 | 32896 | 258 | 8.28 | 1009408 | 35.24 |
| `palu_g4_sharded_r96` | eager | 53.07 | 53.99 | 1169 | 53.1 / 53.0 / 53.2 | 28672 | 192 | 7.22 | 1148032 | 35.24 |
| `palu_g4_sharded_r64` | eager | 49.39 | 51.21 | 1235 | 49.4 / 50.5 / 49.3 | 24640 | 129 | 6.20 | 1348480 | 35.23 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 132.44 | 6.000 | 0.00 |
| `palu_g4_sharded_r64` | 132.44 | 4.031 | 0.00 |

### batch 128

| Arm | Graph | Decode step p50 ms | Decode step mean ms | decode tok/s | steps each (p50 ms) | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 52.42 | 52.36 | 2378 | 52.6 / 52.4 / 52.3 | 32768 | 256 | 16.50 | 1024032 | 35.19 |
| `basis_v96` | eager | 47.28 | 47.72 | 2604 | 47.3 / 47.3 / 47.2 | 28672 | 192 | 14.44 | 1151328 | 35.20 |
| `basis_v64` | eager | 42.31 | 42.73 | 2898 | 41.2 / 42.3 / 42.6 | 24576 | 128 | 12.38 | 1354304 | 35.20 |
| `palu_g4_r96` | eager | 57.28 | 57.86 | 2181 | 57.3 / 58.8 / 57.3 | 40960 | 384 | 20.62 | 801408 | 35.27 |
| `palu_g4_r64` | eager | 51.52 | 52.31 | 2394 | 52.4 / 51.5 / 50.6 | 32896 | 258 | 16.56 | 1008384 | 35.28 |
| `palu_g4_sharded_r96` | eager | 92.01 | 93.35 | 1359 | 92.0 / 92.0 / 91.9 | 28672 | 192 | 14.44 | 1146944 | 35.28 |
| `palu_g4_sharded_r64` | eager | 84.16 | 85.57 | 1482 | 85.4 / 83.8 / 84.2 | 24640 | 129 | 12.41 | 1347136 | 35.27 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 264.89 | 12.000 | 0.00 |
| `palu_g4_sharded_r64` | 264.89 | 8.062 | 0.00 |

## request

### batch 32

| Arm | Graph | E2E s | TTFT p50 ms | TPOT p50 ms | tok/s | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 14.09 | 6135 | 61.3 | 291 | 32768 | 256 | 4.12 | 1026208 | 35.50 |
| `basis_v96` | eager | 12.78 | 4973 | 59.2 | 320 | 28672 | 192 | 3.61 | 1153088 | 35.49 |
| `basis_v64` | eager | 11.68 | 4582 | 54.3 | 351 | 24576 | 128 | 3.09 | 1356352 | 35.49 |
| `palu_g4_r96` | eager | 15.47 | 6794 | 66.9 | 265 | 40960 | 384 | 5.16 | 802656 | 35.56 |
| `palu_g4_r64` | eager | 15.09 | 6605 | 65.3 | 271 | 32896 | 258 | 4.14 | 1009920 | 35.57 |
| `palu_g4_sharded_r96` | eager | 19.06 | 7802 | 86.4 | 215 | 28672 | 192 | 3.61 | 1148704 | 35.57 |
| `palu_g4_sharded_r64` | eager | 18.73 | 7534 | 85.9 | 219 | 24640 | 129 | 3.10 | 1349184 | 35.56 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 58.62 | 2.664 | 2241.29 |
| `palu_g4_sharded_r64` | 58.62 | 1.790 | 1505.87 |

### batch 64

| Arm | Graph | E2E s | TTFT p50 ms | TPOT p50 ms | tok/s | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 26.45 | 11734 | 112.8 | 310 | 32768 | 256 | 8.25 | 1025696 | 35.50 |
| `basis_v96` | eager | 21.99 | 9446 | 95.7 | 372 | 28672 | 192 | 7.22 | 1152416 | 35.49 |
| `basis_v64` | eager | 20.22 | 8734 | 87.2 | 405 | 24576 | 128 | 6.19 | 1355680 | 35.49 |
| `palu_g4_r96` | eager | 29.11 | 12895 | 124.2 | 281 | 40960 | 384 | 10.31 | 802176 | 35.56 |
| `palu_g4_r64` | eager | 27.83 | 12551 | 117.4 | 294 | 32896 | 258 | 8.28 | 1009408 | 35.57 |
| `palu_g4_sharded_r96` | eager | 34.81 | 14785 | 152.4 | 235 | 28672 | 192 | 7.22 | 1148032 | 35.57 |
| `palu_g4_sharded_r64` | eager | 33.51 | 14292 | 146.3 | 244 | 24640 | 129 | 6.20 | 1348480 | 35.56 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 105.43 | 4.792 | 4437.57 |
| `palu_g4_sharded_r64` | 105.43 | 3.220 | 2981.50 |

### batch 128

| Arm | Graph | E2E s | TTFT p50 ms | TPOT p50 ms | tok/s | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 52.50 | 23147 | 224.9 | 312 | 32768 | 256 | 16.50 | 1024032 | 35.49 |
| `basis_v96` | eager | 41.63 | 18463 | 175.6 | 394 | 28672 | 192 | 14.44 | 1151328 | 35.49 |
| `basis_v64` | eager | 38.39 | 17056 | 161.8 | 427 | 24576 | 128 | 12.38 | 1354304 | 35.49 |
| `palu_g4_r96` | eager | 55.98 | 25143 | 235.7 | 293 | 40960 | 384 | 20.62 | 801408 | 35.56 |
| `palu_g4_r64` | eager | 54.09 | 24485 | 226.4 | 303 | 32896 | 258 | 16.56 | 1008384 | 35.57 |
| `palu_g4_sharded_r96` | eager | 66.86 | 28670 | 290.5 | 245 | 28672 | 192 | 14.44 | 1146944 | 35.58 |
| `palu_g4_sharded_r64` | eager | 64.58 | 27935 | 278.6 | 254 | 24640 | 129 | 12.41 | 1347136 | 35.56 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 175.54 | 7.979 | 8167.14 |
| `palu_g4_sharded_r64` | 175.54 | 5.361 | 5487.29 |

