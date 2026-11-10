# PaLU G-LRD4 TP4 grid: results/rebuttal/v2q8_palu_g4_tp4/formal_a5000/eager

Llama-3.1-8B-Instruct BF16, TP4, 4096-token prompts, 128 generated tokens, greedy. Medians over the repetitions inside one process after a warmup trial. Per-token bytes come from the deployed cache layout per rank (Keys are dense in every arm); pool capacity is the number of tokens the pre-allocated KV pool holds for that layout.

## decode

### batch 32

| Arm | Graph | Decode step p50 ms | Decode step mean ms | decode tok/s | steps each (p50 ms) | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 37.15 | 37.24 | 850 | 37.2 / 37.0 / 37.1 | 32768 | 256 | 4.12 | 479520 | 18.47 |
| `basis_v96` | eager | 32.77 | 32.86 | 962 | 32.7 / 32.8 / 33.0 | 28672 | 192 | 3.61 | 528288 | 18.46 |
| `basis_v64` | eager | 31.01 | 31.05 | 1017 | 31.0 / 30.3 / 31.1 | 24576 | 128 | 3.09 | 627424 | 18.45 |
| `palu_g4_r96` | eager | 41.23 | 41.23 | 769 | 41.4 / 40.9 / 41.2 | 40960 | 384 | 5.16 | 365280 | 18.52 |
| `palu_g4_r64` | eager | 38.62 | 38.70 | 817 | 38.6 / 38.7 / 37.2 | 32896 | 258 | 4.14 | 465344 | 18.54 |
| `palu_g4_sharded_r96` | eager | 111.07 | 111.25 | 287 | 110.9 / 111.1 / 111.1 | 28672 | 192 | 3.61 | 523904 | 18.54 |
| `palu_g4_sharded_r64` | eager | 107.58 | 107.84 | 296 | 107.5 / 107.6 / 107.6 | 24640 | 129 | 3.10 | 622144 | 18.52 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 66.22 | 3.000 | 0.00 |
| `palu_g4_sharded_r64` | 66.22 | 2.016 | 0.00 |

### batch 64

| Arm | Graph | Decode step p50 ms | Decode step mean ms | decode tok/s | steps each (p50 ms) | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 55.38 | 55.33 | 1140 | 55.0 / 55.5 / 55.4 | 32768 | 256 | 8.25 | 479008 | 18.49 |
| `basis_v96` | eager | 49.18 | 49.37 | 1275 | 48.3 / 49.2 / 50.3 | 28672 | 192 | 7.22 | 527616 | 18.47 |
| `basis_v64` | eager | 46.92 | 47.05 | 1337 | 46.9 / 46.9 / 47.0 | 24576 | 128 | 6.19 | 626752 | 18.47 |
| `palu_g4_r96` | eager | 61.96 | 62.09 | 1017 | 62.0 / 61.2 / 62.3 | 40960 | 384 | 10.31 | 364832 | 18.54 |
| `palu_g4_r64` | eager | 59.47 | 59.54 | 1060 | 59.5 / 59.6 / 58.8 | 32896 | 258 | 8.28 | 464864 | 18.56 |
| `palu_g4_sharded_r96` | eager | 198.18 | 198.35 | 321 | 198.2 / 198.2 / 201.2 | 28672 | 192 | 7.22 | 523232 | 18.55 |
| `palu_g4_sharded_r64` | eager | 193.65 | 193.80 | 329 | 194.9 / 193.1 / 193.6 | 24640 | 129 | 6.20 | 621472 | 18.54 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 132.44 | 6.000 | 0.00 |
| `palu_g4_sharded_r64` | 132.44 | 4.031 | 0.00 |

## request

### batch 32

| Arm | Graph | E2E s | TTFT p50 ms | TPOT p50 ms | tok/s | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 71.07 | 37187 | 265.2 | 58 | 32768 | 256 | 4.12 | 479520 | 18.82 |
| `basis_v96` | eager | 56.51 | 29373 | 211.9 | 72 | 28672 | 192 | 3.61 | 528288 | 18.81 |
| `basis_v64` | eager | 51.67 | 26640 | 193.6 | 79 | 24576 | 128 | 3.09 | 627424 | 18.80 |
| `palu_g4_r96` | eager | 74.40 | 38908 | 277.4 | 55 | 40960 | 384 | 5.16 | 365280 | 18.87 |
| `palu_g4_r64` | eager | 73.99 | 38731 | 276.0 | 55 | 32896 | 258 | 4.14 | 465344 | 18.88 |
| `palu_g4_sharded_r96` | eager | 88.47 | 41958 | 363.4 | 46 | 28672 | 192 | 3.61 | 523904 | 18.89 |
| `palu_g4_sharded_r64` | eager | 87.12 | 41271 | 358.2 | 47 | 24640 | 129 | 3.10 | 622144 | 18.87 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 58.62 | 2.664 | 2241.29 |
| `palu_g4_sharded_r64` | 58.62 | 1.790 | 1505.87 |

### batch 64

| Arm | Graph | E2E s | TTFT p50 ms | TPOT p50 ms | tok/s | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | eager | 142.20 | 71506 | 552.9 | 58 | 32768 | 256 | 8.25 | 479008 | 18.82 |
| `basis_v96` | eager | 113.20 | 56851 | 440.4 | 72 | 28672 | 192 | 7.22 | 527616 | 18.81 |
| `basis_v64` | eager | 102.43 | 51180 | 396.2 | 80 | 24576 | 128 | 6.19 | 626752 | 18.81 |
| `palu_g4_r96` | eager | 148.55 | 74267 | 581.0 | 55 | 40960 | 384 | 10.31 | 364832 | 18.87 |
| `palu_g4_r64` | eager | 148.21 | 74571 | 575.0 | 55 | 32896 | 258 | 8.28 | 464864 | 18.89 |
| `palu_g4_sharded_r96` | eager | 173.22 | 79611 | 729.7 | 47 | 28672 | 192 | 7.22 | 523232 | 18.89 |
| `palu_g4_sharded_r64` | eager | 169.18 | 77276 | 715.8 | 48 | 24640 | 129 | 6.20 | 621472 | 18.87 |

Pair exchange of the sharded arm (actual tensor bytes, per rank):

| Arm | Weights MiB/step | Partial outputs MiB/step | Prefill shards MiB total |
|---|---|---|---|
| `palu_g4_sharded_r96` | 105.43 | 4.792 | 4437.57 |
| `palu_g4_sharded_r64` | 105.43 | 3.220 | 2981.50 |

