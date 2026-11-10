# PaLU G-LRD4 TP4 grid: results/rebuttal/v2q8_palu_g4_tp4/formal_a5000/full

Llama-3.1-8B-Instruct BF16, TP4, 4096-token prompts, 128 generated tokens, greedy. Medians over the repetitions inside one process after a warmup trial. Per-token bytes come from the deployed cache layout per rank (Keys are dense in every arm); pool capacity is the number of tokens the pre-allocated KV pool holds for that layout.

## decode

### batch 32

| Arm | Graph | Decode step p50 ms | Decode step mean ms | decode tok/s | steps each (p50 ms) | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | full | 36.83 | 36.77 | 861 | 36.9 / 36.6 / 36.8 | 32768 | 256 | 4.12 | 475232 | 18.35 |
| `basis_v96` | full | 32.97 | 33.00 | 957 | 33.4 / 33.0 / 32.4 | 28672 | 192 | 3.61 | 524128 | 18.35 |
| `basis_v64` | full | 30.33 | 30.34 | 1041 | 30.1 / 30.5 / 30.3 | 24576 | 128 | 3.09 | 622752 | 18.35 |
| `palu_g4_r96` | full | 40.20 | 40.37 | 785 | 40.0 / 40.2 / 40.8 | 40960 | 384 | 5.16 | 361152 | 18.32 |
| `palu_g4_r64` | full | 40.62 | 42.96 | 737 | 39.2 / 40.8 / 40.6 | 32896 | 258 | 4.14 | 460320 | 18.34 |

### batch 64

| Arm | Graph | Decode step p50 ms | Decode step mean ms | decode tok/s | steps each (p50 ms) | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | full | 55.55 | 55.70 | 1133 | 55.6 / 55.8 / 55.0 | 32768 | 256 | 8.25 | 473888 | 18.34 |
| `basis_v96` | full | 48.39 | 48.69 | 1292 | 48.4 / 48.4 / 50.1 | 28672 | 192 | 7.22 | 522944 | 18.36 |
| `basis_v64` | full | 47.59 | 47.86 | 1315 | 47.5 / 47.7 / 47.6 | 24576 | 128 | 6.19 | 621024 | 18.35 |
| `palu_g4_r96` | full | 63.78 | 64.10 | 986 | 63.8 / 63.7 / 63.8 | 40960 | 384 | 10.31 | 360288 | 18.37 |
| `palu_g4_r64` | full | 59.06 | 58.99 | 1070 | 59.1 / 59.1 / 58.8 | 32896 | 258 | 8.28 | 459296 | 18.39 |

## request

### batch 32

| Arm | Graph | E2E s | TTFT p50 ms | TPOT p50 ms | tok/s | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | full | 71.09 | 37478 | 264.4 | 58 | 32768 | 256 | 4.12 | 475232 | 18.66 |
| `basis_v96` | full | 57.55 | 30125 | 215.1 | 71 | 28672 | 192 | 3.61 | 524128 | 18.66 |
| `basis_v64` | full | 50.71 | 26328 | 191.0 | 81 | 24576 | 128 | 3.09 | 622752 | 18.66 |
| `palu_g4_r96` | full | 75.16 | 39118 | 280.1 | 54 | 40960 | 384 | 5.16 | 361152 | 18.67 |
| `palu_g4_r64` | full | 74.24 | 38872 | 275.2 | 55 | 32896 | 258 | 4.14 | 460320 | 18.69 |

### batch 64

| Arm | Graph | E2E s | TTFT p50 ms | TPOT p50 ms | tok/s | KV B/token/rank | mean V width/rank | workload KV GiB/rank | pool tokens | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | full | 142.49 | 71719 | 553.9 | 57 | 32768 | 256 | 8.25 | 473888 | 18.63 |
| `basis_v96` | full | 112.83 | 56353 | 441.7 | 73 | 28672 | 192 | 7.22 | 522912 | 18.64 |
| `basis_v64` | full | 104.91 | 52410 | 410.6 | 78 | 24576 | 128 | 6.19 | 621024 | 18.64 |
| `palu_g4_r96` | full | 153.07 | 76835 | 596.6 | 54 | 40960 | 384 | 10.31 | 360288 | 18.71 |
| `palu_g4_r64` | full | 149.66 | 75240 | 582.5 | 55 | 32896 | 258 | 8.28 | 459296 | 18.73 |

