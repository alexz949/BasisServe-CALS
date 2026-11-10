# PaLU G-LRD4 under TP4: measured deployment cost (RTX A5000)

Rebuttal request: "A unified accuracy-memory-latency Pareto comparison, including measured PaLU
latency, is still needed." This directory measures PaLU G-LRD4 deployed with tensor parallelism,
next to Dense and BasisKV, in the same vLLM engine.

**Hardware caveat.** These numbers are from 4x RTX A5000 24 GB (node2, PCIe, NCCL P2P disabled).
The paper's system numbers are on L40S; the L40S run of the same grid is described in
[RUN_L40S.md](RUN_L40S.md) and is not included here yet.

## Setup

Llama-3.1-8B-Instruct, BF16, vLLM 0.30.0, TP4, 4096-token prompts, 128 generated tokens, greedy,
`gpu_memory_utilization=0.8`, block size 32, `NCCL_P2P_DISABLE=1`. Each cell is a fresh vLLM
process with one warmup trial and three measured repetitions; medians are reported.

| Arm | Value cache per rank | Attention path |
|---|---|---|
| Dense | 2 KV heads x 128 | FlashAttention 2 |
| BasisKV V96 / V64 | 2 heads x 96 / 64 (uniform factors) | upstream fused DiffKV Triton kernel |
| PaLU-A R96 / R64 | full rank-r group latent replicated on every rank | upstream fused DiffKV kernel, latent broadcast to both KV heads |
| PaLU-B R96 / R64 | group latent split across the rank pair that shares a group | explicit attention weights, pair exchange of weights and partial outputs; eager only |

TP4 places 2 KV heads on each rank, while a PaLU G-LRD4 group spans 4 KV heads, i.e. two ranks.
Scheme A pays memory to avoid communication; Scheme B pays communication to keep memory low.

## Numerical validation

Teacher-forced next-token log-probabilities at 32 tail positions of 4 prompts, vLLM TP4 versus a
single-GPU Hugging Face reference with the same weights
(`validation/tp4_node2_a5000/compare_*.json`).

| Comparison | Top-1 agreement | Mean abs log-prob diff |
|---|---|---|
| Dense vLLM vs Dense reference (engine baseline) | 1.000 | 0.018 |
| PaLU-A R96 vs PaLU R96 reference | 1.000 | 0.021 |
| PaLU-B R96 vs PaLU R96 reference (code as benchmarked) | 0.992 | 0.020 |

Both PaLU deployments reproduce the PaLU model to within the ordinary engine gap.

## Decode step (ms, p50, eager)

| Arm | Batch 32 | vs Dense | Batch 64 | vs Dense |
|---|---|---|---|---|
| Dense | 37.15 | 1.00x | 55.38 | 1.00x |
| BasisKV V96 | 32.77 | 0.88x | 49.18 | 0.89x |
| BasisKV V64 | 31.01 | 0.83x | 46.92 | 0.85x |
| PaLU-A R96 | 41.23 | 1.11x | 61.96 | 1.12x |
| PaLU-A R64 | 38.62 | 1.04x | 59.47 | 1.07x |
| PaLU-B R96 | 111.07 | 2.99x | 198.18 | 3.58x |
| PaLU-B R64 | 107.58 | 2.90x | 193.65 | 3.50x |

With CUDA Graphs (not possible for PaLU-B) every other arm is within about 2 ms of its eager
number and the ordering is unchanged; see `formal_a5000/full/SUMMARY.md`.

## Request latency (eager)

32 or 64 requests released together; end-to-end time of the cohort, TTFT and TPOT medians.

| Arm | E2E s, B32 | TPOT ms, B32 | E2E s, B64 | TPOT ms, B64 |
|---|---|---|---|---|
| Dense | 71.1 | 265.2 | 142.2 | 552.9 |
| BasisKV V96 | 56.5 | 211.9 | 113.2 | 440.4 |
| BasisKV V64 | 51.7 | 193.6 | 102.4 | 396.2 |
| PaLU-A R96 | 74.4 | 277.4 | 148.5 | 581.0 |
| PaLU-A R64 | 74.0 | 276.0 | 148.2 | 575.0 |
| PaLU-B R96 | 88.5 | 363.4 | 173.2 | 729.7 |
| PaLU-B R64 | 87.1 | 358.2 | 169.2 | 715.8 |

Prefill dominates this 4096/128 workload, so end-to-end gaps are smaller than decode gaps.

## Memory

vLLM pre-allocates the KV pool up to the utilization limit, so peak allocated memory is about
18.5 GiB per rank for every arm and says nothing about the method. The deployed cache layout does:

| Arm | KV bytes / token / rank | vs Dense | Pool capacity, tokens | Mean Value width / rank |
|---|---|---|---|---|
| Dense | 32768 | 1.00x | 479,520 | 256 |
| BasisKV V96 | 28672 | 0.88x | 528,288 | 192 |
| BasisKV V64 | 24576 | 0.75x | 627,424 | 128 |
| PaLU-A R96 | 40960 | 1.25x | 365,280 | 384 |
| PaLU-A R64 | 32896 | 1.00x | 465,344 | 258 |
| PaLU-B R96 | 28672 | 0.88x | 523,904 | 192 |
| PaLU-B R64 | 24640 | 0.75x | 622,144 | 129 |

Bytes per token are summed over 32 layers and include the dense Keys of every arm. Pool capacity
is the number of tokens the same pre-allocated pool holds with that layout (batch 32 eager cell).
Replicating the group latent (Scheme A) makes PaLU R96 larger than Dense; only Scheme B realizes
PaLU's memory saving under TP4.

## Where PaLU-B's time goes

Pair exchange per decode step per rank, actual tensor bytes:

| Batch | Attention weights | Partial outputs, R96 / R64 |
|---|---|---|
| 32 | 66.2 MiB | 3.0 / 2.0 MiB |
| 64 | 132.4 MiB | 6.0 / 4.0 MiB |

The weight exchange is batch x 8 local query heads x context length x 2 bytes x 32 layers, so it
grows with batch and context. Dense and BasisKV only all-reduce hidden states, whose size does not
depend on context length.

GPU time per decode step, mean over the 4 ranks (torch profiler, batch 32, eager, 32 steps;
`formal_a5000/profile_eager/PROFILE.md`):

| Arm | GPU busy ms/step | NCCL ms/step | Attention ms/step | GEMM ms/step |
|---|---|---|---|---|
| Dense | 38.7 | 25.1 | 6.6 | 6.2 |
| PaLU-A R96 | 39.4 | 22.4 | 9.4 | 7.0 |
| PaLU-B R96 | 106.5 | 87.0 | 11.4 | 7.0 |

NCCL kernel time includes waiting for peer ranks. Against PaLU-A, PaLU-B adds about 65 ms of NCCL
time and 2 ms of attention compute per step; communication accounts for nearly all of its slowdown.

## Caveats

- **Interconnect.** PCIe with P2P disabled makes every collective expensive; Dense already spends
  about 25 ms of a 38.7 ms GPU step inside NCCL. PaLU-B's extra cost is almost entirely
  communication, so its penalty depends strongly on the fabric: on a PCIe-limited L40S node with P2P
  on, PaLU-B R96 took 66.98 ms against 38.54 ms for PaLU-A R96 at batch 32 (1.7x, versus 2.7x here);
  see [INTERCONNECT_L40S.md](INTERCONNECT_L40S.md). Its exchange volume still scales with context
  length on any fabric. The interconnect check of that note has not yet been run on node2, and other
  users' jobs occupied node2's remaining GPUs during the grid.
- **NCCL protocol.** NCCL used its LL protocol for these small decode collectives (`RING_LL` in the
  traces). On the L40S node, LL alone added about 120 us per 256 KiB all-reduce; all arms here run
  with the default.
- **BasisKV factors.** Uniform V96 and V64 factors are used, not the per-layer allocated factors
  behind the paper's Table 1 quality numbers.
- **Earlier PaLU-B measurements.** The first PaLU-B batch-32 cells were measured with per-layer
  device synchronizations in the decode path (122.2 / 118.8 ms). They are kept in
  `formal_a5000/eager_superseded_sync/` and are not used above.
- **Profiles** cover one trial of 32 steps and add host overhead; use them for proportions only.

## Layout

- `formal_a5000/{eager,full}/<arm>/p4096_b<batch>/{request,decode}/`: one cell each, with
  `config.json` (full engine configuration, software versions, source hashes, git head),
  `summary.json`, per-trial and per-step records, and `memory_by_rank.jsonl`.
  `SUMMARY.md` and `cells.csv` in each graph directory.
- `formal_a5000/profile_eager/`: torch profiler traces and `PROFILE.md`.
- `validation/tp4_node2_a5000/`: references, served outputs and comparisons.
- `smoke_a5000/`, `logs/`: smoke cells and Slurm logs, including failed attempts.

Generated by `benchmarks/system/run_palu_g4_grid.sh`, `summarize_palu_g4_tp4.py` and
`summarize_palu_profiles.py`. Raw data: Hugging Face `alexz949/BasisServe-CALS`,
`system_benchmarks/v2q8_palu_g4_tp4/`.
