# L40S node 2: PaLU G-LRD4 TP4 grid, eager pass

2026-11-10, Vast.ai 4x L40S instance (a different machine from node 1, see `INTERCONNECT_L40S.md`). Repository
at `abf8550`. The run followed `RUN_L40S.md` sections 5 to 9, with one deviation: the eager pass only (see
"Deviations"). Pipeline wall time 07:46 to 10:11 UTC; the formal eager pass took 2 h 02 min.

## Summary

- **Status.** Validation passed (3/3 rows) and smoke passed (7/7 cells). All 42 formal eager cells completed on
  the first attempt: no failures, no reruns, no out-of-memory at batch 128. Profiles: 3/3.
- **Sanity checks.** For every arm, "KV B/token/rank" and "mean V width/rank" match the A5000 table exactly.
- **Ordering.**
  - At batches 64 and 128 the arms rank as on the A5000: BasisKV V64 < V96 < dense < PaLU-A < PaLU-B (sharded).
  - One exception at batch 128: `palu_g4_r64` is slightly faster than dense, 51.52 ms vs 52.42 ms. The A5000 has
    no batch-128 reference.
  - **At batch 32 the ordering does not hold.** Dense is the fastest arm at 22.10 ms. BasisKV V96 and V64 take
    24.81 ms and 25.34 ms, i.e. 12 % and 15 % slower than dense. PaLU-A r96 is 24.39 ms. The three repetitions
    of each cell agree within 0.3 ms, so this is not noise. See "Batch 32".
- **Interconnect.** All-reduce cost is far lower than on node 1: a 4-GPU all-reduce of 256 KiB takes 76.8 µs,
  against 292 µs on node 1. Dense decode at batch 32 takes 22.10 ms here, against 35.96 ms on node 1.
  - The interconnect check still reports **OVERALL FAIL**.
  - Both failing criteria concern GPU pairs on different sockets: those pairs use NCCL's SHM transport, and their
    P2P copies run at 3 to 6 GB/s in most directions. The benchmark runs with `NCCL_P2P_DISABLE=1` (protocol default), so it uses SHM on every pair
    anyway.
  - Both all-reduce performance criteria pass. See "Interconnect".

## Machine

| | |
|---|---|
| GPUs | 4x NVIDIA L40S (46 GB), driver 570.211.01 (max CUDA 12.8), all PCIe Gen4 x16, 0 uncorrected ECC errors |
| CPU | 2x Intel Xeon Platinum 8558, 4 NUMA nodes. GPUs 0 and 1 on NUMA 0 (root port `04:01.0`, PXB); GPUs 2 and 3 on NUMA 2 (root port `93:01.0`, PIX) |
| Software | vLLM 0.30.0+cu129, torch 2.13.0+cu129, Triton 3.7.1, Transformers 5.22.2 (reference: 5.17.0), NCCL 2.29.7 |
| Run settings | `NCCL_P2P_DISABLE=1`, `NVIDIA_TF32_OVERRIDE=0`, `OMP_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, unprivileged container |

## Interconnect

Raw output: `logs/interconnect_l40s_node2.log`, produced with the benchmark environment. A second run with
`/venv/main` (`logs/interconnect_l40s_node2_venvmain.log`) gives the same verdict.

| Check | Result |
|---|---|
| Peer access on every pair | PASS |
| NCCL uses P2P when enabled | FAIL: P2P/CUMEM on pairs 0-1 and 2-3; SHM across the sockets |
| P2P copy >= 20 GB/s on every pair | FAIL: same-socket pairs 26.3 GB/s; cross-socket pairs 3.2 to 22.8 GB/s, depending on direction |
| 4-GPU all-reduce 256 KiB <= 100 µs (P2P on) | PASS, 76.4 µs (76.8 µs with P2P disabled) |
| 4-GPU all-reduce 64 MiB bus bandwidth >= 10 GB/s (P2P on) | PASS, 13.9 GB/s |
| Concurrent host-to-device, 1 / 2 / 4 GPUs | 27.0 / 44.5 / 82.1 GB/s (scales; no shared uplink, unlike node 1) |
| Estimated decode all-reduce time per step (64 all-reduces, P2P off) | batch 32: 4.9 ms; batch 64: 6.9 ms; batch 128: 10.7 ms |

At batch 32, communication is therefore about 22 % of the 22.10 ms dense step. On node 1 it was 19 of 36 ms.

## Validation (section 5)

| Row | Teacher-forced top-1 agreement | Mean abs logprob difference | Verdict |
|---|---|---|---|
| dense | 0.977 | 0.020 | PASS |
| `palu_r96` | 0.992 | 0.022 | PASS |
| `palu_sharded_r96` | 0.984 | 0.022 | PASS |

Greedy continuation agreement is 0.555 for `palu_r96` and 0.820 for `palu_sharded_r96`. As the protocol notes,
this is not a criterion: a single early near-tie flips the rest of the sequence.

## Results: decode step p50, eager (ms)

Full tables, including means, request E2E/TTFT/TPOT, pool capacity and PaLU-B pair exchange, are in
`formal/eager/SUMMARY.md` and `formal/eager/cells.csv`.

| Arm | b32 | b64 | b128 | A5000 b32 | A5000 b64 |
|---|---|---|---|---|---|
| `dense` | 22.10 | 32.06 | 52.42 | 37.15 | 55.38 |
| `basis_v96` | 24.81 | 30.01 | 47.28 | 32.77 | 49.18 |
| `basis_v64` | 25.34 | 27.06 | 42.31 | 31.01 | 46.92 |
| `palu_g4_r96` | 24.39 | 36.82 | 57.28 | 41.23 | 61.96 |
| `palu_g4_r64` | 24.97 | 33.00 | 51.52 | 38.62 | 59.47 |
| `palu_g4_sharded_r96` | 37.67 | 53.07 | 92.01 | 122.18 (superseded) | 198.18 |
| `palu_g4_sharded_r64` | 38.45 | 49.39 | 84.16 | 118.76 (superseded) | 193.65 |

Request cells, TTFT p50: BasisKV is the lowest at every batch. At batch 32: dense 6135 ms, V96 4973 ms,
V64 4582 ms, PaLU-A r96 6794 ms, PaLU-B r96 7802 ms. At batch 128: dense 23147 ms, V96 18463 ms.

## Batch 32

- On this node, dense decode at batch 32 is 40 % faster than on the A5000.
- The compressed arms do not gain as much, and all of them end up behind dense.
- This run does not explain why.
  - The protocol profiles only dense and the two PaLU arms, so there is no BasisKV kernel breakdown at batch 32.
  - One plausible cause: at batch 32 with 4096-token prompts, reading the KV cache is a small part of the step on
    an L40S. The compressed arms' fixed per-layer work in eager mode (Value reconstruction kernels and their
    launches) may then cost more than the bandwidth they save. This is an unverified hypothesis.
  - A BasisKV profile at batch 32, and the CUDA Graphs pass that was not run (see "Deviations"), would test it.
- Batch-32 means are also noisier than the p50s: `basis_v96` mean 27.22 ms against p50 24.81 ms.

## Profiles (section 8)

`formal/profile_eager/PROFILE.md`, batch 32. NCCL time per step, mean over ranks: dense 15.9 ms, PaLU-A r96
14.5 ms, PaLU-B r96 32.6 ms. The A5000 figures are 25.1, 22.4 and 87.0 ms. The PaLU-B penalty is still mostly
communication: its extra AllGather of weights and partial outputs.

## Deviations from `RUN_L40S.md` (agreed with the requester)

- **Eager pass only.** The `formal full` (CUDA Graphs) pass for dense, BasisKV V96/V64 and PaLU-A r96/r64 was
  not run. The sharded arm only exists in the eager pass anyway.
- **Upload path.** The results were uploaded to `system_benchmarks/v2q8_palu_g4_tp4/l40s_node2/`, not to `l40s/`
  as the protocol says, so they stay separate from `l40s_node1/`.
- **Interconnect check.** The check reports OVERALL FAIL, and the run went ahead after the requester reviewed it.
  The failing criteria are the cross-socket P2P ones, discussed above.
- **Working tree.** It contained uncommitted changes to ShadowKV/Table 3/LRQK files for a separate experiment.
  None of them are imported by the vLLM path used here (`benchmarks/system/bench_llama_vllm.py`,
  `launch_vllm_bench.sh`, `basisserve/vllm`, `basisserve/kernels`).

## Files

| Path | Content |
|---|---|
| `formal/eager/` | 42 cells, `SUMMARY.md`, `cells.csv` |
| `formal/profile_eager/` | dense, `palu_g4_r96`, `palu_g4_sharded_r96` at batch 32; `PROFILE.md` |
| `smoke/eager/` | 7 smoke cells |
| `validation/tp4_l40s/` | References, served outputs, comparison JSON |
| `logs/` | Pipeline, setup, validation, smoke, formal, interconnect, GPU check, `git log -1` |
