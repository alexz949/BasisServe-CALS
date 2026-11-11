# Exp 3 (TP1, 64K) and the follow-up trials, 2026-11-11

Vast.ai 4x L40S node 2, Llama-3.1-8B-Instruct BF16, V128. This note covers what was added after round 2
(`formal_v2/exp1`, `formal_v2/exp2`): the formal Exp 3, and several trials that explain the baselines' numbers.
Only section 1 is a formal result. Everything else is a single-run trial or a smoke probe and is labelled so.

Code: `benchmarks/system/bench_v2q8_tp.py` (arms), `run_v2q8_exp_formal.py` (Exp 3), `v2q8_affinity.py`.
The scripts that launched the trials are copied in `diagnostics/trial_scripts/`.

## 1. Exp 3, formal: TP1, 65536 + 128 tokens, B1

Row 0 of the Exp 2 cohorts, GPU 0, serial, request and steady on cohorts 0/1/2 and one breakdown per arm:
28 of 28 runs complete, no OOM. Medians of the three cohorts. Results in `formal_v2/exp3/`.

| | BasisKV `basis_koff_v128_reuse_opt` | ShadowKV upstream | LRQK upstream, K+V offload | LRQK upstream, K offload |
|---|---|---|---|---|
| Prefill until decode-ready | 11.9 s | 27.0 s | 21.8 s | 18.7 s |
| End to end, 128 tokens | 15.6 s | 30.8 s | 76.8 s | 79.3 s |
| Steady decode, median per step | 30.1 ms | 30.6 ms | 369 ms | 430 ms |
| Steady decode, range of cohort medians | 30.0 to 30.1 | 30.6 to 30.7 | 353 to 578 | 385 to 442 |
| GPU resident (allocated) | 19.7 GiB | 16.5 GiB | 22.0 GiB | 26.0 GiB |
| Prefill peak allocated | 23.0 GiB | 23.3 GiB | 22.3 GiB | 26.2 GiB |
| Prefill peak reserved | 23.4 GiB | 25.2 GiB | 29.9 GiB | 37.5 GiB |
| Process RSS | 5.8 GiB | 10.2 GiB | 28.0 GiB | 20.0 GiB |
| Offloaded to pinned host memory | exact K, 4.01 GiB | V, 4.01 GiB | K and V | K |

Prefill breakdown per request (separate instrumented run, cohort 0):

| | BasisKV | ShadowKV | LRQK K+V offload | LRQK K offload |
|---|---|---|---|---|
| Attention | 6.13 s | 5.59 s | 5.55 s | 5.55 s |
| Rest of the model | 5.48 s | 4.98 s | 6.29 s | 6.28 s |
| Method-specific | 0.28 s | 16.4 s | 10.18 s | 6.77 s |

ShadowKV's 16.4 s: `get_svd` 11.04 s (the SVD itself 3.79 s, the rest 7.25 s), landmarks, outliers and Value
offload 4.34 s, moving the factors back to the GPU after prefill 1.02 s. LRQK's: ALS fit 1.5 s, KV store 4.8 to
8.2 s. LRQK decode per step (instrumented): factor update 139 to 158 ms, `empty_cache` 67 to 86 ms, CPU gather of
missed K/V 55 to 75 ms, hit/miss bookkeeping 33 to 39 ms, attention 7 ms.

LRQK uses its paper's RULER-128K settings (rank 32, top-k 2048, 64 lite tokens). Both LRQK placements are stable
here; at 130048 tokens (Exp 1) K-only offload is out of memory in 7 of 7 runs and K+V offload in 5 of 7.

## 2. Trial: upstream ShadowKV's per-layer cleanup inside `get_svd`

Upstream calls `synchronize`, `gc.collect()` and `empty_cache()` before every layer's SVD. Trial arms turn the two
cleanup calls into no-ops while `get_svd` runs; upstream's source is not edited. TP1, 64K, B1, cohort 0, one run
each (`smoke_v3/exp3/*_cleanup_trial_*`). The 128 generated tokens are identical in all four.

| | Decode-ready | Prefill peak reserved |
|---|---|---|
| Upstream as is | 29.2 s | 25.2 GiB |
| `gc.collect()` off (`shadowkv_upstream_nogc_v128`) | 21.6 s | 25.2 GiB |
| `empty_cache()` off (`shadowkv_upstream_noempty_v128`) | 28.4 s | 27.8 GiB |
| Both off (`shadowkv_upstream_nohk_v128`) | 21.2 s | 27.8 GiB |

`gc.collect()` accounts for about 7.6 s of the 32 calls; `empty_cache()` for almost nothing. The same cohort ran
28.5 s in the formal Exp 3, so single runs differ by about 1 s. No formal run of these arms exists; the formal
ShadowKV baseline stays upstream as is.

## 3. ShadowKV TP SVD placement modes (TP4), checks and probes only

Arms `shadowkv_upstream_v128` (centralized), `shadowkv_upstream_reqpar_v128` (request_parallel) and
`shadowkv_upstream_repl_v128` (replicated); see `basisserve/core/shadowkv_upstream_tp.py`. Results in `svd_modes/`.

- Checks, all passed: layout with traceable Keys (W=4 NCCL, 8 ranks with gloo on CPU tensors); real-model check at
  8192 tokens (replicated full-K projection bitwise equal to the owning ranks; reconstruction error equal in all
  modes); teacher-forced check at batch 8 (top-1 agreement 0.9658 in all three arms, dense reference 0.9912).
- 65536-token B16 probes, one run each with the smoke settings: decode-ready 243.8 s centralized, 195.2 s
  request_parallel, 240.8 s replicated; prefill peak allocated 18.6 to 18.8 GiB per GPU.
- The formal comparison has not been run. Command: `python benchmarks/system/run_shadowkv_svd_modes_formal.py`
  (21 runs, about 3.3 h, output to `svd_modes/formal/`).

## 4. LRQK decode speed against its paper

The paper (arXiv 2510.23649, Table 9: A100 40 GB, batch 1, rank 16, top-k 1024, 16 lite tokens) reports 6.8
tokens/s decode at 64K, 147 ms per step, and about 16 s prefill. Trials at TP1, 64K, B1, cohort 0, K+V offload
unless stated (`smoke_v3/exp3/lrqk_upstream_*_t9_*`):

| Trial | Result |
|---|---|
| Table 9 settings (`lrqk_upstream_kvoff_t9_v128`) | prefill 21.2 s; decode 431 ms mean over a request, 336 ms steady median. K-only offload: 277 ms mean |
| CPU cores for the process: 4 / 12 / 24 / 48 physical cores | steady median 302 / 354 / 315 / 513 ms (formal policy, 12 cores with hyperthreads: 336 ms) |
| `torch.compile` off (`TORCH_COMPILE_DISABLE=1`), Table 9 settings | steady median 267 ms, slowest step 429 ms; tokens differ from compile-on after the 10th |

Ruled out as the cause of the gap to the paper: the settings, PCIe (both GPUs are PCIe 4.0 x16; pinned host-to-GPU
copies measure 25 GiB/s here), the CPU model (Xeon Platinum 8558 here, EPYC 7742 in the paper) and the core count.
Upstream targets PyTorch 2.5.1 and transformers 4.47.1; this environment has 2.13.0 and 5.17.0. In isolation the
compiled decode solver takes 2.45 ms per layer call and the same function uncompiled 1.54 ms, with equal results
to about 1e-8 relative; every LRQK log has Dynamo's `recompile_limit` warning.

`torch.compile` off at the formal settings, **stopped, not a result**: arms `lrqk_upstream_kvoff_nocompile_v128`
and `lrqk_upstream_koff_nocompile_v128` were started as a formal Exp 3 rerun. The first two runs (K+V offload,
cohort 0, kept in `formal_v2/exp3/`) decoded at 3759 ms per step (request mean) and 1352 ms (steady median),
several times slower than compile on, so the rerun was stopped by hand after them. The formal LRQK numbers remain
the compile-on ones in section 1.

Diagnosis right after (10:44 to 11:02 UTC, same geometry and formal settings, instrumented breakdown runs back to
back, `smoke_v3/exp3/*_diag*`): the slowdown was the machine in that period, not the compile switch.

| Run, in order | Decode per step | of which `empty_cache` | CPUs busy for other tenants (median) |
|---|---|---|---|
| compile off | 1136 ms | 355 ms | 72 |
| compile on | 1129 ms | 327 ms | 69 |
| compile off | 669 ms | 170 ms | 54 |
| compile on, formal Exp 3 breakdown at 08:50 UTC, for reference | 422 ms | 86 ms | not recorded |

Compile on was as slow as compile off in the same window, every component was slower by a similar factor, and
the third run sped up as the other tenants' load fell. So the two stopped `nocompile` runs say nothing about
`torch.compile`, and whether compile off helps at the formal settings is still open.

## 5. Caveat: the host's CPUs are shared

With none of our jobs running, about 74 of the node's 192 logical CPUs were busy (load average about 80), including
35% of the 24 CPUs a GPU-0 job is pinned to. During the LRQK runs sampled in `logs/cpu_usage_exp3_lrqk_nocompile.jsonl`
the LRQK process used 12 to 19 CPUs in prefill and about 9 in decode while other tenants used 70 to 73. The load
during the formal runs was not recorded. Between 10:20 and 11:00 UTC upstream LRQK decoded 2 to 3 times slower
than in the formal Exp 3 two hours earlier with the same code and settings (section 4), so an LRQK timing from
this node needs repeats spread over time. CPU-bound parts (LRQK decode, ShadowKV's `gc.collect()`) are the ones
exposed to it; BasisKV's and ShadowKV's decode repeat within 0.1 ms across cohorts.
