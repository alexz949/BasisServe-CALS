# TP1 retest on node6: BasisKV, ShadowKV and LRQK at 64K and 128K

2026-11-11, Llama-3.1-8B-Instruct (BF16, original V128/Wo, revision `0e9e39f2`), TP1, batch 1, one NVIDIA L40S
per run on Slurm node `node6`. This note puts the three result folders of the day side by side:

| Folder | Content |
|---|---|
| `tp1_128k_retest/` | 130048 prompt tokens: BasisKV and modified ShadowKV, formal; LRQK, failed probes; checks |
| `tp1_64k_retest/` | 65536 prompt tokens: BasisKV and modified ShadowKV, formal |
| `tp1_lrqk_retest/` | 65536 prompt tokens: LRQK with the prefill stream-order fix (`REPORT.md` there has the details) |

Each folder has its generated `SUMMARY.md`, `runs.csv`, `summary.csv`, `manifest.json` and
`correctness-summary.json`.

## Bottom line

- **128K: BasisKV is ready to decode 11.0 s earlier than the modified ShadowKV** (35.1 s against 46.1 s) and
  finishes a 128-token request in 39.3 s against 50.1 s. Its decode step is 1.5 ms slower (33.2 ms against
  31.7 ms) and it holds 6.6 GiB more on the GPU.
- **128K: LRQK does not run on one L40S.** Both placements ran out of GPU memory in the first prefill, inside
  upstream's ALS factor fit. Each was tried once.
- **64K: BasisKV is ready 6.8 s earlier than the modified ShadowKV** (11.47 s against 18.25 s); the decode steps
  are within 0.5 ms (29.4 ms and 29.9 ms). **LRQK decodes at 303 to 371 ms per step**, 10 to 13 times the
  BasisKV step.
- **The ShadowKV numbers are for a modified runtime**, with upstream's per-layer `gc.collect()` and
  `torch.cuda.empty_cache()` before each SVD turned off. Unmodified ShadowKV was only run as short probes;
  those suggest the cleanup costs about 5 s of prefill at every length.

## What was run

| Arm | What it is | Offloaded to pinned host memory | CUDA allocator |
|---|---|---|---|
| `basis_koff_v128_reuse_opt` | BasisKV: B16R16 router, Page32, 32 sink + 1952 routed + 64 recent tokens, Base read from V, lean forward, fast router | exact K | `expandable_segments:True` |
| `shadowkv_upstream_nohk_v128` | ShadowKV with per-layer pre-SVD GC and empty-cache calls disabled; otherwise upstream `ByteDance-Seed/ShadowKV@e51904cd`, rank 160, chunk 8, budget 2048 | V | PyTorch default |
| `lrqk_upstream_koff_v128` | LRQK with the prefill stream-order fix `be7c8a2`; upstream `tenghuilee/LRQK@caf16293`, rank 32, top-k 2048, 64 lite tokens | exact K | PyTorch default |
| `lrqk_upstream_kvoff_v128` | Same, upstream's own placement | K and V | PyTorch default |

- **Protocol.** One fresh process per run: same-shape warmup request, fresh state, then one measured request that
  generates 128 tokens. No component timers. Medians are over three different prompts (cohorts 0, 1, 2), one run
  each; ranges are across prompts.
- **Request mode only** for BasisKV and ShadowKV. There are no steady runs for them, so no per-step percentiles
  and no inventory of pinned host memory.
- **Inputs.** 128K: `tp1_offline/inputs/prompts.safetensors`, 130048 tokens per cohort, the TP1 Exp 1 inputs.
  64K: row 0 of cohorts 0, 1, 2 of `tp4_64k/inputs/prompts.safetensors`, the Exp 3 inputs. The 64K prompts are
  not truncations of the 128K ones.
- **Host.** Xeon Gold 6338, driver 550.54.15, conda env `basis` (PyTorch 2.13.0+cu129, Transformers 5.17.0),
  24 threads on 12 physical cores of the GPU's NUMA node. Repository `be7c8a2` plus uncommitted changes.

## 128K (130048 prompt tokens)

| | BasisKV | ShadowKV, modified | LRQK, K offload | LRQK, K and V offload |
|---|---|---|---|---|
| Prefill end (TTFT) | 35.12 s (34.86 to 35.29) | 44.98 s (44.95 to 45.03) | out of memory | out of memory |
| Placement after prefill | none | 1.12 s (1.10 to 1.16) | | |
| Decode-ready | 35.12 s (34.86 to 35.29) | 46.11 s (46.10 to 46.13) | | |
| Request end to end, 128 tokens | 39.32 s (39.08 to 39.52) | 50.13 s (50.12 to 50.19) | | |
| Mean decode step | 33.20 ms (33.06 to 33.35) | 31.66 ms (31.61 to 31.98) | | |
| GPU resident, allocated | 24.23 GiB | 17.60 GiB | | |
| Prefill peak, allocated / reserved | 30.75 / 31.44 GiB | 20.61 / 21.72 GiB | | |
| Decode peak reserved | 31.44 GiB | 18.42 GiB | | |
| Process RSS | 9.71 GiB | 10.18 GiB | | |

All six formal runs ran on the same physical GPU (PCI `4F:00.0`), BasisKV and ShadowKV back to back within each
cohort. Memory figures are identical across cohorts.

**LRQK at 128K.** Both placements failed in the first prefill (the warmup request), before any decode step:
`torch.OutOfMemoryError` in upstream `lrqk_attention.py:399`, the convergence check of the ALS factor fit
(`_lrqk_prefill_inv_w1`), asking for 508 MiB with 42.2 GiB already allocated by PyTorch on the 44.5 GiB card.
K-only had 46 MiB free at that point; K and V offload had 506 MiB free and ran about 40 s longer before failing.
The stream-order check passed in the same session, so the fixed code was the one loaded. This is one attempt
per placement, with PyTorch's default allocator, on this implementation and configuration; it is not a statement
that the method cannot handle 128K.

## 64K (65536 prompt tokens)

| | BasisKV | ShadowKV, modified | LRQK, K offload | LRQK, K and V offload |
|---|---|---|---|---|
| Prefill end (TTFT) | 11.47 s (11.34 to 11.61) | 17.42 s (17.35 to 17.44) | 15.73 s (15.70 to 15.76) | 18.07 s (17.98 to 18.36) |
| Placement after prefill | none | 0.82 s (0.78 to 0.87) | none | none |
| Decode-ready | 11.47 s (11.34 to 11.61) | 18.25 s (18.13 to 18.31) | 15.73 s (15.70 to 15.76) | 18.07 s (17.98 to 18.36) |
| Request end to end, 128 tokens | 15.22 s (15.08 to 15.35) | 22.05 s (21.92 to 22.09) | 54.23 s (54.09 to 55.87) | 65.45 s (64.84 to 66.48) |
| Mean decode step | 29.44 ms (29.43 to 29.58) | 29.86 ms (29.78 to 29.94) | 302.9 ms (302.2 to 316.1) | 370.8 ms (369.0 to 381.1) |
| GPU resident, allocated | 19.72 GiB | 16.46 GiB | 26.04 GiB | 22.03 GiB |
| Prefill peak, allocated / reserved | 23.00 / 23.35 GiB | 23.30 / 27.79 GiB | 26.20 / 37.51 GiB | 22.31 / 29.90 GiB |
| Process RSS | 5.77 GiB | 10.18 GiB | 19.99 GiB | 27.97 GiB |

- The LRQK columns are its request-mode runs from `tp1_lrqk_retest/`, measured about two hours earlier on the
  same host. LRQK also has steady runs there: p50 314.4 ms (K offload) and 332.5 ms (K and V offload).
- LRQK's K and V offload prefill peak reserved was 29.90 GiB in two cohorts and 31.08 GiB in one.
- **Trial, not the baseline:** with upstream's per-layer `empty_cache()` in decode turned off, LRQK's request step
  was 234.7 ms (K offload) and 242.9 ms (K and V offload) on cohort 0, one run each, with identical output and
  reserved GPU memory of 39.1 GiB and 36.6 GiB during decode.

## ShadowKV: what turning off the pre-SVD cleanup changes

Unmodified ShadowKV was not part of the formal runs. The only same-host comparison is one short probe per length
(full prefill, 8 generated tokens, cohort 0, both arms back to back on one GPU). These are single runs.

| Probe | Prefill end, upstream | Prefill end, modified | H2D, upstream / modified | Prefill peak reserved, upstream / modified |
|---|---|---|---|---|
| 8192 tokens | 7.74 s | 2.72 s | 0.35 / 0.37 s | 16.48 / 16.89 GiB |
| 65536 tokens | 22.05 s | 17.14 s | 0.69 / 0.92 s | 24.82 / 27.79 GiB |
| 130048 tokens | 51.03 s | 46.00 s | 1.20 / 1.28 s | 22.18 / 21.72 GiB |

1. **Prefill.** About 5.0 s shorter at every length (5.02, 4.91 and 5.03 s), so the cost is per layer, not per
   token.
2. **H2D.** 0.02 to 0.23 s longer in the modified arm. It does not absorb the 5 s.
3. **Decode.** The probes have only 7 decode steps (31.6 and 31.3 ms at 128K); there is no formal control, so no
   claim.
4. **Peak reserved.** Higher at 8K (+0.4 GiB) and 64K (+3.0 GiB), lower at 128K (-0.5 GiB).
5. **Output.** Identical. The modified arm, fed the upstream arm's tokens, predicts the same token at 8 of 8
   positions with a maximum log-probability difference of 0, at all three lengths.

The scope of the modification was checked by counting calls (8192 tokens, not timed): inside `get_svd`,
upstream makes 32 `gc.collect()`, 32 `empty_cache()`, 64 `synchronize()` and 32 `torch.svd` calls; the modified
arm makes 0, 0, 64 and 32. `H2D` keeps its 2 + 2 cleanup calls in both, and the two functions are bound back
after an exception inside `get_svd`.

## Limits

- **Different GPUs and sessions within the 64K table.** BasisKV and ShadowKV: cohort 0 on the GPU at PCI
  `4F:00.0`, cohorts 1 and 2 on `D5:00.0` while the 128K job ran on `4F:00.0` (different NUMA nodes and disjoint
  cores). LRQK: all on `D5:00.0`, earlier the same day. Within a cohort, BasisKV and ShadowKV always ran back to
  back on one GPU.
- **Allocator settings differ by arm**, as in the archived runs: BasisKV uses `expandable_segments:True`, the
  upstream baselines PyTorch's default. Reserved-memory figures are affected by this choice.
- **The methods offload different things.** BasisKV keeps V on the GPU and puts exact K on the host; ShadowKV
  puts V on the host and keeps low-rank K factors on the GPU. Pinned host memory was not inventoried in these
  request-mode runs; process RSS is the only host-side figure.
- **The ShadowKV arm is not upstream ShadowKV.** No formal same-host number exists for the unmodified runtime.
- **Not a repeat of the earlier L40S results.** Those used another host (Xeon Platinum 8558, driver 570,
  Transformers 5.22.2). There, BasisKV reached 36.5 s TTFT and 33.7 ms steady p50 at 128K.
- **Quality is not measured here.** The checks cover implementation consistency only.
- **Shared host.** Other users' jobs ran on the other GPUs. Other processes used the 24 pinned CPUs for at most
  2.4 logical CPUs in any 5 s sample of these runs; no other process used the benchmark's GPU.
- **Occupied GPU skipped.** One submission was assigned the GPU with 2 uncorrected ECC errors (PCI `52:00.0`);
  the launcher stopped before measuring, and the job was resubmitted with the same settings while a placeholder
  job held that GPU.
