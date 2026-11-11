<!-- Agent-generated read-only audit notes (2026-11-10). Key claims re-verified in ../COMPATIBILITY.md; treat the rest as leads, not verified facts. -->
# Audit: quality pairing protocol and Value-reuse state accounting

Repo: `/workspace/BasisServe-CALS` (read-only audit, 2026-11-10). HF: `alexz949/BasisServe-CALS` (metadata + small files only, cached under
`/tmp/claude-0/-workspace-BasisServe-CALS/933b9de8-a7b5-4403-a350-5f7c8680deb1/scratchpad/hf_audit/`).
PDF text was extracted with pypdf (installed to scratchpad) because `pdftoppm` is absent; page numbers below are PDF page indices (equal to printed page numbers).
No GPU work, no tests, nothing written inside the repo.

Legend: **[V]** verified directly from records/arithmetic; **[I]** inference.

---

## TASK A — Quality protocol

### A1. Paper Table 2 protocol (PDF p.7, p.8, p.17)

| Item | Paper statement | Location |
|---|---|---|
| Model | Llama-3.1-8B-Instruct (also Qwen3-8B, Qwen3.5-9B) | p.8 Table 2 |
| Value rep | "Routing uses V96 for Llama-3.1-8B"; "Sparse baselines share the compressed-Value checkpoint of Full-K" → Full-K = full attention **with V96**, not dense | p.17 (C.1), p.8 |
| Router | B16R16 = 16-dim Value-derived Base + 16-dim Key residual, **page size 4**, 64 recent tokens excluded from routed prefix and kept exactly | p.7 setup; p.17 C.1 |
| Budget | 2048 routed tokens per KV group on RULER (256 on LongBench) **plus** 64 recent; no sink mentioned | p.7, p.17 |
| LRQK / Loki | per-head top-k 832 / 856; dedup union per KV group "2,111.68 (LRQK), 1,985.09 (Loki)" | p.17 |
| ShadowKV | "retains its native chunk, outlier, and local-support configuration" | p.17, p.8 |
| RULER | "10 tasks with 100 prompts per task", official generator, seed 42, max length 131,072; Table 2 columns S1 S2 MK1 MK2 MQ MV QA1 QA2 VT FWE (no niah_single_3) | p.17, p.8 |

Table 2 Llama RULER (10-task) reported: Full-K 82.75, Loki 55.21, ShadowKV 75.41, LRQK 79.15, **BasisKV 80.32** (p.8). LongBench avgs 47.02 / 46.54 / 46.72 / 46.62 / 46.91.

**[V] Table 2 Llama RULER numbers reproduce exactly from `results/l31-v96-retrievalmix-ruler1100/predictions/*.jsonl`** (my recomputation, 10-task mean excl. niah_single_3):

| Table 2 row | Arm file | 10-task | 11-task | Paper |
|---|---|---:|---:|---:|
| Full-K | `full.jsonl` | 82.7467 | 84.3152 | 82.75 |
| BasisKV | `b16r16_page4_fisher.jsonl` | 80.3217 | 82.0197 | 80.32 |
| LRQK | `lrqk.jsonl` | 79.1517 | 81.0470 | 79.15 |
| ShadowKV | `shadowkv.jsonl` | 75.4150 | 77.4682 | 75.41 |
| Loki | `loki_pre_rope.jsonl` | 55.2083 | 50.1894 | 55.21 |

Per-task BasisKV values (MK1 98, MK2 78, MQ 99.25, MV 94.5, VT 64.8, FWE 46.67, QA1 75, QA2 47) match only the Page4 arm (`summaries/eval1100_p4_summary.txt` lines 4–16), not the Page32 (`b16r16_page_fisher`, 80.51) or Page1 (80.66) arms.

Other paper tables, for disambiguation:
- Table 8 (p.20, C4-only router, **Page32**, 11 tasks): Full-K 84.45, B16R16 80.88, LRQK 80.41, ShadowKV 77.97, Loki 50.85 = `results/l31-v96-ruler128k/README.md` lines 5–11 [V].
- Table 9 (p.20): 30 prompts/task ablation (B16, R16, R32, B16R16 Page4 2048+64). R32 = 79.52, B16R16 = 79.48.
- Table 11 (p.21, dense V128 + original Wo, 11 tasks): Dense 86.81, ShadowKV 84.21, LRQK 83.72, Loki 52.17, BasisKV 84.99 = `results/l31-v128-ruler128k/README.md` lines 5–11 [V].

**[V] Minor paper inconsistency:** the C.1 support figures (2,111.68 LRQK; 1,985.09 Loki) are reproduced from the **Table 8 C4-only run** (`results/l31-v96-ruler128k/predictions/{lrqk,loki}`), not the Table 2 run. In the Table 2 run the LRQK union is 2,106.68 (range 1,064–3,389) and Loki physical support 1,997.55.

### A2. Quality result inventory (128K)

**(a) `results/l31-v96-retrievalmix-ruler1100/` — this is the Table 2 source.**
- Value checkpoint [V]: protocol `identity_sha256 f6fbef8b…` (all arms; `protocols/*.json`). HF `calibration/llama31-8b-instruct-128k/identity/B.json` hashes to exactly `f6fbef8b66094083…`; it names checkpoint `ckpt_B` with original manifest sha `4fa8237e…`, layer_ranks all 96. HF `results/audits-20261109/l31-exact-page4/configuration.json` proves equivalence of that original manifest to the published `checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/manifest.json` (sha `86bcd936…`, verified by my download), condition `c4_retrieval_50_50`, uniform V96, ALS6/CG16, 32×131072 fit tokens.
- Routing bank [V]: per-sample `bank_sha256["0"] = e71777a2…` = LFS sha256 of HF `…_retrievalmix/router_page4_b16r16/layer_000.safetensors`. Bank json: page_size 4, objective page_fisher, excluded_prefix_pages 0, excluded_recent_tokens 64, 32 fit windows, 64 fit queries, **no diagnostic windows (in-sample)**, teacher = deployed V96 model.
- Budget [V]: `protocols/b16r16_page4_fisher.json` → `page_size 4, physical_group_budget 2112, sink 0, pinned_pages 0, recent 64`; per-sample `selected_tokens_mean` 2109–2112 (mean 2111.34).
- Prefill/decode [V]: full causal Triton prefill, sparse decode only; greedy, native EOS, official caps; `memory_policy`: V96 + routing sidecars on GPU, **exact K in pinned CPU memory**, selected K rows fetched. Quality sidecar is a 144-dim "correctness-oracle" `[predicted post-RoPE K (128), residual code (16)]` (`basisserve/core/c1_v_conditional_k_router.py` lines 75–118, docstring lines 88–92: "Only the residual rank is additional persistent information in the deployable formulation; this materialized base is intentionally a small-experiment quality oracle"; called from `evaluation/llama_b16r16_k_offload_p1.py` line 99). Selection by Python `page_support`, not the runtime CUDA router.
- Hardware [V]: all 1100 records per arm `gpu = NVIDIA RTX PRO 6000 Blackwell Server Edition`, 8 shards, single-GPU workers; torch 2.13.0, transformers 5.17.0, triton 3.7.1. Median s/prompt: Full-K 33.6, B16R16-p4 71.4, LRQK 35.1, ShadowKV 35.8 (harness timings, not runtime latency).
- Samples [V]: 1100 = 11 tasks × 100; prompts sha `1f6389dd…` = HF `evaluation/llama31-8b-instruct-ruler128k-1100-seed42/prompts.json`. Input lengths 111,232–130,942 tokens (mean 130,018).
- Other arms in this dir: Page32 sink32 B2048 (`b16r16_page_fisher`, 82.01/80.51; bank `297df459…` = HF `router_b16r16`), Page1 (82.33/80.66; bank `145b8555…` = HF `router_page1_b16r16`), B0R32 (81.84), Page8 diagnostics (partial task subsets).

**(b) `results/l31-v96-ruler128k/` — Table 8 (robustness, not Table 2).** V96 C4-only checkpoint (HF rev `3a774d59…`, `…uniform_v96_128k_als6`, manifest `f0ac1c9a…`), C4-only router, **Page32, sink32 + recent64 inside B2048**; RTX PRO 6000 Blackwell; 11×100. README lines 15–17, 38–42.

**(c) `results/l31-v128-ruler128k/` — Table 11, dense V128 + original Wo.** identity `0cb442b2…`; B16R16 Page32 sink32 recent64 inside 2048, router "separately refitted Base16/Residual16" on dense V128 (bank layer0 `1f542b46…`); Dense arm TP2 (NCCL host transport, P2P disabled), routing arms single L40S; 11×100; 10-task: Dense 85.59, B16R16 83.49, ShadowKV 82.63, LRQK 82.10 (README lines 5–13; `run_metadata.json`).

**(d) HF `results/ruler-a100-20261109/l31-exact-page4/` (+ `results/audits-20261109/l31-exact-page4/`, the paused first part).** Same V96 retrievalmix checkpoint and identity B.json, same 1100 prompts, Page4, 2048 historical + recent64, sink0, max 2112, full causal prefill — **but the selector is exact full-dimensional FP32 QK page scoring ("this differs from the paper B16R16 approximate router", README.txt line 4)**; exact K on CUDA. 397 RTX A6000 + 703 A100-SXM4-80GB records. 11-task 81.668, paper-10 79.935. Paired vs archived B16R16 Page4: −0.35 pp [−1.04, +0.33] (11-task), −0.39 pp [−1.14, +0.37] (10-task) (`paired_archive/summary.json`). `consistency.json` status "failed" on prefill-logit bitwise equality across A6000/A100 (0/8), tokens/scores equal 8/8. Proxy smoke (`proxy-smoke-comparison.json`): archived B16R16 rerun on A6000 vs Blackwell archive diverged at token 7 on sample 800 (same score) → cross-hardware reruns are not token-identical.

**(e) `ICLR-results/llama31-8b-instruct/` (local and HF).** Value-compression quality only (WT2/C4 PPL, commonsense; C1/PaLU R64/80/96, factor banks, calibration). **No RULER/routing quality.**

**Answer to "Is Page4, 2048+64, no sink the Table 2 point?"** [V] **Yes for BasisKV**: it is exactly the Table 2 BasisKV point (V96 retrievalmix checkpoint, `router_page4_b16r16` Page-Fisher refit, sink0, 2048+64 = 2112 max). It is **not** the operating point of the existing Table 3 runtime, nor of the rebuttal `_p4` runtime arm (see A4). The Page32/sink32/inside-2048 point is the Table 8 / Table 11 / Table 3-runtime point.

### A3. LRQK and ShadowKV quality results

| Run | Context / n | Value rep | LRQK config (support) | ShadowKV config (support) | Hardware | LRQK / ShadowKV score |
|---|---|---|---|---|---|---|
| retrievalmix-ruler1100 (**Table 2**) | 128K, 11×100 | common V96 retrievalmix ("allocated C1 V") | rank32, top832/q-head + recent64, 2 prefill/2 decode iters, tol 0.01; GQA union mean **2106.68**/KV group; key codes `[1,32,T,32]` BF16 = 256 MiB/layer | rank160, chunk8, routed2048 + 48 outlier chunks (384 tok) + native local (~35.5) + generated; physical mean **2485.4**/group; in-repo `basisserve/core/c1_shadowkv.py` | RTX PRO 6000 Blackwell | 79.15 / 75.41 (10-task); 81.05 / 77.47 (11) |
| l31-v96-ruler128k (Table 8) | 128K, 11×100 | V96 C4-only | same; union 2111.68 | same | RTX PRO 6000 Blackwell | 80.41 / 77.97 (11) |
| l31-v128-ruler128k (Table 11) | 128K, 11×100 | **dense V128 native** + original Wo | top832 + recent64; union 2114.07; prefill K/V on CPU restored per layer | **upstream official `ShadowKVCache_CPU` + CUDA kernels**, pinned-CPU V128; routed2048 + 384 outlier + 91 local + generated | L40S | 82.10 / 82.63 (10); 83.72 / 84.21 (11) |
| l31-v96-ruler64k | **64K**, 11×**30** | a different V96 (C4 64×65536 calibration, identity `ddceb704…`) | rank32, **2048/q-head** + recent64 (union **4465.6**, not support-matched) | rank160 chunk8 routed2048 + 48 outliers | L40S | 88.00 / 85.44 (11) |
| archived / evaluation 64K pilots | 64K, 11×**8** | dense V128 | various | various | — | pilot only |

**64K availability [V]:** only `results/l31-v96-ruler64k` (30/task; ours = Page32 sink32 inside B2048; Full-K 88.91, ours 88.03, Loki 86.80, LRQK 88.00, ShadowKV 85.44; prompts 59,900–65,413 tokens, sha `ab7a64a1…`) and 8-prompt/task dense-V pilots (`results/archived_runs/llama31_8b_64k_densev`, `results/evaluation/llama31_8b_chunk8_ruler64k_*`). None uses the Table 2 checkpoint/router/budget. **No TP4 routing-quality result exists for Llama** (search of `tensor_parallel_size: 4` found only the PaLU/V96 4K serving benchmark).

### A4. What the existing runtimes actually compute

- Table 3 / rebuttal runtime [V]: dense **V128**, B16R16 **Page32**, **sink32 pinned + 1952 routed + 64 recent = 2048** support (`results/rebuttal/table3_k_offload_128k/README.md` lines 17–18; `benchmarks/system/bench_tp1_sparse_full_v6.py` lines 49–52; `benchmarks/system/table3_page_select.cu` lines 23–25 `kRoutedPages = 1984/kPageSize`, `kPinnedPages = 32/kPageSize`, lines 321–323 pin pages 0..kPinnedPages−1; `support_ids [1,8,2048]` in `inventory.csv`). Router bank = V128 bank (HF `system_benchmarks/table3_k_offload_128k/factors/ours_b16r16/layer_000.safetensors` LFS sha `e8a79558…`; json: value_feature_mode dense-v128, page 32, excluded_prefix_pages 1, identity `36c6f53b…`). This hash differs from the Table 11 quality bank (`1f542b46…`) and appears nowhere else in the repo → **no RULER quality exists for the exact runtime router bank**.
- Rebuttal `_p4` arms [V]: page-size-general router at page 4 using the **same Page32-fitted V128 factors**, still sink32 (8 pinned pages) + 1952 routed + 64 recent = 2048 (`bench_table3_k_offload.py` lines 112–121). ≠ Table 2 Page4 (Page4-refit V96 bank, sink0, 2112).
- Runtime prompts [V]: "Three fixed prompt cohorts use calibration windows 0/1/2. This is a timing study, not a held-out quality evaluation." (`results/system_benchmarks/tp1_offline/SUMMARY.md` line 11).
- Runtime baselines [V]: LRQK "CPU exact K/V, rank 32, active 2048 plus lite 64"; ShadowKV "default CPU V / rank-160 reconstructed K path" (same file line 72). LRQK runtime config ≠ quality config (top832+recent64). GPU-local LRQK OOMs at 64K and 130048 (`tp1_lrqk_local/SUMMARY.md`).
- No V96 TP1 long-context runtime exists (V96 runtimes are TP8 4K serving only) [V, by search].

### A5. Conclusion — what can be paired exactly

1. **Pairable now (quality side complete), TP1 ~128K, Table-2-aligned:** V96 retrievalmix checkpoint (`…_als6_retrievalmix`, manifest `86bcd936`/orig `4fa8237e`, identity `f6fbef8b`), `router_page4_b16r16` (layer0 `e71777a2`), Page4, sink0, 2048 historical + 64 recent (≤2112), full causal prefill, greedy/native EOS/official caps, 1100 frozen prompts (`1f6389dd`/`d7118489`); Full-K-V96 82.75, BasisKV 80.32, LRQK (rank32, top832+r64) 79.15, ShadowKV (rank160 chunk8 routed2048+48 outliers+local) 75.41 (10-task; 11-task 84.32/82.02/81.05/77.47). **But no runtime implements this point**; a new latency arm must implement V96 Value path + Page4 bank + sink0 + 2112 support, and baselines with the quality configs (LRQK top832+r64, in-repo V96 ShadowKV). Because the quality harness uses a 144-dim oracle sidecar, Python selection, K on pinned CPU and Blackwell hardware, "same deployed computation" requires re-running RULER through the new runtime (or a recorded token-equality audit against the archived jsonl); the A6000 proxy smoke shows divergence is possible.
2. **Closest to the existing V128 runtime:** Table 11 (dense V128, Page32 sink32 inside 2048) — same budget semantics, different router bank (`1f542b46` vs `e8a79558`), different ShadowKV/LRQK runtime configs for LRQK. Not exact.
3. **Gaps:** (i) V96 vs V128 mismatch between Table 2 quality and Table 3 runtime; (ii) router-bank mismatch even for V128; (iii) Page4/sink0/2112 vs Page32/sink32/2048 budget mismatch; (iv) **no 64K quality** at the Table 2 configuration (only a 30-prompt/task Page32 run with a different V96 checkpoint and unmatched LRQK support) and **no TP4 quality**; (v) no quality for the no-Base (Value-reuse) or K-offload runtime arms; (vi) runtime prompts are C4 calibration windows, not RULER.

---

## TASK B — Value-reuse state accounting (existing data only)

Sources: `results/rebuttal/table3_k_offload_128k/{README.md, runtime_manifest.json, arms/formal/{SUMMARY.md,summary.csv,runs.csv,inventory.csv,outcomes.json,manifest.json}, pages/formal/*, pages/router_validation.json, arms/check/placement_check.json, pages/check/placement_check.json}`; arithmetic in `/tmp/claude-0/-workspace-BasisServe-CALS/933b9de8-a7b5-4403-a350-5f7c8680deb1/scratchpad/audit/arith_out.txt`.

### B1. Arms [V]
Setup: Llama-3.1-8B-Instruct BF16, TP1/B1, one L40S (driver 550.54.15, torch 2.13.0+cu126), 130,048-token prompts (cohorts 0/1/2), 16 conditioning + 128 measured decode steps, prefill excluded; **dense V128**, B16R16 full scan, 1984 routed + 64 recent (README lines 13–21).

| Arm | Keys | Base code | Router |
|---|---|---|---|
| `dense_resident` | GPU | — | dense attention |
| `basis_resident` | GPU | **separate `base_cache` [1,8,T,16]** | archived w16u1 Page32 |
| `basis_resident_nobase` | GPU | view `value_cache[..., :16]` of rotated V | archived Page32 |
| `basis_k_offload_nobase` | pinned host + GPU slots | view of rotated V | archived Page32 |
| `*_p32` / `*_p4` (pages/) | resident / offload, both nobase | view | page-size-general router, same Page32-fitted factors |

All 12 + 12 formal trials complete (`outcomes.json`, returncode 0).

### B2. Base-cache saving at 130,048 tokens [V]
Inventory (`arms/formal/inventory.csv` line 8): `basis_resident` base_cache shape `[1, 8, 130193, 16]` bf16 ×32 layers, backing 1,066,541,056 B. Lines 31/54: nobase arms base_cache backing **0 B**, alias `value_cache`.

Arithmetic (32 layers × 8 KV heads × 16 dims × 2 B):
- capacity 130,193 (= 130,048 + 16 + 128 + 1): 1,066,541,056 B = **0.993294 GiB** ← matches inventory 0.9933
- stored 130,192: 1,066,532,864 B = 0.993286 GiB (inventory "active")
- exactly 130,048: 1,065,353,216 B = 0.992188 GiB

PyTorch-level fields, `basis_resident − basis_resident_nobase` (summary.csv): decode_resident 32.975116 − 31.975116 = **1.000000 GiB**; post_prefill, decode_peak_allocated, prefill_peak_allocated also 1.000000; decode_peak_reserved 1.0117; device_used 0.9766. The exact 1.000 GiB is allocator rounding: each per-layer tensor is 33,329,408 B → 16 × 2 MiB blocks = 32 MiB; ×32 = 1 GiB. Inventory GPU totals: 32.952 − 31.959 = 0.993 GiB. So: **tensor bytes eliminated 0.9933 GiB; allocated-memory drop 1.000 GiB**.

### B3. Persistent incremental routing state [V arithmetic; R32 analytical only]
(over 32 layers × 8 KV heads, BF16; Value cache counted as already present)

| Configuration | Persistent extra dims/token/KV-head | Bytes/token | @130,048 | @capacity 130,193 | Dims scanned/token/KV-head |
|---|---:|---:|---:|---:|---:|
| B16R16, separate Base cache (measured) | 16 Base + 16 residual = 32 | 16,384 | 2,130,706,432 B = 1.984375 GiB | 1.986588 GiB (inv. 0.9933+0.9933) | 32 |
| B16R16, no-Base / Value reuse (measured) | 16 residual | 8,192 | 1,065,353,216 B = 0.992188 GiB | 0.993294 GiB (inv. residual_cache) | 32 (16 from V + 16 residual) |
| Hypothetical independent R32 (**analytical, not measured**) | 32 Key-code | 16,384 | 1.984375 GiB | 1.986588 GiB | 32 |
| reference: exact K (128) | — | 65,536 | 7.9375 GiB | 7.9464 GiB | — |

Value reuse halves persistent routing state (16 KiB → 8 KiB/token; −0.99 GiB at 130K); it does not change dims scanned (32/token). No R32 latency exists or is implied. [V, quality record] LRQK's quality-path key codes are `[1,32,T,32]` BF16 = 64 KiB/token over 32 layers — same size as exact K.

### B4. Router-scan and step timings [V]
| ms/step | basis_resident | nobase | Δ | k_offload_nobase |
|---|---:|---:|---:|---:|
| router_scan | 11.750 | 11.896 | **+0.146 (+1.24%)** | 11.905 |
| page_selection | 0.922 | 1.013 | +0.091 | 1.011 |
| attention_block | 14.177 | 14.411 | +0.234 (+1.65%) | 16.301 |
| full_model wall | 37.400 | 37.830 | +0.429 (+1.15%) | 39.481 |

Per-cohort attention block: 14.19/14.16/14.18 vs 14.37/14.41/14.47 (no overlap). Pages: p32 nobase 11.889 / p4 nobase 13.357 router_scan; p4 costs +1.99 ms attention, +2.01 ms wall vs p32. **No speedup from Value reuse; slightly slower.** [I] Plausible cause: same 32 dims/token scanned, Base read from a strided view (32 B every 256 B) instead of a contiguous buffer.

### B5. K-offload is relocation, separate from Base saving [V]
`basis_resident_nobase` → `basis_k_offload_nobase`: decode_resident 31.975 → 24.289 = **7.6866 GiB** less GPU; host pinned Key 7.94635 GiB (= 32×8×130,193×128×2 = 8,532,328,448 B; inventory line 56, pinned True). New GPU state: slot_key_cache 0.125 GiB + slot maps 0.1339 GiB (143,805,440 B) → inventory net 7.9464 − 0.2589 = 7.6874 GiB. GPU+host = 24.29 + 7.95 = 32.23 GiB > 31.98 resident: **bytes moved, not eliminated** (README lines 65–67; SUMMARY lines 28–29). Both arms are nobase, so the K-offload delta excludes the Base saving; `basis_resident → basis_k_offload_nobase` = 8.6866 = 1.000 (Base) + 7.687 (offload). Cost: 131,705 of 521,589 selected Keys fetched per step (32.15 MiB), +1.044× wall, +1.131× attention block. Placement check: offload vs resident-nobase is bitwise identical (support 16/16 steps, K/V/residual caches, attention, logits 0.0).

### B6. Numerical caveat of the no-Base rotation [V]
- Implementation: `benchmarks/system/bench_table3_k_offload.py` lines 81–91: complete QR of `base_left` in FP64, rotate `v_proj` rows by Qᵀ and `o_proj` columns by Q, `base_right ← R[:16]·base_right`, `base_left ← I[128,16]`, results copied back into **BF16** weights; `base_cache = value_cache[..., :16]`.
- Recorded caveat: README lines 35–38 ("algebraically lossless but re-rounds the weights to BF16 … identical argmax, max logit difference 0.67").
- Measured (`arms/check/placement_check.json` lines 23–44; 130,048 tokens, cohort 0, 16 teacher-forced steps): `steps_with_identical_support 0/16`, `cache_identical` key/value/residual all false, attention max |Δ| 5.40, logits max |Δ| 0.672, mean |Δ| 0.0568, argmax agreement 1.0. Smoke (24,576 tokens, 4 steps): attention 3.625, logits max 0.344, mean 0.0483, argmax 1.0. [I] Value-cache difference is expected by construction (rotated basis); Key/residual differences come from downstream propagation.
- `pages/router_validation.json` validates the page-size-general router (page scores ≤7.2e-7 from token scores; p32 bitwise equal to archived; selection exact), **not** the rotation. No nobase-vs-Base check exists for p4. **No RULER/quality measurement of any nobase or K-offload arm exists.**

---

## Open uncertainties
1. Provenance of the runtime V128 router bank (`e8a79558`, identity `36c6f53b`, windows `80c204be`): not matched to any quality record; calibration mix unknown from local data.
2. Whether a V96 Value-reuse runtime would behave like V128 (Base map acts on V96 latents per `c1_v_conditional_k_router.py` line 89–90; rotation of the latent basis is analogous [I], unmeasured).
3. Whether Page4 runtime with a Page4-refit V96 bank and 2112 support changes router-scan cost beyond the measured Page4/Page32 difference with V128 factors.
4. Cross-hardware token stability (Blackwell → L40S) for any re-pairing; A6000 proxy smoke diverged on 1 of 2 samples.
