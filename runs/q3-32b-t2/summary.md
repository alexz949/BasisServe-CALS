# Qwen3-32B, RULER-128K under the Table 2 protocol (Full-K vs BasisKV B16R16 Page4)

Status (2026-11-03): preparation. The paper has no Qwen3-32B RULER row; this run applies the Table 2 recipe
(Llama-3.1-8B-Instruct / Qwen3-8B rows) to Qwen3-32B on one node with 8 x A100-80GB (NVLink), no Slurm, conda `lowrank`.

## Protocol (from the paper, Section 5 + Appendix C.1, and the recorded Qwen3-8B Table 2 artifacts)

| Item | Setting |
|---|---|
| Model | `Qwen/Qwen3-32B` revision `9216db57`, BF16, 64 layers, 64 Q / 8 KV heads (GQA 8), head_dim 128 |
| RoPE at 128K | static YaRN factor 4, `original_max_position_embeddings` 32768 (official Qwen recipe; `routing_config(rope='yarn4')`) |
| Value | uniform V96 on all 64 layers (75% Value retention), dense K; joint V/O ALS: 6 sweeps, CG 16 fixed, covariance damping 1e-7, FP32 work, BF16 factors |
| Value calibration | 48 x 131072-token windows: fit 0-15 C4 (seed 20260921), 16-31 synthetic retrieval (seed 20260928 / haystack 20260929), held-out 32-47 C4 |
| Router | Base16 (V-code -> pre-RoPE K affine reduced-rank map) + page-Fisher Residual16, page 4, no pinned sink, recent 64 excluded; 64 fit queries (4 strata x 16); ALS 40, PCG 100, relative damping 1e-5; deployed-V96 teacher; 32 fit windows, 0 diagnostic |
| Runtime | 2048 routed tokens + 64 recent per KV group (hard 2112), Page4, GQA max over the 8 query heads, FP32 scores |
| Data | RULER-128K, 11 tasks x 100 prompts, seed 42, base template (`datasets/qwen3-32b-ruler128k-1100`, same tokenizer/sha as the Qwen3-8B set); Table 2 averages 10 tasks (drops `niah_single_3`) |
| Prompt | chat template, `enable_thinking=false`, official answer prefix as assistant text (`--chat-template --prompt-layout chat_answer_prefix`) |
| Decoding | greedy, native EOS, official caps; BF16; chunked-MLP prefill, FlashAttention SDPA prefill with zero-padded V |

Qwen3-8B tokenizer and tokenizer_config are byte-identical to Qwen3-32B's (sha256 `aeb13307...` / `d5d09f07...`), so the
32B windows regenerated with the same seeds should reproduce the 8B window token ids exactly (checked after generation).

## What exists on HF vs what this run builds

| Artifact | HF (`alexz949/BasisServe-CALS`) | This run |
|---|---|---|
| V96 | `checkpoints/qwen3-32b-128k/uniform-v96-als6-cg16`: C4-only 32 x 128K, damping 1e-5, YaRN original 40960 | refit on the 48 mixed windows with the Table 2 recipe (`$R/v96`, `$R/ckpt_v96`, `$R/identity/v96.json`); the C4-only bank is packaged as `$R/ckpt_v96_c4` for smokes/ablation |
| Router | `uniform96-b16r16-als40-pcg100`: Page32, one pinned sink page, C4-only, 32 queries, YaRN original 40960 (Sept-21 attempt scored 0.816 on 579 prompts) | Page4 / no sink / recent 64 / 64 queries / mixed windows (`$R/router/ours_b16r16`) |
| RULER data | `datasets/qwen3-32b-ruler128k-1100` (used as is) | - |
| Loki bank | `loki-r32-32x128k` (dense-model post-RoPE PCA, 32 x 128K C4) | baseline, later |

## Stages (scripts in `runs/q3-32b-t2/scripts/`, logs in `runs/q3-32b-t2/logs/`)

0. `00_windows`: `prepare_llama31_8b_c4_128k_windows.py` (48 windows, 32/16) -> `prepare_retrieval_calibration_windows.py`
   (16 windows, excludes the C4 bank's documents) -> `assemble_mixed_calibration_windows.py`. CPU only.
1. `01_capture_cov.sh`: `capture_attention_o_proj_covariances.py`, dense teacher, 8 GPUs balanced, YaRN4 -> `$R/snapshot`.
2. `02_fit_v96.sh`: `scripts/fit_value.py fit-shard` x 8 (one layer shard per GPU) -> `merge` -> `package_c1_uniform_checkpoint.py`.
3. `03_router.sh`: `fit_k_routing_windowed.py` moments (4 workers x 2 GPUs) -> merge -> base -> fisher (4 x 2 GPUs) -> fit (8 layer shards).
   The fitter was patched to balance the 32B teacher over the visible GPUs (`load_teacher_model`); one A100 cannot hold it.
4. `04_ruler.sh`: `eval_k_routing_ruler_p16.py` smoke -> evaluate (arms `full`, `ours`; 4 replicas x 2 GPUs) -> summarize.

## Open items

- LRQK / Loki / ShadowKV baselines: Table 2 matches LRQK and Loki per-head top-k to ~2048 physical tokens per KV group after
  de-duplication (832 / 856 at GQA 4; the recorded Qwen3-8B run used 704 for LRQK). Qwen3-32B has 8 query heads per group, so
  the matched top-k must be measured on this model (the 64K attempt used 416). Not part of the first pass.
- The existing C4-only V96 differs from the recipe in calibration data and damping; it is kept only as a smoke/ablation arm.

## Smoke results

(filled in below as they complete)

- Stage 0 done (2026-11-03 00:08, 2 min CPU): the three 32B window banks are token-for-token identical to the published
  Qwen3-8B banks (`calibration/qwen3-8b-post-128k/*`); manifests carry the Qwen3-32B config sha.
- Capture smoke (2 windows, 4 x A100, YaRN4): 453 s wall including model load and the 38 GB snapshot write; peak
  42.6 / 42.3 / 42.3 / 39.9 GiB. Full 48-window capture estimated ~2.5 h on 4 GPUs (single pipeline process).
- Evaluator smoke (`eval_k_routing_ruler_p16.py smoke --arm full`, C4-only V96, 2 x A100, 4 generated tokens): 253 s and
  233 s per 128K prompt (prefill-dominated; the model is pipelined over the two GPUs by `device_map=balanced`), peak
  58.6 / 60.2 GiB. So 2 GPUs per replica, 4 replicas. Estimate per arm: ~300 s per prompt with full decode ->
  1100 prompts ~ 23 h; 10 Table-2 tasks x first 10 prompts (100) ~ 2 h.
- Router fitter smoke (layers 0,1; 2 C4 windows; C4-only V96 identity; 2 x A100): moments 215 s / window, fisher 220 s /
  window, peak 55.9 GiB; chain moments -> merge -> base -> fisher -> fit completes. Base16 relative MSE 0.016 / 0.077;
  Residual16 page-Fisher fit NMSE 0.210 / 0.082 (old Page32 32B bank: 0.207 / 0.066; published 8B Page4 bank: 0.483 / 0.196);
  ALS converged by sweep 20 (per-sweep change < 0.1%), PCG max relative residual 0.012 / 0.005.
- Window-sharded capture (`--window-shard/--window-shards`, merge via `evaluation/merge_attention_o_proj_covariances.py`):
  on 4 windows the merged snapshot equals the unsharded one exactly (max relative Frobenius difference 0.0 over 64 layers,
  both splits). 185 s per window + ~75 s load/write; 2 shards x 24 windows on 4 GPUs each -> ~1.3 h.

## Launch plan (pending confirmation)

1. `scripts/01_capture_cov.sh`  ~1.3 h, 8 GPUs (2 shards x 4)
2. `scripts/02_fit_v96.sh`      ALS on covariances, 8 layer shards, then package -> `identity/v96.json`
3. `scripts/03_router.sh`       moments 32 windows / 4 workers x 2 GPUs (~30 min) -> merge -> base -> fisher (~30 min) -> fit (8 shards)
4. `scripts/04_ruler.sh smoke` then `evaluate full`, `evaluate ours` on the 10 Table-2 tasks x first 10 prompts (100), ~2 h per arm

## Run log

- Stage 1 done (2026-11-03 02:45): 2 shards x 24 windows, 4446 / 4470 s each, peak 42.6 GiB per GPU; merged snapshot
  `snapshot/` (64 layers, 32 fit + 16 held-out windows, 6,291,456 rows per layer, windows sha `b3e7142c...` = the
  published Qwen3-8B mixed bank).
- Stage 2 done (2026-11-03 ~03:00): uniform V96, ALS 6 / CG 16 / damping 1e-7 on the mixed snapshot. Fit relative MSE mean
  0.0432 (HF C4-only bank: 0.0432); held-out min 0.0027, median 0.0517, max 0.0908 (layer 30). Packaged to `ckpt_v96/`,
  identity `identity/v96.json`.
- Stage 3 done (2026-11-03 04:52). First attempt stopped at `fisher`: `base` was run without `--window-shards 4`, so its
  protocol recorded 8 shards against the moments' 4 (only difference); base was refit with the matching shard count and the
  chain resumed (`scripts/03b_router_resume.sh`); moments/merge were kept. moments 234 s / window, fisher ~400 s / window,
  peak 55.9 GiB; Fisher records 2048 files, 263 GB (`router/fisher`, intermediate).
  Router bank `router/ours_b16r16` (64 layers; page 4, no sink, recent 64, 64 queries, 32 fit windows, ALS 40, PCG 100):

  | layers | page-Fisher NMSE (this bank) | old HF Page32 32B bank | Base16 relative MSE |
  |---|---|---|---|
  | 0-15 | 0.073 | 0.159 | 0.175 |
  | 16-31 | 0.029 | 0.238 | 0.305 |
  | 32-47 | 0.024 | 0.187 | 0.272 |
  | 48-63 | 0.028 | 0.027 | 0.302 |
  | all (mean / median) | 0.039 / 0.025 | 0.153 / 0.210 | |

  Max NMSE 0.421 at layer 0 (layers 1-6 ~0.10, all others <= 0.06). Published Qwen3-8B Page4 bank: mean 0.039, median 0.019.
  PCG final max relative residual median 0.0009, max 0.0060. Per-layer values in `router/nmse_compare.json`.
  Caveat (Sept-21 32B diagnosis): fit NMSE did not predict retrieval routing quality; RULER is the test.
- Stage 4 first launch stopped at the audit: the evaluator requires `--samples-per-task` to equal the dataset manifest (100),
  and the chain lacked `pipefail`, so the evaluate step also started and failed at the same assertion (no GPU work done).
  Fix: `ruler_data_10/` = first 10 rows of every task's `validation.jsonl`, unchanged and in order, manifest re-hashed with
  the source manifest sha recorded; chain now `set -euo pipefail`, waits on every worker pid, and requires
  "ALL ARM SMOKES VERIFIED". Output `ruler_n10/`.
- RULER smoke + audit passed (05:03): prompts 0 (niah_single_1) and 99 (fwe), 4 tokens each; first tokens and all 4 tokens
  identical between Full-K and BasisKV. Full-K 251-253 s per prompt, peak 58.6 / 60.2 GiB; BasisKV 246-248 s, peak
  67.2 / 69.3 GiB (routing sidecar). Evaluate full started 05:03, 4 replicas x 2 GPUs.
- 05:58 cancelled by request: Full-K stopped at 87/100 prompts (saved under `ruler_n10/full/evaluate`, resumable: the
  evaluator skips verified samples); BasisKV evaluate never started; no summary. TP2 patch not applied.
- TP2 (2026-11-03): `evaluation/routing_tensor_parallel.py` + `--tensor-parallel` / `--sample-indices` in
  `eval_k_routing_ruler_p16.py` (pre-patch copy: `logs/eval_k_routing_ruler_p16.pre_tp.py`). Transformers' Qwen3 TP plan
  shards q/k/v colwise and o_proj/MLP rowwise; the evaluator slices V96 factors and the router bank by KV head / query
  group and all-reduces the compressed-V decoder output. TP2 smoke (prompts 0 and 99, 4 tokens, run twice) matched the
  pipeline smoke token for token on both arms; 130-136 s vs 246-253 s per smoke prompt; peak 56.1 (Full-K) / 65.3
  (BasisKV) GiB per rank. Audit passed 06:05.
- Full-K completed 06:12: 87 pipeline prompts (146 s mean) + 13 TP2 prompts (80 s mean). Full-K 10-task mean 91.83
  (S1 100, S2 100, MK1 100, MK2 100, MQ 100, MV 95, QA1 70, QA2 80, VT 100, FWE 73.3). BasisKV (TP2, 100) started 06:12.
- BasisKV (TP2, 100 prompts) completed 06:42, 68.7 s mean per prompt. Combined result (`combine_n10.py` ->
  `ruler_n10_combined.{md,json}`):

  | Method | S1 | S2 | MK1 | MK2 | MQ | MV | QA1 | QA2 | VT | FWE | Avg. |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Full-K | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 95.00 | 70.00 | 80.00 | 100.00 | 73.33 | 91.83 |
  | BasisKV (B16R16 Page4) | 100.00 | 100.00 | 100.00 | 70.00 | 97.50 | 95.00 | 70.00 | 80.00 | 100.00 | 46.67 | 85.92 |

  BasisKV - Full-K -5.92 (paper deltas: Llama-3.1-8B -2.43, Qwen3-8B -2.21, Qwen3.5-9B -0.50); paired wins/losses 0/10;
  identical generations 69/100. Losses: MK2 3 prompts (two near-miss copies, 8869382 -> 8869530 and 6388956 -> 6388958,
  one wrong needle; the Sept-21 failure mode, but 7/10 vs 0.16 then), MQ one swapped key, FWE 6 prompts (first word kept,
  the other two replaced; FWE is the global-aggregation task). First-token agreement 99/100: prompt 15 (niah_single_2)
  differs in format only (Full-K from the pipeline ' 5443951.' vs BasisKV TP2 ' **5443951**.'), both correct; prefill is
  identical across arms within one protocol, so this is the pipeline-vs-TP2 numeric difference flipping a near tie.
  The existing 32B LRQK TP2 record (`results/q3-32b-v96-ruler128k-lrqk-tp2`) uses the C4-only V96, a different prompt
  template and an unmatched ~3437-token support, so it is not comparable to this table.
- Oracle (2026-11-03, `scripts/04e_ruler_tp2_oracle.sh`): `exact_sparse` arm = FP32 exact-QK Page4 mass with the same GQA
  max rule and hard budget (2048 routed + recent 64 per KV group, no sink), TP2, same protocol and output root
  (`ruler_n10_tp2/exact_sparse`). First attempt stopped at the audit write: auditing arms full,ours,exact_sparse gives a
  6-verified record that `write_json` refuses to put over the existing 4-verified `smoke_audit.json` (all checks had
  passed). Rerun with arms full,exact_sparse writes an identical record; smokes reused. Evaluate 08:45-09:15, 68.8 s mean
  per prompt, peak 56.1 GiB per rank.

  | Method | S1 | S2 | MK1 | MK2 | MQ | MV | QA1 | QA2 | VT | FWE | Avg. |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Full-K | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 95.00 | 70.00 | 80.00 | 100.00 | 73.33 | 91.83 |
  | Oracle exact-QK (Page4, GQA max) | 100.00 | 100.00 | 100.00 | 80.00 | 100.00 | 95.00 | 60.00 | 70.00 | 100.00 | 50.00 | 85.50 |
  | BasisKV (B16R16 Page4) | 100.00 | 100.00 | 100.00 | 70.00 | 97.50 | 95.00 | 70.00 | 80.00 | 100.00 | 46.67 | 85.92 |

  Oracle - Full-K -6.33 (0/10); BasisKV - oracle +0.42 (paired 3/4, first-token agreement 100/100). 7 of the 10 prompts
  BasisKV loses are lost by the oracle too, including both MK2 near-miss copies (identical wrong numbers 8869530 and
  6388958) and 5 of 6 FWE prompts. So at 2048 per KV group with GQA-max sharing over Qwen3-32B's 8 query heads per KV
  head, the router matches exact-QK selection and the -5.9 gap comes from the budget / selection rule, not router error.
  Whether group size 8 (vs 4 on the 8B models) is the cause needs a larger group budget or per-query-head selection.
- Per-query-head selection (2026-11-03, `scripts/04f_ruler_tp2_qhead.sh`, `--ours-selection head`, output
  `ruler_n10_tp2_qh/`): every query head keeps its own 2048 routed + recent 64 (Page4, no sink); the KV-group union is
  measured, not capped. New evaluator source (selection flag + `physical_union_per_kv_group_mean` statistic), so a new
  protocol; Full-K smoke rerun only as the audit reference. Smoke + audit passed 09:47; evaluate 09:48-10:18, 69.0 s mean
  per prompt, peak 65.3 GiB per rank (same as group selection). Physical KV-group support 8739 tokens on average (last
  decode step, mean over prompts and layers; per layer about 5.7k-13k), i.e. about 4.1x the 2112 of group selection.

  | Method | S1 | S2 | MK1 | MK2 | MQ | MV | QA1 | QA2 | VT | FWE | Avg. |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Full-K | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 95.00 | 70.00 | 80.00 | 100.00 | 73.33 | 91.83 |
  | BasisKV, group 2112 | 100.00 | 100.00 | 100.00 | 70.00 | 97.50 | 95.00 | 70.00 | 80.00 | 100.00 | 46.67 | 85.92 |
  | BasisKV, per query head 2112 (union ~8739) | 100.00 | 100.00 | 100.00 | 90.00 | 97.50 | 95.00 | 60.00 | 80.00 | 100.00 | 53.33 | 87.58 |

  Per head - group +1.67 (paired 4/1); per head - Full-K -4.25 (1/9). Recovered: MK2 prompts 31 (wrong needle) and 32
  (near-miss copy 8869530 -> correct), FWE 98 one word; new loss QA1 61 (also lost by the exact-QK oracle). Unchanged: MK2
  38 (still 6388958), MQ 49, and FWE 90/94/95/96/97. FWE is insensitive to the selection rule even at about 4x the
  physical reads and now carries 2.0 of the 4.25-point gap.
- Full-K, all 1000 prompts (2026-11-03, `scripts/05_ruler_tp2.sh full`, output `ruler_tp2/`): 10 tasks x 100, TP2,
  4 replicas, chat template + `chat_answer_prefix`, enable_thinking=false. Smokes (full + ours, prompts 0 and 999) and
  audit passed 10:23-10:35 UTC (one gate covers both arms); evaluate 10:35-16:35 UTC, 84.8 s mean / 74.7 s median per
  prompt, peak 56.1 GiB per rank, 0 tracebacks. 15 prompts hit the answer cap (QA1 10, QA2 5). `summarize` not run
  yet: it needs both arms and its `summary.json` is immutable once written; scores below are read from the samples.

  | Method | S1 | S2 | MK1 | MK2 | MQ | MV | QA1 | QA2 | VT | FWE | Avg. |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Full-K (1000, TP2) | 100.00 | 100.00 | 99.00 | 83.00 | 99.50 | 95.00 | 63.00 | 51.00 | 100.00 | 77.67 | 86.82 |

  Correction to the 100-prompt tables above: their Full-K row came mostly from the pipeline run (87 pipeline + 13 TP2
  samples). On the same 100 prompts the TP2 Full-K of this run scores 89.83, not 91.83: token ids match the earlier TP2
  Full-K samples 13/13 and the pipeline samples 79/87, and two prompts flip from correct to wrong under TP2 numerics,
  MK2 38 (6388956 -> 6388958, the same near-miss every sparse arm produced) and QA1 61 (the "new loss" of per-head
  selection, also lost by the oracle). Against the matching TP2 Full-K reference (89.83, first-token agreement 100/100
  for every arm): oracle exact-QK 85.50 (-4.33, 0 wins / 8 losses), BasisKV group 85.92 (-3.92, 1/9), BasisKV per query
  head 87.58 (-2.25, 1/7).
- LongBench-v1 ShadowKV-9 Full-K (2026-11-03, `scripts/06_longbench_tp2.sh full`, output `longbench_tp2/`): 1549 rows
  (official prompts > 4096 tokens), TP2, 4 replicas; same protocol as the Qwen3-8B Page4 record (chat template with
  enable_thinking=false, official no-chat tasks lcc/samsum raw, greedy, official caps and metrics; the protocol differs
  from 8B only in model, data hash and evaluator source). Smokes + audit 16:35-16:37 UTC; evaluate 16:37-18:34 UTC,
  17.1 s mean / 8.1 s median per prompt, peak 44.7 GiB per rank, 0 tracebacks. Scores read from the samples
  (`summarize` waits for the BasisKV arm).

  | Model | NQA | MFQA | HQA | MuSiQue | DuReader | GovRep | SAMSum | PRe | LCC | 9-task | LRQK 6-task |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Qwen3-32B Full-K | 29.80 | 46.35 | 54.58 | 39.12 | 23.53 | 34.38 | 36.90 | 100.00 | 12.93 | 41.95 | 43.39 |
  | Qwen3-8B Full-K (reference) | 25.35 | 51.47 | 51.61 | 27.34 | 17.89 | 33.53 | 46.52 | 100.00 | 62.06 | 46.20 | 53.16 |

  Warning: LCC is model behaviour, not KV quality. On the raw (no-chat) LCC prompt Qwen3-32B starts reasoning in prose
  in 86 of 96 prompts ("Okay, I need to figure out what the next line of code ..."), and the official code_sim scorer
  takes the first non-comment line, so those score 10.2 against 36.2 for the 10 code-first outputs. Qwen3-8B continues
  the code under the same prompt. SAMSum (first line scored) is less affected: 7 of 165 start in prose; the 32B often
  answers on the first line and then keeps writing few-shot dialogues to the 128-token cap.
- BasisKV, all 1000 prompts (2026-11-03/04, `scripts/05_ruler_tp2.sh ours`, same output and gate as Full-K): B16R16
  Page4, 2048 routed + recent 64 per KV group (hard 2112, GQA max, no sink; measured union 2111.5). Evaluate 23:03-03:55
  UTC, 68.8 s mean per prompt (Full-K 84.8), peak 65.3 GiB per rank, 0 tracebacks. `summarize` passed (first-token
  agreement 1000/1000; `ruler_tp2/summary.{json,md}`, log `logs/05_summarize.log`).

  | Method | S1 | S2 | MK1 | MK2 | MQ | MV | QA1 | QA2 | VT | FWE | Avg. |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Full-K (TP2) | 100.00 | 100.00 | 99.00 | 83.00 | 99.50 | 95.00 | 63.00 | 51.00 | 100.00 | 77.67 | 86.82 |
  | BasisKV (B16R16 Page4, 2112) | 100.00 | 100.00 | 99.00 | 66.00 | 99.00 | 96.00 | 57.00 | 48.00 | 99.60 | 56.00 | 82.06 |

  BasisKV - Full-K -4.76, 95% bootstrap CI [-6.12, -3.48], win/loss 17/96. FWE (-21.67, 3/56) and MK2 (-17, 2/19)
  carry 3.87 of the 4.76 points. FWE losses are mostly one of three words (44 of 56 lose 1/3): the frequency count over
  the whole context is a global aggregation that a 2112-token support does not see. MK2: 7 of 19 losses are near-miss
  copies (<= 2 digits wrong, e.g. 8869382 -> 8869530), the rest a wrong needle. For scale, Qwen3-8B Page4 (group size
  4, same budget) loses 2.01 on 11 tasks (FWE -7.7, MK2 -8); Qwen3-32B has 8 query heads per KV group.
- LongBench BasisKV (2026-11-04, `scripts/06_longbench_tp2.sh ours`, same output and gate as Full-K): B16R16 Page4,
  256 routed + recent 64 per KV group (hard 320, GQA max, no sink). Evaluate 04:25-06:29 UTC, 18.5 s mean per prompt
  (Full-K 17.1; the evaluator's routed decode is not a fast kernel), peak 49.4 GiB per rank, 0 tracebacks. `summarize`
  passed (`longbench_tp2/summary.{json,md}`, log `logs/06_summarize.log`); first-token agreement 1549/1549.

  | Method | NQA | MFQA | HQA | MuSiQue | DuReader | GovRep | SAMSum | PRe | LCC | 9-task | LRQK 6-task |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Full-K | 29.80 | 46.35 | 54.58 | 39.12 | 23.53 | 34.38 | 36.90 | 100.00 | 12.93 | 41.95 | 43.39 |
  | BasisKV (B16R16 Page4, 320) | 30.60 | 46.26 | 53.37 | 38.65 | 23.99 | 33.08 | 36.72 | 100.00 | 12.94 | 41.73 | 43.27 |

  BasisKV - Full-K -0.22 (task-balanced), 95% bootstrap CI [-0.76, +0.33], win/loss 356/392, identical generations
  598/1549: no significant loss at 320 tokens per KV group (Qwen3-8B Page4: -0.93, CI [-1.54, -0.26]). Largest task
  drops GovRep -1.30 and HotpotQA -1.21; LCC is unchanged (prose-reasoning caveat above applies to both arms).
