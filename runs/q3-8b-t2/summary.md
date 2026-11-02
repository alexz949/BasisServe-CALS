# Table 2, Qwen3-8B (post-trained), 128K: Full-K vs BasisKV (uniform V96, Page4 B16R16 router)

Paper row (10 RULER tasks, 100 samples each): Full-K 80.84, BasisKV 78.63. LongBench: 46.20 / 45.27.
Recorded arms in `results/q3-8b-post-v96-retrievalmix-ruler128k/`: `full` and `b16r16_page4_fisher`
(11 tasks there; the paper averages 10). Per-sample predictions: `predictions/<arm>.jsonl`.

## Protocol (from `protocols/b16r16_page4_fisher.json`)
- Model Qwen/Qwen3-8B snapshot `b968826d9c46dd6066d109eabc6255188de91218`, BF16, YaRN factor 4
  (original 32768), sequence 131072, MLP prefill chunk 1024.
- Prompt: chat template, user turn = RULER `input`, `enable_thinking=false`, answer prefix appended after
  the generation prompt (`eval.ruler --chat-template auto`; tokens verified identical to the paper evaluator).
- Value: uniform V96 on all 36 layers. Router: base 16 + residual 16, page 4, 2048 routed + 64 recent, no sink.
  Sidecar (predicted K, residual code) is computed and stored in the cache dtype (BF16); scores are FP32.
- Data: RULER 128K, 11 tasks x 100, seed 42, base template (HF `evaluation/qwen3-8b-ruler128k-1100-seed42`).

## Artifacts (`runs/q3-8b-t2/`)
- `hf/` HF download: `attention_v96` (36 layers), `router_page4_b16r16` (release format, used as is), RULER data.
- `paper_v96/` the V96 bank converted to `basiskv.value_factors.v1` by `convert_v96.py` (layer files linked,
  checksums verified against the HF manifest).
- `yarn4/config.json` Qwen3-8B config with `rope_parameters = yarn, factor 4, original 32768`, max positions 131072.

## Commands (conda `lowrank` + transformers 5.17 via PYTHONPATH, run from `BasisKV/`)
```
COMMON="--model $MODEL --config runs/q3-8b-t2/yarn4 --value-factors runs/q3-8b-t2/paper_v96 \
  --data runs/q3-8b-t2/hf/evaluation/qwen3-8b-ruler128k-1100-seed42 --chat-template auto --device-map balanced"
python -m eval.ruler $COMMON --method full    --output runs/q3-8b-t2/ruler_full
python -m eval.ruler $COMMON --method basiskv --router runs/q3-8b-t2/hf/.../router_page4_b16r16 --output runs/q3-8b-t2/ruler_basiskv
```

## Smoke (niah_single_1, A5000 24 GB, node1)
| run | GPUs | result |
|---|---|---|
| full, no prefill chunking | 2 | OOM in the MLP (3 GiB intermediate) -> added `--prefill-chunk-tokens 1024` (MLPs, then also the layer norms and the 128K Q/K projection+norm+RoPE in `inference.attention`) |
| full | 2 | OOM in attention (model half + KV half already 17 GB per GPU) |
| full | 3 | ok, 1 sample in ~1.5 min, peak 24.0 GB on GPU 1, prediction matches the paper record |
| basiskv, FP32 sidecar | 3 | OOM |
| basiskv, FP32 sidecar | 4 | ok, 3 samples ~1 min 50 s each, peaks 17.9/24.1/24.1/23.5 GB; 3/3 predictions identical to the paper's `b16r16_page4_fisher` |
| basiskv, BF16 sidecar (paper dtype) | 3 | OOM: the 2 GiB temporaries were the decoder RMSNorms (FP32 copies of [131072, 4096]) and the Q norm/RoPE |
| full, prefill fully chunked (norms, MLPs, Q/K projection+norm+RoPE) | 2 | ok, peaks 21.7 / 24.2 GB (GPU 0 carried 3.5 GB of another user's process), ~2.5 min per sample |
| full, prefill fully chunked | 3 | ok, peaks 14.9 / 18.6 / 18.3 GB |
| basiskv, BF16 sidecar, prefill fully chunked | 3 | ok, peaks 17.2 / 22.9 / 22.1 GB, ~1.8 min per sample; 3/3 predictions identical to the paper |
Job 232632 (node3). Decision: 3 GPUs per worker for both arms.

Release fixes made while bringing this up: `eval.common` chat template path (transformers 5.17 returns
BatchEncoding), `--prefill-chunk-tokens`, `eval.ruler --chat-template`, sidecar dtype back to the cache dtype.

## Refit (requested 2026-11-01): regenerate the V96 bank and the Page4 router from the paper's windows
Windows: HF `calibration/qwen3-8b-post-128k/c4-retrieval-50-50` (48 x 131072 int32; fit 0..15 C4 seed 20260921
+ 16..31 synthetic retrieval seeds 20260928/20260929; held-out 32..47 C4). Downloaded to `hf/calibration/`.
Recipe (Llama retrieval-mix record, matches the bank name `als6` and the release defaults):
V96 uniform, ALS 6 sweeps, CG 16, covariance damping 1e-7, FP32 work, BF16 factors, all 131072 positions per window.
Router: replay on the deployed V96 model, 32 fit windows, 64 fit queries (4 strata x 16), no diagnostic windows
(`--heldout-windows 0`), page 4, pinned 0, recent 64, B16R16, ALS 40 / PCG 100 (release `core.fit_router` defaults).
The paper's own Qwen3-8B V96 fit record (`source_results_sha256 c2661cfb...`) is not in git; the bank carries no fit errors.

Chain (from `BasisKV/`, conda lowrank + transformers 5.17):
```
W=runs/q3-8b-t2/hf/calibration/qwen3-8b-post-128k/c4-retrieval-50-50/windows.safetensors
COMMON="--model $MODEL --config runs/q3-8b-t2/yarn4 --device-map balanced --windows $W"
python -m calibration.value $COMMON --output runs/q3-8b-t2/refit/snap --window-shard-index I --window-shard-count N   # x N workers
python -m calibration.value $COMMON --output runs/q3-8b-t2/refit/snap --merge
python -m core.fit_value fit-shard --snapshot-dir runs/q3-8b-t2/refit/snap --output-dir runs/q3-8b-t2/refit/V96 --rank 96 --layer-shard-index I --layer-shard-count N
python -m core.fit_value merge --output-dir runs/q3-8b-t2/refit/V96
python -m calibration.router moments $COMMON --value-factors runs/q3-8b-t2/refit/V96 --heldout-windows 0 --output runs/q3-8b-t2/refit/router_cap --base-rank 16 --residual-rank 16 [--window-shard-index I --window-shard-count N]
python -m core.fit_router base --statistics-dir runs/q3-8b-t2/refit/router_cap --output-dir runs/q3-8b-t2/refit/router_cap/base --base-rank 16
python -m calibration.router fisher $COMMON --value-factors runs/q3-8b-t2/refit/V96 --heldout-windows 0 --output runs/q3-8b-t2/refit/router_cap --base-rank 16 --residual-rank 16 [sharded]
python -m core.fit_router fit --statistics-dir runs/q3-8b-t2/refit/router_cap --output-dir runs/q3-8b-t2/refit/router --residual-rank 16 --device cuda:0 --sweeps 40 --pcg-iterations 100
python -m eval.ruler ... --method basiskv --value-factors runs/q3-8b-t2/refit/V96 --router runs/q3-8b-t2/refit/router --output runs/q3-8b-t2/ruler_basiskv_refit
```
Smoke (job 232628, 2 x A5000): value capture 506 s per 128K window (FP64 Gram accumulation dominates on A5000),
peaks 22.1 / 22.5 GB; fit_value layers 0-1 at rank 96: held-out 0.0085 / 0.0192 in ~6 s per layer; router moments 129 s
per window, peaks 20.7 / 19.5 GB, 117 MB of records per window. (`--merge` on a single-shard capture has no parts/ to
merge and stops on its assertion; the chain captures in 2 shards and merges them as in Table 1.)
Release changes for 128K calibration: `--prefill-chunk-tokens` on the teacher (shared `inference.tp.chunk_tokenwise_prefill`),
`--max-memory-per-gpu-gib` removed (device_map balanced uses the real capacity).

Submitted 2026-11-01 (`submit_refit_chain.sh`, ids in `jobids`), chained afterok behind the calibration smoke 232615:
01_snap 232616 (2 x 3 GPUs, 24 windows each) -> 02_snap_merge 232617 -> 03_fit 232618 (3 x 1 GPU, 12 layers each) ->
04_fit_merge 232619 -> 05_moments 232620 (2 x 3 GPUs) -> 06_moments_merge 232621 -> 07_base 232622 (CPU) ->
08_fisher 232623 (2 x 3 GPUs) -> 09_router_fit 232624 (4 x 1 GPU, 9 layers each). Outputs under `refit/{snap,V96,router_cap,router}`.
The RULER evaluation of the refit factors is not queued yet (sample count per task to be decided).

## LongBench (Table 2 right half)
Data: official THUDM/LongBench v1 `data.zip` extracted to `longbench/data/{task}.jsonl` (nine tasks). Under the Qwen3
tokenizer 1,549 official prompts exceed 4,096 tokens, matching the paper's count; per task kept / mean tokens / max tokens:
narrativeqa 200 / 29.9K / 65.4K, multifieldqa_en 111 / 8.7K / 16.5K, hotpotqa 195 / 13.7K / 17.7K, musique 200 / 16.3K / 17.9K,
dureader 200 / 10.6K / 19.2K, gov_report 182 / 11.4K / 52.6K, samsum 165 / 10.5K / 18.0K, passage_retrieval_en 200 / 12.9K / 15.6K,
lcc 96 / 7.5K / 32.6K. 22.1M prompt tokens and up to 198K generated tokens per arm. Protocol: chat template (auto; samsum and
lcc raw), 256 routed + 64 recent tokens, 2 GPUs per worker. Official metrics need `fuzzywuzzy` and `rouge` (installed into the
transformers-5.17 overlay). Paper: Full-K 46.20, BasisKV 45.27.
RULER Table 2 tasks (10): niah_single_1, niah_single_2, niah_multikey_1, niah_multikey_2, niah_multiquery, niah_multivalue,
qa_1, qa_2, vt, fwe (niah_single_3 is dropped; the recorded per-task scores reproduce 80.84 / 78.63 exactly).
LongBench smoke (job 232642, 2 x A5000, first 3 records per task, 23 kept): both arms run; peaks 11.2 / 13.0 GB (Full-K) and
12.3 / 14.4 GB (BasisKV); ~9 s per sample Full-K, ~17 s BasisKV (routing per decode step) -> ~4 h / ~7.5 h per arm for 1,549 prompts.
Scores look sane (passage_retrieval 1.0 x3, lcc 1.0, samsum 0.8/0.44, gov_report 0.35-0.39); the two arms agree on most samples
and differ on a few generations, as expected for sparse routing.

## RULER runs (decided 2026-11-01): 10 tasks x first 10 samples of the HF seed-42 data per arm, no LongBench
`python -m eval.ruler --model $MODEL --config runs/q3-8b-t2/yarn4 --value-factors runs/q3-8b-t2/paper_v96 \
  --data runs/q3-8b-t2/hf/evaluation/qwen3-8b-ruler128k-1100-seed42 --chat-template auto --device-map balanced \
  --tasks niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,qa_1,qa_2,vt,fwe \
  --samples-per-task 10 --method full --output runs/q3-8b-t2/ruler_full`
and the same with `--method basiskv --router .../router_page4_b16r16 --output runs/q3-8b-t2/ruler_basiskv`; 3 GPUs each.
Comparison target: the paper's per-sample predictions (`results/q3-8b-post-v96-retrievalmix-ruler128k/predictions/{full,b16r16_page4_fisher}.jsonl`)
on the same 100 prompts, plus the 10-task means of the paper's 100-sample run (80.84 / 78.63).

### V96 refit done (2026-11-02 00:18, stages 01-04)
Capture 3h16m / 3h21m per worker (24 windows each, ~8.3 min per 128K window), merge 2 min, ALS ~6 s per layer, bank `refit/V96`.
held-out relative MSE over 36 layers: min 0.0079 median 0.0508 max 0.0730; fit: min 0.0077 median 0.0462 max 0.0686.
Against the published bank: relative Frobenius distance of the effective V->output map E·D per query head, min 0.0030, median 0.0051,
max 0.0366 (layer 34; layer 35 0.019; all others <= 0.010). Same recipe (ALS 6, CG 16, damping 1e-7) on the same 48 windows; the
residual difference is the A5000 BF16/FP32 arithmetic and the FP64 Gram accumulation order.

### RULER Full-K done (job 232665, 3h03m on 3 x A5000, 100 prompts)
10-task mean 80.48; the paper's records on the same 100 prompts give 81.28 and its 100-prompt run 80.84.
Per task (ours / paper on the same prompts): S1 100/100, S2 100/100, MK1 90/90, MK2 60/70, MQ 92.5/92.5, MV 95/95, QA1 50/50,
QA2 30/30, VT 94/92, FWE 93.33/93.33. Identical generations on 86/100 prompts; the 14 text differences change two task scores
by one prompt each (MK2 -10, VT +2) and come from different GPUs/kernels (paper: RTX PRO 6000, single GPU; here 3 x A5000,
chunked prefill) in BF16 greedy decoding.

### RULER BasisKV with the published factors done (job 232666, ~3h20m on 3 x A5000, 100 prompts)
10-task mean 78.60; the paper's records on the same 100 prompts give 78.73 and its 100-prompt run 78.63 (Table 2).
Per task (ours / paper on the same prompts): S1 100/100, S2 100/100, MK1 90/90, MK2 60/60, MQ 95/95, MV 95/95, QA1 40/40,
QA2 30/30, VT 96/94, FWE 80/83.33. Identical generations on 82/100 prompts; the differences are near-tie BF16 greedy flips
(last digit of a needle, hallucinated FWE counts, formatting), the same mechanism as the Full-K arm.
Table 2 Qwen3-8B row reproduced on the 100-prompt subset: Full-K 80.48 vs 80.84, BasisKV 78.60 vs 78.63 (gap -1.88 vs -2.21).

### Router refit, stages 05-07 done (2026-11-02 03:32)
moments: 37 / 39 min per worker (16 windows each, ~125 s per 128K window); merge 1h33m (36 layers, 2 shards; `parts/` 73 GB,
merged `moments/` 117 MB); base fit 10 s (rank 16, fit relative MSE e.g. layer 35 0.098).
Base vs the published router, compared in dense-Value coordinates (E·left·right per KV head, since each bank's V-code basis
differs): relative Frobenius diff min 0.0020, median 0.0044, max 0.0141 (layer 34); bias within 0.14%.
08_fisher waits for 2 x 3 GPUs (other users hold 14 of 16); then 09_router_fit and the refit RULER run 232803.
08_fisher: 51 min per worker (16 windows each, ~193 s per 128K window with 64 queries), 74 GB of per-window Fisher records.
09_router_fit started 04:43 (4 shards x 9 layers, 1 GPU each); `compare_router.py` compares the result with the published router.

### Router refit done (2026-11-02 05:02): 09_router_fit 17 min per shard (9 layers, ALS 40 / PCG 100), bank `refit/router`
Fit page-Fisher NMSE per layer matches the published records to ~0.001-0.009 (layer 0: 0.492 vs 0.483; layer 1: 0.194 vs 0.196;
layers 7-35 identical to 3 decimals). The raw residual score maps Q_res·E_res^T differ by ~100% in unweighted Frobenius norm
(the rank-16 fit leaves low-Fisher directions unconstrained; see the diagnosis below). The functional test is the refit RULER run 232803.
Diagnosis of the residual-map difference: per layer the score-map norms agree (ratio 0.99-1.03, layer 13 1.26) and the leading
principal directions of both the residual encoder span and the residual query span coincide (top-4 cosines 0.93-1.00), while the
trailing directions do not (bottom-4 cosines 0.06-0.8). The rank-16 Page-Fisher objective pins down the high-Fisher directions and
leaves the rest nearly degenerate, so two equally good ALS solutions differ there; the unweighted Frobenius comparison counts those
directions. Equal NMSE + matching leading subspaces is the expected signature; page selection quality is judged by the RULER run.

### RULER BasisKV with the refit factors done (job 232803, 3h10m, 100 prompts) — Table 2 Qwen3-8B row complete
| arm | ours (100 prompts) | paper, same prompts | paper Table 2 (1000) |
|---|---|---|---|
| Full-K | 80.48 | 81.28 | 80.84 |
| BasisKV, published V96 + router | 78.60 | 78.73 | 78.63 |
| BasisKV, refit V96 + router (release calibration chain) | 79.73 | 78.73 | 78.63 |
Refit arm per task (ours / paper same prompts): S1 100/100, S2 100/100, MK1 90/90, MK2 60/60, MQ 95/95, MV 95/95, QA1 40/40,
QA2 40/30, VT 94/94, FWE 83.33/83.33 — nine of ten task scores identical to the paper's records; QA2 differs by one prompt.
Against our published-factor run: 93/100 identical scores, 72/100 identical generations; the refit router differs from the
published one in its low-Fisher residual directions only, and that does not change page selection on these prompts.
Disk: `refit/router_cap/parts` 73 GB and `refit/router_cap/fisher` 74 GB are intermediate and can be removed once the bank is kept.
