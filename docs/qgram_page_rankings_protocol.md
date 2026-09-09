# Stratified Q32 versus terminal Q32: page overlap and ranking audit

## Scope

This is a new evaluation of the frozen stratified Query-Gram Q32/R8 checkpoint against the old terminal8K Q32/R8 checkpoint. Same Qwen3-8B-Base, C1-V80, closed-form Base16, Page32/B2048, pinned page0, native BF16 proxy path. No fitting or model forward. Inputs are the existing dense-teacher C4 diagnostic captures: windows64–79, 32 common terminal8K query positions24831:256:32767, all36 layers and all8 GQA groups, 147456 conditions per arm. All layers, including0/1, use the same sparse rule here.

Exact reference: FP32 exact QK of BF16 captures, per-head non-pinned page mass normalization then GQA max; same selector/budget as proxy. Overlap is intersection/64, not IoU or attention mass. The non-pinned version is (intersection-1)/63.

Both old and new arms are recomputed. Summary checks every old exact/q32 table, including selected sets, scores, inclusive tie-rank intervals and cutoffs, bitwise against results/evaluation/page_overlap_all. No sampler-specific Q positions enter this shared evaluation.

## Detailed outputs

- evaluate/l{layer}_g{group}/pages.safetensors: complete page tables, one row per document/query, up to1024 pages. Page indices >=page_count are invalid padding.
- Each arm exact/q32/qgram32: group_score, rank_min, rank_max, owner, selected mask, selected_ids and cutoff. Proxy arms additionally store intersection_ids, missed_ids and extra_ids (-1 padded).
- Exact teacher_mass and teacher_non_sink_mass are per-page head means; teacher_mass_by_head and teacher_non_sink_mass_by_head retain all4 heads. Group routing score is the head max of non-sink mass, not mean teacher mass.
- Pinned page0 has rank0 and group_score negative infinity; its owner is not meaningful for selection. Routed cutoff is the63rd score. Ties retain inclusive rank intervals; actual selected masks determine selection at ties.
- layer_overlap.csv and summary.md: all36 layer averages and overall averages.
- missed_page_rankings.csv: top20 new-router misses per layer by head-mean full teacher mass, with all three arms, ranks, selection categories, global owner heads, cutoff scores and score-minus-cutoff. This diagnostic subset is not the whole distribution.
- evaluation/export_page_rankings.py exports ALL valid pages for any selected layer/group/document/query into a readable CSV without inference. Complete binary tables are retained for every condition to avoid expanding hundreds of millions of page rows into text.

Each page row is recoverable from layer/group/document/query_position/page; token span is [32*page,32*page+31]. A missed page is exact-selected but proxy-unselected; an extra page is the reverse.

## Tests and launch

Four CPU tests passed in basis: identical sets, swaps and mass accounting, ties/pinned ranks, CSV rank and selection categories. Syntax and whitespace checks passed.

Smoke8300842 completed successfully on one L40S; layer15/group7, document64, Q32767. Exact and old Q32 saved tables match the historical row bitwise. Old overlap41/64=64.0625%, new42/64=65.625%. This single condition is not an estimate of the full-layer change. Only known Qwen3RotaryEmbedding device deprecation warning observed.

Formal job8300843 array0–3 uses four L40S workers, each2 CPUs and48GiB host RAM; nine layers × eight groups per worker. Summary8300847 depends on all workers succeeding, with2 CPUs/12GiB host RAM. Invalid dependencies cancel downstream summary. Temporary sbatch scripts removed after submission. No unrelated jobs modified and no GitHub upload authorized.

Output root: results/evaluation/qgram_pages. Logs: logs/qgram-pages-{smoke,evaluate,summary}-{job}_{task}.{out,err}.

## Commands

Environment: /home/zhangal/.conda/envs/basis. Working directory: /deac/csc/yangGrp/zhangal/BasisServe-CALS.

Smoke:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/compare_residual_selected_pages.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --query-capture results/calibration/q32_terminal8k --comparison-bank q32=results/checkpoints/mse_base_q32_r8 --comparison-bank qgram32=results/checkpoints/mse_base_qgram32_r8 --output-dir results/evaluation/qgram_pages --stage smoke --layer 15 --group 7
```

Formal workers:

```bash
for ((comparison_layer=SLURM_ARRAY_TASK_ID; comparison_layer<36; comparison_layer+=4)); do
  for ((comparison_group=0; comparison_group<8; comparison_group++)); do
    /home/zhangal/.conda/envs/basis/bin/python -u evaluation/compare_residual_selected_pages.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --query-capture results/calibration/q32_terminal8k --comparison-bank q32=results/checkpoints/mse_base_q32_r8 --comparison-bank qgram32=results/checkpoints/mse_base_qgram32_r8 --output-dir results/evaluation/qgram_pages --stage evaluate --layer "$comparison_layer" --group "$comparison_group"
  done
done
```

Summary:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/summarize_page_overlap.py --root results/evaluation/qgram_pages --reference-root results/evaluation/page_overlap_all --ranking-arm qgram32
```

Example full page-ranking export (after the corresponding group completes):

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/export_page_rankings.py --root results/evaluation/qgram_pages --layer 15 --group 7 --document 72 --query-position 31231 --output results/evaluation/qgram_pages/l15_g7_w72_q31231.csv
```

