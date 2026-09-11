# Uniform V96: paired RULER comparison

Qwen3-8B-Base and Llama-3.1-8B Base. BF16, 11 tasks x8 examples/model; primary87 excludes global index86.
Every new arm passed protocol, prompt identity, EOS/cap, decoding and scoring checks, and its first generated token matches the uniform ShadowKV reference.
Within-model prompts are paired. Cross-model prompts differ because their tokenizers differ.

| Model | Full-K | Base16/R16 extra64 | Base16/R16 fixed2048 | LRQK | ShadowKV |
|---|---:|---:|---:|---:|---:|
| qwen3 | 81.877395 | 81.494253 | 82.068966 | 83.448276 | 78.716475 |
| llama31 | 83.390805 | 82.624521 | 82.241379 | 85.306513 | 82.107280 |

## qwen3

| Scope | full | recent_extra | recent_fixed | lrqk | shadowkv |
|---|---:|---:|---:|---:|---:|
| All88 | 80.946970 | 80.568182 | 81.136364 | 82.500000 | 77.821970 |
| Delta versus Full-K (87) | +0.000000 | -0.383142 | +0.191571 | +1.570881 | -3.160920 |

Per-task scores use all8 examples, including sample86 in QA2.

| Task | full | recent_extra | recent_fixed | lrqk | shadowkv |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| niah_single_2 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| niah_single_3 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| niah_multikey_1 | 87.500000 | 87.500000 | 87.500000 | 87.500000 | 87.500000 |
| niah_multikey_2 | 75.000000 | 75.000000 | 75.000000 | 75.000000 | 50.000000 |
| niah_multiquery | 87.500000 | 84.375000 | 87.500000 | 87.500000 | 87.500000 |
| niah_multivalue | 93.750000 | 96.875000 | 100.000000 | 100.000000 | 84.375000 |
| vt | 92.500000 | 92.500000 | 92.500000 | 95.000000 | 92.500000 |
| fwe | 66.666667 | 62.500000 | 62.500000 | 75.000000 | 66.666667 |
| qa_1 | 62.500000 | 50.000000 | 50.000000 | 50.000000 | 50.000000 |
| qa_2 | 25.000000 | 37.500000 | 37.500000 | 37.500000 | 37.500000 |

Bank: 36 authenticated layers. Mean layer fit/diagnostic Page-Fisher NMSE: 0.074820 / 0.131411.
Maximum recorded final query-PCG relative residual: 0.0175897; 36 layers exceed the requested1e-5 tolerance. Fixed40 sweeps/PCG100 were retained; no factor selection used validation.

## llama31

| Scope | full | recent_extra | recent_fixed | lrqk | shadowkv |
|---|---:|---:|---:|---:|---:|
| All88 | 82.443182 | 81.685606 | 81.306818 | 84.337121 | 81.174242 |
| Delta versus Full-K (87) | +0.000000 | -0.766284 | -1.149425 | +1.915709 | -1.283525 |

Per-task scores use all8 examples, including sample86 in QA2.

| Task | full | recent_extra | recent_fixed | lrqk | shadowkv |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| niah_single_2 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| niah_single_3 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| niah_multikey_1 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| niah_multikey_2 | 62.500000 | 62.500000 | 62.500000 | 87.500000 | 75.000000 |
| niah_multiquery | 96.875000 | 96.875000 | 96.875000 | 96.875000 | 100.000000 |
| niah_multivalue | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 81.250000 |
| vt | 72.500000 | 72.500000 | 72.500000 | 72.500000 | 70.000000 |
| fwe | 87.500000 | 79.166667 | 75.000000 | 83.333333 | 79.166667 |
| qa_1 | 50.000000 | 50.000000 | 50.000000 | 50.000000 | 50.000000 |
| qa_2 | 37.500000 | 37.500000 | 37.500000 | 37.500000 | 37.500000 |

Bank: 32 authenticated layers. Mean layer fit/diagnostic Page-Fisher NMSE: 0.097251 / 0.165807.
Maximum recorded final query-PCG relative residual: 0.00958217; 32 layers exceed the requested1e-5 tolerance. Fixed40 sweeps/PCG100 were retained; no factor selection used validation.

## Protocol and execution

HF revision `f1a6253b5d5c747a2475cbf9e704a67d97930b31`, uniform C1 V96 at every layer/head. Each model has a separately fitted Base16+Page-Fisher R16 bank:64x32768 fit,16x32768 diagnostics, Query-Gram Q32 in four8K bins.
Base16/R16 extra64 selects2048 page tokens including sink32 then unions sliding recent64 (max2112); fixed2048 reserves sink32 and recent64 within2048. LRQK is R32/k1152/recent64 with FP32 routing and2/2 iterations; its GQA union is not capped at2048. ShadowKV rank160/chunk8/routed2048 has additional outlier/local/generated tokens.

Environment `lowrank`, direct shell execution without Slurm, two OMP/MKL threads per worker. Initial fitting GPUs Qwen3/6, Llama4/7; baseline queues Qwen0, Llama2. Qwen even-layer fitting migrated from externally busy GPU3 to GPU6 after layer10 saved, preserving factors and replaying dense hidden states. Ours runs after the banks complete: Qwen2/6 (recent_extra redirected from queued GPU0 to idle GPU2), Llama4/7.

```bash
python -u evaluation/calibrate_uniform96_router.py --model-family FAMILY --num-shards 2 --shard-index SHARD
python -u evaluation/eval_uniform96_compare.py --model-family FAMILY --arm ARM
python -u evaluation/eval_uniform96_compare.py --model-family FAMILY --stage summarize
python -u evaluation/summarize_uniform96_compare.py
```

`FAMILY`: qwen3/llama31; `SHARD`:0/1; `ARM`:full/lrqk/recent_extra/recent_fixed. ShadowKV completed earlier and is reused with hashes.
Logs: `results/logs/uniform96_compare/`. Detailed configuration: `docs/uniform96_compare_protocol.md`.
Initial baseline smoke had a cache-constructor argument error before writing results; fixed and rerun with appended logs. Six native-adapter/recent-budget tests passed. The earlier Qwen ShadowKV run had an SVD warning and automatic solver fallback, then completed successfully.

These are resident accuracy measurements. They do not establish throughput or offload memory cost. The uploaded uniform V factor banks were fitted on256x2048 with64x2048 validation; the old Qwen L40S protocol records32x32768 fit plus4x32768 diagnostics. Thus even their V-factor fitting settings differ. The new64x32K router fit is separate from the frozen HF V-factor fit.
