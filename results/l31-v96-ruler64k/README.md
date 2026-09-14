# Llama-3.1-8B-Instruct V96 RULER 64K

Final audited deterministic evaluation: 11 tasks × 30 samples × 5 arms = 1,650 predictions. Scores below are percentages; JSON scores are fractions. All arms use the same uniform V96 checkpoint, including Full-K. Full-K is not an uncompressed Dense model.

| Task | Full-K | Ours | Loki | LRQK | ShadowKV |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_3 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multikey_1 | 100.00 | 100.00 | 100.00 | 100.00 | 96.67 |
| niah_multikey_2 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multiquery | 97.50 | 97.50 | 98.33 | 99.17 | 95.00 |
| niah_multivalue | 89.17 | 85.00 | 85.83 | 84.17 | 84.17 |
| vt | 91.33 | 88.00 | 87.33 | 91.33 | 77.33 |
| fwe | 76.67 | 77.78 | 63.33 | 70.00 | 73.33 |
| qa_1 | 70.00 | 66.67 | 66.67 | 70.00 | 63.33 |
| qa_2 | 53.33 | 53.33 | 53.33 | 53.33 | 50.00 |
| Mean | 88.91 | 88.03 | 86.80 | 88.00 | 85.44 |

Calibration: C4, 64 × 65,536 training tokens plus 16 independent diagnostic windows; V96 ALS6 / encoder CG16; Base16 + residual16, 40 sweeps / PCG cap100. Iteration caps do not imply convergence.

Routing: ours uses 2,048 physical tokens per GQA group including sink32/recent64; Loki and LRQK use 2,048 historical tokens per query head plus recent64. ShadowKV uses rank160, chunk8, routed2,048 and 48 outlier chunks plus native local/generated support. These budgets have different semantics.

Environment: basis (`/workspace/conda/envs/basis/bin/python`), NVIDIA L40S, BF16; deterministic algorithms enabled, TF32 disabled, CUBLAS_WORKSPACE_CONFIG=:4096:8. Our Triton attention dispatch counts were audited. See each summary protocol for source/checkpoint hashes and exact settings.

Files: `tasks.csv` contains task-level scores. Each `*_predictions.jsonl` contains all 330 original prediction objects, including reference answers, generated text/token IDs, per-question scores, routing/kernel counters, timing, exact command and environment. Each `*_summary.json` is the original audited summary. `prompts.json` contains prompt metadata and hashes; the large prompt tensor and model weights remain in the workspace.

All 1,650 source prediction hashes were checked against the audited summaries and all 55 task means were recomputed during packaging. JSONL objects preserve the source JSON content; summary result paths refer to original workspace files. `manifest.json` hashes the packaged files.

This package contains the final deterministic rerun. The earlier provisional ours run had two first-token mismatches and is not used here. The final run passed all paired first-token checks; Full-K reproduced all 330 original generated token sequences. This does not establish the cause of the earlier mismatch.

Example actual evaluation command (other arms/shards are recorded per prediction):
```bash
/workspace/conda/envs/basis/bin/python /workspace/BasisServe-CALS/evaluation/eval_uniform_c1_ruler.py evaluate --identity /workspace/runs/l31-v96/identity.json --data /workspace/runs/l31-v96/ruler --bank /workspace/runs/l31-v96/router/ours_b16r16 --loki /workspace/runs/l31-v96/loki --output /workspace/runs/l31-v96/eval-deterministic --arm full --shard 0 --shards 4
```
