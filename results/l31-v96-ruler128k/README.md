# Llama-3.1-8B-Instruct RULER 128K: V96

All five arms completed 1,100 frozen prompts: 11 tasks × 100 samples. All 5,500 predictions passed the evaluator audit.

| Arm | RULER mean |
|---|---:|
| Full-K | 84.4515 |
| B16R16 | 80.8758 |
| LRQK | 80.4076 |
| ShadowKV | 77.9712 |
| Loki | 50.8470 |

## Configuration

- V96: reused HF checkpoint `alexz949/BasisServe-CALS` at revision `3a774d5996e1d2a0ea0f64b94320b578205572a5`, path `checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6`. All 32 layers use rank 96. Fit: 32 × 131072 tokens, ALS 6, encoder CG 16; validation: 16 × 131072.
- B16R16: 32 fitting and 16 diagnostic windows, 131072 tokens each, ALS 60, PCG 50. Page size 32; sink32 and recent64 both count inside B2048. Maximum support is 2048 tokens.
- LRQK: rank32, top832 per query head plus recent64. ShadowKV: rank160, chunk8, routed2048 plus 48 outlier chunks, reconstructed historical K. Loki: PCA32, top856 per query head, recent0; calibrated on 32 × 128K C4 fitting windows.
- Environment: `/workspace/conda/envs/basis`; BF16. Exact commands and environment paths are retained in each arm’s `run_metadata.json` and fitting logs.

## Per-task scores

| Task | Full-K | B16R16 | LRQK | ShadowKV | Loki |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.0000 | 100.0000 | 100.0000 | 96.0000 | 80.0000 |
| niah_single_2 | 99.0000 | 100.0000 | 99.0000 | 99.0000 | 94.0000 |
| niah_single_3 | 99.0000 | 99.0000 | 99.0000 | 96.0000 | 0.0000 |
| niah_multikey_1 | 99.0000 | 99.0000 | 99.0000 | 98.0000 | 92.0000 |
| niah_multikey_2 | 91.0000 | 75.0000 | 88.0000 | 84.0000 | 41.0000 |
| niah_multiquery | 97.7500 | 97.0000 | 97.5000 | 93.2500 | 55.7500 |
| niah_multivalue | 89.7500 | 88.5000 | 91.2500 | 87.5000 | 56.5000 |
| vt | 66.8000 | 56.8000 | 44.4000 | 26.6000 | 20.4000 |
| fwe | 65.6667 | 56.3333 | 44.3333 | 54.3333 | 40.6667 |
| qa_1 | 76.0000 | 76.0000 | 77.0000 | 78.0000 | 53.0000 |
| qa_2 | 45.0000 | 42.0000 | 45.0000 | 45.0000 | 26.0000 |

## Execution and limitations

Evaluation switched to batch size 2 while preserving completed batch1 samples. Full-K/B16R16/LRQK/ShadowKV retain 49/42/45/45 batch1 samples respectively; their remaining samples used the batch2 evaluator. Loki used the batch2 evaluator for all 1100 samples. Final odd batches may contain one sample. Token-equality comparison was waived by the user; the evaluator audited EOS, generation caps, scores and kernel dispatch.

Loki scored 0 on niah_single_3; this observed quality failure is retained in the full 11-task mean.

The committed dense V128 Full-K reference is 86.8061, versus V96 Full-K 84.4515 (−2.3546 points). The dense V128 reference used TP2, while this V96 run used single-GPU workers; this is an observed run-to-run comparison rather than a strictly isolated V-only ablation.

## Artifacts

- `core/summary.json`: original audited summary and protocols.
- `core/prompts.json`: frozen prompt metadata and tensor hash.
- `predictions/<arm>/predictions.jsonl`: complete per-sample results; repeated metadata is referenced by index in `run_metadata.json`. Each record retains its original source filename and SHA-256. Reconstruction was checked against all 5500 original JSON records.
- `v96/manifest.json`, `identity.json`, `router/ours_b16r16/*.json`, `loki/manifest.json`: checkpoint identities, fitting configuration, per-layer losses and PCA calibration metadata.
- `logs/*.log`: fitting, evaluation and final audit logs.
- `files.sha256`: checksums for every exported file except the checksum list itself.

Model/checkpoint tensors and prompt token tensors are not included. Existing source code changes are outside this result export; recorded source hashes identify the evaluated implementations.

Final audit command (basis environment):

```bash
python -u -m evaluation.summarize_llama_v96 \
  --identity /workspace/runs/l31-v96/identity.json \
  --data /workspace/runs/l31-ruler100/ruler \
  --bank /workspace/runs/l31-v96/router/ours_b16r16 \
  --output /workspace/runs/l31-v96/core
```
