# Uniform V96 versus KL96: ShadowKV on paired local RULER prompts

Both uniform runs completed with exit0. All176 uniform records passed the saved protocol, prompt identity, EOS/cap, decoding and official scoring audit. Each model reuses its own previously evaluated88 prompts; primary mean excludes global index86. Cross-model prompts are not paired.

| Model | KL96 ShadowKV (87) | Uniform96 ShadowKV (87) | Delta | KL96 (88) | Uniform96 (88) |
|---|---:|---:|---:|---:|---:|
| qwen3 | 73.045977 | 78.716475 | +5.670498 | 72.215909 | 77.821970 |
| llama31 | 77.298851 | 82.107280 | +4.808429 | 77.556818 | 81.174242 |

## qwen3: per-task scores (all8 per task)

| Task | KL96 | Uniform96 | Delta |
|---|---:|---:|---:|
| niah_single_1 | 100.000000 | 100.000000 | +0.000000 |
| niah_single_2 | 100.000000 | 100.000000 | +0.000000 |
| niah_single_3 | 100.000000 | 100.000000 | +0.000000 |
| niah_multikey_1 | 87.500000 | 87.500000 | +0.000000 |
| niah_multikey_2 | 37.500000 | 50.000000 | +12.500000 |
| niah_multiquery | 87.500000 | 87.500000 | +0.000000 |
| niah_multivalue | 84.375000 | 84.375000 | +0.000000 |
| vt | 85.000000 | 92.500000 | +7.500000 |
| fwe | 37.500000 | 66.666667 | +29.166667 |
| qa_1 | 50.000000 | 50.000000 | +0.000000 |
| qa_2 | 25.000000 | 37.500000 | +12.500000 |

## llama31: per-task scores (all8 per task)

| Task | KL96 | Uniform96 | Delta |
|---|---:|---:|---:|
| niah_single_1 | 100.000000 | 100.000000 | +0.000000 |
| niah_single_2 | 100.000000 | 100.000000 | +0.000000 |
| niah_single_3 | 87.500000 | 100.000000 | +12.500000 |
| niah_multikey_1 | 100.000000 | 100.000000 | +0.000000 |
| niah_multikey_2 | 87.500000 | 75.000000 | -12.500000 |
| niah_multiquery | 59.375000 | 100.000000 | +40.625000 |
| niah_multivalue | 68.750000 | 81.250000 | +12.500000 |
| vt | 87.500000 | 70.000000 | -17.500000 |
| fwe | 75.000000 | 79.166667 | +4.166667 |
| qa_1 | 50.000000 | 50.000000 | +0.000000 |
| qa_2 | 37.500000 | 37.500000 | +0.000000 |

## Interpretation and execution

Uniform factors improved absolute ShadowKV scores on these paired prompts. This changes the V checkpoint, including its factors and rank schedule; it does not isolate allocation alone. No uniform Full-K arm was requested, so the reconstruction/routing penalty against uniform Full-K is unknown. Equality to the old remote L40S checkpoint is not established.

HF revision `f1a6253b5d5c747a2475cbf9e704a67d97930b31`; C1U-R96, every layer/head96, encoder sweep6 with refitted decoder, BF16. ShadowKV rank160/chunk8/routed2048 plus48 outlier chunks and local/generated tokens.

Environment `lowrank`; direct shell without Slurm; GPU3 qwen3, GPU6 llama31; OMP/MKL threads2.

```bash
python -u evaluation/eval_uniform96_shadow.py --model-family qwen3
python -u evaluation/eval_uniform96_shadow.py --model-family llama31
```

Qwen emitted one PyTorch SVD convergence warning: the library automatically used a more accurate solver and the run completed. No evaluation failures. Four-token smoke scores are truncated-generation diagnostics, excluded from accuracy.

Logs: `results/logs/uniform96_shadow/{qwen3,llama31,download,summary}.log`. Protocol: `docs/uniform96_shadow_protocol.md`. Raw records and results: `results/evaluation/{qwen3,llama31}_uniform96_shadow/`.
