# LongBench: two-sided KL versus uniform C1 at average rank80

Qwen3-8B-Base, BF16, basis, four L40S workers. All arms use ORIGINAL dense prefill. C1 arms convert the dense V cache and then use full-exact-K compact C1 decode with the SDPA backend. No routing, sparse attention, offload or refitting.

| Task | Dense decode | Uniform V80 decode | Two-sided KL avg80 decode |
|---|---:|---:|---:|
| qasper | 39.3018 | 34.9041 | 35.2802 |
| multifieldqa_en | 52.8498 | 47.9694 | 48.6936 |
| hotpotqa | 60.8872 | 59.3363 | 62.1941 |
| 2wikimqa | 50.1190 | 47.2545 | 46.9940 |
| gov_report | 29.1896 | 29.3136 | 28.6770 |
| qmsum | 26.1460 | 27.3302 | 26.9735 |
| Mean | 43.0822 | 41.0180 | 41.4687 |

Same 192 frozen LongBench-v1 prompts, six tasks with 32 each. Official QA F1 and summary ROUGE-L, on a 0–100 scale; six-task arithmetic mean. Not full LongBench.
Same input tokens, greedy decoding, EOS and task caps. Input-plus-generation cap 32K; actual prompts 1,192–30,431 tokens.
Frozen allocation: alpha=1, anchor64, probes32/96, rank bank32/48/64/80/96/112/128. Rank counts: 3/3/11/6/5/3/5 layers. Average rank exactly80, same total cache budget as uniform80.
Factor bank: C4 32 x 32K fit, 4 x 32K held-out local MSE, six ALS sweeps. Separate allocation profile: 32 x 32K C4 windows; confirmation: 12 x 32K; 1,024 sampled terminal positions/window. No LongBench calibration.

Comparison qualification: the existing allocation export canonicalized encoder coordinates and performed a closed-form decoder refit for non-anchor, non-full-rank layers. The six layers still assigned rank80 do not have bitwise-identical encoder/decoder tensors to the uniform checkpoint. This is a comparison of the two exported checkpoints, not a strict rank-index-only ablation. No new fitting was performed for this evaluation.

Each layer retains its actual rank without padding to128. First token comes from original dense prefill; C1 computes the second and subsequent generated tokens.

First-token agreement with dense: 192/192.
Generation-cap exits without EOS: 42/192.
Maximum allocated GPU memory: 23.370 GiB.

Formal worker durations: 9:25, 7:44, 9:30 and 10:24. CPU summary/audit: 18 seconds. Smoke, all four workers and summary completed with exit code0, without retries. Optional FuzzyWuzzy acceleration warning only; no non-finite logits or assertion failures. Commands are recorded in `docs/longbench_c1_twosided_denseprefill_protocol.md` and per-sample JSON files.

## Layer ranks (0–35)

[128, 128, 32, 48, 48, 48, 80, 64, 128, 128, 112, 96, 96, 112, 64, 96, 80, 64, 80, 80, 64, 64, 112, 128, 96, 64, 64, 64, 32, 64, 80, 64, 64, 96, 80, 32]

## Paired score changes

```json
{
  "dense": {
    "improvements": 42,
    "regressions": 56,
    "ties": 94,
    "mean_delta_pp": -1.6134839833561359
  },
  "uniform80": {
    "improvements": 46,
    "regressions": 37,
    "ties": 109,
    "mean_delta_pp": 0.45073866932559
  }
}
```
