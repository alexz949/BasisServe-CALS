# LongBench: dense prefill followed by C1-V80 decode

Qwen3-8B-Base; basis; BF16; four independent L40S workers; same 192 frozen prompts, six tasks with 32 prompts each.

| Task | Dense prefill + dense decode | C1 prefill + C1 decode | Dense prefill + C1 decode |
|---|---:|---:|---:|
| qasper | 39.3018 | 19.6094 | 34.9041 |
| multifieldqa_en | 52.8498 | 30.7387 | 47.9694 |
| hotpotqa | 60.8872 | 29.7403 | 59.3363 |
| 2wikimqa | 50.1190 | 31.2642 | 47.2545 |
| gov_report | 29.1896 | 27.2987 | 29.3136 |
| qmsum | 26.1460 | 26.3270 | 27.3302 |
| Mean | 43.0822 | 27.4964 | 41.0180 |

Original Qwen3Attention modules perform every prefill. Cached BF16 V128 is then multiplied by the frozen C1 encoder to obtain V80; K tensors are unchanged.
Decode scans all K and C1-V tokens with the C1 output decoder and SDPA backend. No routing, sparse pages, sidecar, CPU offload or refitting.
The first generated token is dense-prefill argmax, verified against the saved dense baseline on all 192 prompts. C1 first participates in computing the second generated token.
Same uniform C1-V80 checkpoint, C4 32 x 32K fit / 4 x 32K held-out. Greedy decoding and unchanged EOS/task caps.
32K is the input-plus-reserved-output cap, not a fixed prompt length. Actual prompts: 1,192–30,431 tokens. This is a six-task pilot, not full LongBench.
QA scores are official F1; summary scores are official ROUGE-L, all on a 0–100 scale. Mean is the arithmetic mean across six tasks.
Old results are reused without modification. Switching prefill also changes subsequent hidden states and therefore the keys produced during decode.

First-token agreement with dense: 192/192.
Generation-cap exits without EOS: 56/192.
Maximum allocated GPU memory: 23.366 GiB.

## Paired score changes

```json
{
  "dense": {
    "improvements": 42,
    "regressions": 60,
    "ties": 90,
    "mean_delta_pp": -2.0642226526817247
  },
  "c1_prefill": {
    "improvements": 96,
    "regressions": 45,
    "ties": 51,
    "mean_delta_pp": 13.521629334928312
  }
}
```
