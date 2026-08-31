# Qwen3 C1 K-only Reverse ShadowKV block-scheduled NLL/PPL

Dense-C1 and Reverse ShadowKV score identical fixed tokens with the same direct target-block commit schedule.

Windows: `1 x 256`; block length: `32`; full-exact layers: `[0, 1]`.

| Budget | Tokens | Dense-C1 PPL | Sparse PPL | PPL ratio | Mean NLL delta | Paired SE | Top-1 agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 255 | 14.19965376 | 56.56586553 | 3.98360879 | +1.38218814e+00 | n/a | 0.57254902 |
| 64 | 255 | 14.19965376 | 26.42967045 | 1.86128978 | +6.21269679e-01 | n/a | 0.74117647 |

This is an all-layer, end-to-end teacher-forced quality oracle. Exact K remains on GPU and QUEST metadata is rebuilt in Python; runtime is therefore not a CPU-offload latency measurement.
