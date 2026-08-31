# Qwen3 C1 K-only Reverse ShadowKV block-scheduled NLL/PPL

Dense-C1 and Reverse ShadowKV score identical fixed tokens with the same direct target-block commit schedule.

Windows: `1 x 4096`; block length: `32`; full-exact layers: `[0, 1]`.

| Budget | Tokens | Dense-C1 PPL | Sparse PPL | PPL ratio | Mean NLL delta | Paired SE | Top-1 agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 15.64798524 | 24.11448018 | 1.54105975 | +4.32470327e-01 | n/a | 0.75000000 |
| 1024 | 128 | 15.64798524 | 20.00413372 | 1.27838398 | +2.45596768e-01 | n/a | 0.76562500 |

This is an all-layer, end-to-end teacher-forced quality oracle. Exact K remains on GPU and QUEST metadata is rebuilt in Python; runtime is therefore not a CPU-offload latency measurement.
