# Qwen3 C1 K-only Reverse ShadowKV block-scheduled NLL/PPL

Dense-C1 and Reverse ShadowKV score identical fixed tokens with the same direct target-block commit schedule.

Windows: `1 x 256`; block length: `32`; full-exact layers: `[0, 1]`.

| QUEST support | Budget | Tokens | Dense-C1 PPL | Sparse PPL | PPL ratio | Mean NLL delta | Paired SE | Physical K fraction | Query support fraction | Top-1 agreement |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| physical_shared | 64 | 128 | 10.82088629 | 12.17901129 | 1.12550959 | +1.18235902e-01 | n/a | 0.29350649 | 0.29350649 | 0.89062500 |
| physical_shared | 128 | 128 | 10.82088629 | 11.37709330 | 1.05140124 | +5.01237920e-02 | n/a | 0.62597403 | 0.62597403 | 0.96093750 |
| per_query_head | 64 | 128 | 10.82088629 | 12.02483723 | 1.11126177 | +1.05496097e-01 | n/a | 0.43514610 | 0.29350649 | 0.91406250 |
| per_query_head | 128 | 128 | 10.82088629 | 11.11322243 | 1.02701591 | +2.66574265e-02 | n/a | 0.79340145 | 0.62597403 | 0.95312500 |

This is an all-layer, end-to-end teacher-forced quality oracle. Exact K remains on GPU and QUEST metadata is rebuilt in Python; runtime is therefore not a CPU-offload latency measurement.
