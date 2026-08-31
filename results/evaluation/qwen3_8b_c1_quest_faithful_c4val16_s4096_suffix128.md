# Qwen3 C1 K-only Reverse ShadowKV block-scheduled NLL/PPL

Dense-C1 and Reverse ShadowKV score identical fixed tokens with the same direct target-block commit schedule.

Windows: `16 x 4096`; block length: `32`; full-exact layers: `[0, 1]`.

| QUEST support | Budget | Tokens | Dense-C1 PPL | Sparse PPL | PPL ratio | Mean NLL delta | Paired SE | Physical K fraction | Query support fraction | Top-1 agreement |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| physical_shared | 512 | 2048 | 6.02844838 | 6.16361316 | 1.02242116 | +2.21734971e-02 | 8.654e-03 | 0.12510849 | 0.12510849 | 0.89794922 |
| physical_shared | 1024 | 2048 | 6.02844838 | 6.04754811 | 1.00316827 | +3.16325884e-03 | 4.019e-03 | 0.25207688 | 0.25207688 | 0.92675781 |
| physical_shared | 2048 | 2048 | 6.02844838 | 6.02397882 | 0.99925859 | -7.41685657e-04 | 2.584e-03 | 0.50601364 | 0.50601364 | 0.95410156 |
| per_query_head | 512 | 2048 | 6.02844838 | 6.14147806 | 1.01874938 | +1.85757794e-02 | 7.547e-03 | 0.21500024 | 0.12510849 | 0.91015625 |
| per_query_head | 1024 | 2048 | 6.02844838 | 6.05122244 | 1.00377777 | +3.77064769e-03 | 2.497e-03 | 0.39214268 | 0.25207688 | 0.93359375 |
| per_query_head | 2048 | 2048 | 6.02844838 | 6.03143576 | 1.00049555 | +4.95424596e-04 | 2.593e-03 | 0.67268149 | 0.50601364 | 0.95898438 |

This is an all-layer, end-to-end teacher-forced quality oracle. Exact K remains on GPU and QUEST metadata is rebuilt in Python; runtime is therefore not a CPU-offload latency measurement.
