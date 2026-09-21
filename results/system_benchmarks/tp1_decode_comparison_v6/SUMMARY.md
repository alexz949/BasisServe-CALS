# TP1 dense controls and BasisKV comparison

Llama-3.1-8B-Instruct, TP1 B1, exact Flash-SDPA prefill, fixed teacher-forced decode inputs. Full-model values aggregate two runs; component values use a separate event-instrumented run and therefore are not substituted into the primary latency numbers.

- Primary runs: 16 warmup + 128 measured tokens, two repeats
- Breakdown runs: 16 warmup + 64 measured tokens
- Dense-offload copies the complete exact K and V cache from mapped host memory to one shared GPU staging buffer at every layer and decode token.
- Environment: `basis` conda environment, PyTorch 2.13.0, 8x NVIDIA L40S
- Command: `conda run --no-capture-output -n basis python benchmarks/system/run_tp1_dense_comparison_v6.py --warmup-steps 16 --measure-steps 128 --breakdown-steps 64`
- Failures: none (32/32 new-control trials completed; all logits finite)

## Full-model decode

| Context | Dense local | Dense offload | Basis local | Basis K-offload | Local speedup | Offload speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16K | 27.106 | 106.662 | 29.705 | 32.689 | 0.91x | 3.26x |
| 32K | 30.081 | 194.148 | 31.156 | 34.260 | 0.97x | 5.67x |
| 64K | 35.944 | 355.897 | 34.320 | 37.149 | 1.05x | 9.58x |
| 128K | 47.473 | 686.124 | 40.347 | 43.592 | 1.18x | 15.74x |

## Attention block

This interval begins after Q/K/V projection and RoPE, and ends before W_O. It includes cache append and all method-specific routing or transfer work.

| Context | Dense local | Dense offload | Basis local | Basis K-offload |
|---:|---:|---:|---:|---:|
| 16K | 3.802 | 82.750 | 6.492 | 9.360 |
| 32K | 6.693 | 170.255 | 7.954 | 11.034 |
| 64K | 12.547 | 332.001 | 11.016 | 13.838 |
| 128K | 23.925 | 661.905 | 17.081 | 20.160 |

## KV memory and host traffic

Values exclude model weights. Each offload cell is `GPU / host / host-to-GPU GiB per decode token`. Basis K-offload traffic is the theoretical exact-K support read for 2,048 physical tokens.

| Context | Dense local GPU | Dense offload | Basis local GPU | Basis K-offload |
|---:|---:|:---|---:|:---|
| 16K | 2.000 | 0.062 / 2.000 / 2.000 | 2.250 | 1.250 / 1.000 / 0.125 |
| 32K | 4.000 | 0.125 / 4.000 / 4.000 | 4.500 | 2.500 / 2.000 / 0.125 |
| 64K | 8.000 | 0.250 / 8.000 / 8.000 | 9.000 | 5.000 / 4.000 / 0.125 |
| 128K | 16.000 | 0.500 / 16.000 / 16.000 | 18.000 | 10.000 / 8.000 / 0.125 |

## Main observations

- Basis-local crosses Dense-local between 32K and 64K. At 128K it is 1.18x faster end to end; its attention block is 17.081 ms versus 23.925 ms.
- The selected sparse-attention kernel remains nearly flat at about 2.45 ms across all contexts for local K, while the B16R16 router scan grows from 2.40 ms at 16K to 12.67 ms at 128K.
- K-offload adds about 3 ms to the fixed sparse-attention stage, while full-KV Dense-offload host traffic grows from 2 to 16 GiB per token.
- Dense local/offload produced identical token outputs at every context; Basis local/offload also matched exactly. Dense and Basis are not expected to match because Basis uses routed sparse support.

Dense-local and Dense-offload are systems controls, not accuracy baselines. ShadowKV and LRQK remain the next paper-faithful external-baseline stage.
