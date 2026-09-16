# Fused append results

Llama-3.1-8B base, single L40S, TP1, batch1, 64K input (window64), Dense V128 and original Wo, B16R16, two-stage 512 candidates, hard2048 including sink32/recent64, CPU K offload and persistent slots. Basis environment. Only GPU append+metadata update changed; CPU K write and routing are unchanged.

| Run order | Steady CUDA median ms/step |
|---|---:|
| Reference repeat 10 | 31.2980 |
| Fused repeat 10 | 30.3288 |
| Fused repeat 11 | 30.2904 |
| Reference repeat 11 | 31.3175 |

Mean of two run medians: reference **31.3078 ms**, fused **30.3096 ms**. Reduction **3.19%**, saving **0.9981 ms/step**. Each run generates 100 continuous decode steps, first10 discarded for timing. This is decode timing, excluding prefill, not end-to-end request speedup. ABBA order uses one GPU sequentially.

All four recorded 101-token input sequences match exactly and all logits are finite. Slot hit fractions are close but not bit-identical; floating-point reduction differences mean routing support should not be claimed bitwise identical. This single-window check is not a replacement for RULER quality validation.

32 layer factor sets passed 66-step synthetic append tests, alternating batch1/2 and non-contiguous QKV. V/ring/historical min/max match exactly. Worst per-case relative MSE: Base 3.596e-11, Residual 1.224e-08. The 8K model smoke verified selected-support attention against the FP32 reference in all32 layers.

The isolated CUDA-graph append microbenchmark (batch2, last layer, synthetic data) measured 26.88 us reference versus 3.69 us fused. It excludes CPU K append and does not substitute for the measured full decode result.

Commands:

```bash
python -m benchmarks.system.validate_fused_append
python -m benchmarks.system.run_fused_append
```

Completed decode job: 8327353. Initial attempts exposed unsupported tl.gather in the installed Triton and a missing factor local variable after extracting append_codes. Both were fixed and validation/smoke rerun; final jobs completed successfully. Logs retain the failed attempts.

The candidate is enabled explicitly by --fused-append in bench_two_stage. It has not been promoted to the default RULER runtime. No commit or push.
