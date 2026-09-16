# Fused two-stage cache append

The candidate combines Dense V storage, V128 -> Base16, Base16 -> predicted K128, bias and RoPE, residual subtraction and R16 projection, recent64 ring replacement, and Page32 min/max update in one Triton program per batch/KV head. CPU K append remains separate. Routing, candidate count, hard2048 support, sink32/recent64 and slot policy are unchanged.

The historical summary consumes the displaced ring element K[t-64], not current K[t]. Base matmul, reconstruction, bias, the two RoPE products, RoPE addition, residual subtraction, and residual projection retain the original BF16 round boundaries. FP contraction is disabled. Parallel reduction order can differ from cuBLAS, so equality is measured rather than assumed.

The native cache's existing update was extracted into append_codes without changing its operations. Only the benchmark's --fused-append switch installs the new append implementation; prefill retains the existing path. No default promotion yet.

Validation uses all 32 real fitted factor sets, alternating batch1/2, non-contiguous QKV views, and 66 consecutive updates crossing page and ring boundaries. It checks exact V/ring/minmax and records Base/Residual relative MSE, max error, and bitwise equality. Synthetic activations are used here; model smoke checks attention against an FP32 selected-support reference.

The CUDA-graph microbenchmark isolates GPU codes+metadata append, excluding CPU K storage. It repeatedly overwrites one position and measures operator cost, not evolving slot-cache behavior. The model comparison uses 64K, batch1, 100 continuous decode steps and old/new/new/old order, discarding the first 10 steps for steady-state timing.

Environment: basis. Commands:

```bash
python -m benchmarks.system.validate_fused_append
python -m benchmarks.system.run_fused_append
```

The installed Triton lacks tl.gather; paired RoPE coordinates are computed from the shifted decoder columns. No environment upgrades were needed.
