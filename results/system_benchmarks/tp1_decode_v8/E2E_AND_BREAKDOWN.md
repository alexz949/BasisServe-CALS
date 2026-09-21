# TP1 128-token E2E latency and decode breakdown

## Measurement boundary

This report uses Llama-3.1-8B-Instruct, TP1, batch one, and the same frozen
prompt and teacher-forced decode-token windows for every method. Request
latency is defined as:

```text
E2E-128 = prefill + post-prefill cache placement + 128 full-model decode steps
```

Model loading, tokenization, sampling, and detokenization are excluded. The
decode term is the sum of the 128 measured wall-time samples after 16 warmup
steps. Prefill and decode each pool two fresh-process repeats. With two equal
repeats, the reported median E2E is also the mean of the two per-repeat totals.

## E2E-128 result

| Context | Dense local | Dense offload | BasisKV v8 | ShadowKV default | LRQK default |
|---:|---:|---:|---:|---:|---:|
| 16K | 5.813 s | 16.049 s | **6.324 s** | 17.834 s | 39.904 s |
| 32K | 8.760 s | 29.754 s | **8.811 s** | 20.961 s | 50.799 s |
| 64K | 17.165 s | 58.164 s | **16.343 s** | 34.956 s | 75.050 s |
| 128K | 42.839 s | 125.688 s | **40.779 s** | 62.681 s | OOM |

BasisKV v8 is within 8.79% of Dense-local at 16K and within 0.59% at 32K. It
becomes 1.050x faster than Dense-local at both 64K and 128K. Relative to the
paper-faithful ShadowKV-default runtime, its E2E speedup is 2.820x, 2.379x,
2.139x, and 1.537x from 16K through 128K.

## E2E phase breakdown

| Context | Method | Prefill | Placement | 128-token decode | E2E-128 |
|---:|:---|---:|---:|---:|---:|
| 16K | Dense local | 2.337 s | - | 3.476 s | 5.813 s |
| 16K | Dense offload | 2.383 s | - | 13.666 s | 16.049 s |
| 16K | **BasisKV v8** | **2.475 s** | **-** | **3.848 s** | **6.324 s** |
| 16K | ShadowKV default | 13.642 s | 0.525 s | 3.667 s | 17.834 s |
| 16K | LRQK default | 10.448 s | - | 29.457 s | 39.904 s |
| 32K | Dense local | 4.906 s | - | 3.854 s | 8.760 s |
| 32K | Dense offload | 4.889 s | - | 24.865 s | 29.754 s |
| 32K | **BasisKV v8** | **4.987 s** | **-** | **3.824 s** | **8.811 s** |
| 32K | ShadowKV default | 16.658 s | 0.606 s | 3.698 s | 20.961 s |
| 32K | LRQK default | 17.586 s | - | 33.214 s | 50.799 s |
| 64K | Dense local | 12.562 s | - | 4.603 s | 17.165 s |
| 64K | Dense offload | 12.592 s | - | 45.572 s | 58.164 s |
| 64K | **BasisKV v8** | **12.410 s** | **-** | **3.932 s** | **16.343 s** |
| 64K | ShadowKV default | 28.405 s | 2.581 s | 3.970 s | 34.956 s |
| 64K | LRQK default | 39.140 s | - | 35.910 s | 75.050 s |
| 128K | Dense local | 36.759 s | - | 6.080 s | 42.839 s |
| 128K | Dense offload | 37.838 s | - | 87.850 s | 125.688 s |
| 128K | **BasisKV v8** | **36.761 s** | **-** | **4.018 s** | **40.779 s** |
| 128K | ShadowKV default | 57.136 s | 1.310 s | 4.236 s | 62.681 s |
| 128K | LRQK default | completed | - | OOM on first decode | OOM |

BasisKV constructs exact-K storage, Dense-V storage, B16R16 codes, and
two-stage page metadata inside its prefill interval. ShadowKV exposes a
separate `H2D()` cache-placement phase, so that phase is included explicitly.
LRQK completed 128K prefill but OOMed while restoring its official per-layer
low-rank decode state before persisting the prefill duration.

## BasisKV v8 decode breakdown

The following values are medians across 32 profile steps after eight warmup
steps. Each component is summed across all 32 transformer layers.

| Component | 16K | 32K | 64K | 128K |
|:---|---:|---:|---:|---:|
| Cache append | 0.359 ms | 0.359 ms | 0.357 ms | 0.369 ms |
| Two-stage router scan | 2.901 ms | 3.060 ms | 3.364 ms | 4.126 ms |
| Fused selection + support packing | 0.423 ms | 0.419 ms | 0.415 ms | 0.431 ms |
| Persistent K planning + PCIe fetch | 1.851 ms | 1.528 ms | 2.168 ms | 2.196 ms |
| Selected sparse attention | 0.793 ms | 0.800 ms | 0.798 ms | 0.817 ms |
| Attention-block residual | 0.400 ms | 0.399 ms | 0.402 ms | 0.418 ms |
| **Attention block** | **6.721 ms** | **6.563 ms** | **7.513 ms** | **8.378 ms** |
| QKV/RoPE/Wo/MLP/norm/LM head and model overhead | 24.087 ms | 24.058 ms | 24.062 ms | 25.581 ms |
| **Profiled full model** | **30.796 ms** | **30.625 ms** | **31.568 ms** | **33.956 ms** |

The attention-block residual is computed per token as the block event minus
the five named component events, then summarized; the outside-attention term
is computed the same way. This avoids subtracting independently selected
medians. Component medians do not need to add exactly to the block median.

Profiling inserts CUDA events around every component in every layer, so its
full-model totals are higher than the uninstrumented primary latency. The
primary values remain 29.964, 29.806, 30.636, and 31.331 ms/token. Profiled
totals are used only for attribution and are never substituted into E2E.

### Attention crossover against Dense-local

| Context | Dense-local attention block | BasisKV v8 attention block | Dense / Basis ratio |
|---:|---:|---:|---:|
| 16K | 3.802 ms | 6.721 ms | 0.566x |
| 32K | 6.693 ms | 6.563 ms | 1.020x |
| 64K | 12.547 ms | 7.513 ms | 1.670x |
| 128K | 23.925 ms | 8.378 ms | 2.856x |

The fixed-cost two-stage machinery loses at 16K, reaches parity near 32K, and
then wins increasingly as dense K/V reads scale with context. Within BasisKV,
selection/packing and selected attention remain flat. Router scan is the main
context-dependent term, growing from 2.901 ms at 16K to 4.126 ms at 128K.
K-refresh time is governed mainly by persistent-slot hit behavior rather than
context length and stays in the 1.53--2.20 ms range.

## Configuration and commands

- Environment: `basis`
- Hardware: NVIDIA L40S, SM89
- Model: Llama-3.1-8B-Instruct, TP1, batch one
- BasisKV: B16R16, Page32, 512 fine candidates, 2,048 physical support tokens,
  Dense V128, original Wo, CPU-pinned exact K, persistent GPU K slots
- Formal timing: two repeats, 16 warmup steps, 128 measured steps
- Component timing: one repeat, eight warmup steps, 32 measured steps
- Correctness: finite logits, repeat argmax agreement, and 32/32 validated
  sparse-attention layers at all four contexts

Formal timing command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> CUDA_HOME=/usr/local/cuda \
conda run --no-capture-output -n basis \
python benchmarks/system/bench_tp1_sparse_full_v6.py \
  --mode optimized --storage offload --routing two-stage --key-reuse \
  --length <16384|32768|65536|131072> \
  --warmup-steps 16 --measure-steps 128 --validate \
  --repeat <0|1> --tag formal-two-stage-reuse \
  --output-root results/system_benchmarks/tp1_decode_v8
```

Component command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> CUDA_HOME=/usr/local/cuda \
conda run --no-capture-output -n basis \
python benchmarks/system/bench_tp1_sparse_full_v6.py \
  --mode optimized --storage offload --routing two-stage --key-reuse \
  --length <16384|32768|65536|131072> \
  --warmup-steps 8 --measure-steps 32 --profile-components --validate \
  --repeat 0 --tag profile-two-stage-reuse \
  --output-root results/system_benchmarks/tp1_decode_v8
```

All successful runs produced finite logits. No final formal or component trial
failed. LRQK-default at 128K is the only recorded OOM in the comparison.
