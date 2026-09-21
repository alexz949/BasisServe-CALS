# TP1 Llama-3.1-8B decode optimization

This experiment compares the existing full B16R16 Page32 router and direct mapped-host exact-K attention against the optimized decode path:

- coarse exact-K page bounds followed by a 512-page B16R16 fine scan;
- fused radix candidate selection with preallocated output;
- persistent 2,048-token GPU K slots with vectorized PCIe miss fetches;
- metadata ring and page-bound updates fused into the decode append kernel.

Both paths use exact Flash-SDPA prefill, the same Llama-3.1-8B-Instruct checkpoint, teacher-forced decode tokens, 62 routed pages, and the exact recent 64 tokens.

## Results

Each row pools two independent runs with 16 warmup and 128 measured decode steps per run (256 measured samples). Main timing runs do not enable component profiling.

| Context | Path | CUDA median (ms/token) | Mean | P95 | Max | Throughput (tok/s) | Peak GPU GiB |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 64K | Full router + direct PCIe attention | 37.291 | 37.299 | 37.447 | 38.460 | 26.80 | 23.83 |
| 64K | Two-stage + persistent K slots | **31.265** | 31.315 | 31.942 | 32.207 | **31.92** | 24.28 |
| 128K | Full router + direct PCIe attention | 43.577 | 43.618 | 43.961 | 44.209 | 22.92 | 32.67 |
| 128K | Two-stage + persistent K slots | **31.944** | 31.996 | 32.588 | 32.736 | **31.22** | 33.43 |

The optimized path reduces median decode latency by **16.16% at 64K** and **26.70% at 128K**. Throughput increases by **19.12%** and **36.26%**, respectively. The final-step persistent-slot hit fractions are 52.88% at 64K and 55.80% at 128K.

## Correctness and stability

- All logits were finite.
- Both repeats produced identical argmax sequences within each path.
- Compared with full routing, two-stage routing changed 4 of 144 recorded argmax tokens at 64K and 12 of 144 at 128K. This is expected because the 512-page coarse candidate stage is approximate and requires a separate long-context quality evaluation.
- The 4K end-to-end smoke validated all 32 sparse-attention layer outputs against an explicit selected-support reference.
- Candidate-only fine routing matched full routing exactly for the retained 64K candidates, and the fused selector matched the reference selector for 4,096 input pages.
- There were no samples above 2x the group median in the final formal runs.

An earlier run exposed periodic 0.4--0.9 second stalls at page boundaries. `P`, `TAIL`, and `N` had been specialized by Triton during decode. Making them runtime scalars and disabling alignment specialization removed the stalls; a component-profiled step-63 regression completed in 32.43 ms with a 3.37 ms router scan, 0.99 ms selection, 2.34 ms K refresh, and 0.83 ms slot attention across all 32 layers.

## Commands

Environment: `basis` conda environment on NVIDIA L40S (SM89), direct local execution.

Baseline template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> CUDA_HOME=/usr/local/cuda conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_sparse_full_v6.py \
  --mode optimized --storage offload --routing full \
  --length <65536|131072> --warmup-steps 16 --measure-steps 128 \
  --repeat <0|1> --tag formal-full \
  --output-root results/system_benchmarks/tp1_decode_v7
```

Optimized template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> CUDA_HOME=/usr/local/cuda conda run --no-capture-output -n basis \
  python benchmarks/system/bench_tp1_sparse_full_v6.py \
  --mode optimized --storage offload --routing two-stage --key-reuse \
  --length <65536|131072> --warmup-steps 16 --measure-steps 128 \
  --repeat <0|1> --tag formal-two-stage-reuse \
  --output-root results/system_benchmarks/tp1_decode_v7
```

The per-trial directories contain `benchmark.json` and `run.log`. The initial slot validation attempt did not enter the kernel because `CUDA_HOME` was unset; rerunning with `CUDA_HOME=/usr/local/cuda` passed. No final formal trial failed.
