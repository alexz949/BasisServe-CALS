# Register-consumed MMA router

Environment: `basis`; Slurm, one L40S per job on lovelace. Model: meta-llama/Llama-3.1-8B (base, not Instruct), snapshot d04e592bb4f6aa9cfee91e2e20afa771667e1d4b, TP1. K128, GQA4, Page32; Dense V128 and original Wo in the model benchmark. Nominal 2048 budget, existing sink/recent and slot-cache policy unchanged. B8R8 truncates existing B16R16 factors; no refit.

## Implementation

A warp owns 16 tokens. The factor is transposed and RoPE-paired while loading shared memory; no new persistent model-side factor. BF16 m16n8k8 / m16n8k16 reconstructs four pairs per output tile. Each lane consumes two token pairs in registers, preserves explicit BF16 rounding, and accumulates four query scores. Four-lane reductions write only token scores. The residual and Page-LSE formulas are unchanged. The full FP32 reconstructed-K shared buffer is removed. The 4-warp/8-warp CTA covers two/four independent pages. This is a benchmark candidate; production dispatch is unchanged.

## Correctness

Synthetic status: complete; 52 cases. Ranks 8/16, warps 4/8, lengths 1/31/32/33/63/64/65/127/128/129/257/8192/65536. Short cases also use an independent FP32 matrix product with explicit BF16 boundary reference; multiple batches and KV heads are covered. Tolerance: atol=rtol=0.003. Maximum candidate/current page-LSE difference: 8.4400177e-05.

Changing MMA tiling and QK reduction order is not bitwise-equivalent by construction. Real selected-page comparisons are reported below; no new RULER quality evaluation is claimed.

## Synthetic 64K router

| Rank | Warps | Current ms/layer | Candidate ms/layer | Latency reduction |
|---|---|---|---|---|
| 8 | 4 | 0.258355 | 0.201626 | 21.96% |
| 8 | 8 | 0.258509 | 0.178278 | 31.04% |
| 16 | 4 | 0.276480 | 0.211507 | 23.50% |
| 16 | 8 | 0.276378 | 0.198758 | 28.08% |

## Real-query trace

Matched inputs from continuous decode; 352 samples per trace (step 0 and steps 10–19 across 32 layers). Timings sum per-layer CUDA-graph microbench medians at steps 10 and 19, including residual-query projection, excluding selector. Both versions use preallocated outputs and the same C++ API. Probed decode wall time is not a serving metric.

| Rank | Warps | Current ms/step | Candidate ms/step | Reduction | Minimum / mean page overlap | Max score difference |
|---|---|---|---|---|---|---|
| 16 | 4 | 8.8926 | 6.7814 | 23.74% | 1.00000000 / 1.00000000 | 0.120199 |
| 16 | 8 | 8.8761 | 6.3330 | 28.65% | 1.00000000 / 1.00000000 | 0.120199 |
| 8 | 8 | 8.3241 | 5.8733 | 29.44% | 1.00000000 / 1.00000000 | 0.077198 |

## Continuous decode

Each rank uses same-GPU ABBA order: current, candidate, candidate, current; 64K input, batch 1, 100 generated steps, fresh process each run, vector fetch and existing slot attention in both paths. Report the mean of two per-run steady CUDA medians, discarding the first ten steps. Prefill and placement are excluded.

| Rank | Current ms/step | Candidate ms/step | Reduction | Speedup | All recorded token sequences equal |
|---|---|---|---|---|---|
| 8 | 35.5013 | 32.8845 | 7.37% | 1.0796x | True |
| 16 | 36.0806 | 33.2718 | 7.78% | 1.0844x | True |

Per-run steady CUDA median and native decode wall latency (the latter includes startup within the decode loop):

| Rank | Path | Repeat | CUDA ms/step | Native wall ms/step | Finite logits |
|---|---|---|---|---|---|
| 16 | baseline | 0 | 36.0837 | 39.1494 | True |
| 16 | baseline | 1 | 36.0776 | 38.8759 | True |
| 8 | baseline | 0 | 35.4637 | 38.5378 | True |
| 8 | baseline | 1 | 35.5389 | 38.5762 | True |
| 16 | candidate | 0 | 33.2518 | 36.1724 | True |
| 16 | candidate | 1 | 33.2918 | 36.1522 | True |
| 8 | candidate | 0 | 32.8607 | 35.7561 | True |
| 8 | candidate | 1 | 32.9083 | 35.8164 | True |

B16 4-warp trace additionally ran the previous slot/fetch microbench probes; subsequent traces only probe the router. Neither trace wall time is used for speedup. Continuous-decode jobs have no such probes.

Files: `register_router_body.cuh` contains the candidate kernel body; `register_router.py` builds isolated current/candidate extensions; `validate_register_router.py` checks mathematics and boundaries; `bench_register_router.py` and `run_register_router.py` run matched real workloads. `bench_local_kernels.py` now permits the experiment output root and skipping unrelated slot probes, and uses symmetric preallocated C++ calls for router A/B timing.

## Compiled resources

Current kernels use dynamic shared memory (23,680 bytes for B16); cuobjdump reports only their static shared portion. Candidates use static shared.

```
register_router_43ee189a58_b16_w4: REG:40 STACK:0 SHARED:8576 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0
register_router_43ee189a58_b8_w4: REG:40 STACK:0 SHARED:5440 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0
register_router_705a340c3b_b16_w0: REG:49 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0
register_router_705a340c3b_b8_w0: REG:49 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0
register_router_9b5d81179a_b16_w8: REG:48 STACK:0 SHARED:11648 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0
register_router_9b5d81179a_b8_w8: REG:40 STACK:0 SHARED:7488 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0
```

## Commands

Conda environment: `basis`. CUDA 12.3.2, TORCH_CUDA_ARCH_LIST=8.9, MAX_JOBS=2.

```bash
python -m benchmarks.system.validate_register_router
python -m benchmarks.system.run_register_router --phase trace
python -m benchmarks.system.run_register_router --phase decode --rank 16 --warps 8
python -m benchmarks.system.run_register_router --phase decode --rank 8 --warps 8
```

Commands for decode are applicable when the corresponding completed runs appear above. Exact child commands and progress are in trace.log / decode_b*.log and the per-run summary JSON. No commit or push performed.
