# Conditional router: measured phase costs and transformed-query test

Tested on 2026-09-15 with Llama-3.1-8B base, one L40S, batch 1, 64K input, Dense V128, B16R16, Page32 and the existing hard-2048 selector (sink32 and recent64 included). The test uses the first real decode query at each of 32 layers. All existing serving outputs continue to use the original router.

Commands, in the `basis` environment through Slurm:

```bash
python -m benchmarks.system.profile_router_phases --smoke
python -m benchmarks.system.profile_router_phases
```

Smoke job: `8325053`; 64K job: `8325054`. CUDA 12.3.2, SM89, four allocated CPU cores, one GPU, 64 GiB host allocation. Results and logs are under `smoke/` and `64k/` in this directory. The commands are diagnostic measurements, not end-to-end throughput runs; the nested native runtime's throughput includes profiling hooks and must not be used as serving performance.

## Main result

The original Python call path takes **10.482 ms across 32 layers**, close to the earlier 10.54 ms profile. Its isolated page kernel takes **9.709 ms**. The transformed-query prototype takes **24.969 ms**, including **0.057 ms** to prepare position-independent coefficients: **2.57 times the original page-kernel cost**.

The prototype does not reconstruct approximate K, but it must still perform position-dependent query projection separately for each of four query heads. Removing K reconstruction is therefore not equivalent to removing its cost without replacement.

## Source locations and approximate phase attribution

Locations refer to `basisserve/kernels/csrc/conditional_router_page32.cu` at the tested revision.

| Phase | Original source lines | Approximate attributed ms, 32 layers |
|---|---|---:|
| Load Base16 codes and right factor | 136–153 | 1.385 |
| B16 → K128 reconstruction | 156–190 | 1.322 |
| Add bias and BF16 rounding | 200–208 | 1.298 |
| RoPE and its BF16 rounding | 209–225 | 1.584 |
| Load query and residual query code | 228–250 | 1.075 |
| QK matrix product | 252–289 | 1.364 |
| Residual dot and final score rounding | 292–310 | 1.055 |
| Page-LSE | 311–315 | 0.626 |
| **Page-kernel total** | | **9.709** |

These are **attribution estimates**, not eight independently timed kernels. The fine-grained clone separates bias from RoPE and residual scoring from LSE. Its `clock64` block-cycle proportions are scaled by the unmodified kernel's measured time. They include stalls/barriers and are affected by the changed layout. The instrumented clone costs **10.650 ms**, 9.7% more than the original page kernel. Its output matches the original **bitwise in all 32 layers**, in both smoke and 64K tests.

Preparing the residual query code separately takes **0.130 ms**. The roughly **0.643 ms remaining difference** between the eager wrapper and isolated kernels includes wrapper/launch/allocation effects and timing-method differences; it is not an independently timed component.

**V80 → B16 is absent from the history scan.** Base codes are cached. The actual benchmark uses V128; projecting each layer's one newly appended token costs about **0.055 ms across 32 layers** in isolation. The explicitly labelled V80 dimension control costs **0.054 ms**, using sliced values/factors rather than a newly fitted V80 model. Neither number belongs inside the 9.709 ms page scan.

## What the transformed-query version actually removes

It removes the explicit shared-memory K128 reconstruction path and the original per-token K/bias/RoPE BF16 rounding sequence. It also replaces the original QK operation with rank-16 scoring. There was already **no reconstructed-K HBM round trip** to eliminate.

It adds two coefficient arrays per query and then computes, for every key position and query head:

`u[t,r] = sum_i cos[t,i] C[q,r,i] + sin[t,i] S[q,r,i]`.

This projection is shared neither across key positions nor across the four query heads. The original reconstruction is shared across those heads. Useful multiply-accumulate work for a Page32/KV-head changes from 65,536 for reconstruction plus 16,384 useful QK operations to 262,144 for position-dependent transformed coordinates, before bias and residual scoring. Padding and precision implementation add hardware work beyond those useful-operation counts.

The prototype uses TF32x3 products for FP32 coefficients. Its page kernel uses 186 registers/thread, 8,192 bytes of shared memory and zero recorded spills. This is a tested diagnostic implementation, not a proof that every transformed-query implementation must be slower.

## Compiled instruction evidence

The SASS dumps and full opcode inventories are retained locally. Selected **static instruction-site counts** are:

| Instruction family | Original page kernel | Prototype page kernel |
|---|---:|---:|
| BF16 tensor MMA, `HMMA.16816.F32.BF16` | 20 | 0 |
| TF32 tensor MMA, `HMMA.1688.F32.TF32` | 0 | 96 |
| BF16 conversion, `F2FP.BF16.F32.PACK_AB` | 63 | 0 |
| BF16 conversion, `F2F.BF16.F32` | 0 | 4 |
| Block synchronization | 5 | 31 |

These are not dynamic instruction totals or a predicted speed ratio: block grids, loop counts, active warps, instruction shapes and predicates differ. They do show why deleting BF16 reconstruction instructions did not produce a net speed gain in this prototype. The prototype also still uses shared-memory staging internally for matrix products and reductions.

## Numerical equivalence

- Instrumented original versus original: bitwise equal, all 32 layers.
- Prototype versus a floating-point reassociation reference on five sampled pages/layer: worst per-layer RMSE **0.0000823**, worst absolute difference **0.001041**.
- Prototype versus the original BF16-rounded router over all pages: average per-layer score RMSE **0.01688**, maximum score difference **0.49771**.
- Original versus prototype selected-page overlap: **99.401% mean**, **98.589% worst layer**. Both use the same pinned-page selection rule.

The identity `q^T R_t (B^T z_t+b) = z_t^T B R_t^T q + q^T R_t b` holds in real arithmetic. It does not commute through the original intermediate BF16 rounding. High page overlap does not establish unchanged output error or RULER quality; neither was evaluated for the prototype.

## Next optimization target

Retain the original kernel as the serving implementation. Bias/rounding plus RoPE account for approximately **2.88 ms**, compared with approximately **1.32 ms** for reconstruction alone and **0.63 ms** for Page-LSE. A focused next experiment is to reduce conversion/shared-memory traffic and examine the padded QK layout, while preserving the original numerical contract. Such changes need their own same-input correctness and latency tests; the present measurements do not establish their achievable speedup.
