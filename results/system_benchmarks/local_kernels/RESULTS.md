# Local kernel optimization results

Tested on 2026-09-15 in the `basis` environment, CUDA 12.3.2, one L40S, Llama-3.1-8B base, batch 1, Dense V128, original Wo, B16R16, Page32 and hard budget 2048 including sink32 and recent64. No routing formula, page budget, selection rule or cache policy was changed.

## Adopted changes

- `basisserve/kernels/csrc/conditional_router_page32.cu`: retain WMMA Base reconstruction and all existing BF16 rounding points. Each warp handles tokens and reuses the rounded RoPE coordinates in registers across the real query heads. Remove the rotated-K shared-memory buffer and padded QK WMMA stage. QK summation order changes, so score equality is not guaranteed bitwise.
- `basisserve/kernels/csrc/persistent_key_slots.cu`: each active thread fetches/stores one aligned `uint4` (eight BF16 values). The planner and miss list are unchanged. SASS confirms `LDG.E.128` and `STG.E.128` in place of the scalar BF16 load/store path.
- `basisserve/kernels/gqa_slot_attention.py`: implemented and checked as an experimental shared-KV GQA candidate, but **not enabled** because it was slower than the existing attention kernel.

Router dynamic shared memory decreased from 35,968 to 23,680 bytes per block (12 KiB saved). Registers/thread changed from 48 to 49, with zero recorded stack/local memory. Static SASS sites changed from 20 to 4 BF16 HMMA instructions and from five to three block barriers. Static counts are not dynamic instruction totals or predicted speedups; the measured timings below determine adoption.

## Same-input operator comparison

Each candidate saw the same real query, selected IDs, K slots and V data. The original kernels drove the generation trajectory. Measurements covered the first decode step and steps 11–20 across all 32 layers; the steady operator table excludes the first step. Router timing used steps 11 and 20. Times are sums over layers, averaged over the sampled steps.

| Component | Original ms/step | Candidate ms/step | Result |
|---|---:|---:|---|
| Router, including residual-query projection | 9.829 | 8.871 | 9.7% lower latency |
| Missing-K fetch | 1.426 | 1.375 | 3.6% lower latency |
| Slot attention, shared-GQA with 4 warps | 0.669 | 1.305 | Slower; not adopted |
| Slot attention, shared-GQA with 8 warps | 0.669 | 1.084 | Slower; not adopted |

Fetch microbenchmarks repeatedly execute the actual miss list without rerunning the planner, so they do not turn into all-hit tests. However, repeated reads may be cache-warm. They are not substitutes for the continuous-decode results below.

GQA grouping does not guarantee a gain: this implementation reduced the number of programs and changed the reduction/register layout, while the original kernel could already benefit from cache reuse. No bandwidth-counter measurement was made, so these results do not establish a DRAM-traffic reduction or a general limit on shared-GQA implementations. The existing attention kernel was already less than 0.7 ms per model step in this test.

## Continuous decode without A/B timing probes

Router fusion and vector fetch were combined, retaining the original slot attention. A single Slurm allocation ran fresh processes in baseline/optimized/optimized/baseline order. Each run generated 100 tokens. Both variants used the same Python router wrapper, token recording and logit-finiteness checks.

| Variant | Repeat | CUDA median after first 10 steps, ms | CUDA mean after first 10 steps, ms | Whole native decode loop, ms/step |
|---|---:|---:|---:|---:|
| Baseline | 0 | 37.245 | 37.352 | 49.744 |
| Optimized | 0 | 36.047 | 36.155 | 38.863 |
| Optimized | 1 | 36.026 | 36.115 | 38.878 |
| Baseline | 1 | 37.204 | 37.317 | 40.783 |

Mean of the two run medians: **37.225 → 36.036 ms**, a **3.2% latency reduction** (about 1.19 ms), or approximately **1.033× throughput** at this batch size.

The whole-loop column includes initial-step overhead and host sampling, whereas the steady CUDA columns do not. In particular, the first baseline run's whole-loop result was substantially affected by startup overhead; it must not be used to claim a roughly 28% steady speedup. Prefill and initial placement are outside these decode timings. No complete-request or RULER speed/quality gain is claimed.

All four runs recorded **identical sequences of 101 decode inputs**, including the native runtime's extra final inference. All decoded logits were finite. Cache hit fractions were approximately 72.53%.

## Correctness and limits

- 46 synthetic cases passed, covering V80/V128, B8/B16, cold fill, full hits, partial hits, evictions, reload, invalid IDs, empty splits and token/page boundaries.
- Vector fetch reproduced K bitwise. The planner uses atomic free-slot assignment; two independently evolving slot maps can retain different older entries. The synthetic A/B test therefore starts each refresh from the same resident/lookup state. The real-state operator test executes one planner and shares its slots/miss list across candidates.
- GQA attention's largest synthetic difference was about `9.54e-7`; the largest real-trace difference was `0.001953125`. Comparisons against the original attention and the synthetic dense selected-support reference passed the existing `0.003` tolerances.
- Synthetic router cases matched the original exactly. Across **352 real layer/query samples**, the maximum page-score difference was `0.1222896576`, but the selected page sets were **identical in every sample**. The difference arises despite preserving BF16 rounding points because QK accumulation order changes.
- These finite-sample checks do not prove bitwise score equality or unchanged selection for every possible input. B8 passed synthetic router checks; the reported full-model speedup is specifically B16R16 at 64K.
- After adoption, the default CUDA paths were rebuilt and passed an 8K full-model smoke. Its decode input sequence matched the candidate smoke exactly, and all logits were finite.

## Reproduction and provenance

Commands run through Slurm, in `basis`:

```bash
python -m benchmarks.system.validate_local_kernels
python -m benchmarks.system.bench_local_kernels --mode trace --rank 16 --smoke
python -m benchmarks.system.bench_local_kernels --mode trace --rank 16 --length 65536
python -m benchmarks.system.run_local_decode
python -m benchmarks.system.bench_local_kernels --mode deployed --rank 16 --smoke
```

Relevant jobs: validation `8325231`, model smoke `8325232`, same-input 64K trace `8325233`, continuous-decode A/B `8325236`, default-path integration smoke `8325241`.

`local_kernel_candidates.py` pins reference CUDA sources to commit `0d1c847d7799983dcd309ec2b9f4e486cfc6b526`. The pre-adoption source hashes were checked against this commit. Original/candidate generated CUDA, raw per-layer samples, timings, commands, source hashes and logs are retained locally under this directory. `deployment.json` records the before/after hashes. No files from this experiment have been committed or pushed.
