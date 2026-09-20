# C1 L40S benchmark archive and optimization status

This index records the C1 serving experiments run on the eight-L40S PCIe host.
The archive intentionally retains raw JSON, logs, kernel CSVs and compressed
Chrome traces, including unsuccessful exploratory attempts. A recorded result
is evidence for the exact implementation and protocol in its metadata; it is
not automatically evidence of a fully optimized kernel or a performance
ceiling.

## Result sets

| Directory | Scope | Status and valid interpretation |
| --- | --- | --- |
| `results/vllm_tp8/` | Qwen3-8B, TP8, prefill 4096, decode 128, batch 1–256 | Complete paired E2E run for the original unified DiffKV kernel. Valid pre-optimization baseline; not the final optimized C1 result. |
| `results/vllm_32b_tp8/` | Qwen3-32B, same batch sweep | Complete paired E2E run for the original unified DiffKV kernel. Batch 256 is capacity-sensitive and dense preempted. Valid historical baseline, not a final ceiling. |
| `results/vllm_8b_tp8_seqlen/` | Qwen3-8B, batch 32, prefill 512–32640, decode 128 | Complete context sweep for the original unified DiffKV kernel. It exposes the long-context prefill overhead that motivated specialization. |
| `results/diffkv_sm89/` | QK128/V64 isolated prefill tuning, correctness, memcheck and TP8 smoke | Exploratory/specialization evidence. The single-layer CUDA Graph numbers exclude model, scheduler and communication costs and must not be presented as E2E speedups. |
| `results/systems/` | TP8 hardware/software capture, NCCL and prepared-collective measurements | Systems controls from this host. Only the recorded phases are complete. |
| `results/systems_torch214_superseded/` | Earlier systems capture | Retained for provenance; superseded as indicated by its directory and manifest. Do not mix toolchains without an explicit caveat. |

The first three directories include exact commands, environment versions and
metric definitions in their JSON and summary files. Profiles are separate
unmeasured runs; summed rank-zero kernel durations are neither wall time nor an
isolated network measurement. Logs retain backend availability, JIT and process
cleanup warnings. Compressed traces are included so later attribution can be
recomputed instead of relying only on categorized CSV files.

## What changed after the historical E2E runs

The old prefill launcher used `BLOCK_M=16`. With four local Q heads in Qwen3-8B
TP8, a CTA covered only four query tokens and scanned KV in tiles of 32. Trace
analysis at prefill 32640 separated decode CUDA Graph kernels from non-graph
prefill/mixed kernels:

| Rank-zero summed attention kernels | Dense FA2 | Historical C1 |
| --- | ---: | ---: |
| Prefill/mixed | 10.4936 s | 15.2056 s |
| Pure decode, including combine/reduction | 1.7753 s | 1.3778 s |

This diagnosis does not invalidate the old E2E run. It shows that the measured
C1 path had a prefill implementation bottleneck, while decode attention already
benefited from compact V.

The current implementation separates the two paths:

- prefill/mixed batches use `kernel_diffkv_prefill_sm89` with QK128/V64,
  `BLOCK_M=64`, KV tile 64, four warps and two stages;
- query-length-one decode keeps vLLM's split-KV DiffKV path and segment
  reduction;
- the prepared AllGather and single decoder GEMM are unchanged.

On the serial GPU-0 isolated benchmark, the selected prefill configuration took
2.429 ms versus 5.557 ms for the old unified kernel for Q=8192 with a 24448-token
prefix. Dense-width FA2 took 2.571 ms in that harness. For one long prefill plus
31 decode rows, new/old timings were 2.847/5.933 ms. These are preliminary
single-layer resident-input measurements; FA2 pads V64 to V128 and is not a
native DiffKV kernel. They cannot be substituted into the old E2E table.

Correctness coverage includes independent FP32 reference checks, local Q-head
counts four/eight, page sizes 16/32, ragged and empty query sequences, randomized
page mappings, noncontiguous Q/output, output canaries, and CUDA Graph replay
after changing inputs. Compute Sanitizer reported zero memory errors. A real TP8
smoke with 32640 prefill tokens passed for request batches one and three; all
workers reported `C1DiffKVImpl` and nonzero graph captures. This is integration
validation, not a quality evaluation or final performance result.

## Remaining work before a final claim

1. Rerun the full independent-kernel tile/warps/stages sweep under uncontended
   GPU clocks; the exhaustive sweep currently archived used the specialized
   launcher around the upstream kernel, while the independent kernel has the
   quick configuration checks.
2. Rerun dense and current C1 as a matched pair for the 8B batch and context
   sweeps. Reusing the historical dense numbers is useful for diagnosis but is
   not the strictest final comparison.
3. Repeat the matched Qwen3-32B sweep because its eight local Q heads change the
   prefill tile interpretation and its decoder/communication balance differs.
4. Report the historical and optimized results as separate implementation
   versions. Do not silently replace or merge their measurements.

The complete kernel description and reproduction commands are in
`docs/diffkv_sm89_prefill.md`.
