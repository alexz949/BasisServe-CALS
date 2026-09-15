# L40S systems benchmark results

Generated UTC: 2026-09-15T12:35:56.935740+00:00

**Status: completed with the capacity and measurement limitations below.**

Model: Llama-3.1-8B **base**, BF16, TP4 for model/collective tests. Existing two-sided KL allocation averages V96; actual layer ranks are retained. B16R16, Page32, hard 2048 including sink32 and recent64. The earlier Instruct/Dense-V capacity scan is a separate experiment.

Environment: `basis`. Rebuild tables with `python -m benchmarks.system.summarize`, then run `python -m benchmarks.system.write_summary` and `python -m benchmarks.system.plot_figures`. Per-run JSON metadata and E2E trial records contain exact measurement commands, versions, source hashes and occupancy snapshots. [input_identity.json](input_identity.json) records model, factor and calibration-input content hashes at its stated observation time.

## 1. Hardware and topology

Four L40S GPUs on lovelace. Existing external GPU0 occupancy is allowed and recorded per run; these are not empty-machine measurements. See [hardware.txt](hardware.txt) and [hardware.json](hardware.json).

| GPU | PCI bus | NUMA | Local CPUs |
| --- | --- | --- | --- |
| 0 | 0000:28:00.0 | 0 | 0-31 |
| 1 | 0000:45:00.0 | 0 | 0-31 |
| 2 | 0000:a8:00.0 | 1 | 32-63 |
| 3 | 0000:c5:00.0 | 1 | 32-63 |

Pinned allocation status and physical NUMA placement are checked separately with sampled host pages. Detailed reports: `numa_validation_rank*.json`; per-run offload audits verify the actual buffers.

## 2. NCCL and host-to-device bandwidth

NCCL uses default algorithm selection. Bus bandwidth below is the analytical collective convention, not a PCIe hardware counter. Full sweep: [nccl.csv](nccl.csv).

| Collective | Input B/rank | p50 ms | Algorithm GB/s | Analytical bus GB/s |
| --- | --- | --- | --- | --- |
| all_reduce | 256 | 0.030 | 0.009 | 0.013 |
| all_gather | 256 | 0.026 | 0.040 | 0.030 |
| all_reduce | 16384 | 0.027 | 0.615 | 0.923 |
| all_gather | 16384 | 0.026 | 2.560 | 1.920 |
| all_reduce | 1048576 | 0.096 | 10.894 | 16.340 |
| all_gather | 1048576 | 0.164 | 25.600 | 19.200 |
| all_reduce | 16777216 | 1.145 | 14.655 | 21.982 |
| all_gather | 16777216 | 2.332 | 28.782 | 21.586 |

Pinned H2D 1 GiB transfer per GPU; all six transfer sizes are in [h2d.csv](h2d.csv).

| GPU | NUMA | H2D GB/s | p50 ms |
| --- | --- | --- | --- |
| 0 | 0 | 27.046 | 39.700 |
| 1 | 0 | 27.046 | 39.700 |
| 2 | 1 | 27.046 | 39.700 |
| 3 | 1 | 27.046 | 39.700 |

## 3. TP output block

576/576 layer/batch/method records. Values below average the 32 layer p50 values; they are not a full-model latency. [Detailed CSV](tp_collective.csv).

LR-AR embeds the same C1 per-head coordinates in a zero-initialized global coordinate buffer and AllReduces it. LR-AG gathers those coordinates. Both use identical real C1 decoders. This is not an independently fitted global low-rank AR model.

| Batch | Method | Layers | Collective µs | Block µs |
| --- | --- | --- | --- | --- |
| 1 | C1 LR-AG | 32 | 13.216 | 44.566 |
| 1 | C1 global-coordinate AR | 32 | 25.636 | 72.166 |
| 1 | Dense AR | 32 | 25.514 | 37.013 |
| 8 | C1 LR-AG | 32 | 13.318 | 39.404 |
| 8 | C1 global-coordinate AR | 32 | 25.444 | 75.684 |
| 8 | Dense AR | 32 | 25.568 | 39.783 |
| 32 | C1 LR-AG | 32 | 35.840 | 66.773 |
| 32 | C1 global-coordinate AR | 32 | 53.948 | 87.059 |
| 32 | Dense AR | 32 | 58.410 | 68.480 |
| 64 | C1 LR-AG | 32 | 41.187 | 81.890 |
| 64 | C1 global-coordinate AR | 32 | 64.140 | 107.328 |
| 64 | Dense AR | 32 | 69.186 | 80.128 |
| 128 | C1 LR-AG | 32 | 51.172 | 76.211 |
| 128 | C1 global-coordinate AR | 32 | 82.676 | 109.751 |
| 128 | Dense AR | 32 | 95.878 | 103.399 |
| 256 | C1 LR-AG | 32 | 78.002 | 113.138 |
| 256 | C1 global-coordinate AR | 32 | 130.178 | 168.197 |
| 256 | Dense AR | 32 | 163.127 | 176.369 |

Figure A: [PNG](figure_A.png). Synthetic attention-output inputs, actual weights/factors; AR/AG coordinate and output equality is checked per case.

## 4. Router only

45 numeric records out of48 requested cases; three LRQK configurations run out of memory during native prefill. Representative64K, batch1 below. [All lengths and batches](router.csv). Route timing ends at selected IDs; query preprocessing is separately measured and included in joint total.

| Router | Query µs | Route µs | Joint µs | State bytes | Min/query | Max/query |
| --- | --- | --- | --- | --- | --- | --- |
| BasisKV B16R16 | 13.312 | 332.800 | 336.896 | 50538496 | 2048 | 2048 |
| loki | 13.312 | 52.224 | 54.272 | 33816576 | 2048 | 2048 |
| lrqk | 991.744 | 289.792 | 1364.992 | 218956288 | 2112 | 2112 |
| shadow | 0 | 66.560 | 66.560 | 17211904 | 2496 | 2496 |

State accounting includes the Basis Base16/Residual16 coordinate cache, compact RoPE tables and retained factor tensors. At64K/batch1 the coordinate cache alone is32 MiB; the RoPE tables add about16 MiB. The V tail is not scanned. LRQK includes active exact K needed for online query updates and allocated code capacity; its JSON also reports scan state separately. ShadowKV reconstruction factors are excluded from routing state and reported separately in raw JSON. Output workspaces and validation-only copies are excluded. CSV columns separate the added position/factor/fixed-ID accounting. Logical coordinate bytes are not total DRAM traffic.

| Router | Logical coordinate/landmark B | RoPE table B | Logical effective GB/s |
| --- | --- | --- | --- |
| BasisKV B16R16 | 33521664 | 16760832 | 151.089 |
| loki | 33554432 | 0 | 642.510 |
| lrqk | 134084608 | 0 | 462.693 |
| shadow | 16662528 | 0 | 250.338 |

Logical effective GB/s divides unique coordinate/landmark bytes plus position tables by route p50. It excludes small query/parameter reads, output writes and repeated/cache-served accesses; it is not measured DRAM bandwidth.

ShadowKV uses upstream CUTLASS landmark routing, rank160/chunk8. Native CPU-cache alignment gives routed2048 + outlier384 + local64 =2496 at aligned lengths. Loki uses PCA rank32 with independent top2048/query. LRQK uses official rank32/top2048 plus native lite64, after 64 continuous teacher-forced online updates.

LRQK native import permits TF32 for FP32 matrix multiplications, recorded as `torch_matmul_tf32=true` in formal metadata. BF16 stored routing codes and FP32 solve tensors do not imply that every internal multiplication is full IEEE FP32. The upstream numerical settings are preserved.

LRQK formal setup outcomes (a missing timing row must not be treated as zero latency):

| Context | Batch | Outcome | Log |
| --- | --- | --- | --- |
| 131072 | 1 | complete | — |
| 16384 | 1 | complete | — |
| 16384 | 4 | complete | — |
| 16384 | 8 | complete | — |
| 32768 | 1 | complete | — |
| 32768 | 4 | complete | — |
| 32768 | 8 | complete | — |
| 65536 | 1 | complete | — |
| 65536 | 4 | complete | — |
| 131072 | 8 | gpu_oom | results/system_benchmarks/l40s/router_lrqk_capacity_t131072_b8.log |
| 65536 | 8 | gpu_oom | results/system_benchmarks/l40s/router_lrqk_capacity_t65536_b8.log |
| 131072 | 4 | gpu_oom | results/system_benchmarks/l40s/router_lrqk_capacity_t131072_b4.log |

## 5. Exact-K offload

64K, batch1, real router pages, V resident. Staged rows are independently measured components. The mapped kernel fuses host reads, QK, softmax and PV; its total cannot be decomposed by summing the staged measurements.

| Budget | Unique K B | Staged fetch µs | QK µs | Softmax/PV µs | Staged total µs | Mapped fused µs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 2097152 | 138.304 | 15.360 | 29.696 | 187.392 | 95.232 |
| 2048 | 4194304 | 253.296 | 17.408 | 38.912 | 315.392 | 174.080 |
| 4096 | 8388608 | 545.920 | 25.600 | 62.464 | 566.272 | 330.752 |
| 512 | 1048576 | 86.784 | 14.336 | 23.552 | 135.168 | 54.272 |

H2D DMA payload and logical mapped-host reads are reported separately. Actual PCIe bus traffic is unavailable unless explicitly recorded by a profiler.

| Budget | Effective staged payload GB/s |
| --- | --- |
| 1024 | 15.163 |
| 2048 | 16.559 |
| 4096 | 15.366 |
| 512 | 12.083 |

Effective staged payload throughput includes the whole fetch interval, including host gather and dispatch; the separate H2D sanity test measures copy bandwidth.

## 6. Single-layer attention operator

Layer3, full32Q/8KV heads, batch1. Sparse totals include query preprocessing and routing. These are not TP4 full-model speeds.

| Context | Dense local µs | Dense offload µs | Sparse local µs | Sparse offload µs | Offload speedup |
| --- | --- | --- | --- | --- | --- |
| 131072 | 1142.784 | 9987.072 | 719.872 | 839.680 | 11.894 |
| 16384 | 132.096 | 1260.544 | 142.336 | 280.576 | 4.493 |
| 32768 | 299.008 | 2504.704 | 218.112 | 355.328 | 7.049 |
| 65536 | 579.584 | 5008.384 | 368.640 | 505.856 | 9.901 |

Figure B and full raw timing distributions accompany [sparse_operator.csv](sparse_operator.csv).

Staged breakdown below is a separate implementation. Its total uses pre-captured routing/attention around the host-dependent fetch and supersedes the earlier eager-routing staged total. These component medians are not the internal costs of the mapped fused kernel.

| Context | Query µs | Route µs | Fetch µs | QK µs | Softmax/PV µs | Measured staged total µs |
| --- | --- | --- | --- | --- | --- | --- |
| 131072 | 15.264 | 659.456 | 244.624 | 18.432 | 39.936 | 987.136 |
| 16384 | 13.312 | 105.472 | 240.544 | 17.408 | 38.912 | 400.384 |
| 32768 | 12.288 | 182.272 | 239.424 | 17.408 | 39.936 | 482.272 |
| 65536 | 15.264 | 332.800 | 244.368 | 18.432 | 39.936 | 636.928 |

## 7. Common exact-K backend

4/4 methods. Native routing + common exact-K fetch/attention; **ShadowKV here is not a full official serving-system reproduction**. Fetch deduplicates the GQA union but attention retains each query head's original support.

| Router | Query µs | Route µs | Fetch µs | Attention µs | Total µs | Unique K tokens | DMA payload B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| basis | 12.288 | 332.800 | 389.920 | 163.840 | 859.088 | 16384 | 4194304 |
| loki | 12.288 | 53.248 | 164.864 | 99.328 | 780.288 | 29214 | 7478784 |
| shadow | 0 | 66.560 | 166.912 | 118.816 | 678.912 | 19968 | 5111808 |
| lrqk | 925.696 | 271.872 | 684.960 | 168.960 | 2299.904 | 35065 | 8976640 |

## 8. TP4 full-model decode

48/48 terminal configurations currently available. Generate256 tokens; discard the first32 generated tokens, retaining224 decode forwards. TTFT includes prefill and first-token selection. Peak columns are the maximum per-rank PyTorch allocation; host K is summed over four ranks. CUDA-event steady latency and synchronized wall throughput have distinct timing boundaries.

| Context | Batch | Mode | Status | TTFT s | Decode ms | Tokens/s | Prefill GiB | Decode GiB | Host K GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 1 | c1 | complete | 1.115 | 25.715 | 38.886 | 6.376 | 5.949 | 0.000 |
| 16384 | 16 | c1 | complete | 18.428 | 28.342 | 564.538 | 20.467 | 13.675 | 0.000 |
| 16384 | 4 | c1 | complete | 4.241 | 26.530 | 150.765 | 9.182 | 7.478 | 0.000 |
| 16384 | 8 | c1 | complete | 9.094 | 26.927 | 297.088 | 12.974 | 9.574 | 0.000 |
| 32768 | 1 | c1 | complete | 2.205 | 28.969 | 34.518 | 7.327 | 6.501 | 0.000 |
| 32768 | 16 | c1 | complete | 39.690 | 41.864 | 382.192 | 33.858 | 20.807 | 0.000 |
| 32768 | 4 | c1 | complete | 8.966 | 30.358 | 131.782 | 12.641 | 9.366 | 0.000 |
| 32768 | 8 | c1 | complete | 19.333 | 32.468 | 246.391 | 19.740 | 13.206 | 0.000 |
| 65536 | 1 | c1 | complete | 4.898 | 45.946 | 21.764 | 9.265 | 7.641 | 0.000 |
| 65536 | 16 | c1 | gpu_oom | — | — | — | — | — | — |
| 65536 | 4 | c1 | complete | 20.599 | 47.945 | 83.426 | 19.548 | 13.132 | 0.000 |
| 65536 | 8 | c1 | complete | 44.377 | 51.573 | 155.116 | 33.274 | 20.473 | 0.000 |
| 16384 | 1 | dense | complete | 1.253 | 27.277 | 36.664 | 5.561 | 5.125 | 0.000 |
| 16384 | 16 | dense | complete | 20.882 | 28.913 | 553.393 | 19.673 | 12.755 | 0.000 |
| 16384 | 4 | dense | complete | 4.836 | 28.364 | 141.022 | 8.387 | 6.651 | 0.000 |
| 16384 | 8 | dense | complete | 10.309 | 28.987 | 276.225 | 12.211 | 8.749 | 0.000 |
| 32768 | 1 | dense | complete | 2.486 | 27.257 | 36.695 | 6.585 | 5.758 | 0.000 |
| 32768 | 16 | dense | complete | 46.048 | 37.445 | 427.292 | 33.941 | 20.888 | 0.000 |
| 32768 | 4 | dense | complete | 10.154 | 28.318 | 141.255 | 12.059 | 8.784 | 0.000 |
| 32768 | 8 | dense | complete | 21.741 | 28.763 | 278.159 | 19.416 | 12.881 | 0.000 |
| 65536 | 1 | dense | complete | 5.410 | 27.177 | 36.797 | 8.632 | 7.024 | 0.000 |
| 65536 | 16 | dense | gpu_oom | — | — | — | — | — | — |
| 65536 | 4 | dense | complete | 22.778 | 28.629 | 139.724 | 19.404 | 13.050 | 0.000 |
| 65536 | 8 | dense | complete | 50.389 | 34.930 | 229.027 | 33.824 | 21.147 | 0.000 |
| 16384 | 1 | offload | complete | 1.151 | 34.468 | 29.011 | 6.148 | 5.723 | 1.016 |
| 16384 | 16 | offload | complete | 18.660 | 49.170 | 325.404 | 16.932 | 10.130 | 16.250 |
| 16384 | 4 | offload | complete | 4.312 | 37.791 | 105.846 | 8.303 | 6.599 | 4.062 |
| 16384 | 8 | offload | complete | 9.206 | 37.716 | 212.110 | 11.178 | 7.774 | 8.125 |
| 32768 | 1 | offload | complete | 2.251 | 34.412 | 29.058 | 6.889 | 6.066 | 2.016 |
| 32768 | 16 | offload | complete | 39.936 | 59.520 | 268.818 | 26.820 | 13.758 | 32.250 |
| 32768 | 4 | offload | complete | 9.118 | 36.921 | 108.346 | 10.889 | 7.614 | 8.062 |
| 32768 | 8 | offload | complete | 19.552 | 37.913 | 211.007 | 16.191 | 9.653 | 16.125 |
| 65536 | 1 | offload | complete | 4.959 | 34.649 | 28.860 | 8.389 | 6.769 | 4.016 |
| 65536 | 16 | offload | gpu_oom | — | — | — | — | — | — |
| 65536 | 4 | offload | complete | 20.855 | 37.423 | 106.893 | 16.043 | 9.626 | 16.062 |
| 65536 | 8 | offload | complete | 44.815 | 46.837 | 170.804 | 26.225 | 13.420 | 32.125 |
| 16384 | 1 | sparse_local | complete | 1.127 | 34.120 | 29.307 | 6.408 | 5.984 | 0.000 |
| 16384 | 16 | sparse_local | complete | 18.488 | 36.270 | 441.137 | 20.995 | 14.192 | 0.000 |
| 16384 | 4 | sparse_local | complete | 4.266 | 35.656 | 112.185 | 9.314 | 7.610 | 0.000 |
| 16384 | 8 | sparse_local | complete | 9.128 | 35.706 | 224.057 | 13.241 | 9.837 | 0.000 |
| 32768 | 1 | sparse_local | complete | 2.228 | 33.971 | 29.435 | 7.392 | 6.569 | 0.000 |
| 32768 | 16 | sparse_local | complete | 40.289 | 39.898 | 401.029 | 34.883 | 21.821 | 0.000 |
| 32768 | 4 | sparse_local | complete | 9.036 | 35.783 | 111.784 | 12.898 | 9.624 | 0.000 |
| 32768 | 8 | sparse_local | complete | 19.419 | 35.681 | 224.217 | 20.253 | 13.715 | 0.000 |
| 65536 | 1 | sparse_local | complete | 5.003 | 34.089 | 29.334 | 9.393 | 7.772 | 0.000 |
| 65536 | 16 | sparse_local | gpu_oom | — | — | — | — | — | — |
| 65536 | 4 | sparse_local | complete | 20.682 | 36.157 | 110.631 | 20.056 | 13.639 | 0.000 |
| 65536 | 8 | sparse_local | complete | 45.607 | 38.021 | 210.449 | 34.288 | 21.482 | 0.000 |

Decode backend mapping: Dense uses `flash_attn_with_kvcache` with16 splits; C1 full attention uses the general CUDA GPU paged kernel with32 splits and split transformed V; sparse-local uses that GPU kernel on selected tokens; offload uses the mapped-host fused CUDA kernel on the same selected support. All modes use native FlashAttention prefill. Dense versus C1 therefore measures the implemented systems, not an isolated effect of V compression with a matched attention kernel.

See [e2e_tp4.csv](e2e_tp4.csv) and individual `e2e/*/trial.json` files for communication bytes, support counts, external occupancy and commands. OOM is a capacity outcome, not a numeric timing result.

## 9. Profiling and observed bottlenecks

The current Basis router is slower than Loki and ShadowKV in the measured 64K/batch1 route-only cases. Offload can benefit from fetching fewer keys even when routing itself is not the fastest. Communication-only gains do not guarantee a faster total output block; projection, decoder and launch costs remain included.

Nsight Systems records four steady TP4 decode steps at64K/batch1. Mapped K read/QK/softmax/PV is one fused range; separate internal durations would be misleading. Nsight Compute targets only the warmed Basis router. Counter data must distinguish L2/cache traffic from DRAM and logical bytes.


NCU availability: **unavailable_permission_denied**. The user does not have permission to access NVIDIA GPU Performance Counters on target device0; no kernels were profiled. DRAM bytes, measured bandwidth, occupancy, arithmetic intensity and stalls are unavailable, not zero. See [ncu_status.json](ncu_status.json).

Accepted trace: [e2e_trace_nsys2024.nsys-rep](e2e_trace_nsys2024.nsys-rep). Four GPUs each contain128 router calls and128 mapped attention calls across four decode steps. All four processes completed256 generated tokens. The earlier2023 trace was rejected after SIGSEGV and incomplete coverage; it is retained as failure evidence.

Profiler window averages 48.41 ms/step, versus 34.65 ms/step in the formal run. Use the formal run for performance claims. [Coverage review](profile_review.json).

| NVTX stage | Calls over4 GPUs/4 steps | GPU projected mean µs | CPU range mean µs |
| --- | --- | --- | --- |
| NCCL AllGather | 512 | 264.092 | 20.732 |
| mapped K read + exact QK + softmax/PV (fused) | 512 | 117.432 | 36.184 |
| route | 512 | 89.649 | 53.568 |
| decoder | 512 | 40.056 | 29.267 |
| top-k/page selection | 512 | 35.368 | 120.905 |
| low-rank encoder | 512 | 4.452 | 34.766 |
| query preprocessing | 512 | 4.060 | 10.801 |

The trace shows substantial collective/synchronization time and host launch overhead, especially in page selection. NCCL kernel duration can include waiting for another rank; this does not establish a PCIe bandwidth bottleneck. Ranges overlap and their means must not be added to reconstruct the model step.


## 10. Correctness and limits on interpretation

Basis transform uses T=[A16,Q-perp], E′=ET and D′=T⁻¹D; Base16 and the remaining coordinates have separate storage. Per-layer FP64/BF16 checks are in `basis_validation_rank*.json`. Real-capture router IDs are checked against references; poisoning the unused tail checks that routing does not read it.

The 4K batch1/4 full-model smoke compares Dense prefill against native HF, verifies C1 prefill equality across three cache modes, and verifies sparse-local/offload generated-token equality. See [e2e_smoke_validation.json](e2e_smoke_validation.json). This establishes small-case implementation consistency, not language-model quality at long contexts.

The common backend test checks independent query supports, duplicates, invalid IDs, changing requests, buffer reuse and exact-K attention against reference computation. Microbenchmark selected-K validation does not imply equality to dense full-support attention.

Run `python -m benchmarks.system.audit_results` to check raw timing distributions, sampled NUMA pages, recorded correctness flags, E2E timing arithmetic, source hashes and missing grid points. [record_audit.json](record_audit.json) documents the inspected files and coverage; it is not a completion certificate.

**LRQK timing qualification:** the native online update retains eager temporary allocations and convergence-related host synchronization. Its measured preprocessing/joint totals therefore do not satisfy the allocator-excluded microbenchmark protocol and must not be presented as isolated kernel latency. The route-only scan has a separate timing boundary. Native precision and update semantics have not been changed to make timing look better.

Nsight Compute hardware counters are unavailable due to permissions. Hardware PCIe counters are not inferred from tensor size. Existing external GPU activity can affect latency as well as capacity.

LRQK32K/batch8 initially failed an over-strict FP32-rounded top-k check. On one query head, a native BF16 score of−104.0 corresponded to FP32−103.7499847, which rounds to−103.5 and changes one boundary token. Native official IDs remained exact. Validation now checks native BF16 IDs exactly and bounds independent FP32 score error separately; smoke and the original device-index regression passed. The original failed report and diagnostic are retained. The formal configuration also passed an unchanged-gate rerun on GPU0.

All results are systems measurements, with synthetic inputs in the TP-block test and real captured activations in router/operator tests. They do not establish RULER accuracy or prove that a V-coordinate change has no long-context quality impact.

### LRQK allocator-excluded supplementary measurement

Native default solver dispatch could not be captured. This separate experiment prefers cuSOLVER, records the native scalar convergence decisions at a fixed final online state, and captures the unchanged tensor algebra. Each case requires bitwise equality of all update outputs against the native reference. CUDA graph replay excludes allocation and CPU branch synchronization; solver dispatch differs and this is not a general continuous-online implementation. The main tables retain native eager timings. Do not add supplementary update p50 to route p50 and call that a measured joint total.

| Context | Batch | Update p50 µs | Update p95 µs | Native outputs bitwise equal | Raw record |
| --- | --- | --- | --- | --- | --- |
| 16384 | 1 | 395.264 | 396.288 | True | lrqk_frozen_update_cusolver_t16384_b1.json |
| 16384 | 4 | 990.208 | 995.328 | True | lrqk_frozen_update_cusolver_t16384_b4.json |
| 16384 | 8 | 1760.256 | 1766.400 | True | lrqk_frozen_update_cusolver_t16384_b8.json |
| 32768 | 1 | 396.288 | 396.288 | True | lrqk_frozen_update_cusolver_t32768_b1.json |
| 32768 | 4 | 990.208 | 1043.456 | True | lrqk_frozen_update_cusolver_t32768_b4.json |
| 32768 | 8 | 1764.352 | 1871.872 | True | lrqk_frozen_update_cusolver_t32768_b8.json |
| 65536 | 1 | 396.288 | 397.312 | True | lrqk_frozen_update_cusolver.json |
| 65536 | 4 | 1038.336 | 1045.504 | True | lrqk_frozen_update_cusolver_t65536_b4.json |
| 131072 | 1 | 401.408 | 402.432 | True | lrqk_frozen_update_cusolver_t131072_b1.json |

Command: `python -m benchmarks.system.bench_lrqk_frozen_update --cusolver --length T --batch B` in `basis`;100 warmups and500 measured replays. The64K/batch1 run used the equivalent default length/batch arguments.

Joint fixed-state timings, measured directly rather than summed from stages. The same cuSOLVER and frozen-branch qualification applies. The common-backend measurement checks the native IDs and exact selected attention, with unchanged physical support.

| Context | Batch | Query µs | Route µs | Measured joint µs | Measured common backend µs |
| --- | --- | --- | --- | --- | --- |
| 16384 | 1 | 396.288 | 90.112 | 505.856 | — |
| 16384 | 4 | 990.208 | 304.128 | 1280.000 | — |
| 16384 | 8 | 1760.256 | 545.792 | 2291.712 | — |
| 32768 | 1 | 396.288 | 111.616 | 618.496 | — |
| 32768 | 4 | 991.232 | 523.264 | 1502.208 | — |
| 32768 | 8 | 1761.280 | 996.352 | 2744.320 | — |
| 65536 | 1 | 396.288 | 286.720 | 745.472 | 1857.488 |
| 65536 | 4 | 1040.384 | 970.752 | 1953.792 | — |
| 131072 | 1 | 401.408 | 507.904 | 972.800 | — |

Command: `python -m benchmarks.system.bench_lrqk_frozen_pipeline --length T --batch B`, with `--common` at64K/batch1; `basis`,100 warmups/500 samples. Raw records are `lrqk_frozen_pipeline_t*_b*.json`.
