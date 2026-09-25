# TP1 capacity and K-offload results

Llama-3.1-8B-Instruct BF16, TP1 on one L40S, environment `basis`.
Aggregate decode throughput below is the median of three independent runs,
each with eight warmup steps and 128 measured decode forwards. Prefill is
excluded. All requests remain active; this is not scheduler throughput or E2E.

| Context | Batch | Dense-local tok/s | Dense-K-offload tok/s | Basis-K-offload tok/s |
|---|---:|---:|---:|---:|
| 64K | 1 | 27.66 | 5.08 | 29.36 |
| 64K | 2 | 41.12 | 5.42 | 46.10 |
| 64K | 4 | Allocation OOM | 5.62 | 68.18 |
| 128K | 1 | 20.94 | 2.72 | 24.59 |
| 128K | 2 | Allocation OOM | 2.81 | **35.55 (chunked RoPE)** |

Only the final Basis point was retested with chunked prefill RoPE. Other values
are from the original grid. Do not describe this table as a complete, same-revision
paired rerun. The original 128K/B2 Basis prefill OOM remains in the historical
report and plot; the supplement demonstrates three successful runs after the
prefill-only engineering change. No routing algorithm or MLP change was made.
The revised implementation was not tested at larger batches.

- [Original grid, timing definitions and failures](formal/SUMMARY.md)
- [Original grid figure](formal/throughput.png) (includes the historical prefill OOM)
- [Chunked-RoPE implementation and validation](rope_chunked/SUMMARY.md)
- [128K/B2 three-repeat formal supplement](rope_chunked/formal/SUMMARY.md)
- [Protocol and commands](PROTOCOL.md)

## Raw artifacts and reproduction

[HF raw-data archive](https://huggingface.co/alexz949/BasisServe-CALS/tree/0d0b7a569eb3adcfbf5d63d78a8beb767980fcad/system_benchmarks/tp1_capacity)
contains raw JSON, all logs, smoke records, audits and per-run source archives.
Immutable revision: `0d0b7a569eb3adcfbf5d63d78a8beb767980fcad`.
No SHA256 validation was performed.

Each run's `source.tar.gz` is authoritative for the exact measured implementation;
use it in a separate directory when reproducing historical results. GitHub retains
the remote's more general stride-aware indexed-attention output implementation
instead of reverting it to the older contiguous-only implementation in the
measured snapshots. The published drivers use contiguous output buffers.
Model weights, calibration windows and routing factors remain external inputs
at the paths recorded in the benchmark sources. The HF snapshots do not duplicate
those inputs. Publication tests do not constitute new full-model benchmark runs.
