# Qwen3 TP8 V-only 16K Batch Sweep

Status: the formal 18-trial batch sweep completed with 14 successes and four recorded cache-state OOMs. This is a separate experiment from the completed 36-trial context grid in the parent directory.

## Frozen Protocol

- Arms: BasisKV V64 and the complete STAR-KV V-adaptive export, labeled STAR-KV V-only. Both use dense exact K, full attention, BF16, TP8 / DP1 / PP1, and no CPU KV offload.
- Prefill: 16384 tokens, 256-token chunks for every arm and batch. Actual generation: 128 greedy output tokens without early EOS stopping. The first token is chosen from prefill logits; the other 127 use decode forwards. Cache capacity reserves 128 decode slots.
- Batches: 1, 2, 4, 8, 16, 32, 64, 128, 256. One preselected Qwen-tokenized LongBench-v2 cohort supplies eight source rows. Batches above eight repeat those rows in order; this is a memory-capacity test, not a request-diversity or quality test.
- Total prompt-plus-output length is 16512 tokens, within the Qwen3-8B-Base native 32768-token context.
- Success means all eight TP ranks complete prefill and generate all 128 output tokens. OOMs remain explicit failures, with their phase and rank logs retained. The main resident metric is the maximum rank's decode-ready NVML process memory; decode-end and PyTorch peaks are separate.
- The STAR export's actual global Value-rank retention is 54.0473%, not strict 50%. Its shared V latent remains replicated per TP rank. Basis V64 remains source-local.

The fixed grid is `benchmarks/system/run_qwen3_8b_tp8_v_only_batch_sweep.py`. Its summarizer is `benchmarks/system/summarize_qwen3_8b_tp8_v_only_batch_sweep.py`.

## Formal Result

| Batch | BasisKV V64 decode-ready NVML (GiB/GPU) | STAR-KV V-only decode-ready NVML (GiB/GPU) |
| ---: | ---: | ---: |
| 1 | 3.980 | 4.482 |
| 8 | 5.652 | 9.322 |
| 32 | 11.150 | 25.867 |
| 64 | 18.463 | OOM |
| 128 | 33.160 | OOM |
| 256 | OOM | OOM |

The largest successful tested batch is 128 for BasisKV V64 and 32 for STAR-KV V-only. STAR-KV B64/B128/B256 reported cache-state allocation OOM on all eight ranks. BasisKV B256 reported the same OOM on seven ranks before torchrun terminated the eighth. These failures occurred before prefill; no decode-ready memory is imputed. The STAR-KV B64 failure requested another 1.84 GiB on a GPU already using about 43.41 GiB. Successful B128 BasisKV used 33.160 GiB/GPU at decode-ready (maximum rank). This is a single-cohort capacity comparison, not a statistical throughput or quality result.

`grid_trials.json` records all commands and attempts. `summary.csv`, `failures.csv`, `batch_frontier.csv`, and `SUMMARY.md` hold validated measurements and provenance. The figure is `plots/batch_sweep_16k_memory.pdf` (also PNG). Launcher and per-rank logs are under `raw/`.

Exact commands from `/workspace/BasisServe-CALS-opt`, using conda environment `basis`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_qwen3_8b_tp8_v_only_batch_sweep.py \
  --output-root results/system_benchmarks/tp8_external_baselines/starkv_v_only/batch_sweep_16k \
  > results/system_benchmarks/tp8_external_baselines/starkv_v_only/batch_sweep_16k/formal_grid.log 2>&1

/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python -m benchmarks.system.summarize_qwen3_8b_tp8_v_only_batch_sweep \
  --output-root results/system_benchmarks/tp8_external_baselines/starkv_v_only/batch_sweep_16k \
  > results/system_benchmarks/tp8_external_baselines/starkv_v_only/batch_sweep_16k/summary_generation.log 2>&1
```

## Preflight

Four TP8 smokes passed: Basis and STAR at 4K/B1 and 4K/B16, all with the new 256-token chunk and 128 actual output tokens. Each has eight rank JSON records, identical generated token IDs across ranks, and no failure. B16 records `prompt_repeated=true`; the repeated input rows generated matching outputs. STAR's B1 first two tokens `[323, 279]` match the preceding 4K formal context-grid record. These are implementation checks, not formal 16K measurements. Launcher and per-rank logs are under `smoke/`.

The CPU checks passed 14/14 in `cpu_tests.log`, covering the batch grid, direct CLI entrypoint, prompt repetition, cache-byte accounting, eight-rank 128-token validation, partial summary, and OOM plotting. The source snapshot is in `freeze/`.
