# TP1 capacity and K-offload experiment

Publication: [latest combined summary](README.md) and [immutable raw artifacts](https://huggingface.co/alexz949/BasisServe-CALS/tree/0d0b7a569eb3adcfbf5d63d78a8beb767980fcad/system_benchmarks/tp1_capacity).
The original grid below is preserved; the chunked-RoPE supplement is separate.

Status: formal grid completed on 2026-09-25 UTC, 42 trials: 36 successful and
6 GPU OOM. All successful configurations completed three repeats. The outcome
audit passed, including active-batch checks, nine exact-Dense token-parity pairs,
repeat token agreement, and the run-time source byte comparison. Unit test: one
passed. See `formal/SUMMARY.md`, `formal/summary.csv`, `formal/throughput.png`, and
`formal/audit.json`.

Important result: Basis 128K/B2 failed in prefill RoPE temporary allocation and
never reached decode. Its largest successful tested batch is therefore 1 at
128K, versus 2 for Dense-K-offload. At 64K/B4, both offload methods succeeded
while Dense-local failed; Basis achieved 68.18 tok/s versus 5.62 tok/s for
Dense-K-offload (three-repeat medians). Raw records are published at the HF link above.

## Scope

Llama-3.1-8B-Instruct BF16, TP1 on GPU 0 (one NVIDIA L40S), conda environment `basis`.
Contexts: 65,536 and 131,072 tokens. Batch sweep: 1, 2, 4, 8, 16, 32;
each method/context stops at its first GPU OOM. Three fresh-process repeats per
successful point, eight warmup steps and 128 measured decode forwards.

- Dense-local: complete exact K and V on GPU.
- Dense-K-offload: exact K in pinned, CUDA-mapped host memory; complete V on GPU;
  copy historical K into a shared GPU staging buffer for each layer/step.
- BasisKV-K-offload: exact K in pinned, CUDA-mapped host memory; complete V128
  and B16R16 routing codes on GPU; full-scan Page32 selection; 1,984 routed
  tokens plus 64 recent tokens; persistent GPU K slots reuse previous selections.

No two-stage routing, no MLP changes, no matched-placement traffic experiment.
The source snapshot includes legacy experimental implementations as dependencies;
the executed Basis configuration explicitly selects full scan.

## Admission and timing

Allocate independent KV rows for the whole batch, then prefill one request at a
time into its final row. This limits simultaneous prefill intermediates without
removing any admitted request's caches. All B requests then execute together in
every full-model decode forward; there is no request scheduling or batch reduction.
Use the first B calibration windows, not expanded views of a shared cache.

Measure autoregressive greedy decode with no EOS stopping. The 128 measured
forwards exclude the first token produced by prefill. Thus this is not the
128-output-token complete-request metric of the separate ShadowKV comparison.
Reset cache lengths and K-slot state after warmup. Decode wall timing includes
logits, argmax, a finite-logit reduction, and per-step CUDA synchronization.
Throughput is B * 128 divided by measured decode wall time; prefill is excluded.

Record the actual batch per step, generated tokens, per-step latency, allocation
and decode-resident GPU memory, host RSS/peak RSS, and explicit host-K bytes.
GPU values are PyTorch allocation statistics, not total device usage. Host peak
RSS includes model loading and file-backed pages, not just pinned K.
Keep OOM phase separate: model load, cache allocation, prefill, warmup, or decode.
Non-OOM failures stop the runner and are not interpreted as capacity limits.

## Validation

The 4K B1/B2 smoke completed all six method/batch points with cache/attention
validation enabled. Basis selection/packing was checked against the existing
full-scan reference; selected attention was checked against explicit attention.
Dense-local and Dense-K-offload generated identical tokens at both batch sizes.
The batch-row view unit test passed. No SHA256 validation was performed.

The formal run archives benchmark/runtime source files in `formal/source.tar.gz`.
It records configuration, device identity, Git revision and dirty status in
`formal/manifest.json`. At completion it compares source bytes with the archive,
without hashes, and records any changes in `formal/source_check.json`.

## Command

Slurm client discovery found no usable controller/configuration. The user
explicitly approved running the following command directly after smoke passed.
Per-trial stdout and stderr are saved as `run.log`; exact commands and progress
are stored beside each result.

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp1_capacity.py --contexts 65536 131072 --batches 1 2 4 8 16 32 --repeats 3 --warmup 8 --steps 128 --output results/system_benchmarks/tp1_capacity/formal
```

## Interpretation

The target is throughput in the range where Dense-local cannot fit but K-offload
can. Basis has additional GPU routing/slot storage, so it need not support a
larger batch than Dense-K-offload. All arms eventually face dense-V capacity limits.
Report largest successful tested batch, not an exact maximum. Sparse versus dense
is an algorithmic deployment comparison requiring separate quality evidence.
Do not attribute prefill or allocation OOM specifically to the decode kernel.
