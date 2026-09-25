# TP1 Basis-K-offload 128K/B2: formal chunked-RoPE results

Raw JSON, logs, audit and source snapshot: [immutable HF archive](https://huggingface.co/alexz949/BasisServe-CALS/tree/0d0b7a569eb3adcfbf5d63d78a8beb767980fcad/system_benchmarks/tp1_capacity/rope_chunked/formal).

Completed on 2026-09-25 UTC. Environment: `basis`, one NVIDIA L40S (GPU 0),
Llama-3.1-8B-Instruct BF16. Three independent processes completed successfully;
zero OOM and zero other failures. This is a single-configuration supplement,
not a rerun of the whole capacity grid or of the Dense arms.

## Protocol

- Context: 131,072 tokens per request; actual active batch: 2 throughout decode.
- Eight warmup steps, then 128 measured decode forwards per repeat. Reset cache
  lengths and persistent K-slot state after warmup. No EOS stopping.
- Historical exact K in pinned mapped host memory; complete V128 on GPU.
- Full-scan B16R16 Page32 routing, 1,984 routed tokens plus 64 recent tokens,
  persistent selected-K GPU slots. No two-stage routing or MLP changes.
- Prefill RoPE uses the existing operator in 2,048-token chunks, writing back
  into Q/K projection buffers. Decode RoPE is unchanged.
- Prefill admission is sequential; decode processes both requests together.
  Timing includes logits, argmax, finite checks and per-step synchronization,
  excludes prefill, and excludes the first token produced by prefill.
- Component profiling and expensive numerical validation are disabled in these
  formal runs. The preceding test and smoke validation are in `../SUMMARY.md`.

## Results

| Repeat | Mean decode ms/step | Aggregate decode tok/s | Prefill peak GiB | Decode-resident GiB |
|---:|---:|---:|---:|---:|
| 0 | 55.938 | 35.752 | 42.336 | 35.741 |
| 1 | 56.505 | 35.392 | 42.336 | 35.741 |
| 2 | 56.255 | 35.550 | 42.336 | 35.741 |
| Median | **56.255** | **35.550** | **42.336** | **35.741** |

Measured decode peak allocation was 35.743 GiB in every repeat. Memory values
are PyTorch allocated memory, not total device usage. The three throughput
values span 35.392-35.752 tok/s. This establishes repeatable successful execution
of 128K/B2 with this implementation, not the exact maximum batch size.

The earlier implementation's prefill OOM is retained in
`../../formal/basis_k_offload_t131072_b2_r0/`. Do not merge this supplement into
the old grid without marking the prefill implementation revision. Dense was
not rerun here, so no same-revision paired speedup is claimed in this report.

## Verification

`audit.json` passed: 3/3 successful repeats, 128 steps each at batch 2,
768 measured decode tokens in total, all logits finite, and metric arithmetic
consistent with the stored timings. All three generated-token arrays are equal
and also equal the preceding validation smoke. Source bytes matched the
run-start `source.tar.gz`; no SHA256 checks were performed. GPU allocations were
released after completion.

`summary.csv` contains per-repeat measurements and smoke token parity.
Raw results, commands, phase markers and stdout/stderr are retained in each
`basis_k_offload_t131072_b2_r*/` directory. Run settings and device information
are in `manifest.json`.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp1_capacity.py --contexts 131072 --batches 2 --methods basis_k_offload --repeats 3 --warmup 8 --steps 128 --output results/system_benchmarks/tp1_capacity/rope_chunked/formal
```

The runner sets `CUDA_HOME=/usr/local/cuda`, `MAX_JOBS=2`, and
`TORCH_CUDA_ARCH_LIST=8.9`. Execution was direct on the approved machine, not
through an unavailable Slurm controller. Raw records are published at the HF
link above; readable summaries and code are published on GitHub.
