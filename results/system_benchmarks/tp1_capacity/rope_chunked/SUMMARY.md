# Chunked prefill RoPE: TP1 capacity verification

Raw JSON, logs, tests and source snapshots: [immutable HF archive](https://huggingface.co/alexz949/BasisServe-CALS/tree/0d0b7a569eb3adcfbf5d63d78a8beb767980fcad/system_benchmarks/tp1_capacity/rope_chunked).

Environment: `basis`, one NVIDIA L40S (GPU 0), Llama-3.1-8B-Instruct BF16.
Status: implementation, validation smoke, and the 128K/B2 three-repeat formal
supplement completed on 2026-09-25 UTC. The full grid and Dense arms were not
rerun. Previous formal results and the original 128K/B2 prefill OOM remain
unchanged in `../formal/`.

## Formal supplement

All three independent 128K/B2 Basis runs passed with eight warmup steps and 128
measured decode forwards each. Median full-model decode: **56.255 ms/step**;
median aggregate throughput: **35.550 tokens/s**. Prefill peak GPU allocation:
42.336 GiB; decode-resident allocation: 35.741 GiB. No OOM. Tokens matched across
all repeats and the preceding smoke; runtime source byte comparison passed.
See [formal report](formal/SUMMARY.md) and [per-repeat CSV](formal/summary.csv).

## Change

Dense and Basis now share `benchmarks/system/chunked_prefill_rope.py`.
During prefill only, the existing Transformers RoPE operator runs on 2,048-token
chunks and writes its outputs back into independent Q/K projection buffers.
This bounds temporary allocations without introducing another full-context
rotated Q/K allocation. The helper is inference-only and preserves input strides.
Decode still calls the original RoPE operator directly.

No routing algorithm, full-scan selection, cache placement, MLP, precision,
attention support budget, checkpoint or allocator configuration was changed.

## Validation

- 14 tests passed (`tests.log`), including the existing independent batch-row test.
- BF16/FP32, CPU/CUDA, B1/B2 and chunk-boundary cases matched the original RoPE
  operator exactly, including integer-view bit-pattern comparisons.
- The 128K BF16 CUDA test also matched bit patterns exactly. Incremental peak
  tensor allocation for the isolated RoPE call fell from 3,221,225,472 bytes
  (3.00 GiB) to 71,303,168 bytes (68 MiB). These are operator-only extra
  allocations, not total model memory or latency measurements.
- All six 4K smoke points passed: three methods, B1/B2. All generated tokens
  matched the corresponding pre-change smoke (`smoke/token_parity.json`).
- Runtime source byte comparisons passed for both smoke runs. No SHA256 checks.

## 128K/B2 Basis result

The previously failing configuration completed both prompt admissions, validation,
and 128 measured decode forwards at actual active batch 2, with all logits finite.
Full-scan selection/packing and selected attention were validated during warmup.
The allocated cache length was 131,200, matching the original formal configuration.

| Metric | Single validation smoke |
|---|---:|
| Context tokens | 131,072 |
| Active batch | 2 |
| Warmup steps | 2 |
| Measured decode steps | 128 |
| Mean full-model decode ms/step | 56.246 |
| Aggregate decode tokens/s | 35.556 |
| Prefill peak GPU allocation, GiB | 42.337 |
| Decode-resident GPU allocation, GiB | 35.741 |
| Measured decode peak GPU allocation, GiB | 35.743 |

GPU allocation numbers are PyTorch statistics, not total device usage.
This establishes that the current implementation can execute 128K/B2 after
bounding RoPE temporaries; it does not establish an exact maximum batch.
The old OOM did not establish a fundamental decode-resident capacity limit.
This one validation run is separate from the formal supplement above and must
not be combined with the old grid as if all configurations were retested. The
formal protocol uses eight warmup steps instead of two.

## Commands

Tests:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q -s tests/test_chunked_prefill_rope.py tests/test_tp1_capacity.py
```

Six-point smoke:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp1_capacity.py --contexts 4096 --batches 1 2 --repeats 1 --warmup 2 --steps 4 --validate --output results/system_benchmarks/tp1_capacity/rope_chunked/smoke
```

128K/B2 verification:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp1_capacity.py --contexts 131072 --batches 2 --methods basis_k_offload --repeats 1 --warmup 2 --steps 128 --validate --output results/system_benchmarks/tp1_capacity/rope_chunked/long_smoke
```

The runner sets `CUDA_HOME=/usr/local/cuda`, `MAX_JOBS=2` and
`TORCH_CUDA_ARCH_LIST=8.9`. Each smoke directory contains its source archive,
manifest, source byte-check report, outcomes, and per-trial `run.log`,
`command.json`, `progress.json` and `result.json`. These raw records are published
at the HF link above; readable summaries and code are published on GitHub.
