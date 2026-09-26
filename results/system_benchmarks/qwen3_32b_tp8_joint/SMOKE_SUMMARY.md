# Qwen3-32B TP8 Joint V96 Full-Scan Smoke

Status: integration smoke passed; no formal long-context speedup claim yet.

## Published Artifacts

This document preserves the initial 4K smoke. Subsequent 64K/128K Joint pilots
and the corrected Flash SDPA Dense comparisons are in the
[current comparison table](../tp8_dense_flash_rerun/SUMMARY.md);
[peak memory](../tp8_dense_flash_rerun/MEMORY_SUMMARY.md) is reported separately.
The older pilot summaries retain their explicitly labeled custom-Dense baseline.

Raw pilot results, logs and factor audit are in the
[HF archive](https://huggingface.co/alexz949/BasisServe-CALS/resolve/61958dd0106b32802a628c1596c54d95e6283a85/results/system_benchmarks/qwen3_32b_tp8_joint/raw.tar.gz),
with a [438-file inventory](https://huggingface.co/alexz949/BasisServe-CALS/blob/61958dd0106b32802a628c1596c54d95e6283a85/results/system_benchmarks/qwen3_32b_tp8_joint/raw_manifest.json).
Archive members retain their repository-relative paths. Unrelated kernel-tuning
and component-profile experiments are not included in this upload.

## Configuration

- Eight L40S GPUs, TP8/B1, BF16; `basis` conda environment.
- 4096 input tokens, 8 conditioning forwards, 16 measured decode forwards.
- One trial per arm; same Qwen-tokenized calibration prompt for smoke only.
- Fixed-batch Transformers/custom execution, not vLLM or continuous batching.
- Qwen Q/K normalization is retained; static YaRN factor 4, original limit 40960.
- Full-scan B16R16 routing, page size 32, 62 selected pages including sink,
  plus 64 recent tokens, nominal support 2048 per KV head. No two-stage shortlist.
- GQA8: each TP rank owns one KV head and eight query heads.
- Dense and ALS-full keep K/V on GPU. Basis keeps historical exact K in pinned
  host memory, selected K in persistent GPU slots, and V96 on GPU.
- No new MLP optimization. The existing matched benchmark's tokenwise chunking
  is used for all three arms.

## Smoke Measurements

| Arm | Decode ms/step | Max-rank prefill wall s | Max-rank prefill peak allocated GiB |
| --- | ---: | ---: | ---: |
| Dense | 69.076 | 1.398 | 9.238 |
| ALS-full V96 | 47.510 | 1.509 | 13.803 |
| Basis Joint V96 | 57.359 | 1.465 | 13.736 |

Decode timing is the mean of per-step maximum CUDA-event time across ranks,
including the driver's token selection and checks. Basis/Dense speedup is
approximately 1.204x at this smoke point; Basis is slower than ALS-full here.
These are not three-repeat formal measurements, request E2E latencies, or
evidence for 64K/128K scaling. Prefill peak includes temporary allocations and
the replicated compressed output decoder, not only KV memory.

All 24 rank result files completed, with finite logits and agreement on generated
token IDs across ranks within each arm. Equality across different arms is not
expected and was not asserted. No OOM occurred. GPUs were released after smoke.

## Factors and Validation

HF repository: `alexz949/BasisServe-CALS`, pinned revision
`3cd67a4d96070a1cb8ce6c15512359b17638cd64`.

- Value factors: `checkpoints/qwen3-32b-128k/uniform-v96-als6-cg16`.
- Matching routing bank: `checkpoints/qwen3-32b-128k/uniform96-b16r16-als40-pcg100`.
- These are the paired 128K-calibrated banks, not the R96-S6 factors from the
  earlier vLLM V-only batch sweep.
- All 64 layers passed structure, finite-value, and FP64 coordinate-map probes.
- BF16-exported factor probes, evaluated in FP32, had maximum layer relative
  RMSE 0.0024563 versus the original factor maps. This is not downstream quality
  validation or a guarantee of end-to-end numerical equivalence.
- Full-scan routing, selection, persistent slots, and sparse attention tests:
  54 passed, covering GQA4/GQA8 and contiguous/feature-major attention outputs.
- No SHA256 calculation or validation was performed by this preparation/run.
- CPU affinity follows GPU NUMA locality. Container `set_mempolicy` returned
  Operation not permitted; host allocation NUMA placement is NOT verified.

## Reproduction

Working directory: `/workspace/BasisServe-CALS-opt`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_qwen3_32b_joint_smoke.py --output results/system_benchmarks/qwen3_32b_tp8_joint/smoke
```

The runner records each exact arm command in `smoke/trials.json` and writes
`smoke/{dense,als_full,basis_joint}.log`. Per-rank JSON/logs are under
`smoke/smoke_<arm>_p4096_b1_r0/`.

Other evidence: `prepare.log`, `bf16_factor_audit.json`, `kernel_tests.log`,
`all_kernel_tests.log`, and `smoke_launcher.log` in this directory.

Next: matched long-context runs with these frozen factors and the same RoPE,
support, placement, and execution stack for all arms. Formal jobs and publication
have not been launched or performed as part of this smoke.
