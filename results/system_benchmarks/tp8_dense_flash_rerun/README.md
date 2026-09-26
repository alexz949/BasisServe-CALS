# TP8 Dense Flash SDPA rerun

Status: results and OOM outcomes are summarized in [SUMMARY.md](SUMMARY.md).
This directory contains new Dense measurements, not replacements for historical
raw files. Raw `queue.json` and per-trial logs are in the HF archive linked there.

## Scope

- Environment: `basis`, 8 L40S GPUs, TP8, BF16.
- Dense decode explicitly selects PyTorch `SDPBackend.FLASH_ATTENTION`.
- This is the custom full-model steady-decode harness, not vLLM or E2E timing.
- Llama-3.1-8B-Instruct: original 16 configurations, 3 cohorts, 48 Dense trials.
  Context 4096: batches 1/8/32/128. Contexts 16384/65536/130048:
  batches 1/4/8/16 each. Original saved prompts are reused.
- Qwen3-32B: contexts 65536/130048, batches 1/2/4/8/16, one trial per point.
  Stop increasing batch for that context after the first GPU OOM.
  This remains a capacity pilot, not a three-repeat formal grid.
- Both use 16 conditioning forwards and 128 measured decode forwards.
- Llama first runs a 4096/B1 smoke with 2 conditioning and 8 measured forwards.
- Basis and ALS are not rerun. No new/old generated-token equivalence checks.
- No SHA256 checks. Historical Dense measurements used a custom paged kernel;
  comparisons must explicitly identify the Dense backend.

## Command

Working directory: `/workspace/BasisServe-CALS-opt`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_tp8_dense_flash_rerun.py
```

The queue serializes jobs and records exact child commands in `queue.json`.
Each model grid has a launcher log, trial logs, and rank JSON results.
OOM points are failures, not latency measurements. Results were published
separately from the runner; see the fixed HF revision in the summary.
