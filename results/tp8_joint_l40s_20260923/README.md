# L40S TP8 Joint-ALS + key routing results

Completed on 2026-09-23. This directory archives the completed vLLM measurements
from `results/tp8_vllm_joint`, preserving the original result filenames and bytes.

## Configuration

- Model: Llama-3.1-8B-Instruct; 8 NVIDIA L40S GPUs; tensor parallel size 8.
- Conda environment: `lowrank`; vLLM 0.30.0; PyTorch 2.13.0+cu130;
  Triton 3.7.1; CUDA compiler 13.0.88; CUDA compatibility library 13.4
  with NVIDIA kernel driver 535.183.06.
- BF16; Page32; synchronous scheduling; prefix caching disabled;
  80% GPU memory budget per rank; `FULL_DECODE_ONLY` CUDA Graph.
- Dense: native full attention. ALS-full: V96 ALS6 + C1 latent AllGather.
  Joint: V96 + ALS40 PCG100 B16R16 key routing + C1 latent AllGather.
- Routing: up to 512 candidate pages, 62 historical pages and 64 recent tokens,
  2048 persistent GPU K slots per request; exact K mirror in mapped host memory.

## Results

All values are medians over three fixed prompt cohorts, following a separate
warmup cohort. Batch size is independent of TP size.

### Steady decode, batch size 8

| Prompt tokens | Dense ms/step | ALS-full ms/step | Joint ms/step | Dense / Joint | ALS-full / Joint |
|---|---:|---:|---:|---:|---:|
| 65536 | 19.219 | 18.592 | 16.600 | 1.158 | 1.120 |
| 130048 | 30.748 | 29.046 | 17.480 | 1.759 | 1.662 |

### Complete requests, batch size 8, 128 generated tokens

| Prompt tokens | Dense seconds | ALS-full seconds | Joint seconds |
|---|---:|---:|---:|
| 65536 | 49.778 | 46.226 | 47.714 |
| 130048 | 109.644 | 113.073 | 114.740 |

Key routing improves long-context steady decode over ALS-full + C1 in this
configuration. Complete-request time remains slower than ALS-full at both
lengths. The shorter-context pilot does not show an incremental routing
speedup; see [pilot results](pilot/SUMMARY.md). These are speed measurements,
not a downstream model-quality evaluation or kernel-only timing.

## Measurement and limitations

- Experiment A measures complete-request wall time, including prefill, for
  128 generated tokens.
- Experiment B waits for full-batch decode, conditions for 16 steps, and measures
  128 intervals using CUDA events. It includes sampling and scheduler gaps and
  reports the maximum interval across ranks. Each window requires 128 actual
  full-graph replays. Starting context lengths are recorded in each JSON.
- Three arms use the same prompt cohorts. All completed runs have zero
  preemptions. Validation kernels are disabled during performance measurement.
- Exact K storage uses host memory as well as GPU slots; this is not an entirely
  GPU-resident cache. Chunked prefill host transfers are included in wall time.
- Earlier smoke checks passed cache checks, but greedy eager/graph output tokens
  were not always identical at near-tied decisions. These performance results
  do not establish model quality.
- Logs contain shutdown warnings about forced process cleanup and leaked
  semaphore/shared-memory resources after measurements were saved. All six
  result JSON files report `complete`.

## Commands

Original launch commands:

```bash
bash results/tp8_vllm_joint/run.sh
bash results/tp8_vllm_joint/run_long.sh
```

Each arm used the following benchmark invocation in `lowrank`, with local
checkpoint/prompt paths provided by the original `inputs.sh`:

```bash
conda run --no-capture-output -n lowrank \
  python benchmarks/system/bench_vllm_tp8_joint.py \
  --arm "$arm" --model "$MODEL" --value-bank "$VALUE_BANK" \
  --router-bank "$ROUTER_BANK" --prompt-bank "$PROMPT_BANK" \
  --prefill-tokens 65536 130048 --batch-sizes 8 --decode-tokens 128 \
  --chunk-tokens 8192 --gpu-memory-utilization 0.8 --experiments A B \
  --output "results/tp8_vllm_joint/long/$arm.json"
```

`arm` is `dense`, `als_full`, or `basis_joint`. The pilot instead uses
`--prefill-tokens 4096 16384 --batch-sizes 1 8` and writes under `pilot/`.
Raw JSON includes the actual command, configuration, software versions,
per-request metrics, and per-rank measurement evidence. This directory is a
results archive; the local implementation changes are outside this upload.

## File inventory

Root: `README.md` and `.gitignore` (allows the archived JSON and logs).
Each of `pilot/` and `long/` contains exactly:

- `dense.json`, `als_full.json`, `basis_joint.json`
- `dense.log`, `als_full.log`, `basis_joint.log`
- `SUMMARY.md`, `summary.csv`

See [long-context results](long/SUMMARY.md) and [CSV](long/summary.csv).
