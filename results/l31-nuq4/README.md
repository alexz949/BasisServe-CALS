# Llama NUQ4 Calibration

Llama-3.1-8B-Instruct, existing C1 R64/R96 S6 factors, structure-only validation.
No SHA256. This is a separate calibration task; the first serving integration
target remains Qwen3-8B-Base.

## Status

Both smoke jobs completed successfully in the `basis` environment, on GPU 0
(R64) and GPU 1 (R96). Each fitted 64 K/V codebooks using one 128-token WT2 train
window, then calibrated 32 decoder input A8 scales with the same train window.
The two-window, 256-token forward checks produced finite smoke PPL values:
14.772776 (R64) and 11.307814 (R96). These short smoke values are **not formal
quality results**, and smoke codebooks must not be used for formal serving.

Calibration is not CPU-only: GPU forward/backward collects activations and
Fisher weights, CPU fits codebooks, then GPU forwards calibrate A8 scales.
The current implementation keeps the model on GPU during CPU fitting.
Formal calibration was approved and launched for 16 x 2048 WT2 train tokens
per rank. **Both formal jobs completed successfully**, including 16/16 GPU
Fisher windows, all 64 CPU-fitted codebooks, and 16/16 GPU scale-calibration
windows. Each rank has 64 finite codebooks and 32 positive decoder input
scales. The final CPU audit passed for both ranks, including source-byte
agreement and scale/observed-amax consistency. No SHA256 was used. There is
no formal PPL evaluation in this job, and these are not quality conclusions.

Formal serving artifacts (do not use the smoke artifacts):

| Rank | Codebooks | Decoder A8 scales | Train tokens | Audit |
|---:|---|---|---:|---|
| 64 | `formal/r64/quantizers.pt` | `formal/r64/a8_scales.json` | 32768 | passed |
| 96 | `formal/r96/quantizers.pt` | `formal/r96/a8_scales.json` | 32768 | passed |

Each codebook file is 8,698,641 bytes. Details: `formal/audit.json` and
`audit.log`. Both calibration workers exited with status 0; all eight GPUs
were idle after completion. The Llama serving backend and full PPL still
require separate integration/evaluation; only Qwen serving was smoke-tested.

K is per-channel before RoPE. V is dynamic per-token across all active KV
heads. Both use official Fisher-weighted NUQ4 with the 0.99 outlier rule,
no rotation and no first-token exclusion. Encoder/decoder remain BF16 during
scale collection under matching KV4. Only decoder input scales are calibrated
for the A8 boundary; there is no encoder FP8 conversion.

## Executed Commands

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/calibrate_llama_nuq4.py --phase smoke --rank 64 --output results/l31-nuq4 > results/l31-nuq4/smoke_r64.log 2>&1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/calibrate_llama_nuq4.py --phase smoke --rank 96 --output results/l31-nuq4 > results/l31-nuq4/smoke_r96.log 2>&1
```

Slurm remains unavailable (configuration/DNS failure); short smoke jobs ran
directly as previously agreed. Tokenization emits a warning about the length
of the whole WT2 token vector, but only the recorded short windows are fed
to the model. There were no smoke OOMs or worker failures.

Artifacts: `smoke/r{64,96}/manifest.json`, `quantizers.pt`, `a8_scales.json`,
`smoke_ppl.json`, `complete.json` and source snapshots; top-level worker logs.
Nothing has been committed or uploaded.

## Formal Commands

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/calibrate_llama_nuq4.py --phase formal --rank 64 --output results/l31-nuq4 > results/l31-nuq4/formal_r64.log 2>&1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/calibrate_llama_nuq4.py --phase formal --rank 96 --output results/l31-nuq4 > results/l31-nuq4/formal_r96.log 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/audit_llama_nuq4.py --root results/l31-nuq4 > results/l31-nuq4/audit.log 2>&1
```
