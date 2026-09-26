# V96 Objective Ablation: Incomplete

Status: real-model smoke FAILED; formal fitting and PPL were NOT run.
Model: Qwen3-8B-Base. Shared V96 encoders, TP8 mathematical source partition.

The proposed comparison retains identical shared encoder/decoder structure,
calibration, two initializations and optimization budget. Independent fitting
zeros only cross-source covariance blocks. Joint fitting retains the full
covariance. Neither iterative fit is a certified global optimum.

Three synthetic tests passed in the `basis` environment. The real smoke failed
at factor serialization because `decoder_fp64` was non-contiguous. This backup
preserves that exact unfinished implementation; it is not a validated runtime.
No smoke PPL or formal quality result exists. Before resuming, make the exported
tensors contiguous, rerun tests and smoke, and obtain formal-run approval.
Preserve the failed log and partial artifacts rather than silently overwriting.

Smoke command (working directory `/workspace/BasisServe-CALS`):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/run_v96_objective.py --phase smoke --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --covariance results/tp-source-svd/formal/covariance --output results/v96-objective/smoke > results/v96-objective/smoke.log 2>&1
```

Quality execution materializes structured FP64 E@D once as BF16 o_proj. It
isolates the fitting objective, not compressed-cache runtime rounding/speed.
The failed smoke log and partial configuration are included in the server
backup's other-results archive on HF after upload completion.
