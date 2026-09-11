# Linear-only V-conditioned K Base

Implemented `fit_bias=False` in the closed-form reduced-rank solver and `_fit_base_maps`. This uses uncentered input/cross moments, fits new factors, and stores an exactly zero output intercept. The default affine mode still centers moments and restores the fitted intercept. Rank applies to the weight in both modes. Runtime consumes the zero-intercept bank through its existing Base-before-RoPE path.

The local V96 calibration entry point accepts `--no-base-fit-bias`. Its protocol records the mode, saves zero `base_bias_b16`, and rebuilds the Page-Fisher statistics and residual factors using the new Base. The default affine bank directory is rejected for this mode. Existing artifacts are not overwritten.

## Checks completed

Environment `lowrank`, CPU, OMP/MKL two threads:

```bash
python -m pytest -q tests/test_linear_only_k_base.py tests/test_c1_v_conditional_k_router.py
```

12 tests passed. Log: `results/logs/linear_base/unit_tests.log`. New cases compare to independent uncentered least squares and prediction-space truncated SVD, test rank-zero and singular inputs, distinguish refitting from clearing an affine bias, and verify Base/RoPE/residual runtime arithmetic. These are numerical tests, not model-quality results.

## Proposed real-data smoke, awaiting launch confirmation

Local available artifacts are Qwen3-8B-Base, Two-sided V96, Base16 + residual R16. The earlier V80 calibration directories are not present under the inspected local paths. User's configuration question remains pending.

The first proposed run fits layer 0 using all original 64 fitting and 16 diagnostic C4 windows of length 32768, Q32, Base16 and Page-Fisher R16, 40 residual sweeps, PCG cap 100, unchanged damping/tolerance. No diagnostic selection of factors. GPU 6 (least utilized at preparation time), two CPU threads, direct execution (no Slurm). Existing layerwise calibration allocates approximately 20 GiB for hidden states plus 10 GiB current-layer V/K and model/temporary host storage; allow at least 50 GiB available host RAM.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
python -u -m evaluation.calibrate_v96kl_router \
  --model /home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --checkpoint results/hf/ICLR-results/qwen3-8b/checkpoints/Q3-8B-C1-R96 \
  --calibration results/calibration/v96kl_64x32k \
  --bank results/checkpoints/v96kl_linear_b16r16 \
  --no-base-fit-bias --stop-after-layer 0 \
  > results/logs/linear_base/layer0.log 2>&1
```

Compare the new layer's fit/held-out reconstruction and Fisher residual diagnostics to the affine layer in `results/checkpoints/v96kl_b16r16`, verifying equal window identity and query positions. A full-model routing benchmark requires fitting all 36 layers first; it must not use a mixture of the new Base and old residual factors. No GPU fitting or downstream routing scores have been produced for this ablation yet.
