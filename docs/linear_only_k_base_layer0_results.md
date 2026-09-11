# Linear-only versus affine K Base: layer 0

The authorized layer-0 run completed and exited with code 0. This uses the Qwen3-8B-Base Two-sided V96 payload, Base16 and newly refitted Page-Fisher R16. V96 denotes the model-wide average; this layer's resident V rank is 112, giving Base left/right shapes `[8,112,16]` and `[8,16,128]`.

## Base reconstruction

Both arms use identical 64 fitting and 16 diagnostic C4 windows, sequence length 32768, and identical 32 query positions. The table uses the longest sampled prefix, ending at query position 32639. Error is `sum((exact_post_rope_K - predicted_post_rope_Base)^2) / sum(exact_post_rope_K^2)`, before residual correction. It is neither the bias magnitude nor a full-model average.

| Base | Fit relative MSE | Diagnostic relative MSE |
|---|---:|---:|
| Affine, fitted bias | 0.0070221184 (0.7022%) | 0.0071903949 (0.7190%) |
| Linear-only, refitted weights and zero bias | 0.2319548416 (23.1955%) | 0.2403944323 (24.0394%) |
| Linear / affine | 33.0320x | 33.4327x |

The mean over all 32 sampled-prefix relative errors is 0.0069991141 / 0.0073032237 for affine fit/diagnostic, versus 0.2313984729 / 0.2471798035 for linear-only. These prefixes overlap; their mean is a descriptive query-prefix summary, not an independent sample mean.

For this layer, the fitted intercept is important to Base prediction. Removing it and correctly refitting the rank-constrained weight substantially worsens reconstruction. This does not establish the full-model routing or downstream score gap. The original bias L2 norms per KV head range from approximately 93.26 to 185.44; these are raw activation units and should not be interpreted as percentages or errors. The new stored bias is exactly zero in all 1024 entries.

## Residual fit and limitations

Both residual banks were fitted for 40 sweeps with PCG cap 100 and tolerance 1e-5. Linear-only's logged objective fell from 6868.39 to approximately 1801.89, with small nonmonotonic fluctuations later in fitting. Its final query solve reached 100 iterations and maximum relative residual 0.01284235, so completion does not establish convergence. The affine reference also reached 100 iterations, with residual 0.00638456.

| Diagnostic | Affine | Linear-only |
|---|---:|---:|
| Fit residual Page-Fisher NMSE | 0.17598332 | 0.04439728 |
| Diagnostic residual Page-Fisher NMSE | 0.44801240 | 0.09576516 |
| Raw training loss after sweep-40 encoder update | 1087.77673 | 1801.88733 |

**The residual NMSE denominator is each Base's own residual-score energy.** The smaller linear-only NMSE therefore does not imply better absolute routing approximation. The raw loss row identifies a common optimization endpoint before the final query refit, not a final held-out routing metric. No page-recall or downstream benchmark has been run with the linear-only bank; only layer 0 has been fitted.

## Verification and execution

The comparison audit verified source tensor hashes, finite factors, strict zero linear-only bias, identical window/checkpoint hashes, query positions and exact-Key energy at every reported prefix. Protocol differences are confined to the Base objective/mode and the three source files changed to implement it. Existing affine outputs were preserved. Twelve numerical/runtime tests passed before launch.

Environment `lowrank`; direct execution on GPU 6, OMP/MKL two threads. Layer processing took approximately 1136.64 seconds; fitting/statistics after capture took 1036.02 seconds. The only startup warning was the deprecated `torch_dtype` argument.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
python -u -m evaluation.calibrate_v96kl_router \
  --bank results/checkpoints/v96kl_linear_b16r16 \
  --no-base-fit-bias --stop-after-layer 0 \
  > results/logs/linear_base/layer0.log 2>&1
```

Model, payload and calibration paths are recorded in `docs/linear_only_k_base_protocol.md` and the shared defaults in `evaluation/v96kl_common.py`.

- New factors/diagnostics: `results/checkpoints/v96kl_linear_b16r16/layer_000.safetensors` and `layer_000.json`.
- Affine reference: `results/checkpoints/v96kl_b16r16/layer_000.json`.
- Comparison audit: `results/logs/linear_base/layer0_comparison.json`.
- Run log: `results/logs/linear_base/layer0.log`.
- Unit-test log: `results/logs/linear_base/unit_tests.log`.
