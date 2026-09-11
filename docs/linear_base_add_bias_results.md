# Fixed linear-only K Base weights plus fitted bias

Adding a newly fitted intercept to the frozen linear-only weights reduced layer-0 held-out Base reconstruction relative MSE from 24.0394% to 11.2903%, a 53.03% reduction. The original affine fit remains much better at 0.7190%. A post-fit intercept corrects the mean residual but does not recover the weight/subspace solution obtained by fitting with an intercept.

## Method and metrics

Qwen3-8B-Base, Two-sided average V96 checkpoint; layer 0 actually uses V rank 112 and Base rank 16. Freeze both saved linear-only factors L and R, and fit only `b = mean(K_pre_rope) - mean(C_V) @ L @ R` on the original 64 training windows. Evaluate on the original 16 diagnostic windows; no diagnostic data are used to fit bias. Apply bias before RoPE.

The table uses the longest sampled prefix, ending at position 32639 (32640 tokens). Relative MSE is total squared post-RoPE K prediction error divided by exact K squared energy, before residual correction.

| Base configuration | Training relative MSE | Diagnostic relative MSE |
|---|---:|---:|
| Original affine fit | 0.0070221184 | 0.0071903949 |
| New linear-only weights, zero bias | 0.2319548416 | 0.2403944323 |
| Same new linear-only weights, fitted bias | 0.1087304402 | 0.1129025055 |

Mean relative MSE over the same 32 overlapping sampled prefixes:

| Base configuration | Training | Diagnostic |
|---|---:|---:|
| Original affine fit | 0.0069991141 | 0.0073032237 |
| Linear-only | 0.2313984729 | 0.2471798035 |
| Linear-only plus bias | 0.1083805778 | 0.1158271006 |

Both original arms' reconstruction metrics were reproduced, including exact K energy. Source bank hashes remained unchanged. Saved left/right tensors are exactly equal to the linear-only source tensors, and all saved factors are finite. Only Base factors are saved; residual factors were not refitted. This is a layer-0 reconstruction ablation, not a routing or downstream benchmark, and the changed Base should not be paired with the old residual factors without refitting.

## Execution

Environment: `lowrank`. Direct execution on GPU 6, two OMP/MKL threads, exit code 0. Script-reported elapsed time: 29.11 seconds. The startup `torch_dtype` deprecation warning did not affect completion.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
python -u -m evaluation.check_linear_base_bias \
  --linear-bank results/checkpoints/v96kl_linear_b16r16 \
  --output results/evaluation/linear_base_bias \
  > results/logs/linear_base/add_bias.log 2>&1
```

- Script: `evaluation/check_linear_base_bias.py`.
- Log: `results/logs/linear_base/add_bias.log`.
- Audited metrics and provenance: `results/evaluation/linear_base_bias/result.json`.
- Frozen weights plus new bias, Base only: `results/evaluation/linear_base_bias/base_only.safetensors`.
