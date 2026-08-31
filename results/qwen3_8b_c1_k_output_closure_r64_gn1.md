# Qwen3-8B C1-K Output-Closure Rank-64

## Result

This experiment fixes the existing uniform C1-V64 encoder/decoder and replaces
the post-RoPE KQ-SVD objective with full-layer causal attention-output closure:

\[
\left\|Y_{\text{Dense-K,C1-V64}}-
\sum_{g,h}\operatorname{softmax}
\left((Q_hF_h)(K_gE_g)^\top/\sqrt{128}\right)C_gD_h\right\|_F^2.
\]

The Key projector is shared by the four Query heads in each physical KV group;
the Query projector is head-specific. Both K and V use rank 64, retaining 50%
of the original KV-cache width.

| Arm | WikiText-2 PPL | Change vs. Dense-K | Change vs. KQ-SVD |
| --- | ---: | ---: | ---: |
| Dense-K + C1-V64 | 8.429266443 | reference | -30.80% |
| KQ-SVD K64 + C1-V64 | 12.180634841 | +44.50% | reference |
| C1-K K64 + C1-V64 | **10.222752867** | +21.28% | **-16.07%** |

C1-K recovers 52.19% of the excess PPL introduced by KQ-SVD K64, but it does
not yet close the remaining gap to Dense-K + C1-V64.

## Fresh-heldout closure metrics

The fit set contains C4 windows 0--127. Selection uses 64 disjoint, fresh C4
windows 336--399, rather than the validation documents used to select the
fixed C1-V64 factors.

| Boundary | Weighted output MSE | Weighted causal QK score error |
| --- | ---: | ---: |
| Initial KQ-SVD | 0.064783001 | 0.009394647 |
| After head-specific Q update | 0.049339199 | 0.013734009 |
| After shared-K update | **0.046119336** | 0.013782089 |

The output objective improves 28.81% while causal score error becomes 46.70%
worse. This is direct evidence that preserving the final decoded attention
output is materially different from preserving the QK score matrix.

All 36 layers selected the post-K boundary on fresh heldout data. All 36 Key
updates accepted step 1. Thirty-five Query updates accepted step 1; layer 17
used step 0.5. Every block used the full four-iteration CG budget without
negative curvature. None reached the requested `1e-4` relative residual within
four iterations, so additional CG iterations or another outer sweep remain
possible follow-up experiments.

## Protocol

- Model: Qwen3-8B-Base.
- Fixed value path: uniform C1-V64 ALS5.
- Initialization: rank-64 post-RoPE KQ-SVD.
- Calibration: 128 C4 fit documents and 64 fresh heldout documents, all 2048
  tokens.
- Solver: one alternating head-specific-Q/shared-K Gauss--Newton sweep;
  matrix-free softmax JVP/VJP; four CG iterations; zero damping; backtracking.
- Gauge fixing: per-KV-group QR on K with exact compensation in each Query
  projector.
- PPL: WikiText-2 test, 2048-token chunks, batch size 1, BF16 model, FP32 loss.
- Runtime: `basis` conda environment, one L40S on `yangGrp/lovelace`.
- Slurm job: `8284035`, completed in 01:20:51 with exit code 0.
- Calibration time: 4639.87 seconds; three-arm PPL time: 185.08 seconds.
- Peak allocated CUDA memory during calibration: 15.61 GiB.

## Commands

```bash
python -u evaluation/fit_qwen3_8b_c1_k_output_closure.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --windows results/calibration/qwen3_8b_c4_336prefix64fresh_s2048/windows.safetensors \
  --c1-factor-dir results/checkpoints/qwen3_8b_c1_v64_als5 \
  --kq-factor-dir results/checkpoints/qwen3_8b_post_rope_kqsvd_r64_c4_128f64h \
  --output-dir results/checkpoints/qwen3_8b_c1_k_output_closure_r64_gn1 \
  --fit-start 0 --fit-windows 128 \
  --heldout-start 336 --heldout-windows 64 \
  --sequence-length 2048 --rank 64 \
  --batch-size 1 --query-chunk-size 32 \
  --gn-sweeps 1 --cg-iterations 4 \
  --cg-relative-tolerance 1e-4 \
  --damping 0 --maximum-backtracks 5 \
  --model-dtype bfloat16 --torch-num-threads 4

python -u evaluation/eval_qwen3_8b_post_rope_kqsvd_c1_wikitext.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-factor-dir results/checkpoints/qwen3_8b_c1_v64_als5 \
  --kq-factor-dir results/checkpoints/qwen3_8b_post_rope_kqsvd_r64_c4_128f64h \
  --c1-k-factor-dir results/checkpoints/qwen3_8b_c1_k_output_closure_r64_gn1 \
  --output-json results/evaluation/qwen3_8b_c1_k_output_closure_r64_gn1_wikitext.json \
  --arms dense_k,kq_svd,c1_k \
  --dataset wikitext2 --split test \
  --seqlen 2048 --batch-size 1 \
  --model-dtype bfloat16 --device-map balanced \
  --max-memory-per-gpu-gib 44 --torch-num-threads 4
```

## Artifacts

- Checkpoint: `results/checkpoints/qwen3_8b_c1_k_output_closure_r64_gn1/`
- Factors SHA-256: `3caf8eac10d21ea05f9fde90888cd4c60f77d6fffa41144f65977fed03026d0c`
- Checkpoint metadata SHA-256: `15176d59b165cb8c36ce02412ade082351bc62bd02a9a9c2748ed41a590faa9d`
- PPL JSON: `results/evaluation/qwen3_8b_c1_k_output_closure_r64_gn1_wikitext.json`
- PPL JSON SHA-256: `f5535ebfd69d52dcb3e23b869c5c4ad99613fb580887058d4e547771c9c34223`
- Log: `logs/qwen3_8b_c1_k_r64_gn1_8284035.out`

The evaluator emits the existing tokenizer warning that the concatenated
WikiText token stream exceeds the model maximum. The evaluator slices that
stream into 2048-token chunks before model execution; no overlength sequence
is passed to the model.
