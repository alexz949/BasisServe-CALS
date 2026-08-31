# Qwen3-8B Wo-C1 cross-source covariance WikiText-2 PPL

The two compressed arms use identical stored BF16 joint-s20 encoders and identical private-AllGather communication. Only the decoder covariance model differs. Factors are folded into equivalent dense BF16 `o_proj` weights, isolating quality from runtime kernels.

| Arm | WikiText-2 PPL | Change vs dense |
|---|---:|---:|
| Dense | 7.00202554 | +0.000% |
| Full covariance | 7.28214639 | +4.001% |
| Block diagonal | 7.28660005 | +4.064% |

Full covariance change relative to block diagonal: `-0.061%` PPL.

Protocol:

- `wikitext2` `test`, concatenated non-overlapping 2048-token chunks.
- BFLOAT16 model execution, SDPA attention, FP32 cross-entropy accumulation.
- V and the KV cache remain dense for every arm.

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_wo_c1_cross_source_covariance_ppl.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_cross_source_covariance_joint20 --output-dir results/evaluation/qwen3_8b_wo_c1_cross_source_covariance_ppl_joint20 --dataset wikitext2 --split test --seqlen 2048 --batch-size 1 --model-dtype bfloat16 --attn-implementation sdpa --device cuda:0 --torch-num-threads 4 --local-files-only
```
