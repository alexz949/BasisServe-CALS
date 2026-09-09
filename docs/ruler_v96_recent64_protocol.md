# RULER V96 Base16/R16 plus extra recent64

Same frozen full-window Q32 Base16/R16 and C1-V96 as ruler_v96_r16_fp16. Preserve the original Page32/B2048 selector, pinned page0, and all selected tokens. Append the exact recent64 tokens including the current token, deduplicated against the selected pages; no page rounding and no subtraction from routing budget. At most2112 valid tokens/group, depending on overlap and partial pages.

Accuracy reference native attention implementation in basisserve/core/c1_conditional_recent_attention.py; existing production kernels/files unchanged. Same FP16 selected attention arithmetic, expanded token slots with duplicate slots masked. Tests: exact union at sequence/page boundaries, unchanged full-support output, sparse union output against explicit masked dense reference. All3 passed.

Same88 RULER32K prompts and V100 FP16 settings. Reference full-K/LRQK result reused by hash. No refit. Original no-recent score86.1553; final comparisons must use saved full precision scores.

Environment:basis. Slurm GPU workers2CPUs/64GiB memory,4 formal shards after smoke. No H200 access.

## Jobs and commands

### smoke: 8301893

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_c1_v96_recent64.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
```

### evaluate: 8301894

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_c1_v96_recent64.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index $SLURM_ARRAY_TASK_ID
```

### summarize: 8301895

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_c1_v96_recent64.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```

Output:results/evaluation/ruler_v96_r16_recent64_fp16. Logs:recent64_<stage>_<job>_<array>.log. No new final score at submission.

