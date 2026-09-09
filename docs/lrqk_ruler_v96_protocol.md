# LRQK RULER32K with matched C1-V96

Qwen3-8B-Base. The existing32K RULER11-task subset,8 samples/task,88 paired prompts; not the full suite or untouched held-out data. Both arms use uniform C1-V96 from results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6.

Full exact-K and LRQK R32, per-query-head k1152 plus recent64. Model, exact K and C1 payload FP16; LRQK state/factors/codes/score scan FP32. Same original online2/2 updates, seed0. Both arms use full C1 FP16 memory-efficient SDPA prefill; different fresh caches and model processes, identical first-token verification. No dense V128 arm, no offline router refit, no direct reuse of older V80 RULER scores. Resident-cache accuracy reference, not the official CPU-ring offload system. LRQK selection is token-level, not Page32; physical GQA union recorded at the final decode step per prompt, not a hard B2048 budget.

All jobs use normal Slurm scheduling on V100, never direct SSH execution. Environment:/home/zhangal/.conda/envs/basis/bin/python. GPU workers2CPUs/64GiB host memory each. Smoke2 arms; formal8 single-GPU shards, max4 concurrent; CPU summary follows success. Smoke uses4-token caps for samples0 and32 and is not an accuracy result.

## Jobs and commands

### smoke: 8301695

```bash
ARMS=(full k1152); /home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_lrqk_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}"
```

### eval: 8301696

```bash
ARMS=(full k1152); ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 4)); SHARD=$((SLURM_ARRAY_TASK_ID % 4)); /home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_lrqk_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --arm "${ARMS[$ARM_INDEX]}" --shard-index "$SHARD"
```

### sum: 8301697

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_lrqk_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```


Results:results/evaluation/ruler_lrqk_v96_fp16. Logs:lruler_<stage>_<job>_<array>.log. Per-sample commands/protocol hashes retained. Summary verifies all176 records, scores, EOS/caps, token decoding, first-token agreement and shard coverage. No final accuracy available at submission.

