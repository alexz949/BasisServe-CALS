# L40S BF16 RULER: full K, LRQK and Base16/R16 plus recent64

User requested all three arms on L40S, then explicitly changed LRQK routing state from FP32 to BF16 before submission. All model/K/C1-V96 payload caches are BF16. Our Base/residual sidecar and routing arithmetic are BF16; LRQK persistent factors/codes and scan are BF16. LRQK internal factor solves remain FP32 then cast to BF16, matching the original core implementation.

Same88 RULER32K prompts,11 tasks x8, frozen C1-V96 checkpoint and full-window Query-Gram Q32 Base16/R16 bank. No fitting. All arms use the same BF16 C1 Triton prefill kernel. Full-K uses SDPA decode; LRQK R32/k1152/recent64 uses token-level exact selected attention; ours preserves Page32/B2048 and pinned page0 then appends exact recent64 tokens with deduplication and no subtraction, max2112 valid tokens/group.

Separate fresh caches and requests; smoke compares prefill with full K and repeats generation/logits. Our helper checks immutable prefix, actual sparse-call counts and BF16 cache/sidecar shapes. LRQK state dtype explicitly checked. Summary verifies264 records, task scores, EOS/caps, decoded text, identical first tokens across all arms, checkpoint hashes and shard coverage.

Environment:basis. Normal Slurm scheduling on yangGrp/L40S as explicitly requested. Existing jobs left unchanged.2CPUs/64GiB per GPU worker; formal12 shards max4 concurrent; all stages success-dependent. CPU preflight and summary use small partition.

## Commands and jobs

### preflight: 8302334

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_v96_l40s.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage preflight
```

### smoke: 8302335

```bash
ARMS=(full k1152 ours); /home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_v96_l40s.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}"
```

### evaluate: 8302336

```bash
ARMS=(full k1152 ours); ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 4)); SHARD=$((SLURM_ARRAY_TASK_ID % 4)); /home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_v96_l40s.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --arm "${ARMS[$ARM_INDEX]}" --shard-index "$SHARD"
```

### summarize: 8302337

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_v96_l40s.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```

Output:results/evaluation/ruler_v96_l40s_bf16. Logs:l40ruler_<stage>_<job>_<array>.log. No final scores at submission.

