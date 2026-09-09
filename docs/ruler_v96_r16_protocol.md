# RULER V96 Base16/R16

Same88 RULER32K prompts and C1-V96 as results/evaluation/ruler_lrqk_v96_fp16. Reuse verified full exact-K and LRQK references; run only our arm. Frozen results/checkpoints/c1_v96_b16r16_qgram: Base16, R16, full-window Query-Gram Q32,64 fit/16 diagnostic C4 windows of32K,40BCD/PCG100. No refit.

V100 FP16 model/K/V/sidecar, C1-V96 full memory-efficient SDPA prefill then native Page32/B2048 sparse decode. Pin page0; all36 layers routed. LongBench generation helper reused with cache, immutable-prefix and actual sparse-call assertions. First generated token must equal the completed full-K reference. Smoke samples0/32,4-token caps, repeated exact logits. Summary verifies88 records plus the176-record reference, task scores, EOS/caps, token decoding and shard coverage.

Environment:basis. Slurm only; GPU workers2CPUs,64GiB host memory each.4 formal shards. No H200 access or jobs.

## Commands and jobs

### smoke: 8301777

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_c1_v96_r16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
```

### evaluate: 8301778

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_c1_v96_r16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index $SLURM_ARRAY_TASK_ID
```

### summarize: 8301779

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_c1_v96_r16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```

Output:results/evaluation/ruler_v96_r16_fp16. Logs:r16ruler_<stage>_<job>_<array>.log. No final score asserted at submission.

