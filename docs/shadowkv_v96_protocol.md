# ShadowKV Key path with resident C1-V96

Official source: https://github.com/ByteDance-Seed/ShadowKV revision e51904cdeab7d4d34013370f09f2cf5fcd655e15, cloned at external/ShadowKV. Adapt models/kv_cache.py ShadowKVCache accuracy implementation, not ShadowKVCache_CPU. Apache-2.0 source attribution retained in the new core.

Qwen3-8B-Base; preserve Q/K RMSNorm and RoPE. Per prompt/layer, concatenate normalized pre-RoPE K across8 KV heads into T x1024. FP32 torch.svd, truncate rank160, store U and SV in BF16. Chunk size8. Exclude last4 full chunks plus remainder from historical chunk candidates (32–39 prompt-local tokens). Mean exact post-RoPE K per chunk. Protect48 chunks/head having lowest minimum token-to-mean cosine similarity. Remove them from the landmarks. Per-Q-head landmark softmax in FP32 then BF16, GQA maximum, Top256 chunks=2048 routed tokens. Reconstruct selected pre-RoPE K using U/SV, apply original positional RoPE. Attention includes exact local/outlier K, reconstructed routed K and all newly generated exact K.

Payload substitution only: V uses the frozen C1-V96 encoder/decoder and remains GPU resident; no V offload. Exact full K is also retained by the diagnostic DynamicCache, but selected historical attention uses reconstructed K, not gathered exact K. This is an accuracy adaptation, not a memory/PCIe benchmark or native Qwen3 support claim.

The resident accuracy path does not apply the CPU implementation's chunk-count alignment to multiples of8. No extra prefix sink or user-defined recent64 is added. Physical tokens per KV group per step are2048+384+(32–39)+generated_count; report actual support rather than call it total B2048.

Both full-K reference and ShadowKV use full C1-V96 BF16 Triton prefill. Same32K RULER11 tasks x8 prompts,88 paired examples. Full88 reference generated with same protocol. Smoke samples0 and32,4-token cap; compares full-prefill logits and repeats greedy logits. Summary verifies176 records, scores, EOS/caps, first tokens, shard coverage and source hashes.

Two CPU tests passed: direct execution of the official ShadowKVCache class gives identical SVD factors, landmarks/IDs, selection and reconstruction on a small FP32 fixture; resident V96 incremental decode agrees with explicit attention and has unique support IDs. BF16 real-model smoke is still pending at submission.

Environment:basis. Normal Slurm L40S/yangGrp scheduling,2CPUs/64GiB host RAM per worker, max4 concurrent formal shards. Existing jobs not modified.

## Jobs and commands

### smoke: 8302419

```bash
ARMS=(full shadowkv); /home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_shadowkv_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}"
```

### evaluate: 8302420

```bash
ARMS=(full shadowkv); ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 4)); SHARD=$((SLURM_ARRAY_TASK_ID % 4)); /home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_shadowkv_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --arm "${ARMS[$ARM_INDEX]}" --shard-index "$SHARD"
```

### summarize: 8302421

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_ruler_shadowkv_v96.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```

Output:results/evaluation/ruler_shadowkv_v96_bf16. Logs:shadow96_<stage>_<job>_<array>.log. No final results yet.

## Zero-decode statistics repair

Both smoke arms passed. Full-K88/88 and ShadowKV87/88 completed. ShadowKV sample86 emitted EOS as its first token and therefore never called decode; the statistics function incorrectly required a selected token set. No failed model generation or CUDA failure was observed in that traceback.

Statistics now return zero decode support for zero-step requests. Physical-budget means exclude such requests because no last decode step exists; the prompt still participates normally in accuracy. Three CPU tests pass, including a new zero-step test; original official-equation parity remains exact on the fixture.

Existing175 sample records and7 shard manifests are preserved byte-for-byte. Their original protocol and file hashes are captured in repair_inputs.json. The evaluator accepts those exact recorded artifacts only, checking that semantic protocol fields and all unaffected source hashes match. New records retain the repaired source hashes; summary records all input hashes plus the repair manifest hash.

Resume job8302587: same evaluation command with --arm shadowkv --shard-index2; skip verified completed records, generate sample86. Existing summary job8302421 has its success dependency updated to8302587. No existing GPU job cancelled. Both wait for normal Slurm scheduling as needed.
