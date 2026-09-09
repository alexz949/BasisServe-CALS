# Official QUEST accuracy forward with C1-V80

## Configuration

Qwen3-8B-Base, frozen C1-V80 ALS6, BF16, basis environment, L40S. Same existing
RULER 32K pilot: 11 tasks x eight prompts, task-specific generation caps and EOS,
greedy decoding, shared full C1 prefill and first token. No calibration refit.

Arms: full exact K + C1-V80; Q8-Base16/Q16-Page-Fisher R8 with first two layers
full; official QUEST with first two layers full. Layers are zero-indexed: 0-1
full, 2-35 sparse. Our router uses Page32, 2048 physical tokens per KV group,
including page0. QUEST uses Page32, 2048 tokens independently per Q head, without
pinning. The GQA union is measured, not capped to 2048. This is not an equal
physical-budget comparison. Page32 retains the previous experiment's page size;
the official passkey shell example uses Page16, which is not used here.

## Upstream and bridge

Official repository: https://github.com/mit-han-lab/Quest

Clone: `/deac/csc/yangGrp/zhangal/Quest`, including recursive submodules.
Commit: `01c1623bf9395009520874e989e29f683203b357`.
The source `evaluation/quest_attention.py` is imported directly, with file bytes
verified against that commit. Bounds, page selection, attention masking, softmax
and weighted Value computation execute in its unmodified `forward` function.

The official model installer only recognizes Llama/Mistral, not Qwen3. A tensor
bridge preserves the existing Qwen3 Q/K normalization, RoPE, cache and C1 output
decoder. It passes already-normalized, rotated Q/K through identity projections
and identity rotary embeddings. The removed Transformers position_ids argument
is adapted at the imported rotary helper call, without changing source bytes.
The legacy upstream cache gets all but the last token and appends that last
token exactly once; the actual C1 prefix is not modified. C1-V80 is zero-padded
to K128 to satisfy upstream's equal-width interface; the padded output is sliced
back to V80 before the existing C1 decoder. Upstream's global layer counter is
not used: layers 0 and 1 use the existing full attention module directly.

The official `local_heavy_hitter_mask` is observed, without changing its result,
to count unique valid token IDs in each GQA group's union. Statistics average
over sparse layers and generated decode steps only; they exclude the two full
layers. The upstream accuracy forward computes full exact QK before masking,
so its runtime is not an optimized sparse-kernel or PCIe benchmark. All exact
K and C1-V remain GPU-resident. This is an official-algorithm C1 payload/model
adaptation, not a reproduction on the original paper's models.

## Checks

Two CPU tests in basis passed: upstream support/output versus an independent
per-head min/max + masked-attention reference, and tiny-model full-support
logit equivalence with only later layers invoking upstream, repeated cache
replay, and unchanged prefix tensors. The full-support test uses Page1 because
upstream first clamps budget to length, then floors budget/page_size: with a
partial page an oversized requested budget does not necessarily select all keys.
This upstream behavior is retained in the evaluation.

GPU smoke must replay all three arms with exactly equal logits and generated
IDs from the same prefix. Each sample checks finite logits, cache length and
immutable prefix. QUEST additionally checks the per-layer call counts: zero in
layers 0-1 and one per decoded query in layers 2-35.

## Execution

Environment: `/home/zhangal/.conda/envs/basis/bin/python`.
Output: `results/evaluation/quest_official_ruler32k`.
Four L40S workers, eight CPUs, 256 GiB host memory, one-hour time limit.
L40S exists only on lovelace/yangGrp in the current partition inventory; it is
kept consistent with the previous comparison. Other GPU partitions have no
L40S devices.

From the BasisServe-CALS root, the command is:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_official_quest_ruler.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

After smoke succeeds, evaluate uses the same command with `--stage evaluate`
and shard indices 0-3; summary uses `--stage summarize` after all workers succeed.
Logs use `logs/quest-official-<jobid>` with smoke and per-worker suffixes.

## Completed results

Slurm job 8300540 completed on 2026-09-05 with exit 0:0. Combined GPU smoke,
four-worker evaluation and summary took 11 minutes 45 seconds. Largest recorded
per-worker PyTorch allocated-memory peak was 33.663857 GiB. No GPU OOM, NaN or
runtime error occurred in the Slurm job.

| Method | Task-balanced score | Mean physical tokens / sparse KV group |
|---|---:|---:|
| Full exact K + C1-V80 | 85.2083% | Full context |
| Base16 + Fisher R8, layers 0-1 full | 80.7197% | 2033.744 |
| Official QUEST forward, layers 0-1 full | 50.7386% | 3579.526 |

Full means full-sequence attention, not replacement of C1-V80 by dense V128.
The Value payload is C1-V80 in every layer of every arm. The first two full
layers are excluded from the sparse selection-count averages in the table.

Prior comparison values were 80.4356% for our all-36-sparse router and 34.4886%
for the all-36-sparse shared-GQA QUEST adaptation. The latter differs in both
layer policy and support policy (GQA aggregation, shared budget and pinned
prefix), so 34.4886% to 50.7386% is not an isolated first-two-layer ablation.

Post-run audit:

- 88 unique prompt keys, with eight completed samples for each of 11 tasks.
- All saved samples have identical protocol/source hashes.
- All 88 full exact-K generated-ID sequences exactly match the earlier
  `routing_baselines_ruler32k` comparison.
- All 264 saved prediction strings exactly match decoding their generated IDs.
- Every QUEST record has layer call counts `[0, 0, n, ..., n]`, where
  `n = generated_tokens - 1` and all 34 later layers have count n.
- Independently averaging the 88 per-sample scores reproduces all three means.
- GPU smoke replayed all three arms with identical logits/IDs and unchanged
  prefix; all formal samples passed finite-logit and cache checks.

Artifacts:

- [Per-task summary](../results/evaluation/quest_official_ruler32k/summary.md)
- [Protocol, hashes, paired scores and all generated outputs](../results/evaluation/quest_official_ruler32k/result.json)
- Main log: `logs/quest-official-8300540.out` and `.err`.
- Smoke: `logs/quest-official-8300540_smoke.out` and `.err`.
- Four workers: `logs/quest-official-8300540_0.out/.err` through
  `logs/quest-official-8300540_3.out/.err`.
