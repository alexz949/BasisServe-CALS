# C1-V80 routing comparison on RULER 32K

## Configuration

Qwen3-8B-Base, frozen C1-V80 ALS6, all 36 layers, BF16 and L40S. Use the
existing 88 RULER prompts (11 tasks x 8 examples), original caps/EOS, greedy
generation, shared full C1 Triton prefill and shared first output token.
Each decode arm forks the immutable prefix. This is the previously used pilot,
not an untouched final set or the full RULER suite.

| Arm | Selection | Budget |
|---|---|---|
| c1_exact_k | Full exact K, C1-V80 | All cached tokens |
| uniform_r8 | Existing Q8-Base16 + Q16 Page-Fisher R8; normalized Page-LSE, GQA max | 64 Page32 including page0 |
| quest_page | Exact post-RoPE min/max; raw upper-bound max across GQA heads | 64 Page32 including page0 |
| loki_page_r32 | Existing Loki PCA; normalized Page-LSE, GQA max | 64 Page32 including page0 |
| loki_token_r32 | Original independent per-query-head PCA token Top2048 | Physical GQA union measured; up to 8192 tokens/group |

QUEST and Loki-page are controlled GQA adaptations at the same physical page
budget as our router. All methods use all 36 layers. This is not a reproduction
of QUEST's original first-two-full-layer policy or original paper accuracy.
Loki-token has no pinned prefix and serves as a separate algorithm reference;
it is not an equal physical-budget comparator.

The Loki checkpoint uses the same first 64 C4 32768-token windows as the router.
It fits centered Key PCA after Key RMSNorm and before RoPE, then applies the
basis to post-RoPE Q/K at runtime, following the recorded Loki implementation.
No PCA or Base/residual factors are refitted for this comparison.

Sources:

- [QUEST](https://arxiv.org/html/2406.10774v2)
- [Loki](https://arxiv.org/html/2406.02542v2)

## Implementation and numerical checks

The existing C1 attention module retains QKV projection, Q/K normalization,
RoPE, incremental exact-K/C1-V cache updates and output decoding. Baseline
selection is installed temporarily around each independent decode.
QUEST extrema are initialized from the prefix and updated only for appended
tokens. Loki sidecars are projected once for the prefix and then appended.
Explicit full-support boolean decode masks select native BF16 attention.

Four CPU tests in basis passed: incremental page extrema and score upper
bounds, raw-bound GQA aggregation and prefix pinning, selected attention versus
an independently constructed dense mask, and tiny-model full-support replay
across all baselines including cache isolation and cleanup. Artifact/source
preflight verified the current C1, Base/R8, PCA and shared C4-window identities.
GPU smoke additionally replays every arm and requires identical generated IDs
and logits across two runs from the same prefix.

Exact K remains GPU-resident. Report actual materialized routing metadata:
Base128+R8 for our accuracy oracle, R32 for Loki, and page min/max for QUEST.
The theoretical residual-only extra storage of R8 must not be confused with the
current materialized Base128 cache. Neither runtime nor metadata numbers from
this oracle constitute optimized PCIe offload measurements.
Metadata bytes count persistent token/page routing caches only, excluding fixed
factor matrices and transient scoring, selection, gather and attention buffers.
Selected physical tokens are the unique valid token IDs needed per KV group,
not hardware memory transactions. Aggregate means are weighted by decode steps
and KV groups along each arm's own generated trajectory.

## Execution

Entry: `evaluation/eval_qwen3_8b_routing_baselines_ruler.py`.
Output: `results/evaluation/routing_baselines_ruler32k`.
Environment: `/home/zhangal/.conda/envs/basis/bin/python`.
Stages: smoke on one selected device within a four-L40S allocation, four L40S
evaluation shards, CPU summary.
L40S node `lovelace` currently belongs to partition `yangGrp`; the device is kept
consistent with the previous matched evaluations.

The same command is used with stage smoke, evaluate (shard indices 0-3), and
summarize:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_routing_baselines_ruler.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/q8_qbase_fisher16_r8 \
  --loki-checkpoint results/checkpoints/qwen3_8b_loki_r32_c4_64f_s32768 \
  --output-dir results/evaluation/routing_baselines_ruler32k \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

## Completed results and audit

Completed on 2026-09-05 in the basis environment. Slurm job 8300534 used four
L40S devices, eight CPUs and 256 GiB host memory. Evaluation plus summary took
11 minutes 49 seconds; exit code was 0:0. The largest per-process PyTorch peak
allocation reported across the 88 prompts was 25.195785 GiB. This is allocated
memory, not total device memory occupancy.

| Method | Task-balanced accuracy | Mean selected physical tokens / KV group |
|---|---:|---:|
| Full exact K + C1-V80 | 85.2083% | Full context |
| Q8 Base16 + Q16 Page-Fisher R8 | 80.4356% | 2033.688 |
| QUEST-page, shared GQA raw-bound max | 34.4886% | 2045.969 |
| Loki-page R32, shared normalized Page-LSE | 57.6136% | 2038.433 |
| Loki-token R32, independent per-Q-head Top2048 | 79.9053% | 3865.413 |

Page arms select 64 pages including page0; a partial last page explains counts
below 2048. Loki-token uses independent query-head support and a larger physical
GQA union, so it is not a matched physical-budget result. These runs do not
measure attention-mass recall or optimized offload latency.

Full per-task scores and routing-cache sizes are in
[summary.md](../results/evaluation/routing_baselines_ruler32k/summary.md).
The machine-readable protocol, source/artifact hashes, per-sample outputs,
paired score differences and statistics are in
[result.json](../results/evaluation/routing_baselines_ruler32k/result.json).

Post-run checks:

- All 88 prompt keys are unique; all 11 tasks have eight completed samples.
- Both full exact-K and Base16+R8 generated token IDs match the previous Q16
  experiment on all 88 prompts, not merely its aggregate scores.
- All 440 saved prediction strings match decoding their generated token IDs.
- Independently averaged scores reproduce all five reported aggregate scores.
- Each sample passed immutable-prefix checks, cache-length checks and the
  prescribed selection-budget checks. Each saved sample has matching protocol
  and GPU identity; the summary revalidated per-sample scores and generation caps.
- The successful GPU smoke (job 8300533, 27 seconds, exit 0:0) replayed all five
  arms with exactly matching logits and token IDs from the same prefix.
- Initial smoke job 8300528 failed during prefill with GPU out-of-memory while
  unrelated processes occupied the selected GPU. No scored result came from
  that job. The retry used the same settings/output directory and selected a
  free GPU within its allocation; no unrelated workload was terminated.
- The formal four-GPU run completed without runtime errors or OOM.

Logs are `logs/routecmp-8300534.out`, `logs/routecmp-8300534.err`, and the four
worker log pairs `logs/routecmp-8300534_0.out/.err` through
`logs/routecmp-8300534_3.out/.err`. Smoke logs retain their respective job IDs.
