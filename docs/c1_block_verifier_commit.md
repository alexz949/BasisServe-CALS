# C1 exact block-verifier KV commit

## Scope

This implementation evaluates whether exact C1 KV produced by a multi-token
block verifier can be committed directly instead of replaying every accepted
token. It currently provides:

- `strict_replay`, the unchanged sequential exact-C1 reference;
- `direct_block`, which commits accepted pending target KV directly;
- an exact-proposal block-versus-sequential numerical oracle;
- block-scheduled teacher-forced NLL/PPL evaluation.

Margin guarding, MCQ evaluation, CUDA kernels and exact-Key offloading are not
part of this stage.

## State and token alignment

Every cache layer owns three disjoint transactional states:

```text
committed:
    exact target K
    quantized Shadow K derived from committed exact K
    exact target C1 V

draft/provisional:
    Shadow-path K
    Shadow-path C1 V

target/pending:
    exact block-verifier K
    exact block-verifier C1 V
```

For proposal tokens `y[0:m]`, the verifier receives the full proposal block.
The logits used to verify those tokens are aligned as:

```text
[carried_target_logits, verifier_logits[0:m-1]]
```

The final verifier logit predicts the token after `y[m-1]` and becomes the
next-round exact seed after a fully accepted direct block.

## Commit policies

### `strict_replay`

Pending block KV is discarded. Each emitted token is processed by one exact
one-token C1 forward, and only the resulting sequential KV is committed.

Contract:

```text
output token IDs == ordinary sequential exact-Key BF16 C1 greedy token IDs
```

### `direct_block`

Suppose the verifier accepts `y[:a]` and rejects `y[a]` in favor of correction
token `c`.

```text
commit pending exact target KV for y[:a]
    -> discard provisional draft state and rejected pending suffix
    -> run one exact target forward for c
    -> commit exact KV for c
```

Accepted candidates are never replayed. A fully accepted block commits all
pending exact target KV and performs no extra exact target forward.

The pending state comes from exact C1 attention, not from Shadow-Key drafting.
Direct commit is mathematically the same C1 model, but BF16 block-shaped kernels
can produce a different numerical trajectory from one-token execution. It is
therefore not labeled lossless.

The cache transaction checks that:

- committed length advances by exactly the requested accepted prefix;
- every layer has the same committed length;
- provisional draft and pending target tensors are cleared;
- emitted-token and committed-position counts agree after every round.

## Exact-proposal numerical oracle

`evaluation/eval_c1_block_commit_numerics.py` first generates continuation
tokens with the same transactional one-token exact-Key C1 path used by strict
replay. It then scores and commits those fixed tokens through two exact target
schedules:

```text
branch A: one-token sequential target forwards
branch B: exact multi-token block forwards with direct pending-KV commit
```

Shadow-Key is not used. This isolates execution schedule from quantized proposal
error.

The oracle reports:

- immediate top-1 agreement and first disagreement offset;
- sequential-to-block KL and top-5 overlap;
- NLL delta and PPL ratio on the exact proposal tokens;
- layerwise K and C1-V relative L2, maximum absolute error and cosine similarity;
- the worst cache-drift records.

Example command:

```bash
python evaluation/eval_c1_block_commit_numerics.py \
  --model-path /path/to/qwen3 \
  --c1-export /path/to/c1/export \
  --prompt-file evaluation/prompts/c1_shadow_k_smoke.jsonl \
  --block-lengths 2,4,8,16 \
  --max-new-tokens 128 \
  --dtype bfloat16 \
  --device cuda:0 \
  --output-json results/evaluation/c1_block_commit_numerics.json \
  --output-markdown results/evaluation/c1_block_commit_numerics.md
```

## Block-scheduled teacher-forced NLL/PPL

`evaluation/eval_c1_block_scheduled_ppl.py` scores the same fixed corpus tokens
under:

```text
sequential schedule: block length 1
direct schedules:    block lengths 2, 4, 8, 16
```

Only logits entering cross entropy are promoted to FP32. Model execution stays
at the selected dtype. The report includes mean NLL delta, PPL ratio, top-1
agreement and tail statistics over paired samples.

Example command:

```bash
python evaluation/eval_c1_block_scheduled_ppl.py \
  --model-path /path/to/qwen3 \
  --c1-export /path/to/c1/export \
  --dataset wikitext2 \
  --split test \
  --sequence-length 512 \
  --prefill-tokens 1 \
  --max-samples 8 \
  --block-lengths 2,4,8,16 \
  --dtype bfloat16 \
  --device cuda:0 \
  --output-json results/evaluation/c1_block_scheduled_ppl.json \
  --output-markdown results/evaluation/c1_block_scheduled_ppl.md
```

This is schedule-conditioned teacher-forced PPL. It is not free-running
generation and does not establish MCQ answer or task-accuracy equivalence.

## Current limitations

- single GPU and batch size one;
- eager Qwen3 attention only;
- uniform-rank ALS C1 runtime only;
- no margin guard or fallback calibration;
- no free-running task evaluation;
- no packed low-bit attention or exact-Key streaming;
- Python/Hugging Face call timings are not production throughput measurements.
