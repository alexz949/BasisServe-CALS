# C1 shadow-Key speculative verification oracle

This correctness oracle asks whether a deployed C1 Value representation can
remain the only committed Value cache while a cheaper shadow Key view drafts
tokens and exact Keys verify them. It is not a production offload benchmark or
a new speculative-decoding framework.

## Target and draft semantics

For the already-deployed C1 Value cache and decoder, the target attention is

```text
softmax(Q @ K_exact.T) @ V_C1
```

and the draft attention is

```text
softmax(Q @ K_shadow.T) @ V_C1.
```

The two paths use the same model weights, C1 Value projection, and C1 output
decoder. Only the representation read for committed historical Keys differs.
Keys are quantized after RoPE. A configured recent suffix is read exactly.
Draft tokens create temporary K/C1-V state, but that state is never committed.

“Lossless” means identical token IDs to greedy decoding from the exact-Key C1
target. It does not mean equivalence to the original dense checkpoint.

## Why only Keys are shadowed

C1 has already made the target Value cache narrower and folded the associated
decoder into the deployed model. Creating another persistent approximate Value
cache would duplicate state and blur the C1 target definition. The intended
future memory placement is therefore:

```text
GPU:       shadow K + recent exact K + target C1 V
CPU/store: exact K
```

The current oracle keeps exact K on GPU as well. Consequently its memory usage
is larger than the target baseline and its timings are not serving-throughput
claims. Its purpose is to measure correctness and accepted-run length before
packed low-bit attention or offload work begins.

## Cache transactions

Every layer has three disjoint states:

```text
committed: exact target K, quantized shadow K, target C1 V
draft:     provisional draft K and provisional draft C1 V
verify:    pending exact target K and pending target C1 V
```

The Transformers-compatible cache has explicit `target_prefill`, `draft`, and
`target_verify` modes. Each model call is wrapped in a forward transaction that
checks that every layer is updated exactly once and grows by the declared query
length. An exception restores all layers to their pre-forward lengths.

The default `strict_replay` policy treats block verification tensors as
diagnostic, discards them, and commits emitted tokens through one-token exact
target forwards. An explicitly selected `direct_block` policy instead commits
the accepted prefix of pending exact target K/C1-V and runs one additional
target forward only for a correction token. BF16 block and sequential GEMMs can
choose different top-1 tokens near a tie, so only strict replay retains the
ordinary sequential implementation-level contract. Truncation updates exact K,
shadow K, and C1 V together.

## Corrected greedy loop

The oracle carries the exact target logits that predict the next token. Those
logits seed the first proposal in every round; later proposals are generated
autoregressively with shadow Keys. Metrics report both the full accepted prefix
and the accepted shadow continuation after removing this forced seed.

For proposals `y[0:m]`, a target forward over the whole block produces pending
target K/V and logits after each proposal. Verification predictions are aligned
as:

```text
[carried_target_logits, verify_logits[0:m-1]]
```

If the whole block matches under strict sequential verification, every
proposal is replayed and committed one token at a time. No unmaterialized bonus
token is emitted. Block logits remain available for block-versus-sequential
numerical diagnostics.

At the first mismatch `j`, the oracle:

1. sequentially commits exact target tensors for `y[:j]`;
2. emits the verifier's correction token at `j`;
3. discards every draft tensor and block-pending suffix;
4. runs one exact target forward for the correction token;
5. commits its exact K and C1 V and carries its logits.

The sequential synchronization is intentionally explicit. Block-pending state
contains proposal KV computed with block-shaped BF16 kernels, while the strict
target is ordinary one-token greedy execution. Committing block state would
make the cache numerically drift even for identity shadow Keys.

## Reference quantization

The implementation supports identity 16-bit mode and symmetric groupwise 8-
and 4-bit modes over the Key head dimension. Runtime tensors use the native HF
layout `[batch, kv_heads, sequence, key_head_dim]`.

Logical four-bit bytes assume two values per byte and include scale storage.
Physical reference bytes count the actual `int8` qvalue tensor and scales. The
reference path does not claim packed INT4 HBM savings.

## Evaluation

```bash
python evaluation/eval_c1_shadow_k_speculative.py \
  --model-path /path/to/qwen3 \
  --c1-export /path/to/c1/export \
  --prompt-file prompts.jsonl \
  --shadow-bits 4 \
  --shadow-group-size 32 \
  --recent-exact-window 256 \
  --draft-lengths 4,8,16,32 \
  --commit-policy strict_replay \
  --max-new-tokens 128 \
  --dtype bfloat16 \
  --device cuda:0 \
  --output-json results/evaluation/c1_shadow_k_acceptance.json \
  --output-markdown results/evaluation/c1_shadow_k_acceptance.md
```

Each JSONL record may be a JSON string or `{"prompt": "..."}`. The evaluator
uses local files, eager Qwen3 attention, batch size one, and an independent
ordinary `DynamicCache` exact-C1 greedy baseline.

The evaluator consumes the repository's finalized uniform ALS layout directly:
`results.json` plus one `layer_XXX.safetensors` file per decoder layer. It folds
each stored Value-coordinate encoder into the checkpoint V projection and
installs the stored head decoders at their true C1 width; it does not use the
older padded-HF evaluation path.

The native C1 replacement calls Hugging Face's eager attention function, so
the model must be loaded with `attn_implementation="eager"`. The installer
rejects a mismatched attention implementation: an SDPA-configured mask factory
may omit the explicit causal mask that eager block verification requires.
Reported shadow top-1 agreement and KL exclude the forced exact-target seed at
the start of each round.

## Current limitations and next steps

- only the uniform-rank `install_qwen3_gqa_vo_export` path supports this cache;
- no sampling, batching, TP8, ragged AllGather timing, or vLLM integration;
- exact Keys are not offloaded;
- low-bit storage and attention are not fused;
- correction synchronization adds one exact target call on every rejected
  block;
- strict replay adds one exact target call per emitted token and is therefore
  an oracle cost, not production speculative throughput;
- block and sequential BF16 divergence is reported explicitly rather than
  allowed to alter the output sequence.

An experimental `--commit-policy direct_block` path now commits accepted
pending exact target KV and replays only correction tokens. Numerical and
teacher-forced PPL evaluation for that policy is documented in
[`c1_block_verifier_commit.md`](c1_block_verifier_commit.md). It has not yet
been promoted to the default.

The next system milestone should be attempted only after real-model acceptance
is promising: isolate an exact-Key store, add pinned-CPU layer-wise staging,
then measure whether exact-K transfer can overlap shadow drafting. TP8 work
should preserve the existing C1 rank schedule, decoder order, and ragged
collective rather than emulate them with padded fake ranks.
