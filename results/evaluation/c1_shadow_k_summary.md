# C1-Compatible Shadow-Key Speculative Decoding Summary

## Executive summary

The current design combines a deployed C1 model with a quantized Shadow-Key cache:

- The deployed **exact-Key C1 model** is the semantic target.
- A groupwise INT8 or INT4 **Shadow-Key** cache generates speculative candidates.
- The exact-Key C1 model verifies candidates and supplies correction tokens.
- Strict one-token replay currently guarantees the same token sequence as ordinary sequential exact-Key C1 greedy decoding.

This is not ordinary KV-cache quantization. Quantization errors reduce speculative acceptance and increase rollback work; they do not change the final sequence. The current implementation is a correctness and acceptance oracle, not a production throughput implementation.

## 1. Target model and cache representations

For Qwen3-32B, the deployed C1 target uses:

- BF16 Key with head dimension 128;
- C1 Value with dimension 64, produced by the rank-64 ALS export;
- BF16 model weights and computation.

The canonical committed state is therefore:

```text
exact-Key C1 state = BF16 K(128) + BF16 C1 V(64)
```

The speculative state replaces only Key with a quantized representation:

```text
Shadow state = groupwise INT8/INT4 K(128) + BF16 C1 V(64)
```

For a Key group `g`, symmetric quantization is:

\[
s_g = \frac{\max |K_g|}{q_{\max}}, \qquad
Q_g = \operatorname{round}(K_g/s_g), \qquad
\widetilde K_g = Q_gs_g.
\]

Here, `q_max` is 127 for INT8 and 7 for signed INT4. The current experiments use group size 32 and BF16 scales.

## 2. Speculative decoding round

For draft length `d`:

1. Use current exact C1 logits for the first proposed token.
2. Use the Shadow-Key state to generate the remaining `d - 1` candidates.
3. Run exact C1 verification against the proposal.
4. Accept matching candidate tokens up to the first mismatch.
5. On a mismatch, discard the rejected suffix and emit the exact C1 correction token.
6. Strictly replay every emitted token through a one-token exact C1 forward before committing its KV state.

```text
Canonical exact-Key C1 state
            |
            +-- exact logits --> first candidate seed
            |
            +-- quantize K --> Shadow-Key --> continuation candidates
            |
            +-- exact verification --> accept / reject / correction
            |
            +-- one-token exact replay --> committed canonical KV
```

The first token is forced from exact target logits. Metrics named `mean_shadow_continuation_accepted` exclude this seed and measure only the continuation contributed by Shadow-Key.

## 3. Correctness semantics

The intended invariant is:

\[
\text{speculative output}
=
\text{ordinary sequential exact-Key C1 greedy output}.
\]

An initial implementation tried to commit KV state produced by block verification. Real GPU tests showed that BF16 block and sequential forwards can occasionally disagree at top-1 because their floating-point execution orders differ. The strict implementation therefore treats block verification KV as diagnostic and discards it. Every emitted token is replayed sequentially through the exact C1 model, and only that state is committed.

This guarantees sequence equivalence, but it adds exact target work and is not yet suitable as the final offloading architecture.

## 4. Difference from ordinary quantization

| Property | Ordinary KV quantization | Shadow-Key speculation |
|---|---|---|
| Quantized cache role | Replaces the original cache | Generates candidates |
| Final authority | Quantized model | Exact-Key C1 model |
| Quantization error | Can change the output trajectory | Causes rejection and rollback |
| Exact sequence guarantee | Normally absent | Preserved by strict exact replay |
| State | One quantized cache | Shadow state plus accessible exact state |
| Primary evaluation | Accuracy, perplexity, throughput | Acceptance, rollback, KL, exact calls/token |

If the exact-Key state is deleted and INT8/INT4 Key becomes authoritative, the method reduces to ordinary KV-cache quantization and loses exact-output equivalence.

The method also differs from conventional speculative decoding: there is no independent small draft model. The same C1 model, running with an approximate Shadow-Key state, acts as the draft path.

## 5. INT8 smoke-test result

The aligned INT8 evaluation ran on one NVIDIA A100 80GB GPU in the `lowrank` Conda environment. It used four fixed prompts, 64 generated tokens per prompt, group size 32, no recent exact-Key window, and draft lengths 2, 4, and 8.

Slurm job: `8274705`; state: `COMPLETED (0:0)`; elapsed time: 2 minutes 14 seconds.

| Draft | Exact target matches | Mean accepted | Full-block acceptance | Shadow top-1 | Mean KL | Rollbacks | Exact calls/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4/4 | 1.9766 | 97.66% | 97.66% | 0.000767 | 3 | 1.5000 |
| 4 | 4/4 | 3.9844 | 98.44% | 99.48% | 0.000911 | 1 | 1.2500 |
| 8 | 4/4 | 7.2000 | 88.57% | 98.19% | 0.001126 | 4 | 1.1367 |

All 12 prompt/draft configurations reproduced the sequential exact-Key C1 target sequence. Draft length 4 gave the most stable full-block behavior, while draft length 8 required the fewest exact calls per output token. These timings come from an unfused reference implementation and must not be interpreted as production speedups.

Artifacts:

- [`c1_shadow_k_smoke_int8_w0.json`](c1_shadow_k_smoke_int8_w0.json)
- [`c1_shadow_k_smoke_int8_w0.md`](c1_shadow_k_smoke_int8_w0.md)
- [`c1_shadow_int8_w0_8274705.out`](../../logs/c1_shadow_int8_w0_8274705.out)
- [`c1_shadow_int8_w0_8274705.err`](../../logs/c1_shadow_int8_w0_8274705.err)

## 6. KV storage accounting

The following accounting is per KV head and token. Qwen3 Key and original Value each have 128 elements. BF16 uses two bytes per element.

### Original BF16 KV

```text
K: 128 * 2 = 256 bytes
V: 128 * 2 = 256 bytes
Total:           512 bytes
```

### Exact-Key C1 KV

```text
K:      128 * 2 = 256 bytes
C1 V:    64 * 2 = 128 bytes
Total:             384 bytes
```

This is a 25% total-KV reduction relative to the original BF16 KV cache.

### Groupwise INT8 Key plus C1 Value

For group size 32, a 128-element Key has four BF16 scales:

```text
INT8 values: 128 * 1 = 128 bytes
BF16 scales:   4 * 2 =   8 bytes
Shadow K:              136 bytes
C1 V:                  128 bytes
Total:                  264 bytes
```

- Key saving relative to BF16 Key: `46.875%`.
- Total saving relative to exact-Key C1 KV: `31.25%`.
- Total saving relative to original BF16 KV: `48.4375%`.

### Packed groupwise INT4 Key plus C1 Value

```text
Packed INT4 values: 128 * 0.5 = 64 bytes
BF16 scales:           4 * 2 =  8 bytes
Shadow K:                      72 bytes
C1 V:                         128 bytes
Total:                         200 bytes
```

- Key saving relative to BF16 Key: `71.875%`.
- Total saving relative to exact-Key C1 KV: `47.9167%`.
- Total saving relative to original BF16 KV: `60.9375%`.

The current reference INT4 implementation stores quantized values in an `int8` tensor rather than packing two values per byte. Its logical INT4 size is 72 bytes in this example, but its current physical reference size is the same as INT8. Realizing the logical saving requires nibble packing and a compatible fused attention kernel.

## 7. Interaction with exact-Key offloading

C1 compresses Value but leaves exact Key unchanged:

```text
Original BF16 KV: K = 128, V = 128
Exact-Key C1 KV:  K = 128, V =  64
```

Consequently, Key grows from 50% of the original KV cache to two-thirds of the exact-Key C1 cache. If exact K is offloaded to CPU memory, every exact attention operation must read the entire historical Key prefix across the CPU/GPU interconnect. The cost grows linearly with context length and can dominate decoding.

The current strict oracle makes this worse for offloading because it performs both diagnostic block verification and sequential exact replay:

- draft 2: 1.50 exact calls per output token;
- draft 4: 1.25 exact calls per output token;
- draft 8: 1.1367 exact calls per output token.

If each exact call reloads offloaded Key, draft length 8 still performs about 13.7% more exact-Key accesses than ordinary sequential C1 decoding. A GPU-resident INT4 Shadow-Key may save hot-cache capacity, but strict replay prevents it from automatically producing an offloading speedup.

## 8. Better production directions

### A. Blockwise exact-Key streaming

Load or stream exact Key once per speculative block and verify all `d` query positions together:

\[
Q_{1:d}K^\top.
\]

Accepted KV should be committed directly from the block verifier, with an additional exact call only when correction is needed. This can amortize exact-Key transfer toward approximately one transfer per block rather than one per token.

The remaining challenge is defining or implementing block verification whose greedy semantics agree with the required sequential target. Possible approaches include a numerically aligned kernel, defining block verification as the deployed target semantics, or falling back to sequential verification only when the top logits have a small margin.

### B. K-local distributed attention

Keep Key sharded at its owner instead of all-gathering or moving the full cache. Send queries to the Key owners, compute local scores and softmax statistics, and reduce only:

- the local maximum;
- the softmax denominator;
- the local weighted-Value numerator.

The communication volume then does not scale with moving the full historical Key cache. C1 is especially complementary here because the weighted-Value numerator has dimension 64 instead of 128.

### C. Quantized canonical Key

Making INT8 or FP8 Key authoritative removes the need to retain or access BF16 exact Key. This is the simplest route to production memory and bandwidth savings, but it changes the semantic target to a quantized C1 model and gives up exact equivalence to BF16 exact-Key C1.

### D. Hybrid exact window

Keeping a recent exact-Key window on GPU and quantizing or offloading older Key may improve Shadow-Key acceptance and reduce near-context error. It does not by itself remove the need to access old exact Key when exact BF16 verification is required.

## 9. Current conclusion

INT8 Shadow-Key has demonstrated high speculative agreement and exact final sequences under strict replay. INT4 remains useful as an acceptance-quality experiment, but an INT4 smoke test alone cannot validate an offloading speedup.

The combination that currently looks most promising is:

```text
C1 Value compression
    + blockwise exact-Key verification or K-local distributed attention
    + compressed Shadow-Key candidate generation
    + correction-only fallback
```

The main architectural objective is to avoid reading the full exact Key history once for every emitted token. Until that is achieved, strict exact replay is best treated as a correctness oracle rather than the production serving path.

## 10. Direct block-commit evaluation

The repository now also contains an explicitly selected `direct_block` policy.
It commits accepted pending exact target K/C1-V directly and performs a
one-token exact forward only for a correction token. The default remains
`strict_replay`; direct block commit is not labeled lossless.

### 10.1 INT8 Shadow-Key free-running generation

The four-prompt, 64-token A100 smoke test directly committed between 97.66% and
99.61% of output tokens from block-verifier KV:

| Draft | Exact sequence matches | Full-block acceptance | Shadow top-1 | Direct tokens | Corrections | Exact calls/token |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2/4 | 99.22% | 99.22% | 255/256 | 1 | 0.503906 |
| 4 | 1/4 | 93.94% | 97.89% | 252/256 | 4 | 0.273438 |
| 8 | 2/4 | 83.78% | 97.26% | 250/256 | 6 | 0.167969 |

This demonstrates the desired target-call reduction, but it also confirms that
direct block KV changes the ordinary sequential BF16 greedy trajectory. The
test therefore measures a new block-scheduled target, not an exactly equivalent
implementation of the old sequential target.

The direct stage completed in Slurm job `8277961` before a later, unrelated
numerics assertion terminated that combined job. Its result files are complete:

- [`c1_shadow_k_smoke_int8_direct_block.json`](c1_shadow_k_smoke_int8_direct_block.json)
- [`c1_shadow_k_smoke_int8_direct_block.md`](c1_shadow_k_smoke_int8_direct_block.md)
- [`c1_block_stage1_8277961.out`](../../logs/c1_block_stage1_8277961.out)
- [`c1_block_stage1_8277961.err`](../../logs/c1_block_stage1_8277961.err)

### 10.2 Exact-proposal block/sequential oracle

To remove Shadow-Key proposal error, one transactional sequential exact-C1 run
generated each proposal and retained that same run's logits and committed cache
as the sequential reference. The block branch consumed identical exact proposal
tokens. This avoids treating a second, independently rounded BF16 replay as the
definition of the reference.

| Block | Tokens | Block proposal matches | Top-1 agreement | Mean NLL delta | PPL ratio | Mean KL |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 256 | 255/256 | 98.44% | +0.00320844 | 1.003214 | 0.001140 |
| 4 | 256 | 254/256 | 97.66% | +0.00001936 | 1.000019 | 0.001116 |
| 8 | 256 | 253/256 | 98.05% | +0.00459975 | 1.004610 | 0.000957 |
| 16 | 256 | 251/256 | 97.27% | +0.00319183 | 1.003197 | 0.001149 |

All 256 sequential-reference predictions reproduce the exact proposals by
construction. The block branch changes only 1--5 proposal predictions out of
256, while mean KL remains near `0.001`. Top-1 is therefore sensitive to small
schedule-dependent logit changes even when the full distributions remain close.

### 10.3 Block-scheduled WikiText-2 NLL/PPL

The teacher-forced smoke test scored four fixed 128-token WikiText-2 samples,
508 target tokens total, under sequential cache construction and under direct
block commits:

| Block | Sequential PPL | Block PPL | PPL change | Mean NLL delta | Top-1 agreement |
|---:|---:|---:|---:|---:|---:|
| 2 | 15.58325 | 15.60957 | +0.1689% | +0.00168743 | 97.05% |
| 4 | 15.58325 | 15.58247 | -0.0050% | -0.00005011 | 97.64% |
| 8 | 15.58325 | 15.63191 | +0.3123% | +0.00311785 | 98.23% |
| 16 | 15.58325 | 15.62022 | +0.2372% | +0.00236934 | 97.64% |

Slurm job `8277973` completed successfully on one A100 80GB in 8 minutes 6
seconds. The JSON reports and logs contain no NaN, infinity, OOM, or traceback.

Artifacts:

- [`c1_block_commit_numerics_smoke.json`](c1_block_commit_numerics_smoke.json)
- [`c1_block_commit_numerics_smoke.md`](c1_block_commit_numerics_smoke.md)
- [`c1_block_scheduled_ppl_smoke.json`](c1_block_scheduled_ppl_smoke.json)
- [`c1_block_scheduled_ppl_smoke.md`](c1_block_scheduled_ppl_smoke.md)
- [`c1_block_shared_ref_8277973.out`](../../logs/c1_block_shared_ref_8277973.out)
- [`c1_block_shared_ref_8277973.err`](../../logs/c1_block_shared_ref_8277973.err)

### 10.4 Interpretation

On this smoke sample, top-1 reversals are substantially more frequent than the
change in aggregate likelihood would suggest: top-1 agreement is about 97--98%,
but PPL changes by at most 0.32%. This supports evaluating MCQ/task accuracy
separately instead of using exact greedy-sequence equality as a quality proxy.
It does not yet establish task equivalence: four short corpus samples are too
small, and neither this oracle nor teacher-forced PPL measures production
throughput. A larger PPL run and matched MCQ evaluation are the appropriate next
statistical tests.
