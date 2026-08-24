# Feature-major one-sided ragged TP prototype

This prototype implements the path

```text
source-ready compact attention
    -> source-local feature-major block [w_s, tokens]
    -> direct or one-sided exact-size source push
    -> next receive-arena slot [sum_s w_s, tokens]
    -> one GEMM: arena.T @ decoder
```

It reuses the existing ``StaticRaggedPlan`` and validated ``PackedC1TPLayer``
boundary while keeping transport implementation separate, so the existing
ragged path remains an executable correctness and performance control.

## Why this removes the global pack

For source `s`, let its token-major compact attention output be

```text
X_s: [tokens, w_s].
```

The transport sends the contiguous feature-major block `X_s.T` and writes it at
row offset

```text
offset_s = sum_{j < s} w_j
```

inside every receiver's arena. After all source signals arrive, the arena is
exactly

```text
arena = cat([X_0.T, X_1.T, ..., X_{P-1}.T], dim=0)
      = cat([X_0,   X_1,   ..., X_{P-1}], dim=1).T
```

Therefore the replicated decoder can run directly as

```python
output = torch.mm(arena.T, decoder)
# arena:   [K, tokens]
# decoder: [K, hidden]
# output:  [tokens, hidden]
```

There is no receiver-side rank-major-to-token-major packing kernel. When the
attention kernel still emits `[tokens, w_s]`, only a source-local transpose is
needed. The RMA API also exposes `local_feature_major_view()`, allowing a future
attention kernel to write directly into the registered source slice and remove
that transpose/copy as well.

## Backends

### `feature_direct`

A correctness baseline using grouped two-sided `ncclSend`/`ncclRecv`. It already
uses the feature-major arena and one decoder GEMM, but a source cannot complete a
push unless the receiver participates in the matching P2P operation.

### `feature_rma`

The actual asynchronous source-push path. It uses NCCL 2.29+ symmetric windows,
`ncclPutSignal`, and one `ncclWaitSignal` over all remote sources.

A low-width rank can call the transport as soon as its local attention finishes:

1. copy/write its block into its registered local source slice;
2. issue one grouped put fan-out to the same source offset in every peer window;
3. signal each peer;
4. wait for one new signal from every other source;
5. run one decoder GEMM.

`ncclWaitSignalDesc_t.opCnt` is passed as `1` on every invocation: each call
waits for one further signal from every remote source. Sequential reuse must be
stress-tested for thousands of iterations on the exact deployment NCCL version
before enabling this backend in serving.

## Safety contract

The prototype makes several constraints explicit:

- `prepare_rma()` is a cold-path collective. Every TP rank must call it with the
  same token count, maximum K, dtype, and call order.
- The RMA workspace is bound to the CUDA stream active during `prepare_rma()`.
  Calls from another stream are rejected. Use a separate RMA context for truly
  concurrent streams/microbatches.
- Every rank must invoke layer plans in the same logical sequence. Widths may
  vary across layers as long as `sum(widths) <= max_total_width`.
- The RMA arena has two alternating slots. This prevents an early rank's next
  layer push from overwriting a slower rank's current decoder input without
  adding a post-decode barrier. Acquire `local_feature_major_view()` again for
  every call; retaining one view across calls defeats the direct-to-current-slot
  zero-copy path. Prepared RMA storage therefore occupies
  `2 * max_total_width * tokens * element_size` bytes per rank.
- Two slots assume one ordered layer-by-layer dependency chain. Concurrent
  microbatches that can advance independently require separate communicators /
  RMA contexts (or a deeper explicitly managed slot ring).
- `close()` is collective and should be called on every rank. The destructor
  aborts rather than attempting unordered collective deregistration.
- Host RMA support depends on NCCL build, driver, and topology. The wrapper
  reports compile/runtime version support; window registration can still reject
  an unsupported topology or return a null window. `feature_direct` remains the
  control path.
- The NCCL 2.29 one-sided host API does not support CUDA Graph capture. Pin and
  test the complete PyTorch/NCCL stack rather than loading a second NCCL version
  into an existing PyTorch process.

## Decoder ordering

The decoder must be source-major. Its row blocks follow process-group rank order:

```text
decoder = cat([D_0, D_1, ..., D_{P-1}], dim=0)
D_s.shape == [w_s, hidden]
```

Within each source block, use exactly the same head/coordinate ordering as the
local compact-attention tensor.

## Build and run

The extension is built lazily through `torch.utils.cpp_extension` against the
same pip/conda NCCL installation used by PyTorch. The current `lowrank`
environment contains NCCL 2.28.9, so it can run `feature_direct` but will report
`feature_rma` as unavailable. RMA requires a coherent PyTorch environment whose
bundled NCCL headers and runtime are both 2.29 or newer.

```bash
export MAX_JOBS=8

torchrun --standalone --nproc-per-node=8 \
  benchmarks/bench_feature_ragged_allgather.py \
  --source-ranks 32,32,48,64,64,80,96,128 \
  --heads-per-source 8 \
  --tokens 256 \
  --hidden 5120 \
  --backends existing_registered,feature_direct,feature_rma,padded_allgather \
  --feature-major-input \
  --artificial-skew-us-per-rank 2.0 \
  --output-json results/feature_ragged_tp8_b256.json
```

Run first without `--feature-major-input` to measure the compatibility path that
performs a source-local transpose. Then enable it to represent an attention
kernel writing feature-major output directly.

## Minimal integration

```python
import torch

from basisserve.core.c1_tp_decode import (
    C1TPFactorLoader,
    assert_distributed_loader_consensus,
)
from basisserve.core.c1_tp_feature_decode import C1FeatureMajorTPDecoder
from basisserve.core.c1_variable_v_attention import (
    C1RankLocalAttentionReference,
    C1StaticKVCache,
)
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator

loader = C1TPFactorLoader(
    factor_dir,
    model_config=model_path,
    tp_size=8,
    expected_result_sha256=expected_result_sha256,
)
assert_distributed_loader_consensus(loader)
comm = FeatureRaggedCommunicator.from_distributed()
packed = loader.load_layer(
    layer_index,
    process_rank=comm.rank,
    device=f"cuda:{comm.device}",
    dtype=torch.bfloat16,
)
comm.prepare_rma(
    tokens=256,
    max_total_width=max_layer_total_width,
    dtype=torch.bfloat16,
)
decode = C1FeatureMajorTPDecoder(
    packed,
    communicator=comm,
    backend="feature_rma",
)

# The existing correctness-reference attention folds A_s into local v_proj,
# stores a variable-width V cache, and emits [batch, tokens, local_width]. Q/K
# projection and RoPE remain upstream inputs to this boundary.
attention = C1RankLocalAttentionReference.from_dense_projection(
    dense_v_projection,
    packed,
    output_dtype=torch.bfloat16,
)
cache = C1StaticKVCache(
    batch_size=batch_size,
    capacity=max_sequence_length,
    head_dim=packed.local_encoder.shape[0],
    value_head_dim=packed.local_rank,
    dtype=torch.bfloat16,
    device=f"cuda:{comm.device}",
)
local = attention(hidden_states, query_states, key_states, cache)
y = decode(local.local_coordinates)

# Future fused CUDA attention path: acquire a fresh slot for every call and
# write [local_width, tokens] directly into registered memory.
local_fm = decode.rma_output_view()
attention_kernel(..., output=local_fm)
y = decode.forward_feature_major(local_fm)
```

## What this prototype does not claim

It does not remove the exact layer-completion lower bound imposed by the final
source. It removes the unnecessary rule that earlier sources must postpone their
transport until that final source is ready. Decoder execution still waits until
all source contributions are present, then runs one large GEMM.

The one-sided transport does not reduce bytes beyond C1's compact wire width.
Each source still fans its block out to every other TP rank. Its purpose is to
move source-ready data earlier and reduce receiver participation, not to provide
an additional compression ratio.
