# Local fusion opportunities after two-stage profiling

These are design candidates, not measured speedups. The event-profile total was 34.92 ms versus 31.29 ms uninstrumented. Individual event spans contain launch gaps. The 2.59 ms cache-side remainder and 1.77 ms outside-layer remainder are mixed categories, not standalone kernels that can be removed wholesale.

## Priority 1: append and summary update

One CTA per batch/KV head can load current V128/K128 once, write V, compute Base16, reconstruct the existing predicted K with the same BF16 round points, compute Residual16, maintain the recent64 ring, and update historical min/max using the displaced K[t-64]. The current K[t] must not enter the historical summary early. A vector store to mapped CPU K can also be included, subject to preserving store visibility before later fetch. This saves launches and intermediate traffic; it does not remove PCIe payload.

An existing conditional_router_append_decode kernel already combines V/Base/Residual updates. It is not used by the current native Python cache update. Reuse it as the starting point, validate its accumulation/rounding against the current path, then add the ring/minmax update and optionally CPU K store. The residual query projection is another small operation that can share the same launch if Q is supplied; its result must be ready before fine scanning.

## Priority 2: candidate selector

Replace the separate GQA log-normalization kernel, torch.topk(512), and ID sort with one selector. Keep per-head normalization over non-forced eligible pages, GQA max, 512 shared candidates, forced sink/recent pages within that count, and ascending original IDs. Test ties and numerical differences. Coarse scoring itself is currently distributed over page tiles; merging all scoring and global selection into one block per KV head can severely reduce parallelism. Keep that producer/consumer boundary in the first version.

## Priority 3: final IDs and planning

Final fine-score normalization/top62, candidate-to-original-page mapping, Page32 token expansion, historical-tail mask, exact recent64 concatenation, and slot planning can be combined by the KV-head owner. Preserve the existing logical support and ordering. At minimum, fuse mapping/expansion/masking/recent concatenation into the planner to eliminate several small tensor operations.

## Boundaries worth retaining initially

- Coarse scoring -> global candidate normalization/top512 requires results from all page-tile blocks. A normal CUDA kernel has no grid-wide barrier.
- Fine scan -> final GQA normalization/top62 has the same issue. Keep the current register-consumed reconstruction/RoPE/QK/residual/LSE fusion inside each fine block.
- Slot planner -> fetch can be combined in a block with a barrier, but one block per KV head would reduce the current highly parallel fetch grid. Fewer launches alone do not establish a win.
- Missing-K fetch -> attention can reuse loaded K in registers/shared memory, but needs explicit GQA ownership and cache-write handling. Independent query-head programs can multiply CPU K reads. This is a larger local kernel redesign, not a simple concatenation.

## Model-side pointwise operations

Residual-add plus RMSNorm is a standard fusion candidate. The native runtime already uses a fused RMSNorm and fused SiLU-and-multiply. Gate/up weights already form one combined projection. Folding activation into a GEMM epilogue requires changing that GEMM implementation, so it is not equivalent to removing a Python expression. Retain BF16 rounding semantics or measure differences.

Outside-layer dispatch, allocations and launch gaps are not all GPU operations. Reducing wrappers, reusing workspaces, or graph capture addresses different overheads and must not be called kernel fusion.

The RULER run uses frozen, previously tested kernels; none of these proposals is enabled during evaluation.
