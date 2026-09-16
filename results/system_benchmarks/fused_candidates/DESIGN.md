# Fused candidate selection

One Triton CTA owns one batch/KV-head row. It loads four query-head page-score rows; reproduces the existing per-head log normalization over eligible non-forced historical pages; merges heads by maximum; and forces sink/recent pages within the 512-candidate budget.

The final implementation transforms FP32 scores into monotone unsigned integer keys and performs exact 32-bit radix threshold selection. Two prefix scans resolve threshold ties and compact chosen original page IDs in ascending order. It writes only the 512 IDs, with no intermediate group-score tensor or separate torch topk/sort launches. Coarse scoring and fine scanning retain separate kernels and their parallel grids.

Cutoff ties choose ascending original page IDs. PyTorch topk does not specify tie order, so equality of score multisets is required; exact ID equality is measured separately. Fixed pages have finite maximal sort priority; padding is excluded from counts. For <=512 pages, all ascending IDs are returned as before.

The first prototype used full bitonic sorting. Installed Triton's float sorting multiplies values by zero internally, so infinities produced NaNs; finite sentinels fixed correctness. That implementation was then replaced because it was slower at 128K. Logs retain those attempts. The radix version is the only current candidate implementation.

Validation: random, tied, and large-magnitude inputs; batch2/8KVheads; 8K-128K including 512/513 and 2048/2049 page transitions. Check ordered unique IDs, all forced pages, and exact selected score multisets versus the original selector. A separate 64K model run compares selectors on every actual decode query; those instrumented timings are excluded. Uninstrumented ABBA runs enable fused append on both sides, isolating selector fusion.

Environment: basis.

```bash
python -m benchmarks.system.validate_fused_candidates
python -m benchmarks.system.run_fused_candidates
```

The candidate is explicitly enabled through --fused-select in bench_two_stage. No default RULER path change or GitHub upload.
