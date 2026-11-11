# Value-reuse state accounting (reviewer concern B)

This uses existing records only; no new benchmark was run. Source: `results/rebuttal/table3_k_offload_128k/`
(`arms/formal/summary.csv`, `arms/formal/inventory.csv`, `arms/check/placement_check.json`, `README.md`).
The workload is Llama-3.1-8B-Instruct BF16 on one L40S, TP1, B1, a 130,048-token prompt, and dense V128 in every
arm. The cache capacity is 130,193 slots. All numbers were re-read from those files on 2026-11-10.

## What the router stores and what it scans

| Router | Scanned per cached token, per KV head | Persistent routing state it adds on top of the Value cache |
|---|---:|---:|
| B16R16, separate Base cache (archived runtime) | 32 dims (16 Base + 16 Key residual) | 32 dims |
| B16R16, Base read from the rotated Value cache | 32 dims (16 Base + 16 Key residual) | **16 dims** (Key residual only) |
| Independent R32 Key router (analytical; not run) | 32 dims | 32 dims |

## Measured state at 130,048 tokens (GiB, from the tensor inventory)

| Arm | Exact Keys (GPU) | Values (GPU) | Base routing state | Key-residual routing state | Exact Keys (host, pinned) | Decode-resident GPU total |
|---|---:|---:|---:|---:|---:|---:|
| `dense_resident` | 7.946 | 7.946 | — | — | 0 | 30.859 |
| `basis_resident` (separate Base) | 7.946 | 7.946 | 0.993 | 0.993 | 0 | 32.975 |
| `basis_resident_nobase` (Value reuse) | 7.946 | 7.946 | **0** (view of the Value cache) | 0.993 | 0 | 31.975 |
| `basis_k_offload_nobase` (Value reuse + Key offload) | 0 | 7.946 | 0 | 0.993 | 7.946 | 24.289 |
| R32, analytical (`[1,8,130193,32]` BF16 × 32 layers) | — | — | — | 1.986 | — | — |

## What this shows

1. **Value reuse removes the Base copy.**
   - The separate Base cache is `[1, 8, 130193, 16]` BF16 × 32 layers = 1,066,541,056 B = **0.993 GiB**.
   - Reading Base from the rotated Value cache lowers measured decode-resident GPU memory by **1.000 GiB**
     (32.975 → 31.975), because the allocator rounds each layer's buffer to 32 MiB.
   - At this workload, incremental persistent routing state is **0.993 GiB with reuse and 1.986 GiB without
     reuse**. An independent R32 router would also add 1.986 GiB.
2. **Value reuse does not reduce routing-scan work or latency.**
   - Every B16R16 arm still scans 32 dims per cached token.
   - Measured router scan: 11.750 ms (separate Base) vs 11.896 ms (reuse), so +1.2%.
   - Full decode step: 37.400 vs 37.830 ms, so +1.15%.
   - We claim **no** speedup and **no** halving of scan width. There is no R32 latency measurement, and none is
     inferred here.
   - Value reuse also does **not** change how many exact Keys are fetched from host per step.
3. **Key offload is a separate effect and moves bytes rather than eliminating them.**
   - Going from `basis_resident_nobase` to `basis_k_offload_nobase` lowers GPU decode-resident memory by 7.686 GiB.
   - The same exact Keys (7.946 GiB) now sit in pinned host memory, so total system memory is not reduced.
   - Do not attribute this 7.686 GiB to Value reuse. Of the 8.686 GiB total GPU reduction from `basis_resident`
     to `basis_k_offload_nobase`, 1.000 GiB comes from Value reuse and 7.686 GiB from moving Keys to host.
   - The offloaded arm is bitwise identical to its resident control in selected pages, fetched Keys, caches,
     attention outputs and logits.
4. **Numerical caveat.**
   - The no-Base path rotates the Value basis: it takes an FP64 QR of `base_left`, rotates `v_proj`/`o_proj`/
     `base_right`, and re-rounds the weights to BF16.
   - This is algebraically equivalent but not bitwise identical to the separate-Base runtime. Over 16
     teacher-forced steps, argmax was identical and the maximum logit difference was 0.67.
5. **Scope.**
   - These records use dense V128, not the paper's compressed V96.
   - With V96, Base still occupies 16 of the 96 cached Value coordinates (the `basis_v96_reuse` arm of the new TP1
     runtime implements this). It has not been measured yet.
   - At V128, BasisKV uses **more** GPU memory than dense (32.975 vs 30.859 GiB), because it keeps full Keys and
     Values plus routing state. Its GPU-memory advantage in deployment comes from Value compression and Key
     offload, not from the router.

## Rebuttal-ready sentence

At 128K tokens on one L40S, reusing Value-derived Base coordinates removes a separately stored Base cache.
This cuts the router's incremental persistent state from 1.99 GiB to 0.99 GiB, half of what an independent R32
Key router would add. The router still scans 32 dimensions per cached token, and the measured scan time does not
improve (11.75 vs 11.90 ms). The benefit is persistent state, not scan bandwidth or latency. Moving exact Keys to
host memory is a separate effect: it frees a further 7.7 GiB of GPU memory by placing the same bytes in pinned
host memory.
