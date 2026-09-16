# Page32 router: computation and measurement plan

Command: `python -m benchmarks.system.profile_router_phases --smoke`, followed by `python -m benchmarks.system.profile_router_phases` after validation. Environment: `basis`, CUDA 12.3.2, one L40S. Model inputs: the existing Llama-3.1-8B base/Dense-V benchmark, B16R16, batch 1, Page32, hard budget 2048 including sink32 and recent64.

## What is inside the previous 10.54 ms

That number was the sum of instrumented calls to `conditional_router_page_lse` over 32 layers, averaged over decode steps 11–20. It includes query-code preparation and wrapper effects. The new isolated measurements use each layer's first real decode query and graph replay to measure GPU kernel time; they are not automatically identical to the earlier whole-call number.

The scan reads already cached Base16 codes. It does **not** project V80 to Base16 across the history on every decode step. The native benchmark actually uses Dense V128. V-to-Base projection occurs when the cache is built and when new tokens are appended. The V80 measurement is explicitly a dimension-only control using sliced inputs, not a different fitted model.

The production scan performs:

1. Load a page of Base codes and the Base right factor into shared memory.
2. WMMA: `Base[32,16] @ right[16,128]`, FP32 accumulation.
3. Round the reconstructed pre-RoPE K to BF16, add bias, and round again.
4. Apply key-position RoPE with intermediate BF16 product/add rounding; write BF16 K to shared memory.
5. Load the current queries and their residual query codes.
6. WMMA: query–key products. GQA=4 is padded to a 16-row query tile; only two of eight warps compute the two 16-token tiles.
7. Add the residual-code dot product, with the existing BF16 rounding and scale order.
8. Warp max/exp/sum/log to emit one score per query head and page.

No reconstructed K or full token-score array is written to global memory by the production router. Removing K reconstruction therefore does not save a pre-existing full-K HBM write/read.

## Transformed-query identity and its cost

Let `z_t` be Base16, `B` the 16-by-128 right factor, and `b` the bias. Ignoring intermediate rounding:

`s_t = q^T R_t (B^T z_t + b) = z_t^T (B R_t^T q) + b^T R_t^T q`.

The transformed query `B R_t^T q` depends on the key position `t`. For the two 64-dimensional RoPE halves, precompute coefficients once per query and rank:

`C[r,i] = B[r,i] q[i] + B[r,i+64] q[i+64]`

`S[r,i] = B[r,i] q[i+64] - B[r,i+64] q[i]`.

Then the position-dependent transformed coordinate is:

`u[t,r] = sum_i cos[t,i] C[r,i] + sin[t,i] S[r,i]`.

The bias uses the same formula with `B[r,:]` replaced by `b[:]`. The diagnostic Triton implementation fuses the position projection, Base16 dot, bias, residual dot and Page-LSE. Coefficients are prepared once per query; there is no global full-length transformed-query tensor.

For each page and KV head, useful arithmetic (not an instruction count) is approximately:

- Original reconstruction: `32 * 16 * 128 = 65,536` multiply-accumulates, shared by four query heads.
- Original useful QK work: `4 * 32 * 128 = 16,384` multiply-accumulates. Tensor-core padding executes additional work.
- Transformed coordinate work: `4 * 32 * 16 * 128 = 262,144` multiply-accumulates, plus bias and rank-16 scoring. The prototype's FP32 coefficient products use TF32x3 for precision, adding hardware work beyond this useful-operation count.

Moving RoPE and the projection to the query side removes the explicit `B16 -> K128` intermediate, but it loses reuse of that reconstructed key across the four query heads. It also does not commute through the original nonlinear BF16 rounding. Performance and selection equivalence must both be measured.

## Reading the phase measurements

The fine-grained instrumented clone splits bias from RoPE and residual scoring from LSE, introducing shared-memory traffic and barriers. It must reproduce the original scores bitwise. `clock64` fractions include scheduling stalls and barriers and come from that altered layout. Multiplying those fractions by original-kernel latency produces an **attribution proxy**, not eight separately measured phase times. The total original and instrumented latencies are reported so the perturbation is visible.

The transformed-query prototype is checked against a floating-point reassociation reference and against the original BF16 router. Score error and selected-page overlap are reported. The original output continues to drive the model; this experiment does not claim end-to-end quality equivalence or switch the serving kernel.
