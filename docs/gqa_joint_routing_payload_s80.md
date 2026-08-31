# S80: fully shared GQA routing/payload latent

S80 learns one rank-80 representation per physical GQA KV group. The same
coordinates are used for approximate routing, Value/payload accumulation,
tensor-parallel AllGather, and final-output decoding. There is no fixed
shared/private coordinate mask.

For one KV group, define the post-RoPE joint token feature and cache as

\[
X_g=[V_g,K_g^{post}], \qquad
C_g=X_gA_g=V_gA_g^V+K_g^{post}A_g^K,
\]

where \(A_g\in\mathbb R^{256\times80}\) for Qwen3. Every Query head has a
payload decoder \(D_h\in\mathbb R^{80\times H}\). During fitting, routing is
parameterized by \(U_h\in\mathbb R^{128\times32}\) and
\(R_g\in\mathbb R^{80\times32}\):

\[
\widehat Y=\sum_hP_hC_{g(h)}D_h,
\qquad
\widehat S_h=(Q_hU_h)(C_{g(h)}R_g)^\top/\sqrt{128}.
\]

The final attention probabilities still come from exact post-RoPE QK. Proxy
scores are exposed only for candidate routing and never silently replace
exact attention.

```text
              post-RoPE K ───────────────┐
                                         │ A_K
hidden → dense V → folded V branch ──┐   │
                                    ├── sum → S80 cache [T, 80]
                                    │
                                    ├── Q U × S80[:32]ᵀ → routing proxy
                                    │
exact QK → teacher / final P ────────┤
                                    └── P × S80 → AG latent [80]
                                                     ↓
                                               joint decoder
```

## Width and memory accounting

The learned representation has

\[
R_{store}=R_{payload}=R_{wire}=80,\qquad R_{route}=32.
\]

Export rotates each group's fitted routing subspace into the first 32 latent
coordinates. The persistent cache remains S80, while a routing scan reads the
contiguous `[..., :32]` view directly. The reference runtime also keeps exact
128-dimensional K for authoritative attention and exact candidate
verification. Its resident cache is consequently `K128 + S80-80 = 208`
features per physical KV head, versus dense `K128 + V128 = 256`: an 18.75%
total KV-cache reduction. CPU offload is intentionally outside this patch.

AllGather width is 80 per Query head. Qwen3-8B TP4 owns eight Query heads per
rank, so each source produces 640 local latent coordinates; the global joint
decoder consumes `32 × 80 = 2560` coordinates. No second routing tensor is
communicated.

## Objectives and statistics

Payload calibration forms

\[
Z_h^X=[P_hV_g,P_hK_g^{post}]
\]

and retains the full cross-head covariance
`[query_heads, query_heads, 256, 256]`. The dense coefficient block is
`[O_h; 0_K]`, so the existing routed-C1 full-layer quadratic and exact D-step
remain valid.

Routing uses the raw-score objective

\[
\sum_{h,s}\operatorname{Tr}
\left[\Delta_h^\top G_{Q,h,s}\Delta_hG_{X,g,s}\right],
\qquad
\Delta_h=M_hA_g^\top-J_K^\top.
\]

Each causal/document shard keeps its Q and X Grams paired. Independently
summing Q and X Grams before multiplying them would add nonexistent
cross-document score terms. The initial collector uses the final query token
of each selected full document, whose visible key prefix is unambiguous.

The optimized objective is

\[
\mathcal L_{S80}=
\mathcal L_{payload}/E_{payload}
+\lambda_{route}\mathcal L_{route}/E_{route}.
\]

The manifest records both normalizers and the explicit routing weight.

At FP32, one Qwen3-8B full payload covariance occupies 256 MiB per layer and
9 GiB per 36-layer split. Fit and validation therefore require about 18 GiB
for payload covariances alone; FP64 doubles this. Collection streams rows and
keeps layer accumulators on the device that owns each sharded model layer.

## Solver

The block-coordinate solver uses this order:

1. exact full-layer payload decoder solve;
2. headwise routing-query-factor matrix-free LSQR updates;
3. groupwise routing-selector matrix-free LSQR updates and QR gauge closure;
4. groupwise joint-encoder matrix-free LSQR updates over the combined
   payload/routing residual;
5. joint-encoder QR gauge canonicalization;
6. final D, U, and R closure.

Every iterative block solves the damped residual problem

\[
\min_\Delta\|J\Delta+r\|_2^2+\lambda\|\Delta\|_2^2,
\qquad X\leftarrow X+\Delta.
\]

LSQR receives only `J` and `Jᵀ` callbacks and never forms the Kronecker normal
operator. Payload decoder row space is reduced to at most `heads_per_group ×
rank` before the encoder solve, so the residual does not carry the full hidden
width. The LSQR callbacks stream raw captured routed `[PV, PK_post]`, final-token
Q, and prefix `[V, K_post]` operands. They do not reconstruct a design from
Gram matrices and do not use TSQR. Tolerance is `1e-5`, the limit is 100
iterations, and relative damping is fixed at `1e-5`. There is no acceptance
gate, interpolation, or objective backtracking.

For `A = QR`, canonicalization applies `A ← Q`, `D ← R D`, and
`M ← M Rᵀ`. This preserves both `X A D` and `Q M Aᵀ Xᵀ`. At export, each
routing selector is completed to an orthogonal basis
`Ω_g=[R_g,R_g^⊥]`, then `A_g←A_gΩ_g` and `D_h←Ω_gᵀD_h`. Consequently
`C_g[:32]=C_g^{old}R_g` while the full payload is unchanged. The serving bank
does not store `R_g`. Export folds `V A_V` into `v_proj`; `K_post A_K` remains
an explicit runtime term.

Initialization preserves the existing C1-V80 payload factors. A rank-32 KQ
initializer is regressed into that latent using the C4 V/K cross covariance,
then represented as a group-shared latent selector and head-specific Query
factor. Subsequent BCD updates retain an 80-dimensional payload latent and a
fixed rank-32 routing map without imposing a hard coordinate split.

## Implementation

- Statistics: `basisserve/calibration/gqa_joint_routing_payload_s80_stats.py`
- Matrix-free LSQR: `basisserve/core/iterative_least_squares.py`
- Solver/folding: `basisserve/core/gqa_joint_routing_payload_s80.py`
- Qwen3 runtime/checkpoint: `basisserve/checkpoint/gqa_joint_routing_payload_s80_qwen3.py`
- Collector: `scripts/collect_qwen3_gqa_joint_routing_payload_s80_stats.py`
- Direct LSQR capture: `scripts/capture_qwen3_gqa_joint_routing_payload_s80_direct.py`
- Builder: `scripts/build_qwen3_gqa_joint_routing_payload_s80.py`

The reference runtime caches exact post-RoPE K as the key cache and the S80
joint latent as the value cache. `compute_routing_proxy_scores` reads only the
first 32 latent coordinates and is a separate API. The runtime uses ordinary
PyTorch SDPA/eager attention and is intended for correctness and quality
evaluation, not as a sparse serving kernel.

## Example commands

Collect fit/validation statistics:

```bash
python scripts/collect_qwen3_gqa_joint_routing_payload_s80_stats.py \
  --model /path/to/Qwen3-8B-Base \
  --windows results/calibration/qwen3_8b_c4_256f64h_s2048/windows.safetensors \
  --fit-start 0 --fit-windows 256 \
  --validation-start 256 --validation-windows 64 \
  --routing-fit-windows 128 --routing-validation-windows 32 \
  --sequence-length 2048 --batch-size 1 \
  --statistics-dtype float32 \
  --output-dir results/calibration/qwen3_8b_s80_c4_p256v64_r128v32_s2048
```

Capture the raw fit operands used by LSQR:

```bash
python scripts/capture_qwen3_gqa_joint_routing_payload_s80_direct.py \
  --model /path/to/Qwen3-8B-Base \
  --windows results/calibration/qwen3_8b_c4_256f64h_s2048/windows.safetensors \
  --layers 7,13 --fit-start 0 --fit-windows 256 \
  --routing-fit-windows 128 --sequence-length 2048 --batch-size 1 \
  --output-dir results/calibration/qwen3_8b_s80_direct_256f_r128_s2048_l7_l13
```

Fit the factor bank:

```bash
python scripts/build_qwen3_gqa_joint_routing_payload_s80.py \
  --model /path/to/Qwen3-8B-Base \
  --stats-dir results/calibration/qwen3_8b_s80_c4_p256v64_r128v32_s2048 \
  --direct-capture-dir results/calibration/qwen3_8b_s80_direct_256f_r128_s2048_l7_l13 \
  --c1-v80-init results/checkpoints/qwen3_8b_c1_v80_als5 \
  --kq-r32-init results/checkpoints/qwen3_8b_post_rope_kqsvd_r64_c4_128f64h \
  --joint-rank 80 --routing-rank 32 --routing-weight 1.0 \
  --max-sweeps 35 --min-sweeps 1 \
  --relative-objective-tolerance 1e-5 --convergence-patience 3 \
  --iterative-max-iterations 100 --iterative-tolerance 1e-5 \
  --relative-damping 1e-5 \
  --output-dir results/checkpoints/qwen3_8b_s80_r32_v80_lsqr100
```

Both commands must be launched through Slurm for real Qwen3 calibration and
fitting. The factor bank records model/config identity, layer coverage,
post-RoPE convention, initializer hashes, normalizers, solver settings,
environment versions, and per-layer diagnostics.

## Deliberate limitations

- Exact K remains resident; there is no CPU offload or asynchronous prefetch.
- Routing quality must be evaluated with page/token recall and captured exact
  attention mass in addition to raw-score reconstruction error.
- There is no PairK, cross-layer sharing, hard S/P/R mask, MLA conversion, or
  custom sparse-attention kernel.
- `routing_weight` is not silently tuned per layer. A later controlled sweep
  should report the payload-quality/routing-recall Pareto frontier.
