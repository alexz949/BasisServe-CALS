# Qwen3-8B S80/R32 Design, Solver, and Routing-Calibration Summary

## 1. Scope

This document records the development of the Qwen3-8B S80 experiment from
the original fully shared 80-dimensional latent through the current
rank-32 routing parameterization, solver implementations, performance
optimizations, and the observed routing-calibration failure. It describes
completed implementations and measured results only.

The model geometry used throughout the experiments is:

| Quantity | Value |
|---|---:|
| Hidden size | 4096 |
| Query heads | 32 |
| Physical KV heads | 8 |
| Query heads per KV group | 4 |
| Query/Key dimension | 128 |
| Value dimension | 128 |
| Joint token dimension | 256 |
| Shared payload latent | 80 |
| Routing rank | 32 in the current design |

All Key factors and routing targets use post-RoPE K. Exact K remains the
authoritative Key cache for final attention and candidate verification.

## 2. Design evolution

### 2.1 Original fully shared S80 design

The original design formed one joint token vector per physical KV group:

\[
X_g=[V_g,K_g^{\mathrm{post}}]\in\mathbb R^{T\times256}.
\]

A group-specific encoder produced an 80-dimensional cached representation:

\[
A_g\in\mathbb R^{256\times80},\qquad
C_g=X_gA_g
=V_gA_g^V+K_g^{\mathrm{post}}A_g^K.
\]

The same 80 coordinates were used by the payload path, routing path, and
tensor-parallel communication path. There was no hard shared/private mask.
The original width identity was

\[
R_{\mathrm{store}}=R_{\mathrm{route}}=R_{\mathrm{payload}}
=R_{\mathrm{wire}}=80.
\]

Each Query head had a payload decoder

\[
D_h\in\mathbb R^{80\times4096}
\]

and an unrestricted routing map

\[
M_h\in\mathbb R^{128\times80}.
\]

The payload and proxy-score predictions were

\[
\widehat Y=\sum_h P_hC_{g(h)}D_h,
\]

\[
\widehat S_h=(Q_hM_h)C_{g(h)}^\top.
\]

The original initializer embedded independent C1-V64 payload factors and
KQ-SVD16 routing factors into 80 coordinates. BCD was allowed to mix those
coordinates after initialization.

### 2.2 Current S80 payload plus rank-32 routing design

The payload latent remained exactly 80-dimensional, but routing was fixed at
rank 32. A group-shared selector chooses a 32-dimensional routing subspace
from the same 80-dimensional latent:

\[
R_g\in\mathbb R^{80\times32},\qquad
U_h\in\mathbb R^{128\times32},
\]

\[
M_h=U_hR_g^\top\in\mathbb R^{128\times80},
\qquad \operatorname{rank}(M_h)\le 32.
\]

The routing proxy is therefore

\[
\widehat S_h
=Q_hU_hR_{g(h)}^\top A_{g(h)}^\top X_{g(h)}^\top.
\]

This is not an 80+32 cache. The stored payload representation is still
80-dimensional, and routing selects 32 coordinates from it. The current
factor shapes per layer are:

| Factor | Shape |
|---|---:|
| Joint encoder `A` | `[8, 256, 80]` |
| Payload decoder `D` | `[32, 80, 4096]` |
| Query routing factor `U` | `[32, 128, 32]` |
| Group routing selector `R` | `[8, 80, 32]` |

The current initializer preserves an existing C1-V80 payload solution. It
regresses KQ-SVD32 into the V80 latent using the joint V/K covariance, then
factorizes the routing map into head-specific `U` and group-shared `R`.
`R` is QR-canonicalized and has orthonormal columns.

## 3. Objectives

### 3.1 Payload objective

For each Query head, calibration forms

\[
Z_h^X=[P_hV_{g(h)},P_hK_{g(h)}^{\mathrm{post}}].
\]

The dense target coefficient is

\[
B_h^{\mathrm{dense}}=
\begin{bmatrix}O_h\\0_K\end{bmatrix}.
\]

The full-layer payload objective is

\[
\mathcal L_{\mathrm{payload}}
=\left\|
\sum_h Z_h^XB_h^{\mathrm{dense}}
-\sum_h Z_h^XA_{g(h)}D_h
\right\|_F^2.
\]

The implementation retains all cross-head covariance blocks
`[32, 32, 256, 256]`; it does not reduce the target to independent headwise
reconstruction.

### 3.2 Routing objective

Let the fixed K selector be

\[
S_K=[0_{128\times128},I_{128}]\in\mathbb R^{128\times256}.
\]

For head `h` in group `g`, the routing-map error is

\[
\Delta_h=U_hR_g^\top A_g^\top-S_K.
\]

For calibration document `s`, define paired sufficient statistics

\[
G_{Q,h,s}=Q_{h,s}^\top Q_{h,s},\qquad
G_{X,g,s}=X_{g,s}^\top X_{g,s}.
\]

The raw-score objective is

\[
\mathcal L_{\mathrm{route}}
=\sum_{h,s}
\operatorname{Tr}
\left(
\Delta_h^\top G_{Q,h,s}\Delta_hG_{X,g(h),s}
\right).
\]

Q and X statistics remain paired by document. Combining Q and X Grams
across documents before multiplying them would introduce nonexistent
cross-document score terms.

### 3.3 Normalized joint objective

The fitted objective is

\[
\mathcal L_{S80}
=\frac{\mathcal L_{\mathrm{payload}}}{E_{\mathrm{payload}}}
+\lambda_{\mathrm{route}}
\frac{\mathcal L_{\mathrm{route}}}{E_{\mathrm{route}}},
\]

with `routing_weight = 1.0` in the reported experiments. Both normalizers
are stored in every factor-bank manifest.

## 4. Calibration layout

The main C4 calibration bank contains full, pretokenized 2048-token windows.
The completed R128/V32 setup used:

| Split | Payload documents | Routing documents | Query rows per routing document | Visible K positions |
|---|---:|---:|---:|---:|
| Fit | 256 | 128 | 1 | 2048 |
| Validation | 64 | 32 | 1 | 2048 |

Payload therefore uses `256 × 2048 = 524,288` activation rows. Routing uses
the last-token Query from each selected document and the complete 2048-token
visible prefix. Per head, routing contains `128 × 2048 = 262,144` score
residuals, but only 128 independent Query vectors.

The direct LSQR capture for each layer stores:

| Tensor | R128 shape | BF16 size |
|---|---:|---:|
| Payload joint rows | `[524288, 32, 256]` | 8 GiB |
| Routing Queries | `[128, 32, 128]` | 1 MiB |
| Routing joint rows | `[128, 2048, 8, 256]` | 1 GiB |

The completed Routing-256 calibration kept the same 256 payload fit
documents, 64 payload validation documents, 32 routing validation documents,
and 2048-token windows. It changed only routing fit coverage from 128 to all
256 fit documents. Its direct routing tensors are
`routing_queries=[256,32,128]` and
`routing_joint_rows=[256,2048,8,256]`; the latter occupies 2 GiB in BF16.

## 5. Solver implementation and attempted variants

### 5.1 BCD order

The current update order is

\[
D\rightarrow U\rightarrow R\rightarrow A\rightarrow\text{gauge},
\]

followed by a deployment closure

\[
D\rightarrow U\rightarrow R\rightarrow\text{gauge}.
\]

The D step is an exact full-layer reduced solve. U is updated one Query head
at a time, R one physical KV group at a time, and A one physical KV group at
a time.

After an R update, thin QR gives

\[
R_g^{\mathrm{proposed}}=Q_gT_g,
\]

then `R_g <- Q_g` and `U_h <- U_h T_g^T` for all heads in the group. This
preserves `U_h R_g^T`. Joint-encoder gauge closure similarly preserves both
`A_gD_h` and `U_hR_g^TA_g^T`.

### 5.2 Original normal-equation CG

The first BCD implementation used matrix-free conjugate gradient on the
conditional normal equations. It used FP64 solves, exact objective
backtracking, and initially zero damping. Later runs fixed relative damping
at `1e-5`, used tolerance `1e-5`, and ran as many as 200 CG iterations.

This implementation exposed strongly coupled encoder blocks. One Layer 13
encoder group produced a relative CG residual of approximately `22.42`; a
backtracked step could still lower the fit objective despite the unusable
linear-system residual. Layer 7 also showed a small final-closure fit
increase of approximately `0.56%` in an earlier run.

Returning the minimum-residual CG iterate prevented a worse-than-zero
direction from being returned, but it did not repair the conditional system
or remove the normal-equation conditioning.

### 5.3 Jacobi-preconditioned CG

An exact Hessian-diagonal Jacobi preconditioner was applied to the
normal-equation CG. This improved several fit objectives and runtimes, but it
did not remove the strong non-diagonal coupling in the difficult encoder
group. It also made the Layer 13 fit/validation gap larger.

### 5.4 Direct matrix-free LSQR

The solver was rebuilt to operate on raw captured residual operands. LSQR
receives exact `J x` and `J^T y` callbacks and does not form `J^T J`. Every
block solves an increment problem:

\[
\min_{\delta}
\|J\delta+r\|_2^2+\lambda\|\delta\|_2^2,
\qquad X\leftarrow X+\delta,
\]

with

\[
\lambda=10^{-5}\operatorname{mean}(\operatorname{diag}(J^TJ)).
\]

The direct implementation preserved the original least-squares objective
while avoiding the squared conditioning of explicit normal equations.

### 5.5 LSQR systems for U and R

For one Query head, fixed A and R define

\[
B_g=A_gR_g\in\mathbb R^{256\times32}.
\]

The U direction has shape `128 × 32` and its Hessian action has the
Kronecker-sum form

\[
\mathcal H_U[Z]
=\sum_sG_{Q,h,s}Z(B_g^TG_{X,g,s}B_g).
\]

There are 32 independent U solves per U stage.

For one KV group, the R direction has shape `80 × 32`. Four Query heads are
fitted jointly, and the Hessian action is

\[
\mathcal H_R[Z]
=\sum_s(A_g^TG_{X,g,s}A_g)Z
\left(
\sum_{h\in\mathcal H_g}U_h^TG_{Q,h,s}U_h
\right).
\]

There are eight independent R solves per R stage.

### 5.6 Direct-LSQR performance implementations

The direct residual path was changed through the following measured
implementations:

1. CPU matrix-free LSQR over direct operands.
2. GPU batched BMM for routing products.
3. GPU-resident per-group payload data rather than repeated host transfers.
4. Large GEMM contractions and a 16,384-row payload chunk.
5. A 60-iteration cap instead of 100 for the optimized smoke runs.

The payload data are sliced to the four heads belonging to the current KV
group. The earlier implementation unnecessarily transformed all 32 heads for
each group.

### 5.7 Two-sided Kronecker right preconditioning

For a matrix least-squares Hessian

\[
\mathcal H(X)=\sum_sL_sXR_s,
\]

partial traces form the approximation

\[
L=\sum_sL_s\frac{\operatorname{Tr}(R_s)}{n},\qquad
R=\sum_sR_s\frac{\operatorname{Tr}(L_s)}{m}.
\]

After balanced Cholesky factors `L = L_c L_c^T` and
`R = R_c R_c^T`, LSQR solves in right-preconditioned coordinates

\[
X=L_c^{-T}YR_c^{-1}.
\]

The ridge term remains an explicit augmented residual block:

\[
\begin{bmatrix}JX\\\sqrt\lambda X\end{bmatrix},
\]

so preconditioning changes the iterative coordinates but not the fitted
damped least-squares objective.

The first implementation applied this preconditioner only to A. The second
also applied it to U and R using:

- U: a `128 × 128` Query-side factor and a `32 × 32` selected-latent factor;
- R: an `80 × 80` joint-latent factor and a `32 × 32` projected-Query factor.

An explicit small-matrix test verifies that the preconditioned augmented
LSQR solution equals the closed-form damped least-squares solution.

## 6. Experimental results

### 6.1 Original S80 and normal-equation CG results

All values below are normalized joint objectives. The five-sweep bank used
the original V64/KQ16 initialization; the 40-sweep rows used the later
R128/V32 calibration bank but still used the historical normal-equation
solver implementation.

| Variant | Layer | Sweeps | Initial fit | Final fit | Validation | Solver elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Original S80, zero-damping CG | 7 | 5 | 0.182364 | 0.140654 | 0.190734 | merged bank |
| Original S80, zero-damping CG | 13 | 5 | 0.203146 | 0.113613 | 0.259110 | merged bank |
| Damped CG | 7 | 40 | 0.183981 | 0.143163 | 0.159635 | 2786.2 s |
| Damped CG | 13 | 40 | 0.203933 | 0.124079 | 0.173710 | 874.6 s |
| Jacobi-PCG | 7 | 40 | 0.183981 | **0.116008** | **0.152470** | 1989.6 s |
| Jacobi-PCG | 13 | 40 | 0.203933 | **0.110471** | 0.205659 | 1085.1 s |

For the complete original five-sweep 36-layer bank, mean final fit was
`0.110670`, while mean validation was `0.238948`. The layerwise validation
range was `0.079969–0.335360`.

### 6.2 Direct-LSQR implementation results

| Variant | Layer | Iteration cap | Final fit | Validation | Solver elapsed |
|---|---:|---:|---:|---:|---:|
| CPU direct LSQR | 7 | 100 | 0.160771 | 0.206266 | 5610.8 s |
| CPU direct LSQR | 13 | 100 | 0.170088 | 0.275213 | 12164.3 s |
| GPU batched BMM | 7 | 100 | 0.160577 | 0.206615 | 1215.4 s |
| GPU batched BMM | 13 | 100 | 0.169878 | 0.276451 | 1218.1 s |
| GPU-resident group cache | 7 | 100 | 0.160577 | 0.206615 | 644.4 s |
| GPU-resident group cache | 13 | 100 | 0.169878 | 0.276451 | 644.2 s |
| Large GEMM direct LSQR | 7 | 60 | 0.166349 | 0.203083 | 295.7 s |
| Large GEMM direct LSQR | 13 | 60 | 0.179326 | 0.261124 | 296.6 s |
| A-only Kronecker preconditioner | 13 | 60 | **0.154266** | **0.226815** | 170.3 s |
| U/R/A Kronecker preconditioners | 13 | 60 | **0.153956** | **0.916995** | 170.1 s |
| U/R/A preconditioners, Routing-256 | 13 | 60 | 0.158341 | **0.202879** | 183.2 s |

The Layer 7 A-preconditioner smoke job failed before solving because the
allocated L40S already contained an unrelated 38.5 GiB process. The failure
was a device-allocation conflict rather than an LSQR failure.

At the time this document was written, the unpreconditioned 35-sweep jobs
were still running. Their completed sweep-19 fit snapshots were `0.126121`
for Layer 7 and `0.124471` for Layer 13. No final closure or validation value
existed for those jobs at that snapshot.

### 6.3 Iteration effects of preconditioning

Layer 13 used a 60-iteration maximum for every U, R, and A block.

| Stage | A-only preconditioner: total/min/max/converged | U/R/A preconditioners: total/min/max/converged |
|---|---:|---:|
| Sweep U, 32 heads | 1920 / 60 / 60 / 0 | 1289 / 29 / 52 / 32 |
| Sweep R, 8 groups | 480 / 60 / 60 / 0 | 129 / 14 / 19 / 8 |
| Sweep A, 8 groups | 203 / 14 / 60 / 7 | 202 / 13 / 60 / 7 |
| Final U, 32 heads | 1920 / 60 / 60 / 0 | 1482 / 33 / 60 / 29 |
| Final R, 8 groups | 480 / 60 / 60 / 0 | 121 / 12 / 18 / 8 |

The three nonconverged final U blocks were heads 29, 30, and 31. Their final
normal residuals were `2.42e-5`, `2.67e-5`, and `1.02e-5`. The difficult A
group remained group 3; with the A preconditioner it ended at 60 iterations,
normal residual `1.28e-3`, and LSQR condition estimate `269.85`.

With Routing-256, the same U/R/A-preconditioned solver produced:

| Stage | Total iterations | Min | Max | Converged blocks |
|---|---:|---:|---:|---:|
| Sweep U, 32 heads | 654 | 17 | 25 | 32 |
| Sweep R, 8 groups | 91 | 9 | 14 | 8 |
| Sweep A, 8 groups | 201 | 14 | 60 | 7 |
| Final U, 32 heads | 799 | 21 | 29 | 32 |
| Final R, 8 groups | 85 | 8 | 13 | 8 |

The fit job used 183.2 solver seconds and 191 seconds of Slurm elapsed time.

## 7. Routing-calibration failure exposed by converged U/R solves

The A-only and U/R/A-preconditioned Layer 13 runs started from the same
factors, used the same 128 routing fit documents, used the same 32 routing
validation documents, and used the same damping coefficient.

| Component | A-only preconditioner | U/R/A preconditioners |
|---|---:|---:|
| Final fit payload | 0.079513 | 0.079295 |
| Final fit routing | 0.074753 | 0.074662 |
| Final fit total | 0.154266 | 0.153956 |
| Validation payload | 0.098025 | 0.097752 |
| Validation routing | 0.128789 | **0.819244** |
| Validation total | 0.226815 | **0.916995** |

Payload behavior stayed stable. The entire validation failure came from the
routing component. The deployed factor norms also changed sharply:

| Factor statistic | A-only preconditioner | U/R/A preconditioners |
|---|---:|---:|
| `||U||_F` | 266.88 | **2069.69** |
| `max(abs(U))` | 6.625 | **121.0** |
| `||R||_F` | 16.00 | 16.00 |

`R` retains a fixed norm because its eight `80 × 32` selectors have
orthonormal columns. The large change is in the head-specific U factors.

The 128-document routing set supplies many score observations through the
2048-token Key prefixes, but it supplies only 128 independent Query vectors
per head. U itself has shape `128 × 32`. Increasing the number of Key
positions does not increase the span of the Query-side calibration matrix.
The unpreconditioned 60-step LSQR runs never reached the conditional optimum
and therefore supplied strong implicit regularization. Once the U/R
preconditioners allowed most blocks to converge, fit routing changed only
slightly while U acquired very large components outside directions well
constrained by the 128 calibration Queries. Those components produced the
observed routing validation increase.

The preconditioned solver was checked against an explicit damped
least-squares solution, so the measured behavior is consistent with solving
the current finite calibration objective more accurately rather than with a
changed ridge objective.

### 7.1 Measured Routing-256 result

Routing-256 used 256 independent last-token Query vectors, each paired with
its complete 2048-token prefix. The payload and validation splits were
unchanged.

| Component | Routing-128 U/R/A preconditioners | Routing-256 U/R/A preconditioners |
|---|---:|---:|
| Initial fit total | 0.313567 | 0.315857 |
| Final fit payload | 0.079295 | 0.079505 |
| Final fit routing | 0.074662 | 0.078836 |
| Final fit total | **0.153956** | 0.158341 |
| Validation payload | 0.097752 | 0.098015 |
| Validation routing | 0.819244 | **0.104864** |
| Validation total | 0.916995 | **0.202879** |
| `||U||_F` | 2069.69 | 1140.85 |
| `max(abs(U))` | 121.0 | 85.0 |

Doubling independent routing Queries increased the normalized fit objective
slightly while reducing validation routing by 87.2%. It also reduced final-U
work from 1482 total iterations (`46.31/head`) to 799 (`24.97/head`). All 32
final U blocks converged within 29 iterations.

## 8. Commands used for the recorded artifacts

All calibration and fit jobs used the `basis` Conda environment. Unit and
solver regression tests used the `lowrank` environment because `basis` did
not contain pytest or Ruff.

### 8.1 R128/V32 statistics collection

```bash
python scripts/collect_qwen3_gqa_joint_routing_payload_s80_stats.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --windows results/calibration/qwen3_8b_c4_256f64h_s2048/windows.safetensors \
  --output-dir results/calibration/qwen3_8b_s80_c4_p256v64_r128v32_s2048_l9_17 \
  --layers 9-17 --fit-start 0 --fit-windows 256 \
  --validation-start 256 --validation-windows 64 \
  --routing-fit-windows 128 --routing-validation-windows 32 \
  --sequence-length 2048 --batch-size 1 --model-dtype bfloat16 \
  --statistics-dtype float32 --device-map single \
  --max-memory-per-gpu-gib 44 --torch-num-threads 4
```

### 8.2 R128 direct residual capture

```bash
python scripts/capture_qwen3_gqa_joint_routing_payload_s80_direct.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --windows results/calibration/qwen3_8b_c4_256f64h_s2048/windows.safetensors \
  --output-dir results/calibration/qwen3_8b_s80_direct_256f_r128_s2048_l7_l13 \
  --layers 7,13 --fit-start 0 --fit-windows 256 \
  --routing-fit-windows 128 --sequence-length 2048 --batch-size 1 \
  --model-dtype bfloat16 --device-map single \
  --max-memory-per-gpu-gib 42 --torch-num-threads 4
```

### 8.3 Historical Jacobi-PCG Layer 13 run

```bash
python scripts/build_qwen3_gqa_joint_routing_payload_s80.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --stats-dir results/calibration/qwen3_8b_s80_c4_p256v64_r128v32_s2048_l9_17 \
  --c1-r64-init results/checkpoints/qwen3_8b_c1_v64_als5 \
  --kq-r16-init results/checkpoints/qwen3_8b_post_rope_kqsvd_r64_c4_128f64h \
  --output-dir results/checkpoints/qwen3_8b_s80_rw1_s40_pcg_rd1e5_l13 \
  --layers 13 --joint-rank 80 --payload-init-rank 64 \
  --routing-init-rank 16 --routing-weight 1 --min-sweeps 40 \
  --max-sweeps 40 --relative-objective-tolerance 1e-5 \
  --cg-max-iterations 200 --cg-tolerance 1e-5 \
  --cg-relative-damping 1e-5 --decoder-jitter 0 \
  --work-dtype float64 --factor-dtype bfloat16 \
  --work-device cpu --torch-num-threads 32
```

This command is historical and records the command line accepted by that
checkpoint's builder version.

### 8.4 Optimized direct-LSQR Layer 13 run

```bash
python scripts/build_qwen3_gqa_joint_routing_payload_s80.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --stats-dir results/calibration/qwen3_8b_s80_c4_p256v64_r128v32_s2048_l9_17 \
  --direct-capture-dir results/calibration/qwen3_8b_s80_direct_256f_r128_s2048_l7_l13 \
  --c1-v80-init results/checkpoints/qwen3_8b_c1_v80_als5 \
  --kq-r32-init results/checkpoints/qwen3_8b_post_rope_kqsvd_r64_c4_128f64h \
  --output-dir results/checkpoints/qwen3_8b_s80_r32_v80_direct_lsqr_gemm_i60_l13_smoke \
  --layers 13 --max-sweeps 1 --min-sweeps 1 \
  --iterative-max-iterations 60 --iterative-tolerance 1e-5 \
  --relative-damping 1e-5 --work-dtype float64 \
  --work-device cuda --torch-num-threads 8
```

### 8.5 A-only Kronecker-preconditioned Layer 13 run

```bash
python scripts/build_qwen3_gqa_joint_routing_payload_s80.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --stats-dir results/calibration/qwen3_8b_s80_c4_p256v64_r128v32_s2048_l9_17 \
  --direct-capture-dir results/calibration/qwen3_8b_s80_direct_256f_r128_s2048_l7_l13 \
  --c1-v80-init results/checkpoints/qwen3_8b_c1_v80_als5 \
  --kq-r32-init results/checkpoints/qwen3_8b_post_rope_kqsvd_r64_c4_128f64h \
  --output-dir results/checkpoints/qwen3_8b_s80_r32_v80_direct_lsqr_kronpc_l13_smoke \
  --layers 13 --max-sweeps 1 --min-sweeps 1 \
  --iterative-max-iterations 60 --iterative-tolerance 1e-5 \
  --relative-damping 1e-5 --work-dtype float64 \
  --work-device cuda --torch-num-threads 8
```

### 8.6 U/R/A Kronecker-preconditioned Layer 13 run

```bash
python scripts/build_qwen3_gqa_joint_routing_payload_s80.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --stats-dir results/calibration/qwen3_8b_s80_c4_p256v64_r128v32_s2048_l9_17 \
  --direct-capture-dir results/calibration/qwen3_8b_s80_direct_256f_r128_s2048_l7_l13 \
  --c1-v80-init results/checkpoints/qwen3_8b_c1_v80_als5 \
  --kq-r32-init results/checkpoints/qwen3_8b_post_rope_kqsvd_r64_c4_128f64h \
  --output-dir results/checkpoints/qwen3_8b_s80_r32_v80_kronpc_ura_l13_smoke \
  --layers 13 --max-sweeps 1 --min-sweeps 1 \
  --iterative-max-iterations 60 --iterative-tolerance 1e-5 \
  --relative-damping 1e-5 --work-dtype float64 \
  --work-device cuda --torch-num-threads 8
```

This run was Slurm job `8288740` on an L40S. Slurm allocated physical GPUs
2 and 3; the process was explicitly restricted to GPU 3 because GPU 2 held
an unrelated 38.5 GiB process.

### 8.7 Routing-256 collection and fit

```bash
python scripts/collect_qwen3_gqa_joint_routing_payload_s80_stats.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --windows results/calibration/qwen3_8b_c4_256f64h_s2048/windows.safetensors \
  --output-dir results/calibration/qwen3_8b_s80_c4_p256v64_r256v32_s2048_l13 \
  --layers 13 --fit-start 0 --fit-windows 256 \
  --validation-start 256 --validation-windows 64 \
  --routing-fit-windows 256 --routing-validation-windows 32 \
  --sequence-length 2048 --batch-size 4 --model-dtype bfloat16 \
  --statistics-dtype float32 --device-map single \
  --max-memory-per-gpu-gib 44 --torch-num-threads 4
```

```bash
python scripts/capture_qwen3_gqa_joint_routing_payload_s80_direct.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --windows results/calibration/qwen3_8b_c4_256f64h_s2048/windows.safetensors \
  --output-dir results/calibration/qwen3_8b_s80_direct_256f_r256_s2048_l13 \
  --layers 13 --fit-start 0 --fit-windows 256 \
  --routing-fit-windows 256 --sequence-length 2048 --batch-size 4 \
  --model-dtype bfloat16 --device-map single \
  --max-memory-per-gpu-gib 44 --torch-num-threads 4
```

```bash
python scripts/build_qwen3_gqa_joint_routing_payload_s80.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --stats-dir results/calibration/qwen3_8b_s80_c4_p256v64_r256v32_s2048_l13 \
  --direct-capture-dir results/calibration/qwen3_8b_s80_direct_256f_r256_s2048_l13 \
  --c1-v80-init results/checkpoints/qwen3_8b_c1_v80_als5 \
  --kq-r32-init results/checkpoints/qwen3_8b_post_rope_kqsvd_r64_c4_128f64h \
  --output-dir results/checkpoints/qwen3_8b_s80_r32_v80_kronpc_ura_r256_l13_smoke \
  --layers 13 --max-sweeps 1 --min-sweeps 1 \
  --iterative-max-iterations 60 --iterative-tolerance 1e-5 \
  --relative-damping 1e-5 --work-dtype float64 \
  --work-device cuda --torch-num-threads 8
```

These were Slurm jobs `8288755`, `8288756`, and `8288757` on an L40S. Their
elapsed times were 88, 92, and 191 seconds. The two A100-80GB submissions
`8288753` and `8288754` remained pending for priority and were cancelled
before execution; they created no output artifacts.

### 8.8 Solver regression tests

```bash
python -m pytest -q \
  tests/test_iterative_least_squares.py \
  tests/test_gqa_joint_routing_payload_s80_stats.py \
  tests/test_gqa_joint_routing_payload_s80_solver.py \
  tests/test_gqa_joint_routing_payload_s80_qwen3.py
```

Result: `20 passed in 8.29s`. After the U/R preconditioner and explicit
damped-objective regression test were added, the focused solver file reported
`11 passed`; Ruff reported no findings.

## 9. Implementation and result artifacts

Core implementation:

- `basisserve/core/gqa_joint_routing_payload_s80.py`
- `basisserve/core/iterative_least_squares.py`
- `basisserve/calibration/gqa_joint_routing_payload_s80_stats.py`
- `basisserve/checkpoint/gqa_joint_routing_payload_s80_qwen3.py`
- `scripts/collect_qwen3_gqa_joint_routing_payload_s80_stats.py`
- `scripts/capture_qwen3_gqa_joint_routing_payload_s80_direct.py`
- `scripts/build_qwen3_gqa_joint_routing_payload_s80.py`

Primary result manifests:

- `results/checkpoints/qwen3_8b_s80_rw1_s5_c4_256f64h/manifest.json`
- `results/checkpoints/qwen3_8b_s80_rw1_s40_pcg_rd1e5_l7/manifest.json`
- `results/checkpoints/qwen3_8b_s80_rw1_s40_pcg_rd1e5_l13/manifest.json`
- `results/checkpoints/qwen3_8b_s80_r32_v80_direct_lsqr_gemm_i60_l7_smoke/manifest.json`
- `results/checkpoints/qwen3_8b_s80_r32_v80_direct_lsqr_gemm_i60_l13_smoke/manifest.json`
- `results/checkpoints/qwen3_8b_s80_r32_v80_direct_lsqr_kronpc_l13_smoke/manifest.json`
- `results/checkpoints/qwen3_8b_s80_r32_v80_kronpc_ura_l13_smoke/manifest.json`
- `results/checkpoints/qwen3_8b_s80_r32_v80_kronpc_ura_r256_l13_smoke/manifest.json`
