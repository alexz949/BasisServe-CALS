# Qwen3.5-9B gated V ALS and frozen-V Private AllGather

This experiment implements the uploaded two-stage specification. Results are
under `results/q35_hybrid`; completion is determined from actual checkpoint and
evaluation artifacts, not this document. No Git upload has been authorized.

## Fixed protocol

- Model: `Qwen/Qwen3.5-9B`, revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Environment: `lowrank`; process-local dependencies in
  `results/q35_hybrid/deps` provide Transformers 5.3.0 without changing other
  jobs' environment packages.
- Hardware: this host's A100 GPUs, selected with `nvidia-smi`. The user explicitly
  authorized local execution without Slurm and permits occupied devices when
  available memory is sufficient. Two CPU threads per process.
- C4: fit 256×2048, held-out 64×2048, KL profile 128×2048, confirmation 16×2048;
  document-disjoint within calibration, sampled with seed 20260909 from the
  recorded local C4 train Arrow shard. Identical fit token IDs for C1 and PaLU.
- PPL: full WikiText-2 raw test, including the final short block; independently
  sampled C4 validation 128×2048. Ordinary one-token-shift NLL, token-weighted.
- V ranks: 64, 80, 96 of native dimension 256, on full-attention layers
  3, 7, 11, 15, 19, 23, 27, 31. K and GDN recurrent state remain native.
- C1 Stage A: six encoder sweeps, final decoder refit, FP32 work, BF16 factors,
  relative damping 1e-5, linear tolerance 1e-5, cap 200 total CG iterations
  per block including residual-replacement corrections. Held-out selection
  uses decoder-closed endpoints only. Quantized factors are reevaluated.
- Stage A damping is a proximal update centered on the current factor. Its
  scale is the current-factor Rayleigh quotient, with a dtype-epsilon floor.
  It is explicitly different from an unconditional zero-centered ridge.
- Encoder preconditioning uses pooled per-group input covariance and the
  gate-aware decoder Gram. Decoder preconditioning uses a positive Jacobi
  approximation. The actual operator retains all cross-group interactions.
- Two-sided KL: alpha 1.25, probes anchor−32 and anchor+32, candidate ranks
  32/48/64/80/96/112/128 plus exact native 256. Exact mean-rank allocation over
  the eight full-attention layers. Each bank is fitted independently; no
  slicing an ALS bank to invent lower-rank candidates.
  The 51 anchor/probe configurations are sharded across up to eight available
  GPUs; sharding preserves every complete 128-window KL measurement exactly once.
- PaLU: MLRD/GLRD2/GLRD4 group physical KV heads by 1/2/4. Use the existing
  official Fisher allocator including block32 rounding and report realized
  ranks. Fisher uses the repository's official-PaLU double-label-shift loss;
  this is **not** the loss used for PPL. Only PaLU Fisher uses backward, with
  no optimizer. Whitening uses all 256×2048 original V-projection inputs.
- Stage B: only after reviewing V-only PPL, recapture on each entire frozen V
  bank. TP4, half the post-gate width: 512 coordinates per source, 2048 total.
  Reuse `fit_qwen35_private_ag_joint_factors` with six sweeps. Both GDN and
  full-attention outputs are included. Every Wo bank pins its upstream V hash.
  Stage B uses FP64 work and BF16 export, retaining the existing covariance
  damping of 1e-5; native moments failed the solver's FP32 SPD criterion.
  Banks advance independently after their own complete V-only PPL review;
  other V ranks can continue fitting while that frozen bank enters Stage B.
- Deliverables: six C1 V-only banks, six corresponding V+Wo combinations,
  nine PaLU banks, and both PPL evaluations for every configuration plus dense.
  Internal rank candidates and calibration statistics are not extra final arms.

## Implementation

- `basisserve/core/qwen35_gated_v_als.py`: chunked joint apply/adjoint, exact
  fixed `W W^T` normal representation, initialization, proximal PCG, QR gauge,
  true-residual diagnostics and held-out endpoint selection.
- `basisserve/core/qwen35_gated_v_runtime.py`: native Q/gate split, Q/K norm
  and partial RoPE; compact V cache; query-output reconstruction before gate;
  grouped PaLU cache support; schema/hash checks, fork/reset/reorder support.
- `evaluation/run_qwen35_hybrid.py`: real-layer smoke, immutable captures,
  layer/rank fitting, dense/V-only/composed PPL.
- `evaluation/qwen35_hybrid_palu.py`: matched Fisher, whitening, grouped PaLU
  factor export. Balanced SVD products are obtained from the left Gram of
  whitened W using the existing PaLU Cholesky treatment.
- `evaluation/qwen35_hybrid_banks.py`: uniform bank assembly, teacher hidden
  capture, full-vocabulary two-sided KL probes and exact-budget assembly;
  independent confirmation compares the frozen choices on 16 held-out windows.
- `evaluation/qwen35_hybrid_wo.py`: frozen-V moment capture and existing Wo ALS.
- C1 PPL and Stage B recapture verify the config and every original model
  shard against the V bank's recorded identities before loading the model.
- `evaluation/qwen35_hybrid_local_errors.py`: common-input V, conditional Wo,
  and composed local errors, preserving their cross term. These use native
  dense inputs and are distinct from Stage B's frozen-V trajectory statistics.
- `basisserve/core/qwen35_hybrid_output_runtime.py`: validates upstream V hash
  and composes the existing GDN/full-attention Private AG adapters.

The Qwen3.5 Wo runtime is a single-process AllGather equivalent. There are no
claims of measured distributed communication speedup. Latent attention uses
query-chunked SDPA with unequal Q/K and V widths; it never reconstructs the
historical V cache. The native GDN implementation currently uses the PyTorch
path because optional FLA/causal-conv libraries are unavailable.

## Commands

Every process uses this prefix, with the physical GPU chosen per job:

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export PYTHONPATH=results/q35_hybrid/deps:.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=2
```

Fit one actual layer (full fixed calibration protocol):

```bash
python -u -m evaluation.run_qwen35_hybrid fit \
  --layers 3 --ranks 64,80,96,32,48,112,128 \
  --encoder-preconditioner separable --chunk-rows 2048 --linear-max-iter 200 \
  --output results/q35_hybrid/factors
```

Assemble and evaluate a uniform bank once all eight layer factors exist:

```bash
python -m evaluation.qwen35_hybrid_banks assemble --uniform --anchor 64 \
  --output results/q35_hybrid/banks
python -m evaluation.run_qwen35_hybrid evaluate \
  --bank results/q35_hybrid/banks/c1_uniform_v64.pt \
  --output results/q35_hybrid/c1_uniform_v64_ppl.json
```

KL profile and assembly:

```bash
python -m evaluation.qwen35_hybrid_banks profile --anchor 64 \
  --output results/q35_hybrid/kl
python -m evaluation.qwen35_hybrid_banks assemble --anchor 64 \
  --output results/q35_hybrid/banks
python -m evaluation.qwen35_hybrid_banks confirm --anchor 64 \
  --output results/q35_hybrid/kl
```

Stage B, after the V-only gate is reviewed:

```bash
python -m evaluation.qwen35_hybrid_wo capture \
  --bank results/q35_hybrid/banks/c1_uniform_v64.pt \
  --output results/q35_hybrid/wo_uniform_v64_moments
python -m evaluation.qwen35_hybrid_wo fit \
  --bank results/q35_hybrid/banks/c1_uniform_v64.pt \
  --moments results/q35_hybrid/wo_uniform_v64_moments \
  --output results/q35_hybrid/wo_uniform_v64 --work-dtype float64
python -m evaluation.run_qwen35_hybrid evaluate \
  --bank results/q35_hybrid/banks/c1_uniform_v64.pt \
  --wo-bank results/q35_hybrid/wo_uniform_v64/wo_bank.pt \
  --output results/q35_hybrid/c1_uniform_v64_wo_ppl.json
```

Use each anchor and both C1 schedule types. Output artifacts are created
atomically without overwriting existing files. Inspect logs and completed
artifacts before resuming. Do not treat an existing process or a partial bank
as evidence that the experiment has finished.

After every requested arm and independent KL confirmation has completed:

```bash
python -m evaluation.summarize_qwen35_hybrid \
  --output results/q35_hybrid/final_summary.json
```

This audit checks all 21 compressed configurations plus dense, actual sweep
histories, upstream V identities, common-input local errors, and original model
files. Missing results prevent it from writing a completion summary.

## Numerical audit findings

The initial Jacobi-only encoder preconditioner hit the 200-iteration cap with
large residuals. A same-subproblem comparison justified switching to a
separable encoder preconditioner; original logs remain available. True residual
replacement stays within the original 200-iteration cap. Loss-increasing
blocks are rejected and diagnostics distinguish rejection, cap and convergence.

An existing Wo ALS test exposed an FP32-rounding discrepancy between the ideal
proposed encoder step and the stored step in an FP64 objective verification.
`gqa_routed_ov_joint.py` now evaluates the quadratic change for the actually
stored update in the loss's precision. The verification tolerance was not relaxed.
Its descent guard now scales with the gradient/curvature terms rather than
an absolute unit floor, which incorrectly rejected valid updates for native
small-weight magnitudes. Six-sweep tests cover both unit and 0.02 weight scales.

The first formal Wo fit rejected a source covariance in FP32: its smallest
eigenvalue was 9.428024e-7, below the existing numerical SPD threshold of
1.938958e-4. Stage B was rerun in FP64 using the same solver, objective,
covariance damping and six sweeps. The precision-aware SPD criterion and its
multiplier are unchanged. The failed attempt remains in the appended fit log.

The capture attention registry also registers the native SDPA mask builder.
Padded native/captured forwards agree in tests; formal calibration uses only
full unpadded windows, whose original causal path was already correct.

Export-metric normalization now streams the target squared norm in row chunks.
The original whole-target FP64 conversion and square temporarily required
about 32 GiB beyond live captures. Layer 7 was restarted before export to avoid
that peak on shared GPU1; its original log is retained by appending the resumed
run. Fitting math, six sweeps, calibration and completed factors are unchanged.
Exact PID/session provenance is in `results/q35_hybrid/metric_memory_restart.json`.

An optional TF32x3 diagnostic gave only about 1.32× GEMM speedup. It is not
enabled in the production fitter; normal matrix multiplication remains FP32.
