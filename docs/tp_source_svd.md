# Full-Covariance TP-Source SVD

## Scope

Implementation of the uploaded Full-Covariance TP-Source SVD handoff. The user
confirmed **Qwen3-8B-Base**, P=8, source ranks 256 and 384. The user subsequently
changed calibration to **newly collected 256 fit windows and no held-out set**.
Do not substitute historical TP4 or shared-Value-encoder quality results.

Each source concatenates four contiguous post-attention heads: width 512.
Free encoders `[8,512,r]` and decoders `[8,r,4096]` mix heads within each source.
They do not compress V or the KV cache. TP8 is a mathematical partition here,
not a distributed quality forward. Q/K/V, attention and MLP stay dense.

## Implementation

- `basisserve/core/tp_source_svd.py`: FP64 support-eigendecomposition and weighted SVD, zero ridge. Full 512 x 512 within-source second moments retain cross-head terms.
- `evaluation/run_tp_source_svd.py`: `capture`, `fit`, `ppl` stages. Standalone fitting requires no joint checkpoint. Geometry derives from the TP layout; ranks and TP size are CLI arguments.
- Existing TP4 independent/joint comparison entry points remain untouched because other analyses import their helpers. This is a separate protocol, not a compatibility wrapper.
- The shared `_eval_ppl_fp32_loss` accepts frozen tokens, avoiding different tokenization across arms. Its original token-loading path remains available to other experiments.

For `C=Q diag(s) Q.T`, keep `s > max(atol, rtol*max(abs(s)))`. Defaults:
support rtol 1e-12, atol 0, negative-eigenvalue tolerance 1e-10 relative to
spectral scale. Materially negative spectra are rejected, never ridge-repaired.
Tiny discarded negatives, discarded positive eigenvalues and their residual
energy are recorded. The exact-optimum claim applies to the retained metric;
original-metric loss is reported separately.

Every source records SVD tail energy, reconstructed FP64 retained-metric loss,
audit tolerance/error, support and effective ranks, finite factor shapes, and
the full-support SVD reconstruction error. Singular support is zero-padded to
the requested factor shape. Complete source-rank endpoints need not reproduce
weight components in the covariance nullspace.

The primary PPL replacement is formed as FP64 `E @ D`, assembled in input-source
order, transposed to PyTorch weight layout and cast **once** to BF16. Separate
diagnostics report BF16 factor-rounding error and BF16 intermediate execution
on deterministic probes. These paths are not conflated.

For residual R in input/output layout and C=Z.T@Z/N, local and complete-output
losses are per-row quadratic energies; raw SSE is energy times N. Relative
loss divides by the corresponding dense target energy. Cross-source error is
computed from equally normalized energies. No held-out loss is reported in
this run; fit error is not evidence of generalization or better PPL.

## Calibration and Quality

C4 train, revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, English, seed
20260821, shuffle buffer 10000. Select 256 distinct document URLs with at least
2048 tokens, one uniformly positioned window per document. Missing URLs are
skipped, not replaced with hashes. This is a newly sampled bank, not a claim
of bitwise identity to an unavailable older Section 3 bank. Token windows,
document IDs and offsets are stored. Capture uncentered FP64 second moments
at the dense `o_proj` input, normalized by all fit rows; no centering or ridge.

Identity checks use exact model snapshot paths/IDs, model geometry, tokenizer
from that same snapshot, calibration metadata and direct equality of each
stored dense weight against the model shard. No SHA256 computation/checking.

Quality uses BF16+SDPA, B1, complete non-overlapping WT2 test windows of 2048.
Cross-entropy reduction is FP32, with summed NLL, scored tokens, available
tokens, complete chunks and discarded tail recorded. Identical token IDs are
saved once and reused. Before every arm, all `o_proj` weights are restored
from their original dense copies; successive approximations are never composed.

## Commands

Working directory `/workspace/BasisServe-CALS`; environment `basis`.
No Slurm configuration is available on this host. The user approved all three
formal stages after smoke; all completed under the direct-run agreement.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/run_tp_source_svd.py --stage capture --phase formal --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --output results/tp-source-svd/formal > results/tp-source-svd/capture_formal.log 2>&1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/run_tp_source_svd.py --stage fit --phase formal --tp-size 8 --ranks 256 384 --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --output results/tp-source-svd/formal > results/tp-source-svd/fit_formal.log 2>&1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/run_tp_source_svd.py --stage ppl --phase formal --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --output results/tp-source-svd/formal > results/tp-source-svd/ppl_formal.log 2>&1
```

Real smoke uses these stages with `--phase smoke`, output `results/tp-source-svd/smoke`,
and separate `*_smoke.log` files. It captures only 2 windows and layer 0, fits
the true source geometry and ranks, then evaluates only two WT2 chunks with
only layer 0 compressed. It is **not** full-model compression quality.

Synthetic integration smoke:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python tests/smoke_tp_source_svd.py --output results/tp-source-svd/synthetic --device cuda:0
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest tests/test_tp_source_svd.py -q
```

## Artifacts

`config.json`, `fit_metrics.json`, `factors/r*/layer_*.safetensors` (FP64 factors
and primary BF16 weight), `ppl_tokens.safetensors`, `ppl.json`, `summary.md`,
stage status records and logs. Calibration lives under `covariance/`.
Completed artifacts are preserved; failure status is recorded separately.
Do not publish smoke numbers as formal PPL. No joint objective ablation is
claimed without a separately fitted matching free-source joint baseline.

## Formal Results

Completed 2026-09-26 in `basis`, GPU 0 (L40S), using the commands above.
All 36 layers were replaced for each compressed arm. The same 146 complete
WT2 test windows yielded 298862 scored next-token targets per arm; 70 trailing
tokens were discarded. No complete windows were omitted.

| Arm | Source rank / width | WT2 PPL | Delta vs Dense | Relative increase |
| --- | --- | ---: | ---: | ---: |
| Dense | 512 / 512 | 7.00217797 | 0 | 0% |
| Independent SVD | 256 / 512 | 7.41655973 | +0.41438176 | +5.9179% |
| Independent SVD | 384 / 512 | 7.04550913 | +0.04333116 | +0.6188% |

Rank 384 preserves PPL much more closely than rank 256 in this experiment.
These are materialized-weight quality results, not factor-runtime speed or KV
cache compression measurements, and not a comparison against joint fitting.

Verification: 14 unit tests, synthetic GPU smoke and real-model first-layer
smoke passed. All 72 layer/rank records and 576 source optimality audits passed.
Every source retained all 512 covariance dimensions; discarded positive energy
was zero. Maximum tail-energy audit error/tolerance was 2.843e-5; maximum
full-support reconstruction loss/target energy was 8.792e-26. No ridge was used.

Recorded stage work times: capture 555.21 s (after data/model preparation),
fit 214.98 s, and three-arm PPL 92.15 s (after data/model preparation).
All stages exited successfully. No held-out covariance was collected by user
request. Capture's unexpected `lm_head.weight` loading notice is expected for
`AutoModel`; evaluation uses `AutoModelForCausalLM`.

Raw results: `results/tp-source-svd/formal/ppl.json` and `fit_metrics.json`.
The generated summary is `results/tp-source-svd/formal/summary.md`.
Logs are `results/tp-source-svd/{capture,fit,ppl}_formal.log`; pre-run source
archive is `results/tp-source-svd/source.tar.gz`. Code and the PPL summary are
included in the server-backup GitHub publication. Large-artifact HF backup is
not complete yet; see `results/shutdown-audit/SUMMARY.md` before deleting local data.
