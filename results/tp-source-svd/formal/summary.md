# Full-Covariance TP-Source SVD

Phase: formal. Model: Qwen/Qwen3-8B-Base.
TP source partition: 8; source ranks: [256, 384].
Free post-attention source encoders; dense Q/K/V and KV cache. Not V64/V96.
Zero ridge; support-truncated FP64 optimum audited separately from BF16 export.
PPL uses FP64-materialized weights cast once to BF16, not BF16 factor execution.

## Commands

```bash
evaluation/run_tp_source_svd.py --stage fit --phase formal --tp-size 8 --ranks 256 384 --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --output results/tp-source-svd/formal
```
Conda environment: basis. All three exact launch commands and log paths are
recorded in [the protocol](../../../docs/tp_source_svd.md#commands).

## Results

| Arm | PPL | Scored tokens | Status |
| --- | ---: | ---: | --- |
| dense | 7.00217797 | 298862 | complete |
| r256 | 7.41655973 | 298862 | complete |
| r384 | 7.04550913 | 298862 | complete |

Compared with Dense, r256 increases PPL by 0.41438176 (+5.9179%); r384
increases PPL by 0.04333116 (+0.6188%). All 36 layers are fitted and replaced.
Each arm uses the same 146 complete 2048-token windows; 70 tail tokens are
discarded, with no complete windows omitted.

## Verification

All three formal stages completed successfully. Calibration used 256 C4 train
windows, totaling 524288 rows. All 72 layer/rank fits and 576 source audits
passed; all covariance supports have dimension 512, with no discarded positive
energy. Maximum tail-energy audit error/tolerance: 2.843e-5. Maximum full-support
reconstruction loss/target energy: 8.792e-26. Zero ridge throughout.

14 unit tests, synthetic GPU smoke and real first-layer smoke passed. Recorded
stage work times: capture 555.21 s, fit 214.98 s, PPL 92.15 s. Capture/PPL times
exclude their initial data/model preparation. Exact commands are also stored
in each stage status JSON. No SHA256 checks were performed.

No joint-vs-independent claim is made: no matched joint factors are fitted here.
Synthetic smoke is not real-model PPL; historical TP4 results are not substituted.
Held-out statistics were not collected, per user request; fit errors are not generalization evidence.
See fit_metrics.json for local/final fit errors, denominators and rounding diagnostics.
Inputs use snapshot identity, structure and direct weight equality checks, not SHA256.
