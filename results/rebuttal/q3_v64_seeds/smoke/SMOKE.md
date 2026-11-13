# Smoke test (2026-11-13)

Small end-to-end run of `scripts/launch_q3_v64_seeds.sh` (now `scripts/run_v64_seeds.sh qwen3_8b launch`) with calibration seeds 20260821 and 20260822,
`FIT_WINDOWS=16 HELDOUT_WINDOWS=8 KL_PROFILE_WINDOWS=16 KL_CONFIRMATION_WINDOWS=16`; conda env `basis`, one A100 per job.
The fitted models are meaningless (16 calibration windows); only the pipeline was checked. Heavy artifacts were deleted,
logs are in `logs/`.

| Check | Outcome |
| --- | --- |
| BasisKV, PaLU M-LRD and PaLU G-LRD4 factors load and evaluate | pass (WikiText-2/C4 PPL for all three, both seeds) |
| Calibration seeds change the window hashes | pass (all banks differ; zero shared documents between seeds) |
| Methods of one seed share the calibration data | pass (whitening and Fisher record the same window hash; the 256 PaLU windows are the verified prefix of the 320 BasisKV windows) |
| Evaluation inputs identical in every run | pass (one C4-validation bank, sha256 `31df55d6...`, equal to the paper's) |
| Accuracy recomputed from the example records equals the evaluator | pass (enforced inside the evaluator; difference 0.0 on all seven tasks) |
| Example IDs pair across methods and seeds | pass (ID and document hash identical for Dense, BasisKV, M-LRD, G-LRD4 on OpenBookQA and for two seeds on all 20,465 examples) |
| Paired bootstrap of a result against itself | pass (difference exactly 0 in all 10,000 replicates) |
| Resume cannot read another seed's results | pass (stages of seed 20260821 are rejected for the seed-20260822 checkpoint; nothing is written) |

Found and fixed: the evaluator's GPU check read `hf_device_map`, which a single-GPU load does not set, so the first Dense
run stopped before evaluating. It now reads the parameter devices.

The full-protocol evaluations of BasisKV and G-LRD4 and the Dense run were stopped on request once the checks above
had what they needed; the cross-method pairing check used OpenBookQA only.
