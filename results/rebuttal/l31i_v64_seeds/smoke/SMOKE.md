# Smoke test (2026-11-12 EST)

`scripts/run_v64_seeds.sh llama31_8b_instruct launch "42"` with
`FIT_WINDOWS=16 HELDOUT_WINDOWS=8 KL_PROFILE_WINDOWS=16 KL_CONFIRMATION_WINDOWS=16 MCQ_LIMIT=10`; conda env `basis`,
one A100 per job. Kept deliberately small: perplexity end to end and ten questions per MCQ task. The fitted models
are meaningless (16 calibration windows). Heavy artifacts were deleted; logs are in `logs/`.

| Check | Outcome |
| --- | --- |
| Dense, BasisKV, PaLU M-LRD and PaLU G-LRD4 load and score WikiText-2 (288,627 tokens) and C4 | pass |
| Dense perplexity against the paper | 7.2165 / 9.7733 here, 7.2164 / 9.7740 in the paper |
| Ten examples per MCQ task are scored and written for all four runs | pass |
| C4-validation bank regenerates the paper's bank | pass (sha256 `3fe79cbd…`) |
| BasisKV banks use damping 0, six sweeps, 16 CG iterations; allocation uses 8 KL shards | pass |

Two attempts failed before the passing one, both from setup rather than the pipeline: the model cache had been moved
without its shared blob store (re-downloaded to `/data/hf/hub`), and the interrupted covariance capture left a
`cov.partial` directory that the capture refuses to overwrite (the stage now removes it before retrying).
