# ShadowKV TP8 Request Benchmark

Grid status: `complete`. Trial statuses: `{"complete": 108, "failed": 0, "failed_oom": 0, "not_run": 0}`.

Three frozen real-text cohorts per method/context/batch/mode are required for a group result.
Request components come from the cohort with median total request time, so the stacked bars add exactly.
The request generates 128 tokens in total; steady decode measures 128 steps after 16 conditioning steps.
Every rank uses a common host monotonic clock and synchronized GPU phase boundaries.
The separate ShadowKV construction category includes gather, SVD, redistribution, and official preparation.
BasisKV projection, encoding, and cache writes remain inside prefill and total request time.
The result is an instrumented synchronized request; timing instrumentation overhead is included.

## Provenance

- Hardware: 8 x NVIDIA L40S, TP8 / DP1 / PP1; topology in `docs/hardware/l40s_tp8_topology.md`.
- Environment: `basis`; BF16, TF32 off, eager; CPU affinity policy from the frozen runner.
- ShadowKV: rank 160, chunk 8, sparse budget 2048, owner layer % 8, pinned CPU V.
- BasisKV: frozen Joint-ALS V96 / B16R16 / Page32 / support 2048 full-scan path.
- Inputs: frozen `p{length}_c{cohort}` real-text windows and the pinned Llama-3.1-8B-Instruct revision.
- Full trial commands, failures, timestamps, and output paths: `grid_trials.json`.
- Dense is the repository's matched TP8 eager control, not a tuned serving engine.
- Code base commit: `bae46c1409d3fa00030bab88df8b359e20b886bf`; worktree dirty: `True`.
- Model path/revision: `/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`.
- ShadowKV source commit (when that arm runs): `e51904cdeab7d4d34013370f09f2cf5fcd655e15`.
- PyTorch `2.13.0+cu130`, CUDA `13.0`, NCCL `[2, 29, 7]`, Transformers `5.17.0`, FlashAttention `2.8.3.post1`.

## Outputs

- `summary.csv`: every planned trial, including missing and failed configurations.
- `request_breakdown.csv`: one additive representative request per complete three-cohort group.
- `decode_summary.csv`: three-cohort medians from independent 16+128 windows.
- `memory_summary.csv`: median max-rank allocated/reserved/NVML snapshots; active state bytes remain in raw rank JSON.
- `failures.csv`: every failed attempt, including earlier failures after a successful retry.
- Primary request plot generated: `True`.

No number from the 4K smoke directories is a formal benchmark result.
OOM during prefill must be reported as prefill OOM, not decode-cache capacity.
