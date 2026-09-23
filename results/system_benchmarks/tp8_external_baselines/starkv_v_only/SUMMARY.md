# STAR-KV V-only TP8 Memory Benchmark

Grid status: `complete_with_failures`. Trial statuses: `{"complete": 35, "failed_oom_cache_state_allocation": 1}`.

Qwen3-8B-Base, TP8 / DP1 / PP1, BF16, exact dense K, full Flash SDPA attention,
fixed 4096-token chunked prefill, one decode step after the full prompt.
One preselected Qwen-tokenized LongBench-v2 cohort (cohort 0, eight real-text requests) is used per configuration.
The primary decode-ready GPU-resident metric is the maximum NVML process memory across the eight TP ranks for that single trial; no cohort median or repeatability estimate is available.
PyTorch allocated/reserved and prefill/decode peaks are reported separately.
Actual STAR V retention is 54.0473%, not a strict 50% result; layers 0, 1, and 31 use dense local V.
STAR's global V latent is replicated per TP rank; Basis V64 is source-local.
The 65536+ contexts exceed Qwen3's native 32768-token context and are memory-only stress tests, not quality claims.
No CPU V offload, K routing, K compression, or substituted checkpoint is used.

## Provenance

- Hardware: 8 x NVIDIA L40S; topology in `docs/hardware/l40s_tp8_topology.md`.
- Environment: `basis`; BF16, TF32 disabled, eager, same CPU affinity policy as the frozen TP8 runner.
- STAR full fused checkpoint: HF `alexz949/BasisServe-CALS`, revision `0ef83dff27205b131c82df6d62636129e9dac7b9`, upstream STAR-KV `c9f0f36e7e386eaf93099c9c9796e168ca1e6504`.
- Basis V64 factors: HF `alexz949/BasisServe-CALS`, revision `0872566b1da66eb4c813d7a1cb3313325f22b287`.
- Qwen3-8B-Base model revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Every trial command, status, and output path is in `grid_trials.json`; all failures are in `failures.csv`.
- Code base commit: `bae46c1409d3fa00030bab88df8b359e20b886bf`; worktree dirty: `True`.
- PyTorch `2.13.0+cu130`, CUDA `13.0`, NCCL `[2, 29, 7]`.
- Transformers `5.17.0`, FlashAttention `2.8.3.post1`, GPU `NVIDIA L40S`.

## Outputs

- `summary.csv`: every trial, including failed and missing configurations.
- `memory_summary.csv`: one max-rank observation per successful configuration; OOM groups have no measured resident value.
- `oom_frontier.csv`: first observed cache/workspace/other OOM and largest success per batch.
- `failures.csv`: every failed attempt, its last rank stages, allocation request when available, and launcher log.
- Memory plot generated: `True`.

Theoretical active V/K bytes are computed from actual stored tensor widths and do not equal total process memory.
A prefill-workspace OOM is not a KV-cache capacity limit.
No 4K smoke measurement is a formal grid result.
