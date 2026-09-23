# Qwen3 TP8 V-only 16K Batch Sweep

Grid status: `complete_with_failures`; trial statuses: `{"complete": 14, "failed_oom_cache_state_allocation": 4}`.

BasisKV V64 versus STAR-KV V-only adaptive export, exact dense K, full attention,
BF16, TP8 / DP1 / PP1 on eight NVIDIA L40S GPUs. One preselected real-text
cohort supplies eight prompts; batches above eight repeat those prompts in order.
Each trial prefills 16384 tokens in 256-token chunks, reserves 128 decode slots,
and actually generates 128 greedy output tokens without EOS stopping.
The first output token comes from prefill; the remaining 127 require decode forwards.
The model's native context is 32768 tokens, so the 16512-token request fits.
No CPU V offload, K routing, K compression, or substituted checkpoint is used.
The STAR export retains 54.0473% actual global V rank, not a strict 50%.

## Outcomes

- basis_v64: largest successful batch `128`; first observed OOM batch `256` (phase `cache_state_allocation`).
- star_v_adaptive: largest successful batch `32`; first observed OOM batch `64` (phase `cache_state_allocation`).

The main plotted value is max-rank decode-ready NVML process GPU memory.
An OOM has no measured decode-ready value. Decode-end NVML memory and peak
prefill/decode PyTorch allocation are separate columns in `summary.csv`.
Theoretical K/V bytes use actual stored widths and are not process memory.
Every attempt and its command are in `grid_trials.json`; all failures are in `failures.csv`.
Memory plot generated: `True`.

## Provenance

- Code base commit: `bae46c1409d3fa00030bab88df8b359e20b886bf`; worktree dirty: `True`.
- PyTorch `2.13.0+cu130`, CUDA `13.0`, NCCL `[2, 29, 7]`.
- Transformers `5.17.0`, FlashAttention `2.8.3.post1`.
- Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- STAR fused checkpoint HF revision `0ef83dff27205b131c82df6d62636129e9dac7b9`.
- Basis V64 factor HF revision `0872566b1da66eb4c813d7a1cb3313325f22b287`.
