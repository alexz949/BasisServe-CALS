# Qwen3.8-27B: Two-sided V128 and mixed Wo compression

Authorized configuration: Two-sided average V rank 128; four source-private Wo encoders, full-attention local rank 768 and GDN local rank 1152. Thinking remains disabled for future task evaluations. This pipeline fits factors and performs independent terminal-KL confirmation; it does not launch full downstream benchmarks.

## Immutable model and HF cache

Model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Standard snapshot:
`/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

All 18 weight shards and 32 downloaded files were verified against download metadata (SHA256 for LFS blobs, Git blob SHA1 for smaller files), then registered in the HF cache with hard-linked blobs and standard relative snapshot symlinks. Existing `results/q38_hybrid/model` paths are preserved and share the same inode data, with no duplicate weight storage. The full model identity was additionally checked against pinned Hub LFS metadata in `evaluation.prepare_qwen38_identity`. Fixed-revision `snapshot_download(..., local_files_only=True)` resolves successfully.

Qwen3.5-9B was also registered in the standard cache using its existing 13 files, including all four weight shards. Its snapshot is:
`/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Existing `results/q35_hybrid/model` files are preserved, sharing storage with cache blobs. No model bytes were modified. Cache logs: `results/q38_hybrid/logs/hf_cache.log` and `results/q35_hybrid/logs/hf_cache.log`.

## Architecture and gate audit

64 layers: 48 GDN and 16 full attention at indices `3,7,...,63`. Hidden size 5120. Full attention has 24 Q heads and four KV heads, head dimension 256. GDN has 48 V heads of dimension 128. Both Wo input widths are 6144, giving four local source widths of 1536. Full768 retains 50% and GDN1152 retains 75% of that local width. These are latent communication-width ratios, not whole-matrix parameter-count ratios.

The config contains `output_gate_type="swish"`. **The earlier interpretation that this necessarily changes the full-attention output gate was not supported by executed code.** The inspected official Transformers and vLLM Qwen3.5-family implementations apply sigmoid for full attention without reading this field. This run preserves that native path. Capture now checks that the captured pre-gate output times captured sigmoid gate exactly reproduces every native Wo input; the real-checkpoint smoke passed this check. GDN retains its native SiLU gated normalization.

Sources inspected:
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next.py

HF jobs use `lowrank`, PyTorch 2.8.0+cu128, with the existing project-local Transformers dependency at `results/q35_hybrid/deps` via PYTHONPATH. No shared dependency package was upgraded. Native GDN uses the torch implementation because the optional fused dependencies are unavailable in this environment.

## Data and fitting

Independently tokenize/sample for the new model using seed 20260909 and the same C4 sampling protocol: 256 fitting, 64 held-out factor-selection, 128 terminal-KL profile, and 16 independent confirmation windows, all length 2048. Separate C4 validation and WikiText test tokens are prepared for later PPL evaluation. Document IDs/hashes and split membership are recorded in `results/q38_hybrid/data/manifest.json`. The tokenizer's whole-WikiText length warning is expected: later PPL windows are split into at most 2048 tokens.

V capture uses the native Dense trajectory. Fit ranks `32,48,64,80,96,112,128,160,192,224` for every full-attention layer: 160 fitted candidates. Rank 256 is the exact native endpoint and requires no fitted adapter. V ALS uses six encoder sweeps, PCG cap 200, separable preconditioning, FP32 work, BF16 exported factors, and chunk rows 2048. Reaching the iteration cap does not establish convergence; solver histories and exported held-out errors are preserved.

Two-sided profiling uses an all-layer rank128 anchor and per-layer probes 96 and 160 (33 profiles total), the full rank bank through 256, exponent 1.25, and total rank budget `16*128=2048`. Freeze the selected schedule, then compare it with uniform128 on the independent 16 confirmation windows without reselection.

Recapture post-gate Wo input second moments on the frozen V trajectory. Keep FP64 accumulated moments on CPU to avoid placing all 64 large Gram matrices alongside model weights on GPU. Fit source-private Wo using six encoder sweeps, FP64 work, BF16 exports, full rank768 and GDN rank1152. Refit all layers from this model's moments; no 9B factors are reused.

## Verification completed before the full pipeline

- 19 relevant tests passed, including complete 33-profile coverage for the 16-layer 27B geometry.
- Real-checkpoint 64-token smoke passed. Identity V hidden-state relative MSE: `2.6883164537139237e-5`; compressed cached-decode versus uncached MSE: `7.062917575240135e-5`.
- Smoke compact cache shapes: K `[1,4,64,256]`, V `[1,4,64,64]`. Rank64 here is only the tiny runtime test, not the target configuration.
- Smoke peak allocated GPU memory: 50.63 GiB. Its deliberately capped eight-iteration toy ALS solves are not convergence evidence for the later fit.
- Shell syntax and `git diff --check` passed.

## Execution and artifacts

Direct execution is authorized because this machine has no Slurm. Entry point:

```bash
bash scripts/run_qwen38_v128_g1152_f768.sh \
  >> results/q38_hybrid/logs/pipeline.log 2>&1
```

The script records exact Python commands to the driver log before running them. All processes use `lowrank` and OMP/MKL two threads. Stage allocation: GPU6 V capture plus GPU5 Dense teacher; GPU2/5/6 V-fit shards; GPU5/6 KL-profile shards; GPU6 confirmation and frozen-V Wo capture; GPU2/5/6 Wo-fit shards; CPU assembly. Existing artifacts are preserved; failed stages halt later stages. Shared GPU availability and long solver runtimes can affect completion time.

Paths under `results/q38_hybrid/`:
- `baseline_summary.json`: authenticated identity only, not a benchmark score.
- `checks/smoke.json`: completed runtime check.
- `capture/`, `teacher/`, `factors/`, `kl/`: intermediate V fitting artifacts.
- `banks/c1_twosided_v128.pt`: intended frozen V bank.
- `checks/confirm_v128.json`: intended independent KL comparison.
- `wo_moments/`, `wo_g1152_f768/wo_bank.pt`: intended frozen-V Wo statistics and final bank.
- `logs/`: download, cache, data, tests, identity, smoke and per-stage logs.

At protocol creation the complete pipeline has been launched at capture/teacher stage. Fitted banks and downstream scores must not be reported as complete until the corresponding artifacts and audits finish.
