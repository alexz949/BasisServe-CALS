# PaLU Fisher with exactly matched C1 calibration tokens

## Scope

Build a V-only PaLU M-LRD Fisher checkpoint on the same C4 32 x 32,768 fit tokens as the existing C1-V80 checkpoint. Full exact K is preserved. No LongBench evaluation, C1 refit, optimizer step, new calibration sampling, or rank-budget correction is included in this experiment.

## Exact data matching and reused whitening

C1 checkpoint: `results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6`. Its recorded covariance snapshot points to `results/calibration/qwen3_8b_c4_32f4h_s32768/windows.safetensors`, containing 36 windows: first 32 fit and last 4 held-out. The endpoint is fixed after six sweeps; these four held-out windows are not used for the PaLU run.

Existing PaLU input: `results/calibration/qwen3_8b_c4_32f_s32768_palu/windows.safetensors`. A full tensor equality check confirms that all 1,048,576 input token IDs equal the C1 source's first 32 windows in the same order, not merely matching sample counts or lengths.

- C1 36-window artifact SHA256: `9c40181831da2645c2348a7970e22ab59885ab2c059d1f531359f4a023906fc8`.
- PaLU 32-window artifact SHA256: `bc604a85e580c9f285e06d996b2ee17131aab4f70912404771d8c52fc781614b`.
- Reused whitening artifact SHA256: `0a08ce435fa552e22c022dd2331d3d062254d4be0e4a5eb718d43d03d8592213`.

The whitening manifest authenticates that exact 32-window artifact. Its model config/snapshot and artifact hashes are verified on every stage. Whitening already exists at `results/calibration/qwen3_8b_c4_32f_s32768_palu_whitening`; it is reused rather than recomputed. Statistics are per-layer input X-transpose-X, summed across windows before Cholesky; accumulation FP32, Cholesky FP64 cast to FP32. The existing full-sequence dense SDPA capture used four L40S workers.

## Fisher objective and execution

Only V projection weights have gradients enabled. No optimizer is constructed and no weights are updated. Each window is processed independently with batch size 1 and full 32K causal context. K remains dense.

The loss preserves this repository's existing official-PaLU reproduction: inputs are tokens `[:-1]` and HF labels are tokens `[1:]`. Because HF shifts the labels again, the effective targets are tokens `[2:]`, paired with hidden positions through sequence length minus three. This is the existing double-shift convention, not standard next-token PPL. No token/query subsampling is applied; each window contributes 32,766 target positions.

For window j and layer l, g(l,j) is the gradient of that window's mean loss with respect to the full V projection weight matrix. The layer importance is:

`s_l = mean_entries(sqrt(sum_j g(l,j)^2 / 32))`.

Four independent full-model GPU workers each process eight windows, using indices `shard_index::4`. They save FP32 elementwise sums of squared gradients on CPU. Merge adds those tensors in FP64, divides by all 32 windows, takes the entrywise square root, and finally averages weight entries. Gradients are never averaged before squaring; per-shard scalar importance is not averaged.

Decoder activation checkpointing is non-reentrant. The installed Transformers implementation requires training mode for checkpointing; the new collector sets training mode and verifies attention dropout and every Dropout module are zero. Smoke and formal runs count decoder calls and assert that all 36 layers are recomputed. Vocabulary logits/loss are processed in 128-position chunks with an assembled hidden gradient, retaining one full-context decoder backward. This changes execution and peak memory, not the mathematical loss.

Existing Palu factorization uses the matched whitening, grouped activation-aware SVD, FP64 CPU working arithmetic, and BF16 saved writer/decoder factors. Fisher determines layer ranks, not the whitening objective. The existing block32 official allocation is retained with requested V retained ratio 0.625 (nominal R80); realized average rank may differ and is reported explicitly.

## Validation and resources

New collector: `evaluation/collect_palu_c1matched_fisher.py`. Existing Fisher collection code is not edited. Two CPU unit tests in `tests/test_palu_c1matched_fisher.py` pass: chunked loss/gradient equivalence to full shifted loss, and sum-of-squares merging before square root.

Single-L40S 32K smoke job `8300964` passed: one window, loss 10.4771519 under the preserved loss convention, 19.627 seconds of window computation, 32.122 GiB peak allocated GPU memory. All 36 V weight gradients were finite and all decoder layers recomputed. Total smoke job elapsed 37 seconds. Smoke output is separate and excluded from formal statistics.

Formal job `8300965`, array 0–3: four L40S on lovelace, two CPUs and 32 GiB host memory per worker, basis environment. CPU merge/build job `8300969` depends on all four succeeding, two CPUs and 24 GiB host memory. Slurm time limits are limits, not runtime estimates. Temporary sbatch files are deleted after submission. No unrelated jobs or existing checkpoints are modified.

Raw statistics and audit metadata: `results/calibration/palu_c1matched_fisher32k`. New checkpoint destination: `results/checkpoints/palu_m_fisher_r80_c1matched32k`. Logs: `logs/palu-match-{smoke,fisher,build}-{job}[_task].out/.err`.

## Completed checkpoint and measurements

All formal Fisher workers completed with exit code 0: shard 0 in 2:57, shards 1–3 in 2:58 each. Merge/build job 8300969 completed with exit code 0 in 2:36; its factorization stage took 129.458 seconds. There were no OOMs, non-finite losses/gradients/factors, failed jobs or retries. No LongBench test was run in this task.

Median formal window time: 19.909 seconds. Maximum allocated GPU memory: 32.122 GiB. Mean preserved-PaLU-convention loss: 10.4220122; this is not standard next-token NLL/PPL. The smoke and formal first-window loss match exactly. Their per-layer gradient norms are not bitwise identical (maximum relative difference 1.890%); no bitwise-backward reproducibility claim is made.

| Rank per KV head | Number of layers |
|---|---:|
| 32 | 1 |
| 64 | 17 |
| 96 | 15 |
| 128 | 3 |

Total rank over 36 layers and 8 KV heads: 23,552. Average rank: 81.7778, versus nominal target 80. Realized V retention: 0.6388889; V compression: 0.3611111. This retains the prior allocator's block32 rounding without enforcing an exact total-rank budget.

Ranks in layer order 0–35 (all 8 KV heads in a layer share its rank):

```text
128, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 64,
64, 64, 64, 64, 64, 96, 64, 96, 96, 96, 128, 64,
64, 64, 64, 64, 64, 64, 64, 32, 64, 64, 96, 128
```

Twenty-two layers differ in rank from the old 256 x 2048 PaLU-M Fisher checkpoint at `ICLR-results/qwen3-8b/checkpoints/Q3-8B-PALUM-R80`. This is a schedule comparison, not an evaluated quality improvement.

Final validation reloaded the 72 factor tensors and verified their BF16 dtype, finiteness and rank-dependent shapes, all 36 layer ranks, factor/manifest/Fisher hashes, matched calibration metadata and unchanged collector/dependency code hashes. K remains dense in the checkpoint manifest.

- Checkpoint directory: `results/checkpoints/palu_m_fisher_r80_c1matched32k`.
- Factor SHA256: `08fe0c6564c6cd3d5b51f8ecd0b27b2708d6adcf81eec719b46a87da14d7bd3a`.
- Checkpoint manifest SHA256: `97a5c294acca8a54ad24dd987fab572d5973ca1e1e87445fe8457c5324f8db62`.
- Fisher result SHA256: `88f726679e67022c202d89c92aac6510c1363a0a8e193bbcb657e3ae823ae04f`.

## Reproduction commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. All commands use the basis Python executable below.

### Smoke

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/collect_palu_c1matched_fisher.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
```

### Formal collection

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/collect_palu_c1matched_fisher.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage collect --shard-index "$SLURM_ARRAY_TASK_ID"
```

### Merge

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/collect_palu_c1matched_fisher.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage merge
```

### Factor fitting

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/build_gqa_palu_m_fisher_checkpoint.py --profile qwen3_8b --target v --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --fisher-result results/calibration/palu_c1matched_fisher32k/fisher.json --whitening-dir results/calibration/qwen3_8b_c4_32f_s32768_palu_whitening --output-dir results/checkpoints/palu_m_fisher_r80_c1matched32k --torch-num-threads 2
```
