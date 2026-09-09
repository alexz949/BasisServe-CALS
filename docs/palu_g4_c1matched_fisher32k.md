# PaLU G-LRD4 Fisher R80 with C1-matched 32K calibration

## Configuration

Qwen3-8B-Base, V-only compression, full exact K preserved. Four adjacent physical KV heads are jointly factorized: heads 0–3 and heads 4–7. Each layer has two 512-dimensional output groups. Nominal equivalent rank 80 means a nominal rank of 320 per four-head group, not rank 80 per group. Fisher assigns different group ranks to different layers; both groups in one layer share its assigned rank.

Reuse `results/calibration/palu_c1matched_fisher32k/fisher.json` and `results/calibration/qwen3_8b_c4_32f_s32768_palu_whitening`. The preceding M-LRD run verified that all 32 x 32,768 input token IDs match the C1-V80 checkpoint's first 32 fit windows exactly. The four C1 held-out windows are excluded. No backward, whitening capture, new calibration sampling, or model weight optimization is repeated for G4.

The importance statistics are gradients of the original dense V weight matrices and do not depend on a downstream M/G4 grouping. Reuse the 36 layer scalars, then rerun the existing official Fisher rank allocator with `head_group_size=4`, retained ratio 0.625 and block32 group-rank rounding. Do not multiply the already rounded M-LRD schedule by four.

Factorization groups each 1024 x 4096 V weight matrix into two 512 x 4096 blocks. For a group weight W and input Cholesky C, compute the truncated SVD of W C, then use a triangular solve to unwhiten the right factor. Working arithmetic is CPU FP64; saved writer and decoder tensors are BF16. Per-layer writer shape is `(2 * group_rank, 4096)` and decoder shape is `(2, 512, group_rank)`.

## Execution and validation

Reused builder: `evaluation/build_qwen3_8b_iclr_v_checkpoint.py`; no changes to existing implementation. Small CPU tests passed for grouped activation-weighted SVD: truncated reconstruction reaches the weighted singular-value-tail optimum, and full-rank factors reconstruct the input matrix.

Slurm job `8300978`, CPU partition `small`, two CPUs, 16 GiB host memory, basis environment. No GPU allocation. Logs: `logs/palu-g4-match-build-8300978.out/.err`. Temporary sbatch file removed after submission. Existing M-LRD checkpoint and calibration artifacts are unchanged. No LongBench/RULER evaluation is included in this checkpoint-building task.

Output directory: `results/checkpoints/palu_g4_fisher_r80_c1matched32k`.

## Completed result

Job 8300978 completed with exit code 0 in 1:29. Factorization itself took 75.845 seconds. There were no reported errors, non-finite factors or retries. All 36 layers and 72 BF16 tensors passed shape/finiteness checks. The artifact hash, Fisher result hash and calibration-window identity were verified against their manifests.

Total latent rank across layers/groups is 23,040, equivalent to exactly 80 dimensions per physical KV head on average. V retention is 62.5% and V-cache compression is 37.5%. The companion M-LRD checkpoint averages 81.7778 because block32 rounding acts on individual heads there, versus four-head groups here. Thus nominal targets match, but realized budgets are not identical.

Per-four-head-group ranks, in layer order 0–35; both groups of a layer use the same rank:

```text
448, 384, 352, 416, 416, 384, 384, 416, 352, 384, 352, 224,
288, 224, 288, 288, 320, 352, 256, 384, 352, 352, 512, 256,
256, 256, 192, 288, 192, 192, 256, 160, 288, 192, 384, 480
```

The corresponding per-head equivalent rank is each listed value divided by four. This schedule is computed from the unchanged dense-model Fisher scalars, not from compressed-model backward passes.

- Factor SHA256: `b7c5f13c75f92d273a3a828b6bd1613a34fbbef08fda3d390eb71cfc0d063b69`.
- Manifest SHA256: `93f246e929cceedbd6aa7f6a29309f25350c3cbe6953b2b0c278a209f0abfada`.
- Reused Fisher SHA256: `88f726679e67022c202d89c92aac6510c1363a0a8e193bbcb657e3ae823ae04f`.

This result establishes a built, validated checkpoint only. No new PPL, LongBench or RULER score is available from this task.

## Reproduction command

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/build_qwen3_8b_iclr_v_checkpoint.py --method palu-fisher --equivalent-rank 80 --head-group-size 4 --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --fisher-result results/calibration/palu_c1matched_fisher32k/fisher.json --whitening-dir results/calibration/qwen3_8b_c4_32f_s32768_palu_whitening --output-dir results/checkpoints/palu_g4_fisher_r80_c1matched32k --torch-num-threads 2
```
