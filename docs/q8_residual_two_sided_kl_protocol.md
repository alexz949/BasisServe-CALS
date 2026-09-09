# Qwen3-8B residual two-sided KL protocol

## Implementation status

The factor-bank completion script, cached-prefix/layer-suffix replay, signed three-rank DP, confirmation evaluator and summary writer are implemented. Six small CPU tests passed in the `basis` environment. Factor-bank completion finished successfully as Slurm array job `8300163`, tasks 0–3, with one L40S, two CPUs and 48 GiB host memory per task on `lovelace`. All four tasks completed with exit code `0:0`. Logs are `logs/rkl-bank-8300163_{0,1,2,3}.{out,err}`. The 32K numerical smoke test, all 64 formal profile windows, exact-budget allocation, and all 16 confirmation windows have also completed successfully.

| Task | Elapsed | State | Exit code |
|---|---:|---|---|
| 0 | 20:39 | COMPLETED | 0:0 |
| 1 | 17:49 | COMPLETED | 0:0 |
| 2 | 23:17 | COMPLETED | 0:0 |
| 3 | 21:49 | COMPLETED | 0:0 |

The completed bank is `results/checkpoints/q8_residual_kl_bank`: 36 layer files, 108 layer/rank combinations and 324 FP32 tensors. Layers 0/13/33 reuse the existing sweep factors; the other 33 layers were fitted. New-fit per-layer time was 130.72–170.66 seconds, median 146.39 seconds.

Post-run CPU verification in `basis` passed: all four shard manifests cover exactly layers 0–35, all artifact SHA256 values match, every tensor has the expected shape and finite values, the Base16 factors in all 36 layers match the frozen source bitwise, and all three reused layer banks match their source tensors exactly. The only stderr message was the existing Transformers `Qwen3RotaryEmbedding(device=...)` deprecation warning; no NaN/Inf tensor, CUDA error, OOM or failed task was observed. No terminal KL, PPL or RULER result is implied by completing this factor bank.

The CPU tests cover prefix KV/sidecar fork isolation, unchanged-anchor replay, modified-layer replay versus complete suffix forward, probe-order independence, full-budget rank independence, causal masking, block versus sequential sparse suffix execution, teacher-KL direction/readout accounting, and signed exact-budget DP versus exhaustive enumeration.

`basis` does not have pytest installed. The six fixture-free test functions were executed directly with `runpy`; no environment packages were installed or changed.

## Terminal measurement execution

The 32K single-window GPU smoke test passed on A100 job `8300182` and independently on all three L40S workers in array `8300185`. Unchanged-anchor replay and the layer-33 R4 intervention replay both matched complete suffix forward bitwise (maximum absolute error zero); the shared prefix cache remained unchanged. A100 smoke computation took 14.54 seconds with 28.41 GiB peak allocated GPU memory. L40S smoke computation took approximately 12.5 seconds with the same peak allocated memory.

The same window produced different terminal KL probe deltas on A100 and L40S, despite exact replay within each device. All three L40S smoke runs agreed. For example, layer-0 R4 delta was -0.00156832 on A100 versus +0.00567830 on L40S; layer-0 R16 delta was -0.00135695 versus +0.00320557. Consequently, heterogeneous GPU results are not pooled in the formal profile. A100 profile job `8300183` and its pending downstream jobs were canceled. Its completed windows 0/4/8 were preserved under `results/evaluation/q8_residual_kl_64x32k/hardware_a100/profile` and are excluded from allocation.

The formal profile and confirmation use L40S exclusively, with the same factor bank, token windows, attention implementation, and per-window teacher/anchor/probe pairing. The dataset and algorithm parameters were not changed.

| Stage | Job | Scope / dependency |
|---|---|---|
| Profile shard 0 | 8300193 | L40S, indices 0,4,...,60 |
| Profile shards 1–3 | 8300185 | Three L40S workers, indices congruent to shard index modulo 4 |
| Allocate | 8300194 | CPU, after both profile jobs succeed; verify 64 L40S windows |
| Confirmation | 8300198 | Four L40S workers, after allocation succeeds |
| Summary | 8300202 | CPU, after confirmation succeeds; verify 16 L40S windows and frozen schedule hash |

These jobs use `basis`. Profile and confirmation workers request two CPUs and 64 GiB host memory each; the CPU-only allocation/summary jobs request two CPUs and 8 GiB host memory. Logs use `logs/rkl-{profile,allocate,confirm,summary}-<job>[_<array-index>].{out,err}`. Downstream jobs used success-only dependencies. All formal jobs completed with exit code `0:0`: profile shard 0 took 14:38, profile shards 1–3 took 15:36 / 15:34 / 15:32, allocation took 00:26, confirmation shards took 00:53 / 00:52 / 00:53 / 00:53, and summary took 00:13. Slurm elapsed times include process startup; the three original L40S profile workers also ran a smoke test.

## Completed allocation and confirmation

The allocation assigns R4 to 16 layers, R8 to 12 layers and R16 to 8 layers, for total layer rank 288 and average R8. All three candidate ranks use their directly measured signed costs. An independent CPU recomputation of all 64-window costs and exact-budget DP matched the saved optimum, predicted additive profile delta KL -0.0033297070. This prediction is not a measured joint-schedule profile result.

| Arm | Confirmation teacher KL | Suffix NLL | Suffix PPL |
|---|---:|---:|---:|
| Same C1-V80, full exact-K teacher | 0.00000000 | 1.90025533 | 6.68760178 |
| Uniform residual R8 | 0.01884865 | 1.91338754 | 6.77600395 |
| Adaptive residual, average R8 | 0.01242355 | 1.90617098 | 6.72728054 |

On the 16 confirmation windows, mean KL decreased by 34.09% and suffix PPL by 0.719%. KL improved in 11/16 windows and NLL in 10/16. Window 64 accounts for 85.05% of the total net KL improvement; the paired-window KL delta standard error is 0.00542688, versus mean delta -0.00642510. The aggregate improvement therefore does not establish a stable or statistically significant gain. No schedule was changed after confirmation.

All formal per-window files were checked for L40S hardware, protocol consistency and unchanged shared-prefix caches. The frozen schedule hash matches all 16 confirmation files; aggregate KL/NLL/PPL recomputed from those files matches the summary. Full results, per-window differences and scope limitations are recorded in [the experiment summary](../results/evaluation/q8_residual_kl_64x32k/summary.md); the frozen allocation is [schedule.json](../results/evaluation/q8_residual_kl_64x32k/schedule.json).

## Fixed protocol

- Model: Qwen3-8B-Base, BF16.
- C1: uniform V80, checkpoint `results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6`; frozen.
- Base: per-group pre-RoPE affine Base16, checkpoint `results/checkpoints/qwen3_8b_v80_32k_base16_p32`; frozen.
- Residual feature: exact post-RoPE K minus the RoPE-transformed Base prediction.
- Residual ranks: R4, R8, R16. One rank per decoder layer, identical for its eight KV groups.
- Residual bank: reuse the matching completed layer 0/13/33 factors; fit the other 33 layers using the existing 64-window captures. Page32 non-sink Fisher BCD, 40 sweeps, final-token queries. The 16 validation windows report diagnostics; they do not choose the fitted factors.
- Windows: existing C4 `64f16h_s32768/windows.safetensors`, 80 rows of 32,768 tokens. Each row packs eight 4096-token source windows without inserted separators; these are not native 32K documents.
- KL profile: indices 0–63. Confirmation: indices 64–79.
- Each window: 32,640-token full-attention C1 prefix, followed by 128 causally masked sparse-attention positions under teacher forcing.
- Teacher: same C1-V80 and shared prefix, full exact-K suffix attention.
- Student selection: Page32 LSE, non-sink normalization per query head, max across the four query heads in each KV group, 64 physical pages including pinned page0; B2048 includes the pinned page.
- Anchor: R8 at all 36 layers.
- Probes: change exactly one layer to R4 or R16; all other layers remain R8. Recompute the changed layer and all downstream layers on a fresh prefix-cache fork.
- KL: full vocabulary, all 128 suffix readout positions, FP32 log-softmax and FP64 reduction.
- NLL: 127 suffix positions whose next-token targets are present. Reported suffix PPL is not full-corpus PPL.
- Allocation: signed measured KL differences at R4/R16 and zero at R8; no clipping, interpolation, Fisher-to-KL factorization, or exponent. Exact-budget DP enforces sum of layer ranks = 288.
- Confirmation: freeze the schedule, compare it against uniform R8 on the 16 confirmation windows. These windows do not enter terminal allocation; they were previously used for factor diagnostics, so they are not an untouched final benchmark.
- Storage: exact K stays on GPU and the oracle materializes Base128+R routing coordinates. This is an accuracy measurement, not a deployment-memory or PCIe latency benchmark.

## Programs

All commands run from `/deac/csc/yangGrp/zhangal/BasisServe-CALS`, under Slurm for GPU work, using `/home/zhangal/.conda/envs/basis/bin/python`.

Factor-bank completion, with `--shard-index` taking 0, 1, 2 and 3:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_qwen3_8b_residual_kl_bank.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --base-root results/checkpoints/qwen3_8b_v80_32k_base16_p32 \
  --reuse-root results/checkpoints/qwen3_8b_v80_base16_residual_rank_sweep \
  --output-dir results/checkpoints/q8_residual_kl_bank \
  --shard-index 0 --num-shards 4
```

The 32K GPU correctness/timing smoke test:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/profile_qwen3_8b_residual_two_sided_kl.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/q8_residual_kl_bank \
  --windows results/calibration/qwen3_8b_c4_64f16h_s32768/windows.safetensors \
  --output-dir results/evaluation/q8_residual_kl_64x32k \
  --suffix-length 128 --query-block-size 8 --shard-index 0 --num-shards 4
```

The same evaluator CLI supports `--stage profile` (four window shards), `--stage allocate` (one CPU process after all profile shards), `--stage confirm` (four window shards), and `--stage summarize` (one CPU process after all confirmation shards). All other settings remain identical. Profile has 72 intervention configurations per window, plus the teacher and anchor; each configuration is not a single-token forward.

Each completed window is written atomically. Restarted work checks the protocol, source hashes and frozen schedule hash before reusing any completed window. The summary includes paired confirmation-window KL differences and their standard error.
