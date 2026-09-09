# Q-aware Base16 + Page-Fisher residual

## Status and scope

Completed as Slurm array `8300348`, tasks 0–3, after the user's confirmation.
Each worker requests one L40S, two CPUs, 48 GiB host memory and two hours on
`lovelace`. Logs are `logs/qbase-fit-8300348_{0,1,2,3}.{out,err}`. The temporary
sbatch file was deleted after submission. The previous MSE-Base RULER run has
not been launched.
The user requested changing Base to per-query Q-aware fitting while retaining
Page-Fisher residual fitting. This program builds a new, separate 36-layer bank;
it does not overwrite the old bank, launch terminal KL allocation, or run RULER.
The old residual factors and adaptive rank schedule cannot be treated as fitted
for this new Base.

Five small CPU tests passed in the `basis` environment: explicit position-rotated
causal QK loss, sink/future masking, query scaling, finite factor gradients,
zero loss for an exact predictor, recomputation of residual Fisher after changing
Base, and three-rank artifact packing. Some test functions cover multiple checks.
All 36 layers completed the full Base + R4/R8/R16 fit. There are 108
layer/residual-rank combinations and 324 FP32 tensors in the new bank.

| Shard | Layers | Elapsed | State | Exit code | Peak host RSS |
|---:|---:|---:|---|---|---:|
| 0 | 0,4,...,32 | 42:18 | COMPLETED | 0:0 | 43679956 KiB |
| 1 | 1,5,...,33 | 42:26 | COMPLETED | 0:0 | 38713016 KiB |
| 2 | 2,6,...,34 | 42:33 | COMPLETED | 0:0 | 39023872 KiB |
| 3 | 3,7,...,35 | 42:58 | COMPLETED | 0:0 | 38826084 KiB |

Independent CPU verification in `basis` passed for the complete bank:
all four manifests cover exactly layers 0–35, current source/input and artifact
hashes match, and all nine tensors/layer have the expected shapes and finite
FP32 values. The represented Base maps changed in all 36 layers, and all 108
residual encoders differ from the old bank. No original bank artifact or
schedule was overwritten. Per-layer fitting time was 213.43–329.81 seconds,
median 293.18 seconds.

Validation raw-QK NMSE decreased in all 36 layers. Its unweighted mean over
layers changed from 0.1028245349 to 0.0990476124; the median layer-relative
decrease was 3.0440%. These are validation-selected local fitting metrics,
not independently held-out routing or generation results. Selected epoch
counts were: epoch 4: 2 layers; 5: 1; 7: 2; 8: 2; 9: 3; 10: 2; 11: 4; 12: 20.

| Layer | Selected epoch | Initial validation raw-QK NMSE | Fitted validation raw-QK NMSE | Residual R4 validation Fisher NMSE | R8 | R16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 0.009114 | 0.009077 | 0.500596 | 0.429236 | 0.381931 |
| 1 | 12 | 0.045893 | 0.044976 | 0.379241 | 0.275366 | 0.254824 |
| 2 | 12 | 0.089519 | 0.087476 | 0.185946 | 0.148799 | 0.101508 |
| 3 | 12 | 0.049061 | 0.047958 | 0.155222 | 0.127053 | 0.104300 |
| 4 | 11 | 0.027117 | 0.026743 | 0.031253 | 0.029831 | 0.028096 |
| 5 | 12 | 0.022438 | 0.021512 | 0.037174 | 0.034482 | 0.031037 |
| 6 | 12 | 0.062706 | 0.061619 | 0.070649 | 0.059191 | 0.053705 |
| 7 | 12 | 0.061988 | 0.059086 | 0.836334 | 0.716138 | 0.551020 |
| 8 | 12 | 0.026463 | 0.025718 | 0.875284 | 0.734196 | 0.594549 |
| 9 | 12 | 0.104785 | 0.101200 | 0.891398 | 0.752716 | 0.522158 |
| 10 | 12 | 0.013454 | 0.012861 | 0.559850 | 0.451694 | 0.365546 |
| 11 | 11 | 0.017847 | 0.017436 | 0.768071 | 0.700987 | 0.477906 |
| 12 | 12 | 0.038309 | 0.036965 | 0.887398 | 0.657635 | 0.506513 |
| 13 | 12 | 0.137321 | 0.124732 | 0.897656 | 0.817657 | 0.563338 |
| 14 | 12 | 0.036181 | 0.033469 | 0.757549 | 0.569205 | 0.447826 |
| 15 | 12 | 0.116996 | 0.106572 | 0.901918 | 0.694824 | 0.528200 |
| 16 | 12 | 0.104475 | 0.099329 | 0.811424 | 0.618467 | 0.424060 |
| 17 | 12 | 0.127230 | 0.119545 | 0.081972 | 0.074167 | 0.057699 |
| 18 | 12 | 0.176975 | 0.166493 | 0.075600 | 0.072618 | 0.061282 |
| 19 | 12 | 0.155367 | 0.145208 | 0.070846 | 0.068051 | 0.055760 |
| 20 | 12 | 0.129285 | 0.120746 | 0.091443 | 0.091177 | 0.073659 |
| 21 | 12 | 0.099681 | 0.092603 | 0.066695 | 0.062383 | 0.051189 |
| 22 | 11 | 0.167026 | 0.160383 | 0.071288 | 0.073079 | 0.053173 |
| 23 | 11 | 0.078025 | 0.075789 | 0.060052 | 0.054548 | 0.043614 |
| 24 | 8 | 0.252221 | 0.244093 | 0.089594 | 0.085153 | 0.063561 |
| 25 | 9 | 0.074037 | 0.071452 | 0.049656 | 0.050745 | 0.036359 |
| 26 | 7 | 0.137341 | 0.134921 | 0.081762 | 0.083229 | 0.062199 |
| 27 | 9 | 0.118661 | 0.115303 | 0.067304 | 0.067770 | 0.046958 |
| 28 | 10 | 0.056611 | 0.054088 | 0.063079 | 0.061580 | 0.047430 |
| 29 | 8 | 0.276988 | 0.271303 | 0.116935 | 0.112663 | 0.073575 |
| 30 | 5 | 0.085193 | 0.083871 | 0.064881 | 0.069528 | 0.050823 |
| 31 | 4 | 0.227651 | 0.224391 | 0.092481 | 0.100570 | 0.059666 |
| 32 | 9 | 0.121284 | 0.119529 | 0.069792 | 0.076517 | 0.050043 |
| 33 | 10 | 0.347855 | 0.344453 | 0.105719 | 0.102904 | 0.068468 |
| 34 | 4 | 0.072213 | 0.071416 | 0.102557 | 0.106417 | 0.065775 |
| 35 | 7 | 0.034373 | 0.033397 | 0.381582 | 0.279384 | 0.213256 |

Residual Fisher NMSE is normalized by this new Base's residual teacher energy.
It is not directly comparable to a different Base's normalized residual loss.
The residual validation measurements do not select a residual checkpoint.

The only stderr message was the existing Transformers
`Qwen3RotaryEmbedding(device=...)` deprecation warning. No RULER accuracy,
terminal KL, or new adaptive schedule has been measured.

## Fixed configuration

| Component | Setting |
|---|---|
| Model | Qwen3-8B-Base; 36 layers, 8 KV groups, 4 query heads per group, head dimension 128 |
| Payload | Frozen C1-V80, `results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6` |
| Base | Per-group affine pre-RoPE prediction, rank 16 |
| Initialization | Only the MSE-Base16 factors from `results/checkpoints/q8_residual_kl_bank`; its residual tensors are not reused |
| Fit | Existing C4 windows 0–63, each 32768 tokens |
| Validation | Existing C4 windows 64–79; selects Base epoch, reports residual diagnostics |
| Base queries | Eight separate post-RoPE Q observations/window: 25599, 26623, 27647, 28671, 29695, 30719, 31743, 32767, zero-based |
| Base key positions | All causally visible non-sink tokens for each sampled Q |
| Base optimizer | Existing factorized Adam implementation, 12 epochs maximum, 4 documents/step, factor LR 0.002, bias LR 0.0005, clip 1, patience 4, seed 73 |
| Residual | Exact post-RoPE K minus the RoPE-transformed NEW Base prediction |
| Residual queries | Last token/window, unchanged from the previous residual allocation bank |
| Residual fitting | Non-sink Page-Fisher BCD, ranks 4/8/16, 40 sweeps, damping/tolerance 1e-5, maximum 100 iterative-solver iterations |
| Pages | Page32; page 0 excluded from fit distribution and pinned during downstream selection |
| Output | `results/checkpoints/q8_qbase_fisher_bank` |
| Intended resources | Four L40S workers, one GPU and two CPUs per worker; nine layers per worker |

The C4 windows pack eight 4096-token source windows without separators; they are
not native 32K documents. The 16 validation windows will select Base factors,
so a later terminal evaluation on these same windows is not an untouched test.

The Q8 files also contain statistics computed for an older C1 checkpoint. This
program reads only `queries_by_head` from them, not the old payload/Fisher Grams.
The capture program runs an unmodified dense teacher with observation hooks;
the auxiliary C1 factors enter the saved output statistics, not the teacher
forward. Current C1-V80 codes are computed offline from the raw V captures.
Window identities, split offsets, query positions, model configuration, C1
factors, initialization bank and source files are recorded/checked separately.

## Objectives

Let `c_i` be the frozen resident C1 value code for a KV group, and let `R_i` be
the exact RoPE map at key position `i`. For each layer and group, fit

\[
\widehat k_i^{pre}=c_i A B+b,\qquad \operatorname{rank}(AB)\le16,
\]

\[
\mathcal L_{base}=\frac12\sum_{w,t,h}\sum_{i=32}^{t}
\left[\frac{(q^{post}_{w,t,h})^\top
\left(R_i\widehat k^{pre}_{w,i,g(h)}-k^{post}_{w,i,g(h)}\right)}{\sqrt{128}}\right]^2.
\]

The sum over `t` uses the eight specified queries, not all 32768 queries.
Each sampled query and its causal key positions remain separate. This is raw
score regression, not softmax KL, and is not a global mean-Q covariance proxy.
Only the small Base factors receive gradients; no model backward is performed.

Freeze this Base and define a new innovation

\[
\delta_i=k_i^{post}-R_i\widehat k_i^{pre}.
\]

The residual score approximation is

\[
\widehat z_i^{res}=\frac{(q^{post}U_h)^\top(\delta_i E_g)}{\sqrt{128}},
\qquad e_i=\widehat z_i^{res}-\frac{(q^{post})^\top\delta_i}{\sqrt{128}}.
\]

For the final query, normalize exact-teacher token probabilities `p_i` over the
non-sink tokens. For each page, define

\[
m_p=\sum_{i\in p}p_i,\qquad
\bar e_p=\sum_{i\in p}\frac{p_i}{m_p}e_i,\qquad
\mathcal L_{res}=\frac12\sum_{w,h,p}m_p
\left(\bar e_p-\sum_{p'}m_{p'}\bar e_{p'}\right)^2.
\]

This is the existing local Page-Fisher quadratic objective, not exact global KL.
Its page masses and within-page weights come from exact K. Its feature Grams
must be regenerated from the new `delta`, even though the teacher weights are
unchanged. R4/R8/R16 are all refitted, with no reuse of the old residual factors.

At inference, the Base factors remain fixed and are applied per token; “Q-aware”
describes their offline objective, not a query-dependent refit. Both Base and
residual remain local to each GQA group.

## Exact fitting command, approved and submitted

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.
Environment: `basis`. Run through Slurm, with `--shard-index` equal to 0, 1, 2,
or 3 on the four workers. Keep log files and remove the temporary sbatch script
after submission.

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_qwen3_8b_qaware_base_fisher_bank.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --calibration-root results/calibration \
  --initial-bank results/checkpoints/q8_residual_kl_bank \
  --output-dir results/checkpoints/q8_qbase_fisher_bank \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

Each complete layer writes a bank artifact and JSON record atomically. Source
hashes and configuration must match to reuse a completed layer. The output
schema supports the existing residual KL/mass/RULER evaluators, but their
previous profile, schedule and accuracy results are not reused.

## Terminal KL continuation: superseded, not submitted

The user subsequently chose uniform R8 and explicitly skipped KL allocation.
The proposal below was not submitted. The active evaluation is documented in
[the uniform RULER protocol](q8_qbase_uniform_ruler_protocol.md).

The next run uses this new bank at all 36 layers, with uniform R8 as the
anchor and single-layer R4/R16 interventions. All 64 fit windows contribute
terminal measurements. Each window has a 32640-token full-attention C1 prefix
and a 128-token causal sparse suffix, processed in blocks of 8. Page32,
B2048 and one pinned prefix page remain fixed. Signed three-point costs feed
the existing exact-budget DP, with total layer rank 288 (average R8).

After freezing the new schedule, compare full exact-K C1, uniform R8 and the
new adaptive schedule on windows 64–79. These windows already selected Base
epochs, so this stage is a confirmation diagnostic, not an untouched held-out
test. The evaluator now carries the full factor-bank protocol in its results
and checks that all layers share it; the measurement and DP mathematics are
unchanged. No previous profile, schedule or confirmation file is overwritten.
The new-bank protocol check and all six existing small CPU replay/DP tests
passed in `basis` after this metadata update. No new terminal GPU run has
been submitted.

Working directory and environment are identical to the fitting command above.
The proposed command for the GPU numerical smoke test is:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/profile_qwen3_8b_residual_two_sided_kl.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/q8_qbase_fisher_bank \
  --windows results/calibration/qwen3_8b_c4_64f16h_s32768/windows.safetensors \
  --output-dir results/evaluation/q8_qbase_residual_kl \
  --suffix-length 128 --query-block-size 8 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

The complete proposed pipeline uses the same command and settings with stage
`smoke`, then `profile` (four GPU workers with shard indices 0–3), `allocate`
(CPU), `confirm` (four GPU workers), and `summarize` (CPU). All stages depend
on successful completion of their predecessors. This continuation still
requires explicit confirmation before job submission; it does not include
RULER or GitHub upload.
