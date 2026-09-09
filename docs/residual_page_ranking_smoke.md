# Residual page-boundary repair: implementation and smoke

## Scope

Frozen closed-form affine MSE-RRR Base16 and C1-V80. Initialize from terminal-Q32 Page-Fisher R8 (`mse_base_q32_r8`). This diagnostic changes only the residual encoder E and query factors U. No Adam, autograd, payload fitting, Base fitting, rank allocation, or deployment selector change.

Real-cache smoke: Qwen3-8B-Base layer 33, KV group 0 and its four query heads. C4 fit windows 0/1; disjoint diagnostic windows 64/65. Query positions 26623, 28671, 30719, 32767; each query uses its full causal prefix. Page32, physical budget 64 pages / 2048 tokens including pinned page0. Raw dense-teacher V/K and Q come from existing captures; resident C1 codes are reconstructed with the frozen encoder. This is not a live end-to-end C1 rollout, all-layer evaluation, or RULER result.

## Objective and local solver

Residual proxy token score:

\[
\hat s_{h,i}=d^{-1/2}\left(q_h^\top k_i^{\rm Base}+(q_h^\top U_h)(E^\top r_i)\right).
\]

Deployment selection is preserved: per-head page LSE, normalization over non-sink pages, max across GQA query heads, then one fixed physical page set with pinned page0. It is not an independent per-head union. Define its equivalent log ranking score:

\[
a_p=\max_h\left[\log\sum_{i\in p}e^{\hat s_{h,i}}-\log\sum_{i\notin\mathrm{sink}}e^{\hat s_{h,i}}\right].
\]

Teacher page mass is computed from full exact FP32 QK softmax using cached BF16 Q/K. The physical-set objective is mean captured full-attention mass across query heads, then queries/windows. A pair weight is the head-mean mass of an omitted page minus that of an included page. Pinned pages are never candidates for removal. Each query pairs up to eight highest-mass omitted pages with the eight lowest-mass selected non-sink pages, retaining only positive teacher-mass gains.

The local surrogate is weighted squared hinge, `w * max(0, 0.05 - (a_positive - a_negative))^2`. The current max-head owner is fixed only while differentiating. Page derivatives include both the within-page softmax-weighted residual mean and the non-sink global mean; the latter cannot be dropped when the two pages have different active heads.

Alternate E and U for two sweeps. Linearize currently active hinge constraints and solve damped least squares in observation space:

\[
\Delta=A^\top(AA^\top+\lambda I)^{-1}b,
\qquad \lambda=0.01\,\operatorname{mean}\operatorname{diag}(AA^\top),
\]

with a numerical floor on the damping scale. A and b include square-root normalized pair weights. The parameter rank stays R8. The implementation uses a small dense linear solve, not a dense parameter-space Hessian or Adam. This observation-space implementation is for the bounded smoke; quadratic growth in the number of constraints has not been addressed for a large fit.

Try step sizes 1, 0.5, 0.25, 0.125. Recompute native BF16 concatenated-sidecar scores and the actual physical page selection for every trial. Accept only if fit mean teacher-mass coverage does not decrease and the fixed-pair native hinge loss decreases. Diagnostic data are not passed to the optimizer or used for step selection. This guarantees only non-decreasing aggregate fit coverage for accepted updates on the fixed fit set, not per-query, diagnostic, or task-accuracy improvement. Pairs are refreshed for every factor block.

## Verification

Five new CPU tests passed: finite-difference checks for both factor derivatives including normalization; pinned exclusion and positive swap weights; frozen inputs and non-decreasing fit coverage; no changes when full budget leaves no omitted pages; and native concatenated BF16 proxy arithmetic. The real-cache smoke additionally compares proxy scores bitwise against the production sidecar construction at every sampled causal query.

The first job (8300753) failed before optimization because the new driver looked for a nonexistent `source` field in the old direct manifest. The driver was corrected to verify its hash against the already-audited Q32 capture input record and to check the model configuration hash. No numerical gate was relaxed. The same settings and output directory were rerun as job 8300755, completed exit 0:0 in 10 seconds including startup. Program wall time was 1.44054 seconds; peak allocated GPU memory 0.18302 GiB. Hardware was one shared NVIDIA A100 80GB PCIe, two CPUs, 32 GiB requested host RAM, using `basis`. A known Transformers rotary-embedding deprecation warning remained. No original checkpoint was overwritten.

## Results

| Metric | Before | After | Change (percentage points) |
| --- | ---: | ---: | ---: |
| Fit full teacher mass | 91.638786% | 91.859531% | +0.220746 |
| Fit non-sink teacher mass | 91.143370% | 91.387480% | +0.244111 |
| Diagnostic full teacher mass | 96.268255% | 96.516240% | +0.247985 |
| Diagnostic non-sink teacher mass | 95.983690% | 96.248573% | +0.264883 |

Updates: sweep 0 E accepted at step 1; sweep 0 U rejected; sweep 1 E accepted at step 0.5; sweep 1 U accepted at step 1. Active constraints were 57, 57, 57 and 55 respectively. Fit improvement is partially enforced by the acceptance gate; diagnostic improvement is not. These tiny-data results verify that the mechanism can change real selected pages without increasing rank, not that it improves RULER or generalizes across layers/groups.

## Files and command

- Core: `basisserve/core/residual_page_ranking.py`.
- Driver: `evaluation/smoke_residual_page_ranking.py`.
- Tests: `tests/test_residual_page_ranking.py`.
- [Result JSON](../results/evaluation/page_rank_smoke/result.json) includes source hashes, accepted steps, metrics, and execution details.
- `results/evaluation/page_rank_smoke/group_factors.safetensors` contains only the fitted single-group E/U, not a deployable full-model bank.
- Logs: `logs/page-rank-smoke-8300755.out` and `.err`; the failed job's logs are retained too. Temporary submission scripts were removed.

Working directory `/deac/csc/yangGrp/zhangal/BasisServe-CALS`:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/smoke_residual_page_ranking.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_q32_r8 --query-capture results/calibration/q32_terminal8k --output-dir results/evaluation/page_rank_smoke
```

No formal multi-layer experiment, RULER job, GitHub commit, or push has been performed for this method.

## Requested layer/group expansion (not run)

This section records the cancelled A100 attempt. The user subsequently requested L40S; that expansion completed as job 8300760. See [the completed two-layer report](residual_page_ranking_l15_l33.md).

The user approved expanding to layers 15 and 33, all eight GQA groups each, keeping two fit/two diagnostic windows, four queries per window and two alternating sweeps unchanged. The driver now accepts `--layer` (15 or 33) and `--group` (0–7), and refuses to overwrite existing result/factor files. The five core tests passed again.

Array job 8300759 (0–15, concurrency two) was submitted to shared `gpu_small` for A100 80GB workers. It remained pending for resources. After the user said to skip if no A100 was free, a read-only node allocation check found all eight A100 80GB and all eight A100 40GB GPUs allocated. The pending array was cancelled before any experiment started. Existing smoke artifacts remain unchanged; no expanded result is available. No other jobs were cancelled.

The proposed command is the smoke command above plus `--layer 15/33 --group 0–7`, with separate output directories `results/evaluation/page_rank_l15_l33/l{layer}_g{group}`. The temporary submission script was removed after submission. No GitHub commit or push performed.
