# Three-sample multikey selected-page trace

Qwen3-8B-Base, frozen C1-V80, closed-form Base16, Q32 Fisher R8. Shared full C1 prefill; independent greedy decode forks. Page32/B2048 including pinned page0, all36 layers. No refitting or parameter selection. Environment: basis; NVIDIA L40S.

Exact-pages uses FP32 exact QK for the same non-pinned head-normalized GQA-max selector. Selected attention remains native BF16 exact-K/C1-V. Full exact-K uses full SDPA output. Each row compares exact/proxy selection on the SAME state within its arm; states across arms differ.

| Sample | Reference | Full exact-K | Exact pages B2048 | Q32 proxy | First Q32 divergent generated token |
| --- | --- | --- | --- | --- | ---: |
| 33 | 2378217 | ` 2378217.` (score 1.00) | ` 2378217.` (score 1.00) | ` 2378920.` (score 0.00) | 6 |
| 37 | 3812733 | ` 3812733.` (score 1.00) | ` 3812733.` (score 1.00) | ` 3812745.` (score 0.00) | 7 |
| 38 | 1368711 | ` 1368711.` (score 1.00) | ` 1368711.` (score 1.00) | ` 1785139.` (score 0.00) | 3 |

First-token numbering includes the shared leading space from full prefill. Matching generated prefixes do not imply identical hidden states/cache across arms.

## Same-state answer-page omissions on the full-attention trajectory

Below are the five largest per-head teacher-mass omissions per sample, up to and including the generated-token position where the independent Q32 run first diverges. These are observational selector comparisons, not layer intervention effects. Exact selected the page in the listed group; proxy did not.

| Sample | Predicted token # | Layer | Group | Answer page | Max head mass | Exact rank | Proxy rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 33 | 6 | 19 | 6 | 161 | 79.7902% | 1 | 313 |
| 33 | 5 | 19 | 6 | 161 | 57.7891% | 1 | 330 |
| 33 | 5 | 30 | 2 | 161 | 57.2196% | 1 | 331 |
| 33 | 3 | 23 | 3 | 161 | 51.8592% | 2 | 145 |
| 33 | 4 | 23 | 3 | 161 | 40.7011% | 2 | 74 |
| 37 | 6 | 15 | 2 | 435 | 92.8187% | 1 | 112 |
| 37 | 7 | 9 | 7 | 435 | 88.7623% | 1 | 76 |
| 37 | 6 | 7 | 1 | 435 | 82.3571% | 1 | 573 |
| 37 | 6 | 20 | 5 | 435 | 80.8199% | 1 | 65 |
| 37 | 7 | 7 | 1 | 435 | 71.5814% | 1 | 579 |
| 38 | 2 | 24 | 1 | 524 | 26.8612% | 1 | 279 |
| 38 | 3 | 24 | 1 | 524 | 17.2008% | 1 | 287 |
| 38 | 3 | 7 | 1 | 524 | 11.8432% | 5 | 76 |
| 38 | 3 | 21 | 4 | 524 | 10.6321% | 4 | 168 |
| 38 | 3 | 30 | 2 | 524 | 10.5823% | 1 | 171 |

## Artifacts and checks

Every sample has per-arm JSON step/layer records and safetensors containing exact/proxy selected IDs, per-head full page masses, normalized GQA-max scores, rank-min, cutoffs, overlap counts, and decode logits. Partial final pages are included. Support sentence spans and old-Q32 distractor spans are recorded. Minimum rank records ties; actual selected IDs remain authoritative.

Full and Q32 generated token IDs reproduce their previous runs exactly. All six reproduction checks, nine artifact hashes, and step/layer row counts passed. A separate 8-token smoke verified bitwise logit equality between traced and untraced full/Q32 paths.

This selected three-failure subset is a diagnosis, not an independent accuracy benchmark. Exact-pages answers all three correctly at the unchanged B2048 budget. This does not establish all-88-prompt accuracy or causal necessity of any individual omitted page.

Run log: formal array 8300813 completed on three L40S in 33–34 seconds per sample; summary job 8300818 completed in 13 seconds. Initial smoke 8300809 exhausted GPU memory because Mock call histories retained KV tensors; direct callable patches removed this retention. The unchanged-setting smoke 8300810 then passed in 34 seconds. Temporary submission files were removed.

## Commands

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/trace_multikey_pages.py --stage evaluate --sample-index 33 --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_q32_r8 --output-dir results/evaluation/multikey_trace
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/trace_multikey_pages.py --stage evaluate --sample-index 37 --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_q32_r8 --output-dir results/evaluation/multikey_trace
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/trace_multikey_pages.py --stage evaluate --sample-index 38 --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_q32_r8 --output-dir results/evaluation/multikey_trace
```
