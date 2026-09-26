# Pre-Shutdown Backup Audit

Date: 2026-09-26. Initial audit followed by user-approved backup preparation.

## Backup Status

The user approved GitHub main publication with message `commit all from run pod`.
The selected SVD/V96 code, tests and summaries are included in this publication.
17 tests passed in the clean publication worktree. V96 real smoke remains FAILED.

HF backup is INCOMPLETE. Fisher loose-file upload encountered the repository's
20000-file limit, followed by a 128-commits/hour rate limit. Do not release the
server based on this commit. Local originals and recovery progress are intact.
Archive replacement requires approval and must wait for the rate-limit window.

Target: https://huggingface.co/alexz949/BasisServe-CALS/tree/main/migration/server-backup-20260926

Local packs: `/workspace/server-backup-20260926/`. The complete Git recovery
pack `workspaces.tar.gz` is LOCAL ONLY, not cleared for public upload. The
public source snapshot omits Git history/credentials and redacts token-shaped
strings in three vendored Transformers test files; originals are unchanged.

This backup scope is not an exhaustive archive of every `/workspace/runs`,
`ICLR-results`, environment or downloaded model on the machine.
Environment: basis. Command: `python /tmp/audit_server_backup.py`.
GitHub main: `370e3a74fea7b1b7d2701d41a767b839510f92a4`.
HF revision: `8ee91f2b2f88a90284b2ef004b8107f06ec03c97`.

## Highest Priority

- New TP-source SVD code, documentation and results are unpublished. Results occupy about 15 GiB: covariance 5.7 GiB, factors 8.6 GiB, plus smoke.
- V96 objective runner/tests are unpublished. Smoke failed saving a non-contiguous decoder tensor; no formal run or PPL.
- Quantization code and text results are already on GitHub in f52a424. Local quantizer/checkpoint binaries are not GitHub tree entries; HF has no corresponding new experiment directories.
- Main worktree contains unresolved rebase conflicts. Preserve its exact state before attempting any cleanup.
- Quest checkout contains local patches; a published patch exists, but its equivalence to the current checkout has not been verified.

## Runs Directory

| Directory | GiB | Files |
| --- | ---: | ---: |
| /workspace/runs/l31-cal128 | 174.80 | 12494 |
| /workspace/runs/l31-ruler220-ruler-src | 0.07 | 67 |
| /workspace/runs/l31-ruler220 | 0.58 | 815 |
| /workspace/runs/l31-router-source | 82.42 | 3720 |
| /workspace/runs/qwen3-32b-densev-ruler30 | 2.48 | 326 |
| /workspace/runs/qwen35-mixed-eval | 18.91 | 579 |
| /workspace/runs/qwen3-32b-joint-v96 | 3.76 | 65 |
| /workspace/runs/qwen3-8b-joint-v96 | 0.85 | 37 |

These totals are NOT all confirmed missing backups. Downloaded models and published checkpoints may be duplicated locally.
Three approximately 82 GiB Fisher directories deserve explicit backup decisions; HF has no corresponding Llama statistics prefix.

## Source Files Absent at Their Paths on Main

Missing at a source path does not rule out inclusion in a previously uploaded source archive. Different files can be older local versions. Do not overwrite main wholesale.

### /workspace/BasisServe-CALS

74 absent paths; 68 differing files.

- `basisserve/core/llama31_70b_vllm_c1.py`
- `basisserve/core/llama31_8b_vllm_c1.py`
- `basisserve/core/tp_source_svd.py`
- `basisserve/vllm/llama31_70b_c1.py`
- `basisserve/vllm/llama31_8b_instruct_c1.py`
- `benchmarks/system/bench_paper_faithful_lrqk.py`
- `benchmarks/system/bench_paper_faithful_shadowkv.py`
- `benchmarks/system/bench_tp1_lrqk_clean.py`
- `benchmarks/system/bench_tp1_page_retrieval.py`
- `benchmarks/system/bench_tp1_page_retrieval_decode.py`
- `benchmarks/system/bench_tp1_shortlist.py`
- `benchmarks/system/bench_tp1_slot_refresh.py`
- `benchmarks/system/lrqk_clean.py`
- `benchmarks/system/page_key_slots.cu`
- `benchmarks/system/profile_tp1_compute.py`
- `benchmarks/system/run_paper_faithful_tp1.py`
- `benchmarks/system/run_tp1_compute_profile.py`
- `benchmarks/system/run_tp1_dense_k_offload.py`
- `benchmarks/system/run_tp1_frozen_decode.py`
- `benchmarks/system/run_tp1_lrqk_clean.py`
- `benchmarks/system/shortlist_candidates.py`
- `benchmarks/system/summarize_tp1_compute_profile.py`
- `benchmarks/system/summarize_tp1_lrqk_aligned.py`
- `benchmarks/system/summarize_tp1_page_retrieval.py`
- `benchmarks/system/tp1_profile_trace.py`
- `benchmarks/system/warp_slot_plan.cuh`
- `docs/tp_source_svd.md`
- `evaluation/audit_llama_cal128_rank_bank.py`
- `evaluation/benchmark_vllm_llama31_70b_c1.py`
- `evaluation/benchmark_vllm_llama31_8b_instruct_c1.py`
- `evaluation/build_llama31_8b_instruct_iclr_v_checkpoint.py`
- `evaluation/build_llama31_8b_instruct_palu_m_checkpoint.py`
- `evaluation/eval_llama31_8b_instruct_iclr_quality.py`
- `evaluation/eval_llama_cal128_longbench.py`
- `evaluation/eval_llama_cal128_longbench_v2.py`
- `evaluation/eval_llama_cal128_longbench_v2_dense.py`
- `evaluation/eval_llama_router_source_ruler.py`
- `evaluation/eval_tp1_decode_v8_ruler.py`
- `evaluation/fit_llama31_8b_instruct_c1_joint.py`
- `evaluation/fit_qwen3_32b_densev_tp2.py`
- `evaluation/llama_cal128_deployment.py`
- `evaluation/prepare_llama31_8b_instruct_c1_windows.py`
- `evaluation/prepare_llama_cal128.py`
- `evaluation/prepare_llama_longbench_full.py`
- `evaluation/prepare_llama_longbench_v2.py`
- `evaluation/prepare_llama_ruler220_subset.py`
- `evaluation/prepare_qwen3_32b_densev_128k_windows.py`
- `evaluation/prepare_qwen3_32b_ruler30_subset.py`
- `evaluation/run_llama31_8b_instruct_c1_two_sided_factorized_kl_sharded.py`
- `evaluation/run_llama_router_source.py`
- `evaluation/run_qwen3_32b_densev_tp2x4.py`
- `evaluation/run_qwen3_32b_tp8_v96.sh`
- `evaluation/run_tp_source_svd.py`
- `evaluation/run_v96_objective.py`
- `evaluation/summarize_llama_cal128_longbench_v2_dense.py`
- `evaluation/summarize_vllm_llama31_70b_c1.py`
- `evaluation/summarize_vllm_llama31_8b_instruct_c1.py`
- `evaluation/verify_llama_ruler220_regeneration.py`
- `paper_figures/tp1/MAIN_FIGURE_CAPTIONS.md`
- `paper_figures/tp1/REQUEST_BREAKDOWN_SUMMARY.md`
- `paper_figures/tp1/fig2_request_breakdown.py`
- `paper_figures/tp1/inputs/request_breakdown/Pasted text.txt`
- `paper_figures/tp1/prepare_request_breakdown.py`
- `paper_figures/tp1/request_breakdown_caption.md`
- `tests/smoke_tp_source_svd.py`
- `tests/test_llama31_70b_vllm_c1.py`
- `tests/test_llama31_8b_vllm_c1.py`
- `tests/test_llama_router_source.py`
- `tests/test_lrqk_clean.py`
- `tests/test_lrqk_clean_fit.py`
- `tests/test_tp1_profile_trace.py`
- `tests/test_tp1_request_breakdown.py`
- `tests/test_tp_source_svd.py`
- `tests/test_v96_objective.py`

### /workspace/BasisServe-CALS-opt

24 absent paths; 46 differing files.

- `basisserve/core/quest_native_tp8.py`
- `basisserve/core/quest_tp8.py`
- `basisserve/core/qwen3_tp8_v_latent.py`
- `basisserve/kernels/csrc/quest_native_bindings.cu`
- `basisserve/kernels/latent_value_decode.py`
- `basisserve/kernels/quest_native.py`
- `basisserve/kernels/quest_paged.py`
- `benchmarks/system/bench_latent_value_decode.py`
- `benchmarks/system/bench_qwen3_tp8_v_latent.py`
- `benchmarks/system/bench_qwen_joint_attention_opt.py`
- `benchmarks/system/bench_qwen_joint_cold_attention.py`
- `benchmarks/system/bench_qwen_joint_kernel_opt.py`
- `benchmarks/system/bench_qwen_joint_model_ablation.py`
- `benchmarks/system/run_qwen3_32b_joint_capacity.py`
- `benchmarks/system/run_qwen3_32b_joint_smoke.py`
- `benchmarks/system/run_tp8_dense_flash_rerun.py`
- `benchmarks/system/summarize_qwen3_32b_joint_capacity.py`
- `benchmarks/system/summarize_tp8_dense_flash_rerun.py`
- `benchmarks/system/summarize_tp8_v_latent.py`
- `docs/qwen32b_tp8_joint_optimization_review.md`
- `tests/test_joint_prefix_append.py`
- `tests/test_quest_native.py`
- `tests/test_quest_paged.py`
- `tests/test_tp_dense_flash_decode.py`

## Verification Limits

No SHA256. Same size is not content verification. Archive contents not inspected; missing loose file does not prove absent from archives. Different source may be older, not unpublished.
Scope: primary and opt source trees, their result binaries, /workspace/runs, and the named HF repository. Other branches, private repos, external backups and every cache were not exhaustively checked.
Do not release/delete this server based on same-size matches alone. All paths and HF size metadata are in inventory.json.
