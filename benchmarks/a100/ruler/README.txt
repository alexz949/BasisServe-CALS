Llama-3.1-8B-Instruct V96 exact-score Page4 RULER-128K
1100 samples, 11 tasks x 100. 397 original RTX A6000 results and 703 resumed A100-SXM4-80GB results.
Page size 4; routed historical budget 2048 plus recent 64; no sink tokens.
Exact full-dimensional QK FP32 page scoring; this differs from the paper B16R16 approximate router.
11-task mean: 81.6681818182%; paper 10-task mean (excluding niah_single_3): 79.935%.
All 1100 records passed the original audit_result checks. Original archive files and configured assets were hash-verified.
8/8 cross-GPU replay generated token sequences and scores match. Prefill logits hashes differ; output-equivalence was explicitly accepted.
No matched proxy rerun is included, so this is not a paired comparison against the paper router.
Environment: lowrank; torch 2.13.0+cu126, transformers 5.17.0, triton 3.7.1.
Run command: bash /mnt/basiskv/ruler/setup/resume.sh
Audit command: python /mnt/basiskv/ruler/setup/audit_final.py
Original HF archive revision: bb3e548586ecd47cddd8201eb3e1f33077600205 (results/audits-20261109).
provenance/source preserves the exact configured source snapshot. GitHub scripts preserve this run's machine-specific paths.
Deploy these orchestration files under /mnt/basiskv/ruler/setup. fetch.py reconstructs the isolated code and input snapshot; inspect env.sh for machine-specific environment paths. resume.sh records the authorization already granted for this completed run.
