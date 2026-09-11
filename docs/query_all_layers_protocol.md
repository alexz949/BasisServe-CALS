# All-layer Q32 CPQR / strong-RRQR check

Status: user-authorized full pipeline completed on 2026-09-10. Both capture workers, merge, CPQR audit, sRRQR audit and final result check exited 0 without retries. This expands only layer coverage to 0–35; query budgets, windows, whitening, position bins and sRRQR bounds remain unchanged. Full results: `results/evaluation/srrqr_all_q32/summary.md`.

Inputs: local dense Qwen3-8B-Base model and the fixed `results/calibration/v96kl_64x32k/windows.safetensors`, fit windows 0–63 only. Each new window artifact contains BF16 `[36,512,32,128]` post-q_norm/post-RoPE Q; approximately 144 MiB/window, 9 GiB total. The existing three-layer capture and results remain unchanged. Capture performs the same dense full-window model-backbone forward; additional hooks save candidate Q for all layers, with no V/Base/residual fitting.

Reuse the existing capture/audit code with explicit layer lists; no implementation or model configuration changes are needed. Stages: two-worker capture, merge/hash/coverage validation, full-feature CPQR comparison, then full-spectrum Gram-root strong-RRQR comparison. Downstream stages run only after predecessors succeed. The expanded audit covers 144 layer/bin cases for CPQR and 288 layer/bin/bound cases for sRRQR. Report disagreements or errors without assuming the three-layer outcome generalizes.

Environment `lowrank`, direct execution authorized for this server. At preparation, GPU 5 had 81153 MiB free and GPU 6 had 28394 MiB free, both 0% utilization. The prior capture measured 20.157 GiB peak allocated per worker. Two processes use one GPU each and two OMP/MKL/OpenBLAS/PyTorch threads each; recheck GPU availability immediately before launch. Saving more layers retains CPU tensors, not additional persistent GPU tensors, but the actual all-layer peak remains to be measured. Plan host capacity 32 GiB per capture worker. CPU audits run sequentially with two threads, no GPU, host capacity budget 8 GiB; these are estimates, not reserved allocations or measured peaks.

Working directory `/home/lz299/BasisServe-CALS`. Exact shared arguments:

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=.
set -o noclobber
query_all_layers=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35
query_model=/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4
```

Capture: launch these as two independent supervised processes with the same setup above (even/odd window IDs). GPU assignments are provisional until the pre-launch nvidia-smi check.

```bash
CUDA_VISIBLE_DEVICES=5 python -u -m evaluation.capture_query_cpqr capture --model "$query_model" --calibration results/calibration/v96kl_64x32k --layers "$query_all_layers" --window-count 64 --num-shards 2 --shard-index 0 --output-dir results/calibration/cpqr_all_candidates > results/logs/query_cpqr/capture_all_0.log 2>&1
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.capture_query_cpqr capture --model "$query_model" --calibration results/calibration/v96kl_64x32k --layers "$query_all_layers" --window-count 64 --num-shards 2 --shard-index 1 --output-dir results/calibration/cpqr_all_candidates > results/logs/query_cpqr/capture_all_1.log 2>&1
```

After both capture processes succeed, merge all 64 artifacts and check hashes, matching protocols, and disjoint complete fit-window coverage:

```bash
CUDA_VISIBLE_DEVICES= python -u -m evaluation.capture_query_cpqr merge --num-shards 2 --output-dir results/calibration/cpqr_all_candidates > results/logs/query_cpqr/capture_all_merge.log 2>&1
```

Run CPQR versus production/explicit-feature Gram pivoting, verifying full FP64 feature-Gram equality and recording initial geometry:

```bash
CUDA_VISIBLE_DEVICES= python -u -m evaluation.audit_query_cpqr --candidate-capture results/calibration/cpqr_all_candidates --layers "$query_all_layers" --queries-per-bin 8 --output-dir results/evaluation/cpqr_all_q32 > results/logs/query_cpqr/audit_cpqr_all_q32.log 2>&1
```

Run the existing Gu–Eisenstat swap test independently from CPQR at f=2 and f=1.01. The full Gram-root initialization must match the explicit-feature CPQR reference, and no spectral truncation is introduced:

```bash
CUDA_VISIBLE_DEVICES= python -u -m evaluation.audit_query_srrqr --candidate-capture results/calibration/cpqr_all_candidates --cpqr-reference results/evaluation/cpqr_all_q32 --layers "$query_all_layers" --bounds 2,1.01 --max-swaps 512 --output-dir results/evaluation/srrqr_all_q32 > results/logs/query_cpqr/audit_srrqr_all_q32.log 2>&1
```

New capture/result directories and noclobber logs preserve previous artifacts. Reusing completed capture records requires identical protocol/source hash and valid saved hashes. Audit output directories must be new. If a stage fails, inspect its existing log before proposing a repair or resume; do not silently replace the baseline. Completion reporting will include per-layer selection changes, swap counts, stopping-condition compliance, max rho, volume/condition/projection metrics, commands, environment, logs and failures. No mass recall or downstream task conclusions can be drawn from this selector-only experiment.

## Completed results

All 144 CPQR layer/bin cases retained the production Gram-pivot sequence (1152 selected position slots). Both sRRQR bounds retained exactly the same selections: 144/144 identical bins and zero swaps at each bound. All 288 bound/bin cases satisfied the stopping condition. The largest initial/final rho was 1.004078783579815 in layer 1, zero-based bin 0, corresponding to approximately 0.407878% potential best single-swap volume improvement, below the f=1.01 threshold. This does not imply global maximum volume or exclude improvements at a tighter threshold.

Final checks verified all 72 result JSONs, their source/input/reference hashes, complete 64-window and 36-layer coverage, and numerical geometry/selection consistency. All 192 overlapping window/layer Q tensors agree bitwise with the prior three-layer capture. Selected-block condition numbers range from 1.110502 to 1.525972. Maximum explicit-feature Gram discrepancy is 6.536993e-12; full-root Gram discrepancy is 6.394885e-13, with no eigenvalues clipped.

Capture ran directly on GPUs 5/6 in lowrank, two CPU threads per worker, peak allocated GPU memory 20.157 GiB each. CPU audits used two threads and no GPU. Captured tensor payload is 9 GiB (9.1G reported filesystem usage). The only logged warnings were the capture-time torch_dtype deprecation messages. No OOM, failed numerical checks, or retries occurred.

Candidate manifest SHA256: `fbbc190c863dd5d8bd44079ee06e7c139fea360829828b47b509447bd382e05d`. Detailed results, 144-row CSV and 72 result hashes are in `results/evaluation/srrqr_all_q32/{summary.md,bins.csv,audit.json}`. Final result-check log: `results/logs/query_cpqr/all_result_check.log`. No residual refit, attention-mass recall, or downstream evaluation was launched.
