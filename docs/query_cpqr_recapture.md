# Query CPQR: local fit-query recapture preparation

Status: one-window repeated capture smoke, formal 64-window capture/merge, and real-query Q32 CPQR audit completed successfully on 2026-09-10. See `results/evaluation/cpqr_q32/summary.md`: all 12 layer/bin combinations selected identical pivot sequences.

Reuse `results/calibration/v96kl_64x32k/windows.safetensors`, fit windows 0–63 only. These are the local fixed windows used by the V96-KL bank, not the missing historical remote captures. Dense Qwen3-8B-Base BF16 SDPA backbone forward, no KV cache or logits; capture post-q_norm/post-RoPE queries in layers 0, 15, 35 at positions 63, 127, ..., 32767. Each window saves `[3,512,32,128]` BF16 queries. Full capture is approximately 768 MiB. No Base, V, or residual fitting occurs.

The model is fully resident per GPU; two independent workers split 64 windows into even/odd IDs. Two CPU/OMP/MKL/OpenBLAS threads per worker. Prepare for roughly 40 GiB GPU capacity per worker (estimate, to be checked in smoke); host capacity budget 32 GiB per worker. One-window repeated capture checks exact reproducibility and records peak allocated GPU memory. It does not prove identity with the old layerwise capture. The later audit compares CPQR with Cholesky recomputed on the same newly captured Q.

The current host has no `sbatch` or `srun` in PATH. The user explicitly authorized direct execution on this server and requested GPU selection with `nvidia-smi`; the Slurm exception is settled. The user subsequently authorized the smoke on GPU 6, then the formal 64-window capture on GPUs 6/5, and then the CPU audit. All four commands/stages below have completed. Recheck resources before any further launch.

Working directory: `/home/lz299/BasisServe-CALS`. Environment: `lowrank`.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=.
mkdir -p results/logs/query_cpqr
set -o noclobber
```

1. One-window smoke, two identical forwards, GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.capture_query_cpqr capture --model /home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --calibration results/calibration/v96kl_64x32k --layers 0,15,35 --window-count 1 --num-shards 1 --shard-index 0 --verify-repeat --output-dir results/calibration/cpqr_smoke > results/logs/query_cpqr/capture_smoke.log 2>&1
```

2. Formal capture: launch these two commands concurrently as independent supervised processes, after smoke passes:

```bash
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.capture_query_cpqr capture --model /home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --calibration results/calibration/v96kl_64x32k --layers 0,15,35 --window-count 64 --num-shards 2 --shard-index 0 --output-dir results/calibration/cpqr_candidates > results/logs/query_cpqr/capture_0.log 2>&1
CUDA_VISIBLE_DEVICES=5 python -u -m evaluation.capture_query_cpqr capture --model /home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --calibration results/calibration/v96kl_64x32k --layers 0,15,35 --window-count 64 --num-shards 2 --shard-index 1 --output-dir results/calibration/cpqr_candidates > results/logs/query_cpqr/capture_1.log 2>&1
```

3. CPU merge and hash/coverage checks, after both workers succeed:

```bash
CUDA_VISIBLE_DEVICES= python -u -m evaluation.capture_query_cpqr merge --num-shards 2 --output-dir results/calibration/cpqr_candidates > results/logs/query_cpqr/capture_merge.log 2>&1
```

4. CPU CPQR audit, two threads, budget 8 GiB host RAM, no GPU:

```bash
CUDA_VISIBLE_DEVICES= python -u -m evaluation.audit_query_cpqr --candidate-capture results/calibration/cpqr_candidates --layers 0,15,35 --queries-per-bin 8 --output-dir results/evaluation/cpqr_q32 > results/logs/query_cpqr/audit_q32.log 2>&1
```

The audit preserves four bins, FP64 per-head whitening and all window/head observations. It records production/feature-Cholesky/CPQR pivots, Gram differences, pivot gaps, selected singular values, numerical rank and projection residual energy. Rank-deficient sets have JSON null condition number. No routing mass recall, Fisher loss, or downstream score is measured at this stage.

Capture manifests and per-window records retain input hashes, code hash, library versions, execution command, GPU model, timing, repetition checks and peak memory. Completed files are immutable; reruns verify existing records and reuse them. Shell noclobber preserves logs; any resume needs an explicit new log name. The audit requires a fresh output directory.

## Completed smoke

Executed the exact stage-1 command above in `lowrank`, directly on A100-SXM4-80GB GPU 6 with two CPU threads. Exit code 0. Window 0, 32768 tokens, layers 0/15/35: saved BF16 shape `[3,512,32,128]`. Two complete dense-backbone forwards produced bitwise identical selected Q. Program-reported window time, including both forwards and artifact saving, was 21.118185 seconds; this excludes model loading. Peak allocated GPU memory was 21,643,899,392 bytes (20.157 GiB). GPU memory was released on completion.

Artifact: `results/calibration/cpqr_smoke/shard_0/window_000.safetensors` (approximately 12 MiB). Independently checked SHA256: `bb2e755ac033dcb05f6793ec0fa3e44e04ffdda27794df07e6591ee8e6fb3771`, matching its JSON record. Log: `results/logs/query_cpqr/capture_smoke.log`. The only logged warning was the Transformers `torch_dtype` deprecation. No OOM, nonfinite-Q assertion, or repetition mismatch occurred. This checks capture execution and repeatability only; it does not establish CPQR equivalence or routing quality.

## Completed 64-window capture

Executed both exact stage-2 commands concurrently, followed by the stage-3 CPU merge, in `lowrank`. GPU 6 captured 32 even window IDs; GPU 5 captured 32 odd window IDs. Each worker used two CPU/OMP/MKL/OpenBLAS threads. Both capture processes and the merge exited 0 without retries. Every saved tensor passed BF16 shape `[3,512,32,128]` and finite-value assertions. The merge rehashed all 64 artifacts, verified identical protocols and disjoint complete coverage of fit IDs 0–63, and wrote the top-level manifest.

Output: `results/calibration/cpqr_candidates` (769 MiB filesystem usage; 768 MiB Q tensor payload). Manifest SHA256: `c7b10fb5cb4088d5ff45f641d508bc3ccfcc31fd7b766bbb5bd8fd87826e0160`. Logs: `results/logs/query_cpqr/capture_0.log`, `capture_1.log`, `capture_merge.log`. Peak allocated GPU memory was 20.157 GiB on each worker. Shared-GPU available memory fell during execution, but neither process encountered OOM. Only the `torch_dtype` deprecation warning appeared. Formal capture performs one forward per window; repeated-forward verification was done in the separate smoke.

These artifacts contain 64 x 512 = 32,768 candidate positions per captured layer, each retaining 32 heads of width 128. No validation windows were captured. The subsequent authorized CPU audit produced CPQR results in `results/evaluation/cpqr_q32`; no residual factors, attention-mass recall, or downstream metrics have been computed in this sequence.
