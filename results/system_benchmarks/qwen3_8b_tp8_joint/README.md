# Qwen3-8B TP8 Joint Batch Sweep

Status: complete. All 28 formal attempts finished: 26 successful and 2 Dense prefill OOM.
All 208 successful rank records were validated. See [full results](RESULTS_SUMMARY.md) and [paired CSV](comparison.csv).

## Published Raw Data and Reproduction

HF repository `alexz949/BasisServe-CALS`, immutable revision `8ee91f2b2f88a90284b2ef004b8107f06ec03c97`:

- [Complete raw archive](https://huggingface.co/alexz949/BasisServe-CALS/resolve/8ee91f2b2f88a90284b2ef004b8107f06ec03c97/results/system_benchmarks/qwen3_8b_tp8_joint/raw.tar.gz)
- [507-file inventory](https://huggingface.co/alexz949/BasisServe-CALS/blob/8ee91f2b2f88a90284b2ef004b8107f06ec03c97/results/system_benchmarks/qwen3_8b_tp8_joint/raw_manifest.json)

The archive contains all formal/smoke rank records, logs (including initial smoke/test failures), exact commands, manifests, and `source.tar.gz`, retaining repository-relative paths. Archive member bytes were compared directly; remote names and sizes were verified. No SHA256 validation was performed.

**For reproduction, restore the archived `source.tar.gz` into a separate checkout before using the commands below.** It contains the actual tested Qwen3-8B runtime, kernels and driver. The shared runtime on GitHub main is intentionally not overwritten by this results publication and is not a substitute for that frozen source. External model/factor/prompt inputs are identified below; they are not duplicated in the raw archive.

GitHub contains only the four preparation/runner/summary/test files and five Markdown/CSV artifacts approved for this publication. Archived Markdown retains its pre-publication status; this README records the subsequent upload.

## Formal Highlights

| Context | Largest successful tested Dense batch | Largest successful tested Basis batch | Dense OOM |
| --- | ---: | ---: | --- |
| 65536 | 14 | 16 | B16, prefill |
| 130048 | 6 | 8 | B8, prefill |

These are whole-run completion results on this grid, not exact maximum batches or decode capacity limits.

- 65536/B14: decode peak allocated memory 18.729 -> 11.507 GiB (38.560% less); decode latency 41.052 -> 36.198 ms/step (1.134x).
- 130048/B6: decode peak allocated memory 16.345 -> 10.268 GiB (37.179% less); decode latency 40.772 -> 32.465 ms/step (1.256x).
- Basis also completed 65536/B16: 36.553 ms/step, 438.028 wall tokens/s, 12.574 GiB decode peak.
- Basis also completed 130048/B8: 32.560 ms/step, 245.920 wall tokens/s, 12.313 GiB decode peak.
- At 65536/B1 Basis has higher decode peak memory (4.536 vs 4.059 GiB); the observed memory crossover is B2. At 130048 it is already lower at B1.
- All values are single-trial measurements. Failed Dense points have no speedup or decode-memory comparison.

## Verification

- All 36 layers passed finite-value, shape and FP64 coordinate-map preservation checks before BF16 export. This is not a model quality evaluation.
- 25 focused tests passed with CUDA_HOME configured. An earlier test invocation omitted CUDA_HOME and failed three extension tests; its log is preserved.
- Initial Dense smoke exposed a missing Qwen3 36-layer geometry entry. Added the matching geometry, then reran the same settings successfully. This was not OOM.
- Both final smoke trials completed on all 8 ranks, with active batch 6. Only 2 conditioning and 8 measured steps: these are functional smoke figures, not formal speed claims.
- CPU affinity was applied, but the container denied host NUMA memory-policy binding; host NUMA placement is not verified.
- After the formal run, `tar -dzf results/system_benchmarks/qwen3_8b_tp8_joint/source.tar.gz` succeeded: archived source files match the working files. No hashes were computed.

| Smoke (4096/B6) | Decode ms/step | Wall tokens/s | Decode peak allocated GiB |
| --- | ---: | ---: | ---: |
| Dense | 42.544 | 141.506 | 3.360 |
| Basis Joint V96 | 33.533 | 179.193 | 4.380 |

## Protocol

- Model: `Qwen/Qwen3-8B`, post-trained, not Base; snapshot `b968826d9c46dd6066d109eabc6255188de91218`.
- Environment: `basis`, TP8, 8 x L40S, BF16. No KV4 or A8.
- Dense: full GPU-resident K/V, PyTorch Flash SDPA.
- Basis: Joint V96, B16R16 full scan, historical exact K in pinned CPU memory with GPU slots. No two-stage routing.
- 65536 tokens: batches 1, 2, 4, 6, 8, 10, 12, 14, 16.
- 130048 tokens: batches 1, 2, 4, 6, 8 only.
- One trial per arm/point, 16 conditioning + 128 measured decode forwards; 28 formal trials.
- Primary metric: maximum single-rank peak decode allocated memory. Also report allocated resident memory, reserved memory, prefill peak, latency, wall throughput, host memory and failure phase.
- Prefill OOM is not a measured decode capacity limit. This is not request E2E or a quality evaluation.
- Matching checkpoint runtime: YaRN4 with original context 32768, maximum 131072, theta 1000000.

## Inputs

HF repository `alexz949/BasisServe-CALS`, revision `8d96d7b135cd82647727fd5ca0675222adb29c1c`:

- `checkpoints/attention_c1/qwen3_8b_post_uniform_v96_128k_als6_retrievalmix/attention_v96`
- Same bank's `router_b16r16`, not the score or smaller-page variants.
- `calibration/qwen3-8b-post-128k/c4-48x128k/windows.safetensors`: first B rows, prefix of requested length, identical between arms. Calibration inputs are used for systems timing, not held-out quality claims.

Factors undergo structure, finite-value and coordinate-map checks; no SHA256 calculation or validation. Existing 32B and Llama results are preserved separately.

## Commands

Working directory: `/workspace/BasisServe-CALS-opt`.

Preparation (CPU, environment `basis`):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/prepare_qwen3_32b_joint_v96.py --model qwen8 --snapshot /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/8d96d7b135cd82647727fd5ca0675222adb29c1c --output /workspace/runs/qwen3-8b-joint-v96/factors > results/system_benchmarks/qwen3_8b_tp8_joint/prepare.log 2>&1
```

Smoke: same launcher below with `--phase smoke`; tests both arms at 4096/B6, 2 conditioning + 8 measured steps. Save as `smoke.log`.

Formal, after smoke and explicit approval:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp8_joint_batch_sweep.py --phase formal --models qwen8 --max-batch-128k 8 --output results/system_benchmarks/qwen3_8b_tp8_joint > results/system_benchmarks/qwen3_8b_tp8_joint/formal.log 2>&1
```

No Slurm configuration on this machine; use the previously agreed direct execution. Per-trial logs and rank records are retained in the published HF archive linked above. Publication was approved separately after completion.

Post-run validation and summary (`basis`):

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m benchmarks.system.summarize_tp8_joint_batch_sweep --output results/system_benchmarks/qwen3_8b_tp8_joint > results/system_benchmarks/qwen3_8b_tp8_joint/summary.log 2>&1
```
