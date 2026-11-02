# A5000 TP8 Experiment B — 4K/B1 pilot

2026-09-22. Status: verified; 9 measured trials, 3 separate full warmups, 96 rank records. This follows the old L40S measurement design, not its exact hardware/software/checkpoint identity or full grid.

## Results

Each summary is the median of three cohort summaries. Each cohort measures 128 complete autoregressive forwards, with max-rank CUDA event times per step and synchronized wall throughput.

| Arm | Three cohort mean steps (ms) | Median mean (ms) | Median tokens/s | vs Dense |
| --- | --- | ---: | ---: | ---: |
| Dense K128/V128 | 142.576 / 142.633 / 142.834 | 142.633 | 7.015 | 1.000× |
| ALS-full V96 | 110.601 / 111.267 / 111.521 | 111.267 | 8.992 | 1.282× |
| Joint V96 + B16R16 | 130.483 / 127.274 / 126.540 | 127.274 | 7.860 | 1.121× |

| Arm | Prefill peak allocated, max rank (GiB) | Decode-ready allocated, max rank (GiB) | Peak reserved, max rank (GiB) | Host exact-K capacity, whole replica (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Dense | 3.878 | 3.657 | 4.770 | 0 |
| ALS-full | 4.508 | 4.287 | 5.105 | 0 |
| Joint | 4.496 | 4.275 | 5.063 | 0.259 |

Memory values were identical across cohorts within each arm; these are PyTorch allocator measurements, not NVML peaks. All ranks passed the 20 GiB reserved-memory gate. At 4K/B1, Joint is approximately 14.4% slower than ALS-full, and neither compressed path reduces total resident allocation versus Dense. Compressed paths include replicated decoders and additional state; these totals are not cache-only storage. Profiling is needed to separate routing, slots, attention and communication costs. Do not extrapolate this cell to long contexts or large batches.

## Protocol and verification

- TP8, greedy own-trajectory decoding, ignore EOS for timing. P=4096, B=1. Warmup uses C4 window 47; measured cohorts use windows 0, 1, 2. Four distinct input hashes, identical across arms.
- Reset request lengths, counters, routing metadata and K-slot resident/lookup state before every trial. Retain compiled kernels and reusable engine allocations.
- Fresh real prefill emits one token, then 16 conditioning and 128 measured decode forwards: 145 tokens total. Cache length is 4112 before the first measured forward and 4240 after the last.
- Both compressed prefills use V96 cuDNN attention directly, without V128 padding. ALS-full uses fixed-capacity caches and a CUDA valid-length scalar. Separate NaN-tail/FP32 correctness smoke passed at 4K and 32K, maximum absolute error 0.000244.
- All 96 rank records passed rank IDs, finite logits, complete within-arm/cohort token agreement, matching prompts, dispatch agreement, max-rank timing reconstruction, and per-arm job consistency. All 19 recorded source hashes matched current source files at verification.
- Per rank/trial: Dense prefill calls=64; compressed prefill calls=32; decode calls=4608. Compressed arms each have 4672 prepared AllGathers. Joint routing, select/pack, persistent slots and compressed-V attention each have 4608 dispatches.
- After warmup/reset, Joint cohort 0 matches the earlier smoke prefix `[3742,315,468,6204,13,31245,1071,11,330,1687,1051,4661]`.
- Asset manifest SHA256: `435e186266ab7006e39786463edf06efc93041c16a61a1af73e96e18a824046e`.

## Environment and exact commands

Conda environment: `basis`. Node4: 8 × RTX A5000, SM86, 24564 MiB/GPU; driver 535.154.05; dual AMD EPYC 7352. Four NVLink pairs with `SYS` links between pairs (`node4_hardware.log`). PyTorch 2.6.0+cu118, Transformers 4.51.3, CUDA runtime 11.8, cuDNN 9.1.0. BF16, TF32 disabled, eager execution. Each allocation used 16 CPUs total; the node was not CPU/filesystem-exclusive.

```bash
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6 OMP_NUM_THREADS=2 MAX_JOBS=2
```

Dense completed under Slurm 209998:

```bash
conda run --no-capture-output -n basis torchrun --standalone --nproc_per_node=8 -m benchmarks.ampere.tp8_joint --assets results/system_benchmarks/ampere_tp8_joint/asset_manifest.json --output results/system_benchmarks/ampere_tp8_joint/decode_b/p4096_b1/dense --build-dir results/system_benchmarks/ampere_tp8_joint/tp8_smoke_4k_b1/build --mode dense --length 4096 --batch 1 --steps 145 --discard-first 16 --engine-warmup --cohorts 3 --warmup-offset 47
```

ALS-full and Joint completed under Slurm 210014, using a non-login shell and the same basis environment's Python directly:

```bash
export PATH=/home/lz299/miniconda3/envs/basis/bin:$PATH
for arm_mode in als_full joint; do
  /home/lz299/miniconda3/envs/basis/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 -m benchmarks.ampere.tp8_joint --assets results/system_benchmarks/ampere_tp8_joint/asset_manifest.json --output results/system_benchmarks/ampere_tp8_joint/decode_b/p4096_b1/${arm_mode} --build-dir results/system_benchmarks/ampere_tp8_joint/tp8_smoke_4k_b1/build --mode ${arm_mode} --length 4096 --batch 1 --steps 145 --discard-first 16 --engine-warmup --cohorts 3 --warmup-offset 47
done
```

Verification, exit 0:

```bash
/home/lz299/miniconda3/envs/basis/bin/python -m benchmarks.ampere.summarize_decode --input results/system_benchmarks/ampere_tp8_joint/decode_b/p4096_b1
```

## Execution warnings and remaining scope

Job 209998 was cancelled after 37m33s during the ALS-full portion after prolonged NFS RPC/conda activation waits. Delayed logs revealed ALS-full had completed warmup before cancellation; `als_full/warmup_209998/` preserves those excluded records. Dense's completed trials were retained. Job 210014 completed both remaining arms in 9m58s, exit 0; all ALS-full/Joint records used here belong to that job. No GPU jobs remain running.

The single-GPU masking smoke (209996) completed in 3m20s. Startup and between-trial filesystem waits are outside the decode timing intervals, but this is a sequential pilot on a node with observed I/O contention, not a noise-controlled publication run. Nonfatal `libnvrtc.so.11.2` load warnings remain. Node4 has no swap; low free RAM was mostly page cache, with ample available memory.

The bank's training-manifest identity still differs from the published retrievalmix manifest (`router_training_manifest_matches_current=false`). Assets are hash-verified, but historical quality-checkpoint equivalence is unconfirmed. L40S numbers remain historical and are not used for these speedups.

Only final-step slot counts and persistent host-K capacity are retained: no interval-average slot-hit rate, host RSS or sampled NVML peaks. This is decode-only Experiment B, not TTFT/E2E, capacity search or profiling. The larger A5000 grid remains unmeasured; next increment is a 16K/B1 probe. B16 needs more distinct prompt windows for three cohorts plus a separate full-batch warmup.

Raw outputs: `comparison.json`, `verify.log`, each arm's `cohorts.json`, `warmup/`, `cohort_0/` through `cohort_2/`, and their eight rank JSONs. Logs: `dense.log`, `als_full.log`, `joint.log`, `slurm.log`.
