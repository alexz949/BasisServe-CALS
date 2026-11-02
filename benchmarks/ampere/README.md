# Ampere TP8 Joint ALS + B16R16

This directory contains the rebuilt Llama-3.1-8B-Instruct TP8 decode path for the current 8 × RTX A5000 node. A5000 is **SM86**. The code also accepts SM80, but SM80 needs its own kernel measurements before any performance claim. The previous L40S results were measured on SM89 with 46 GB per GPU and remain historical results.

## Assets and checkpoint identity

`assets.py` pins a HF commit and verifies every layer of the original C4 V96 checkpoint, the retrievalmix V96 checkpoint, the current ALS40/PCG100 B16R16 bank, the C4 input windows, and the local Llama model config and weight index. It writes `results/system_benchmarks/ampere_tp8_joint/asset_manifest.json`; large tensor files remain in the HF cache.

The router bank was fitted using a retrievalmix V96 teacher, so `joint` and its matched `als_full` comparison use retrievalmix V96. The original C4 V96 is available for a separate `als_full` run. The bank record names training manifest SHA256 `4fa8237e16ff5776d29eed59c263413528414c40abd96c8cf47c360f462a7451`; the currently published retrievalmix manifest SHA256 is `86bcd936655076e03a3ff886befc79ee3181d3e2f6965b60f86b8c12becf3c5a`. This provenance difference is recorded in the asset and trial outputs. Bank to checkpoint equivalence beyond the published file hashes remains unverified.

## Decode path

Each rank owns one physical KV head and four query heads. V96 coordinates come from the frozen encoder. Prefill uses cuDNN causal attention directly with V96, with no V128 padding; Q projection is chunked before the complete prompt attention call. The prepared feature major AllGather assembles 32 query heads before the V decoder matrix. Joint decode keeps exact K in CUDA mapped host memory, ranks pages using the min/max coarse bound, scores at most 512 candidate Page32s with the Ampere WMMA fine router, packs 62 routed pages plus the 64 recent tokens, refreshes 2048 persistent K slots, and runs exact QK with compressed V96 slot attention. The source in `two_stage.py` is generated from the checked in CUDA router at build time and compiled for the active SM capability.

`kernel_smoke.py` checks fine scores against full scores, page and token support, first and repeated slot fetches, and V96 attention against a PyTorch reference at 4K and 32K. `prefill_smoke.py` compares direct V96 cuDNN and Triton attention to a padded Flash reference. At 4K/B1 on A5000, cuDNN took 0.362 ms and padded Flash took 0.370 ms; native Triton took 0.636 ms. These are five-repeat single-operator smoke timings, not a meaningful end-to-end speedup claim. `tp8_joint.py` checks all eight ranks, finite logits, generated token IDs, dispatch counts and source hashes. `--discard-first` excludes that many complete decode forwards, not the prefill-produced first token; `--prompt-offset` selects distinct C4 windows for later cohorts. The runner checks the model context limit before prefill. The SM86 extension source directory is shared across trials by default and can be set with `--build-dir`; Torch extension objects, Triton cache, and CUDA cache are placed on node-local `/tmp` to avoid shared-filesystem lock stalls. On node4, set `CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11` because its default GCC 12 lacks `cc1plus`. Logs and per rank JSON files are stored under `results/system_benchmarks/ampere_tp8_joint/`.

## A5000 workload grid

The first formal comparison follows the previous L40S **Experiment B**: a separate full engine warmup on different inputs, then three distinct measured prompt cohorts. Each cohort has a fresh prefill, 16 conditioning decode forwards and 128 measured decode forwards (`--steps 145 --discard-first 16 --engine-warmup --cohorts 3`). The prefill-produced token is not a decode forward. Reset request lengths, routing metadata, persistent-slot residents/lookups and dispatch counters before every cohort, while retaining compiled kernels and reusable engine allocations. Every arm/workload cell starts a fresh TP8 process. Use identical input cohorts across `dense`, matched `als_full`, and `joint`. ALS-full passes a fixed-capacity cache plus a CUDA valid-length scalar to its existing Triton kernel to avoid per-position compilation.

| Context | Batches to attempt |
| --- | --- |
| 4K | 1, 2, 4, 8, 16 |
| 16K | 1, 2, 4, 8 |
| 32K | 1, 2, 4 |
| 64K | 1, 2 |
| 96K | 1 |
| 130048 | 1 |

The A5000 has 24 GB physical memory. The 130048-token row reserves context for generation within the model's 131072-token limit. Each candidate cell must first pass a short memory probe with the intended cache capacity; keep it only when all eight ranks remain below 20 GiB reserved and token agreement and dispatch assertions pass. The grid above is a candidate set, not yet a measured feasible grid. For three measured input cohorts, choose prompt offsets 0, B and 2B. A separate full-batch warmup also needs B distinct windows: the current 48-window asset supports this protocol only up to B=12, so the B16 candidate requires additional real-text windows before a formal run. At B1 use warmup offset 47; larger batches must choose a disjoint in-range warmup span. This gate determines the executable grid without assuming that an L40S batch fits. The user confirmed proceeding to the 4K/B1 three-arm Experiment B pilot after the successful TP8 smoke.

For each formal trial record: the exact HF revision and asset SHA256, model/config hashes, prompt hash, source hashes, GPU name and capability, per rank token IDs and dispatch counters, prefill latency, warmup and steady decode CUDA event times, steady wall throughput, and peak allocated and reserved memory. Compare arms only within this A5000 grid.

`full_decode_smoke.py` verifies the fixed-capacity V96 decoder against FP32 attention with NaN-poisoned unused tails. On node4, Slurm job 209996 passed at valid lengths 4096, 4240, 32768 and 32912; maximum absolute error was 0.000244. Its startup encountered transient NFS RPC waits, not GPU compilation or swap pressure. Node4 has no swap configured. The hardware capture shows four NVLink pairs (0/1, 2/3, 4/5, 6/7), with `SYS` links between pairs; the old L40S report used a different topology.

The 4K/B1 pilot uses prompt windows 0, 1 and 2 for measurement and window 47 for full warmup. `summarize_decode.py --input results/system_benchmarks/ampere_tp8_joint/decode_b/p4096_b1` checks 96 rank records (three arms × four requests × eight ranks), matching prompts across arms, distinct warmup/cohort hashes, per-rank token and dispatch agreement, every max-rank decode timing, and recorded source hashes against current files. It writes `comparison.json`. Per-step slot-hit statistics, NVML peaks and host RSS are not collected by this pilot; final-step slot counts and persistent host-K allocation capacities are retained and must not be mislabeled as those missing metrics.

## Verified Experiment B pilot — 2026-09-22

The 4K/B1 three-arm pilot completed and passed `summarize_decode.py`: 9 measured trials plus 3 full warmups, 96 rank records. Median cohort mean decode steps were Dense 142.633 ms, ALS-full 111.267 ms (1.282× versus Dense), and Joint 127.274 ms (1.121×). Peak reserved memory was 4.770 / 5.105 / 5.063 GiB respectively; at 4K the compressed paths do not reduce total GPU resident allocation. Full protocol, all three values, exact commands, memory accounting, job history and caveats are in [RESULTS.md](../../results/system_benchmarks/ampere_tp8_joint/decode_b/p4096_b1/RESULTS.md).

Repeated NFS/conda activation waits affected the initial launch. After preserving Dense, the remaining arms completed using `/home/lz299/miniconda3/envs/basis/bin/python -m torch.distributed.run` from a non-login shell, with the same environment and benchmark settings. Use this direct environment entry point for the next jobs. The cancelled ALS-full warmup is archived separately and excluded. No larger workload has been run yet.

## Verified build and TP8 smoke — 2026-09-22

Build the extensions first in a single-GPU allocation on the same node that will run TP8. `build_kernels.py` uses the same loaders and local cache as the runner, and prints the start, completion and elapsed time of each extension. Reuse one `--build-dir` across trials. A quiet compiler is not evidence of a stalled job; inspect its processes and artifact timestamps before cancelling. Cancellation can leave a Torch extension `lock`; only move that exact lock after verifying that its build process has exited.

Environment for both commands: conda `basis`, node4, `CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 OMP_NUM_THREADS=2 MAX_JOBS=2`. Set `BASISSERVE_VERBOSE_BUILD=1` for the prebuild log.

```bash
conda run --no-capture-output -n basis python -m benchmarks.ampere.build_kernels --build-dir results/system_benchmarks/ampere_tp8_joint/tp8_smoke_4k_b1/build
conda run --no-capture-output -n basis torchrun --standalone --nproc_per_node=8 -m benchmarks.ampere.tp8_joint --assets results/system_benchmarks/ampere_tp8_joint/asset_manifest.json --output results/system_benchmarks/ampere_tp8_joint/tp8_smoke_4k_b1_steady --build-dir results/system_benchmarks/ampere_tp8_joint/tp8_smoke_4k_b1/build --mode joint --length 4096 --batch 1 --steps 12 --discard-first 4
```

Slurm job 209944 (1 GPU, 4 CPUs) completed the prebuild in 7m49s: fine-router cache load 3.28s, postprocess 214.73s, persistent slots 86.81s, mapped-host 98.48s, feature AllGather 54.14s. All five modules loaded successfully. Job 209970 (8 GPUs, 16 CPUs) completed the TP8 smoke in 2m03s.

The 4K/B1 joint run produced 12 identical token IDs on all eight ranks. All logits were finite, dispatch counters agreed, and 19 source hashes agreed across ranks and with the current files at verification. Per-rank counts were prefill attention 32, prepared AllGather 416, and each of two-stage routing, select/pack, persistent slots, and compressed-V decode attention 352.

| Smoke metric | Value |
| --- | ---: |
| Measured decode forwards after discarding 4 | 7 |
| Mean max-rank decode step | 130.246 ms |
| Median max-rank decode step | 129.368 ms |
| Synchronized wall throughput, B1 | 7.687 tokens/s |
| Maximum prefill elapsed time across ranks | 1.552 s |
| Maximum prefill/decode reserved memory per rank | 5.0625 GiB |

Runtime: PyTorch 2.6.0+cu118, Transformers 4.51.3, CUDA runtime 11.8, cuDNN 9.1.0. The log contains nonfatal `libnvrtc.so.11.2` load warnings. Prefill directly uses V96 without padding. These are short smoke measurements, not a repeated benchmark or a matched-arm speedup. Prefill is a first invocation in a fresh process and is not a warmed TTFT measurement. The router/checkpoint manifest identity caveat above still applies.

Raw records: `results/system_benchmarks/ampere_tp8_joint/tp8_smoke_4k_b1_steady/rank_0.json` through `rank_7.json`, plus `summary.json`. Build and run logs are `build.log` and `tp8_smoke_4k_b1_steady.log` in the parent directory; the latter retains earlier failed/cancelled attempts. The feasible workload grid beyond 4K/B1 has not yet been probed.
