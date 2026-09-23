# STAR-KV V-only TP8 Memory Grid

Status: the user-confirmed 36-trial formal grid completed with 35 successes and one recorded cache-state OOM. The benchmark compares Qwen3-8B-Base BasisKV V64 with the complete STAR-KV V-adaptive export under TP8; both use dense exact K and full attention.

## Formal Result

The primary metric is decode-ready NVML process memory, taking the maximum across eight TP ranks. Each configuration uses one preselected real-text cohort (cohort 0), so these are single-trial observations without a cohort variance estimate.

| Context / batch | BasisKV V64 (GiB/GPU) | STAR-KV V-only (GiB/GPU) |
| --- | ---: | ---: |
| 130048 / 1 | 5.680 | 9.324 |
| 130048 / 4 | 11.574 | 26.096 |
| 98304 / 8 | 16.217 | 38.102 |
| 130048 / 8 | 19.451 | OOM during V-cache allocation |

STAR-KV's first observed OOM is 130048/B8; its largest successful B8 context is 98304. The failed allocation was a 1.81 GiB BF16 Value-cache tensor at layer 34. All eight ranks reported CUDA OOM at cache allocation, before prefill, so this is not a transient prefill-workspace failure. The GPU had 44.39 GiB total capacity, about 1.51 GiB free at the failed request, and about 42.87 GiB process memory in use. Theoretical active V and dense K state for that failed configuration are 33.401 and 8.930 GiB/GPU respectively, before model weights and other allocations; no decode-ready resident memory is assigned to the OOM point. BasisKV V64 succeeded at all 18 configurations, including 130048/B8.

Formal raw rank JSON and logs are in `raw/`; `grid_trials.json` preserves all 36 attempts. `summary.csv`, `memory_summary.csv`, `oom_frontier.csv`, and `failures.csv` provide trial, residency, frontier, and failure data. The checked figure is `../plots/starkv_tp8_memory.pdf` (also PNG). `SUMMARY.md` records versions and artifact revisions. The [complete raw STAR-KV artifact tree](https://huggingface.co/alexz949/BasisServe-CALS/tree/26b66c9ad858cda646735ef28e1a6b6197e580fd/results/system_benchmarks/tp8_external_baselines/starkv_v_only), including the separate 16K batch sweep, is pinned at HF revision `26b66c9ad858cda646735ef28e1a6b6197e580fd`. The exact formal launch command, from `/workspace/BasisServe-CALS-opt`, was:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_qwen3_8b_tp8_v_only_memory_grid.py \
  --contexts 4096 16384 32768 65536 98304 130048 \
  --batches 1 4 8 --cohorts 0 \
  --arms basis_v64 star_v_adaptive \
  --output-root results/system_benchmarks/tp8_external_baselines/starkv_v_only \
  > results/system_benchmarks/tp8_external_baselines/starkv_v_only/formal_grid.log 2>&1
```

This checkpoint's actual global V-rank retention is 54.0473%, not a strict 50% comparison. Contexts above Qwen3-8B-Base's native 32768-token limit are memory-only stress tests, not quality results. These comparisons are for V-only placement, not the complete STAR-KV method.

## TP8 Placement Smoke

The independent V-only runner uses exact dense K, full Flash SDPA attention, and the same bounded-memory prefill for both arms. Basis V64 stores 64 BF16 Value coordinates per layer/source-local TP rank. STAR stores each exported shared latent on every rank, except layers 0, 1, and 31, which store local dense V128. For Flash SDPA's equal-Q/K/V-width requirement, the Basis V64 cache is temporarily zero-padded to 128 during attention; the persistent cache remains V64. STAR's full local V is reconstructed temporarily from the persistent shared latent. Both operations can affect peak prefill workspace but not decode-ready active V bytes.

| 4K smoke | Basis V64 | STAR V-adaptive export |
| --- | ---: | ---: |
| B1 active V cache / GPU | 18.004 MiB | 134.689 MiB |
| B8 active V cache / GPU | 144.035 MiB | 1077.513 MiB |
| B8 max-rank decode-ready PyTorch allocated | 3.440 GiB | 4.372 GiB |
| B8 max-rank decode-ready NVML process memory | 6.420 GiB | 7.381 GiB |

The B1 and B8 active V ratios are 7.48x and match the exported ranks and TP8 placement formula. Both arms completed 4K/B1 and 4K/B8 prefill plus one decode step across all eight ranks. The shared prefill chunk size is frozen at 4096 tokens for formal runs.

On the same 4K/B1 real-text prompt, 1024-token and 4096-token prefill chunks produced identical two generated tokens on all eight ranks for each arm. Across fixed-position probes of all 36 layers/ranks, Basis K/V cache relative L2 differences were 0.812%/1.121%, and STAR K/V differences were 0.818%/1.772%. These are BF16 numerical differences, not bitwise-identical states. The small-tensor STAR chunk-equivalence, 36-trial grid-accounting, and failure-classification tests passed, 8/8 in `grid_tests_36.log`.

An independent single-GPU forward of the complete fused STAR checkpoint on the same 4K/B1 prompt produced generated IDs `[323, 279]`, exactly matching TP8. It loaded all 432 checkpoint tensors with no missing or unexpected keys. This is a two-token model smoke, not a full quality evaluation; the record and command are in `smoke/single_gpu_reference.json` and `smoke/single_gpu_reference.log`.

The failed smoke attempts are retained. The first Basis 4K/B1 attempt found Flash SDPA's Q/K/V equal-width requirement; `smoke/basis_4k_b1/launcher.log` records it, and `attempt1` succeeded after temporary V padding. STAR's initial attempt exposed an incorrect test assertion for the three dense skipped layers; `attempt1` exposed Transformers TP-plan wildcard matching, which left their V projection unsharded. `attempt2` succeeded after asserting all 36 actual local projection shapes. These were implementation preflight failures, not OOMs or formal results.

Qwen-tokenized LongBench-v2 cohorts are frozen in `prompts/` (six context lengths, three disjoint groups of eight); `prompt_preparation.log` records their selection. The 36-trial formal grid used only preselected cohort 0 for every length, batch, and arm; the other two cohorts remain archived but were not part of this run. Raw source texts longer than the model maximum triggered a tokenizer warning before being cut to at most 130048 tokens.

The Qwen3 memory grid uses `benchmarks/system/run_qwen3_8b_tp8_v_only_memory_grid.py`; `summarize_qwen3_8b_tp8_v_only_memory.py` validates eight-rank JSON and separates cache-state OOM from prefill/decode workspace OOM. A partial-grid dry run over two 4K smoke records passed before the formal run; no formal result is inferred from that dry run.

The pre-grid source archive and provenance are in `freeze/`. The repository base commit alone does not include the uncommitted experiment implementation.

A separate fixed-16K, B1-to-B256 sweep with 128 actual output tokens is complete under `batch_sweep_16k/`. It recorded 14/18 successes: BasisKV V64 completed through B128 and first OOMed at B256; STAR-KV V-only completed through B32 and first OOMed at B64. All four failures were cache-state allocation OOMs before prefill. The sweep uses 256-token chunks, unlike the preceding context grid's 4096-token chunks, and is not pooled with its 36 trials. See `batch_sweep_16k/README.md` and its own `SUMMARY.md` for measurements and commands.

## User-Confirmed Scope

Qwen3-8B-Base, STAR-KV V-adaptive-50 versus BasisKV V64 only. Dense exact K, full attention, BF16 Value state, no Key compression/routing and no asymmetric offload.

## Checkpoint

The exported checkpoint is [`ICLR-results/qwen3-8b/star-v50-adaptive/full/fused.pt`](https://huggingface.co/alexz949/BasisServe-CALS/blob/0ef83dff27205b131c82df6d62636129e9dac7b9/ICLR-results/qwen3-8b/star-v50-adaptive/full/fused.pt) in the `alexz949/BasisServe-CALS` model repository at revision `0ef83dff27205b131c82df6d62636129e9dac7b9`. The 16,277,383,782-byte file was downloaded to the Hugging Face cache; the working tree does not contain another copy.

`torch.load(..., map_location="meta", weights_only=True)` read an `OrderedDict` with 432 BF16 tensors. All 36 layers contain Q and K projection weights. For V, 33 layers contain `v_proj.VS.weight` and `v_proj.U.weight`; layers 0, 1, and 31 contain dense `v_proj.weight`. The 36 V widths match `result.json` exactly. This validates checkpoint structure and metadata, not TP8 inference correctness or GPU memory.

Only this adaptive V export is in scope. No alternate V75/V96/fixed-rank checkpoint, random factors, plain SVD replacement, or retraining will be substituted.

## Metadata Caveat

The available training metadata names upstream commit `c9f0f36e7e386eaf93099c9c9796e168ca1e6504` and records:

- 36 layer ranks totaling 19924; mean rank 553.4444.
- Nominal target: 36 x 512 = 18432.
- Actual global rank retention: 54.0473%, not 50%.
- `within_budget=false`; `eligible_for_v50_comparison=false`.
- Dense skipped layers: 0, 1, 31.

The checkpoint shapes confirm these ranks. Under the specified TP8 placement, the 33 compressed layers replicate 16,852 latent elements per token per rank, while the three skipped layers retain 3 x 128 = 384 source-local V elements. The total is 17,236, or 7.48 times Basis V64's 36 x 64 = 2,304 source-local elements. This is a theoretical active-V ratio, not measured GPU process memory. The eventual experiment must report actual exported ranks and distinguish dense skipped layers from replicated compressed latents. This export is not eligible for a strict 50%-retained comparison.

## Other Required Checks

- Base model is locally available at revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Its native configuration has a 32768-token context limit and no RoPE scaling. Longer-context capacity tests must be explicitly labeled memory-only stress tests, not quality validation.
- Qwen prompts must be tokenized with the Qwen tokenizer; Llama token IDs cannot be reused silently.
- Verify the actual trained checkpoint, including non-Value weights, instead of silently discarding trained weights and retaining only Value factors.
- Choose and freeze one shared bounded-memory prefill chunk size after a smoke test. Separate cache/state OOM from transient workspace OOM.
- Theory must use the loaded artifact's actual placement and ranks; it must never be presented as measured GPU process memory.
