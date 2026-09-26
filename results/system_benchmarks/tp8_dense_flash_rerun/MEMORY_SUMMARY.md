# TP8 Peak GPU Memory

Units: GiB (2^30 bytes). Each entry is the maximum across eight TP ranks of
PyTorch allocated memory, not the sum over GPUs or total nvidia-smi device usage.
Prefill and decode peaks are recorded separately; peak statistics are reset
before the measured decode loop. Raw rank JSON also records reserved memory and
decode-resident allocated/reserved memory.

## Qwen3-32B

Flash SDPA Dense reruns versus retained BasisKV Joint V96 full-scan pilots.
BF16, TP8, single-trial results, not a new paired memory benchmark.
Basis historical exact K is pinned on CPU with GPU slots; V96 remains on GPU.

| Context | Batch | Dense prefill peak | Basis prefill peak | Dense decode peak | Basis decode peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 1 | 12.899 | 16.384 | 10.911 | 14.395 |
| 65,536 | 2 | 16.861 | 19.301 | 12.916 | 15.353 |
| 65,536 | 4 | 24.787 | 25.185 | 16.928 | 17.323 |
| 65,536 | 8 | 40.636 | 36.801 | 24.950 | 21.107 |
| 130,048 | 1 | 16.804 | 19.211 | 12.906 | 15.313 |
| 130,048 | 2 | 24.640 | 24.906 | 16.907 | 17.174 |
| 130,048 | 4 | 40.308 | 36.304 | 24.910 | 20.904 |

Offloading historical K does not guarantee lower total allocated memory at small
batches: factors, decoder weights, communication buffers, routing state and
workspaces also contribute. These totals alone do not isolate each contribution.
At 64K/B8 and approximately 128K/B4, Basis has lower measured peaks than Dense.
Failed/OOM trials are not represented as successful memory measurements here.

Dense sources: `qwen_p{65536,130048}/b{batch}/dense/*/rank*.json` in this directory.
Basis sources: `../qwen3_32b_tp8_joint/{64k_b1,64k_capacity,128k_capacity,128k_b4_basis}`.
Fields: `prefill_peak_allocated_bytes`, `decode_peak_allocated_bytes`.
Commands and environment (`basis`, 8 x L40S) are in [README.md](README.md);
latency comparisons are in [SUMMARY.md](SUMMARY.md).

## Llama-3.1-8B: New Quest Pair

65,536 context, B1, TP8, BF16. All three matched prompt cohorts give the same
maximum-rank allocated-memory figures at the precision below.

| Method | Prefill peak | Decode peak |
| --- | ---: | ---: |
| Quest native CUDA + local GQA/BF16 patch, GPU K128/V128 | 5.406 | 3.806 |
| BasisKV Joint V96, historical K offload | 5.766 | 4.165 |

These are total allocator peaks, not KV-cache-only sizes. At this B1 point Basis
has a higher measured GPU allocated peak despite historical K offload.
See [paired summary](../quest_tp8/paired/RESULTS_SUMMARY.md) for commands, timing,
placement/budget differences and NUMA warnings. No SHA256 checks were performed.
