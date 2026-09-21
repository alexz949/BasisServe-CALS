# Qwen3-32B V96 RULER 128K: LRQK under tensor parallelism

1100 prompts, 11 RULER tasks x 100, sequence length 131072, on top of the frozen
uniform-V96 checkpoint. Four TP2 replicas across eight GPUs, one prompt per replica.

| Task | LRQK TP2 | ShadowKV | Delta |
|---|---:|---:|---:|
| `fwe` | 32.0 | 35.3 | -3.3 |
| `niah_multikey_1` | 97.0 | 95.0 | +2.0 |
| `niah_multikey_2` | 81.0 | 59.0 | +22.0 |
| `niah_multiquery` | 98.0 | 85.5 | +12.5 |
| `niah_multivalue` | 96.5 | 85.2 | +11.2 |
| `niah_single_1` | 100.0 | 100.0 | +0.0 |
| `niah_single_2` | 100.0 | 97.0 | +3.0 |
| `niah_single_3` | 100.0 | 96.0 | +4.0 |
| `qa_1` | 52.0 | 52.0 | +0.0 |
| `qa_2` | 56.0 | 39.0 | +17.0 |
| `vt` | 96.4 | 91.6 | +4.8 |
| **11-task mean** | **82.63** | **75.97** | **+6.66** |

## Budget is not matched

LRQK selects 832 historical plus 64 recent tokens per query
head. After the eight query heads in a KV group are de-duplicated that is about
3437 physical tokens per group, against ShadowKV's 2514. LRQK therefore
works from roughly 36% more cache than the arm it is compared with, and the
+6.66 point gap should not be read as a like-for-like method comparison.

The shape of the gap is still informative: LRQK leads on every needle-retrieval
task, by as much as 22 points on `niah_multikey_2`, draws level on `qa_1`, and
loses on `fwe`, the one task that asks for a global aggregate rather than a
located span.

## Why tensor parallelism

The single-GPU arm keeps a full model per device, leaving no room for the rank-32
key codes, so it streams 536 MB per layer from pinned host memory on every decode
step. Splitting the model halves each rank's share of the codes and lets them stay
resident, which removes 4.36 TB of per-sample PCIe traffic.

| | LRQK TP2 | LRQK single-GPU | ShadowKV single-GPU |
|---|---:|---:|---:|
| Seconds per sample | 108 | 1595 | 153 |
| Decode seconds per step | 0.353 | 11.32 | - |
| Peak GiB per rank | 70.7 | 86.0 | 80.4 |

## Agreement with the single-GPU arm

The single-GPU run had completed 142 of these prompts before it was stopped.
Every one of those samples scores the same here; 32
of them are bitwise identical. The remainder diverge only through the summation
order of the output all-reduce. Each record keeps its own `reference_comparison`.

## Contents

- `predictions/lrqk_tp2.jsonl`: one record per prompt with the sample, generated
  IDs and text, score, timing, peak memory, per-rank per-layer routing statistics,
  the single-GPU comparison, and the SHA256 of the original per-sample file.
- `run_metadata.json`: the protocol shared by every record, plus the routing
  fields that are identical across all layers and samples.
- `summary.json`: the evaluator's own audited task table.

Reproduction environment: `lowrank`.

    torchrun --nnodes=1 --nproc-per-node=2 \
      evaluation/eval_qwen3_32b_v96_lrqk_tp2.py evaluate \
      --model <Qwen3-32B> --factors <uniform-v96-als6-cg16> \
      --prompts <prompts.safetensors> --prompts-json <prompts.json> \
      --output <out> --batch-size 1 --shard $R --shards 4

Batch 2 does not fit: each extra prompt needs 23.6 GiB of resident codes and
values against 21 GiB of headroom on a 94.97 GiB device.
