# Qwen3-8B TP4 Wo-only runtime

## Result

On four NVIDIA L40S GPUs, the Wo-only C1 packed AllGather runtime reduces the
ideal per-rank Wo communication volume by 75% and improves decode throughput by
7.73--8.86% over dense TP4. At the same theoretical communication volume, the
rank-1024 low-rank AllReduce baseline improves decode throughput by only
0.58--0.94%. C1 also retains substantially better language-modeling quality.

The low-rank AllReduce baseline is 1.31--1.88% faster than C1 for the isolated
prefill portion. Once the same 32-token decode tail is included, C1 has
1.31--5.93% lower end-to-end latency than low-rank AllReduce and 8.36--14.32%
lower latency than dense TP4.

## Protocol

- Model: Qwen3-8B-Base
- Hardware: 4x NVIDIA L40S, TP4
- Environment: `basis`
- Precision: BF16
- PyTorch: 2.6.0+cu124
- CUDA: 12.4
- Dense arm: full-width rank-4096 Wo AllReduce
- C1 arm: rank-512 per-source encoder, packed feature-major AllGather, joint
  rank-2048 decoder
- LR arm: rank-1024 encoder, latent AllReduce, shared decoder
- Decode: one-token prompt, eight warmup steps
- Prefill: one warmup and three measured repeats, followed by 32 generated
  tokens
- Slurm job: 8286642 (`COMPLETED`, 10 minutes 13 seconds)

All compressed collectives passed a distributed projection correctness gate
against ordinary `torch.distributed` reference implementations.

## Communication and quality

| Arm | Wo collective | BF16 wire width | Ideal Wo communication reduction | WikiText-2 PPL | C4 validation PPL |
|---|---|---:|---:|---:|---:|
| Dense | full-output AllReduce | 4096 | 0% | 7.002509 | 9.168594 |
| Wo C1 | packed feature-major AllGather | 512 per source | 75% | 7.279747 | 9.320145 |
| Wo LR | latent AllReduce | 1024 | 75% | 7.548429 | 9.736591 |

Relative to dense, C1 changes WikiText-2 PPL by +3.96% and C4 PPL by +1.65%;
the equal-wire LR baseline changes them by +7.80% and +6.20%, respectively.

## Decode throughput

| Batch x decode | Dense tok/s | Wo C1 tok/s | Wo LR tok/s | C1 vs dense | LR vs dense | C1 vs LR |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 128 | 32.142 | 34.724 | 32.417 | +8.03% | +0.85% | +7.12% |
| 8 x 128 | 248.533 | 267.831 | 249.982 | +7.77% | +0.58% | +7.14% |
| 32 x 128 | 984.240 | 1060.296 | 993.494 | +7.73% | +0.94% | +6.72% |
| 64 x 128 | 1962.306 | 2117.976 | 1979.538 | +7.93% | +0.88% | +6.99% |
| 32 x 2048 | 976.018 | 1057.840 | 984.063 | +8.38% | +0.82% | +7.50% |
| 64 x 1024 | 1949.421 | 2122.108 | 1961.009 | +8.86% | +0.59% | +8.22% |

## Prefill throughput

| Batch x prompt | Dense tok/s | Wo C1 tok/s | Wo LR tok/s | C1 vs dense | LR vs dense | C1 vs LR |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 2048 | 15075.607 | 18839.783 | 19201.168 | +24.97% | +27.37% | -1.88% |
| 8 x 2048 | 12913.459 | 15647.906 | 15868.419 | +21.17% | +22.88% | -1.39% |
| 32 x 1024 | 12780.861 | 15440.570 | 15655.453 | +20.81% | +22.49% | -1.37% |
| 64 x 512 | 12797.485 | 15472.304 | 15678.159 | +20.90% | +22.51% | -1.31% |

## End-to-end latency: prefill plus 32 output tokens

| Batch x prompt | Dense ms | Wo C1 ms | Wo LR ms | C1 vs dense | LR vs dense | C1 vs LR |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 2048 | 1106.649 | 1014.113 | 1077.986 | -8.36% | -2.59% | -5.93% |
| 8 x 2048 | 2271.716 | 1987.187 | 2039.082 | -12.52% | -10.24% | -2.55% |
| 32 x 1024 | 3575.865 | 3066.499 | 3107.189 | -14.25% | -13.11% | -1.31% |
| 64 x 512 | 3575.931 | 3064.005 | 3106.371 | -14.32% | -13.13% | -1.36% |

Negative latency differences are improvements.

## Peak allocated memory per rank

| Batch x prompt | Dense GiB | Wo C1 GiB | Wo LR GiB |
|---|---:|---:|---:|
| 1 x 2048 | 4.876 | 5.482 | 5.230 |
| 8 x 2048 | 6.139 | 6.799 | 6.491 |
| 32 x 1024 | 7.602 | 8.325 | 7.954 |
| 64 x 512 | 7.636 | 8.358 | 7.988 |

## Commands

The three values of `ARM` were `dense`, `wo_c1_ag`, and `wo_lr_ar_wire`.

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_qwen3_8b_wo_tp4.py \
  --workload decode --arm "$ARM" \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --phase1-dir results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 \
  --quality-results results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/results.json \
  --configurations 1x128,8x128,32x128,64x128,32x2048,64x1024 \
  --warmup 8 \
  --output-json "results/qwen3_8b_wo_tp4/full/decode_${ARM}.json"

torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_qwen3_8b_wo_tp4.py \
  --workload prefill --arm "$ARM" \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --phase1-dir results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 \
  --quality-results results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/results.json \
  --configurations 1x2048,8x2048,32x1024,64x512 \
  --output-tokens 32 --warmup 1 --repeats 3 \
  --output-json "results/qwen3_8b_wo_tp4/full/prefill_${ARM}.json"
```

## Warnings

- The communication reductions use the ideal ring byte model and cover only
  the Wo collective, not QKV, attention, MLP, or other model traffic.
- The prefill-only comparison favors low-rank AllReduce slightly, whereas the
  complete request favors C1 because C1 has much faster decode.
- C1 uses 0.48--0.72 GiB more peak allocated memory per rank than dense in the
  tested prefill configurations because it retains its factor bank and packed
  communication workspace.
