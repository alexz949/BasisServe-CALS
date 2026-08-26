# Qwen3-8B TP4: 32K Prefill and 128-Token Generation

## Protocol

- Model: Qwen3-8B-Base
- Hardware: 4 x NVIDIA L40S, true TP4
- Environment: `basis`
- Precision: BF16 model, BF16 C1 latent wire
- Batch size: 1
- Prompt length: 32,000 tokens
- Output length: 128 tokens (the prefill produces the first token, followed by 127 decode steps)
- Warmup runs: 1
- Repeated measurements: 3
- Dense attention: PyTorch Flash-SDPA
- C1 attention: custom Triton compressed-V prefill and decode
- C1 output projection: packed feature-major NCCL AllGather followed by one BF16 cuBLAS decoder GEMM
- C1 checkpoint: uniform V64 ALS5

## Results

| Metric | Dense | C1 uniform V64 | C1 relative to Dense |
|---|---:|---:|---:|
| Time to first token | 2819.84 ms | 2528.10 ms | -10.35% latency |
| Prefill throughput | 11,348.16 token/s | 12,657.73 token/s | +11.54% |
| 127-step decode time | 3946.91 ms | 7009.04 ms | +77.58% latency |
| Decode throughput | 32.18 token/s | 18.12 token/s | -43.69% |
| End-to-end latency | 6766.75 ms | 9537.14 ms | +40.94% latency |
| End-to-end output throughput | 18.92 token/s | 13.42 token/s | -29.05% |
| Static KV cache per rank | 1.103 GiB | 0.827 GiB | -25.00% |
| Peak allocated memory per rank | 7.542 GiB | 7.976 GiB | +5.75% |

C1 V64 accelerates the 32K prefill by 1.115x and reduces the static KV cache by exactly 25%, as expected when only Value width changes from 128 to 64 while Key remains dense. The current long-context Triton decode path is the bottleneck: it runs at 0.563x Dense decode throughput, so the complete request runs at 0.710x Dense end-to-end speed.

The handwritten SM89 CUDA decode kernel was not used because its validated tuning table ends at context length 8192. Extending and retuning that kernel through 32K is required for a CUDA-to-CUDA long-context comparison.

## Commands

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_qwen3_8b_tp4_prefill.py \
  --arm dense \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --batch-sizes 1 --prompt-lengths 32000 \
  --max-batch-prompt-tokens 32000 --output-tokens 128 \
  --warmup-runs 1 --repeat-runs 3 --torch-num-threads 4 \
  --output-json results/evaluation/qwen3_8b_tp4_b1_p32000_o128_dense_bf16.json
```

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_qwen3_8b_tp4_prefill.py \
  --arm c1_uniform_r64 \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --factor-dir results/checkpoints/qwen3_8b_c1_v64_als5 \
  --c1-decode-attention triton --c1-wire-dtype bfloat16 \
  --batch-sizes 1 --prompt-lengths 32000 \
  --max-batch-prompt-tokens 32000 --output-tokens 128 \
  --warmup-runs 1 --repeat-runs 3 --torch-num-threads 4 \
  --output-json results/evaluation/qwen3_8b_tp4_b1_p32000_o128_c1_v64_bf16.json
```

The completed result artifacts are the two JSON files referenced by these commands.
