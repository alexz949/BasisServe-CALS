# Qwen3-8B TP4 Fused Conditional-Router Results

## Scope

This experiment replaces the chunked PyTorch C1 Base16+R8 router with a fixed-geometry CUDA kernel. One CUDA block processes one `(batch, KV head, Page32)` tuple and fuses:

1. `V80 -> Base16`;
2. `Base16 -> pre-RoPE K128`;
3. RoPE application;
4. QK and R8 residual scoring;
5. Page32 log-sum-exp reduction.

The kernel uses BF16 WMMA operations on SM80+ GPUs. Page-mass normalization, max aggregation across the four Query heads in each GQA group, and Top-K selection remain unchanged.

## Commands

Microbenchmark and numerical validation:

```bash
python evaluation/validate_conditional_router_page32_lse.py \
  --sequence-length 65536 \
  --correctness-length 4097 \
  --warmup 5 \
  --repeat 50 \
  --reference-repeat 3 \
  --output-json results/evaluation/qwen3_8b_conditional_router_page32/fused_cuda_l40s.json
```

Full-model TP4 RULER evaluation:

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/eval_qwen3_8b_tp4_k_offload_ruler.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-factor-dir results/checkpoints/qwen3_8b_c1_v80_als5 \
  --routing-factor-dir results/checkpoints/qwen3_8b_v80_base16_r8_nonsink_page32_b4096_32k \
  --router c1_base16_r8 \
  --exact-key-storage mapped_host \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_64k_yarn4_hard5_s8 \
  --sequence-length 65536 \
  --yarn-factor 4.0 \
  --page-size 32 \
  --physical-token-budget 4096 \
  --pinned-prefix-pages 1 \
  --quality-oracle-json results/evaluation/qwen3_8b_tp4_k_offload_ruler_64k_hard5/c1_gpu.json \
  --output-json results/evaluation/qwen3_8b_tp4_k_offload_ruler_64k_hard5_fused/c1_mapped.json
```

## Environment

- Conda environment: `basis`
- PyTorch: `2.6.0+cu124`
- CUDA: `12.4`
- GPU: `4 x NVIDIA L40S` for the full-model evaluation
- Slurm job: `8298322`, completed in `00:01:48`
- Precision: BF16 model and routing factors, FP32 page-LSE output

## Router Microbenchmark

| Metric | Result |
| --- | ---: |
| Correctness length | 4,097 tokens |
| Performance length | 65,536 tokens |
| Maximum absolute page-LSE error | `4.7684e-7` |
| Mean absolute page-LSE error | `8.8945e-8` |
| Selected page-ID agreement | `100%` |
| PyTorch routing latency | `8.1739 ms` |
| Fused CUDA routing latency | `0.0700 ms` |
| Router speedup | `116.73x` |

The compiled SM89 kernel uses 40 registers and 32 KiB shared memory per block, with no register spills.

## Full-Model Results

The baseline is the prior mapped-host run with the chunked PyTorch router. All other model, checkpoint, dataset, sparse budget, exact-K attention, and TP4 settings are unchanged.

| Metric | PyTorch router | Fused CUDA router | Change |
| --- | ---: | ---: | ---: |
| Hard-5 balanced accuracy | 70.00% | 70.00% | 0.00 pp |
| Oracle generation matches | 5/5 | 5/5 | unchanged |
| Decode throughput | 3.3320 tok/s | 19.7734 tok/s | `5.93x` |
| Decode time, 191 steps | 57.3222 s | 9.6595 s | -83.15% |
| Prefill time | 34.8764 s | 34.6876 s | -0.54% |
| End-to-end time | 92.1985 s | 44.3471 s | -51.90% |
| Output throughput including prefill | 2.1258 tok/s | 4.4197 tok/s | `2.08x` |
| Exact-K physical read per decode step | 0.28125 GiB | 0.28125 GiB | unchanged |

The fused runtime adds approximately 2.25 MiB of preallocated routing workspace per rank. CPU-pinned exact-K storage and the physical fetch budget are unchanged.

## Result Files

- Microbenchmark: `results/evaluation/qwen3_8b_conditional_router_page32/fused_cuda_l40s.json`
- Full-model evaluation: `results/evaluation/qwen3_8b_tp4_k_offload_ruler_64k_hard5_fused/c1_mapped.json`
