# DeepSeek-V2-Lite mean-DP C1 TP4 runtime

## Protocol

- Model: DeepSeek-V2-Lite base, BF16
- Hardware: one `lovelace` node, 4 x NVIDIA L40S, physical TP4
- Software: `basis` conda environment, PyTorch 2.6.0+cu124, CUDA 12.4
- Workload: batch 64, prompt length 4096, 128 output tokens (127 decode steps after TTFT)
- Prefill: eight sequence chunks of 512 tokens; all chunks are included in TTFT
- Measurement: 3 warmups and 5 timed repetitions
- Attention: SDPA
- MoE: original 64 routed experts, top-6 routing, 2 shared experts, grouped-MM implementation
- C1 checkpoint: Global-KL mean-DP, 8 logical sources, average source rank 128
- C1 runtime: post-attention BF16 source encoding, compiled packed NCCL AllGather, replicated BF16 decoder
- MLA latent KV and KV cache are unchanged in both arms

## Results

| Metric | Dense TP4 | C1 mean-DP TP4 | C1 speedup |
| --- | ---: | ---: | ---: |
| TTFT mean | 16.0897 s | 14.7805 s | 1.0886x |
| Prefill throughput | 16,292.61 tok/s | 17,735.82 tok/s | 1.0886x |
| Decode mean, 127 steps | 22.9175 s | 22.7371 s | 1.0079x |
| Decode throughput | 354.66 tok/s | 357.48 tok/s | 1.0079x |
| End-to-end mean | 39.0073 s | 37.5175 s | 1.0397x |
| End-to-end output throughput | 210.01 tok/s | 218.35 tok/s | 1.0397x |
| Peak allocated per rank | 20.6385 GiB | 20.8444 GiB | +1.00% |
| MLA latent KV cache per rank | 7.8292 GiB | 7.8292 GiB | unchanged |

The five-run distributions were tight:

- Dense end-to-end: 38.9895--39.0216 s, median 39.0042 s.
- C1 end-to-end: 37.5042--37.5354 s, median 37.5133 s.

## Communication accounting

- Physical TP4 maps two consecutive logical checkpoint sources to each process.
- The mean local C1 wire width is 256 BF16 values per token and layer.
- Dense attention-output AllGather width would be 512 BF16 values per process.
- Actual C1 packed-AllGather payload reduction versus dense AllGather: 50%.
- Theoretical ring-byte reduction versus the standard dense row-parallel `o_proj` AllReduce: 75%.
- These percentages are payload accounting; the measured end-to-end speedups above include encoder, collective, decoder, MLA attention, MoE, LM head, and greedy selection.

## Commands

Dense:

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_deepseek_v2_lite_tp4_prefill.py \
  --arm dense \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0 \
  --batch-size 64 --prompt-length 4096 --prefill-chunk-size 512 \
  --output-tokens 128 --warmup-runs 3 --repeat-runs 5 \
  --attn-implementation sdpa --experts-implementation grouped_mm \
  --output-json results/deepseek_v2_lite_tp4_runtime/b64_s4096_o128/dense_bf16.json
```

C1:

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_deepseek_v2_lite_tp4_prefill.py \
  --arm c1_mean_dp \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0 \
  --factor-dir results/deepseek_v2_lite_c1/c1_tp8_avg128_global_kl_als5 \
  --batch-size 64 --prompt-length 4096 --prefill-chunk-size 512 \
  --output-tokens 128 --warmup-runs 3 --repeat-runs 5 \
  --attn-implementation sdpa --experts-implementation grouped_mm \
  --output-json results/deepseek_v2_lite_tp4_runtime/b64_s4096_o128/c1_mean_dp_bf16.json
```

## Run status and warning

- Successful Slurm job: `8283648`, completed in 10m58s with exit code 0.
- An initial unchunked run (`8283646`) failed during Dense prefill because grouped-MoE attempted one additional 12 GiB allocation for all 1,572,864 routed token-expert pairs. It produced no benchmark JSON. Sequence chunking removed that peak without changing batch size, total prompt length, routing top-k, output length, or timed token count.
