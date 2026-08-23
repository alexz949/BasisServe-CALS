# Qwen3-32B TP8 full-decode component profile

GPU: `NVIDIA A100-PCIE-40GB`; batch: `256`; context: `2048`; dtype: `torch.bfloat16`.

| Unchanged component | p50 ms | p95 ms |
|:---|---:|---:|
| input_rmsnorm | 0.028672 | 0.052224 |
| qk_headnorm_rope | 0.258048 | 0.278528 |
| post_attention_residual_rmsnorm | 0.037888 | 0.045056 |
| mlp_gate_up | 0.145408 | 0.147456 |
| mlp_silu_multiply | 0.028672 | 0.034816 |
| mlp_down | 0.08704 | 0.088064 |
| mlp_full | 0.252928 | 0.253952 |
| mlp_residual | 0.017408 | 0.026624 |
| final_rmsnorm | 0.027648 | 0.03072 |
| dense_o_proj_local | 0.038912 | 0.039936 |
| lm_head_local | 0.47104 | 0.473088 |
| greedy_local_argmax | 0.083968 | 0.086016 |

| Source rank | Fused Q/K/compact-V p50 ms |
|---:|---:|
| 32 | 0.049152 |
| 48 | 0.049152 |
| 64 | 0.049152 |
| 80 | 0.049152 |
| 96 | 0.049152 |
| 112 | 0.049152 |
| 128 | 0.049152 |

Decoder entries use the checkpoint's real compact decoder blocks. Attention, ragged AllGather, MLP AllReduce, and distributed-greedy communication are outside this single-GPU profile.
