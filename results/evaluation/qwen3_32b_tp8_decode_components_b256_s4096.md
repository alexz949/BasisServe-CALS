# Qwen3-32B TP8 full-decode component profile

GPU: `NVIDIA A100-PCIE-40GB`; batch: `256`; context: `4096`; dtype: `torch.bfloat16`.

| Unchanged component | p50 ms | p95 ms |
|:---|---:|---:|
| input_rmsnorm | 0.027136 | 0.036864 |
| qk_headnorm_rope | 0.253952 | 0.272384 |
| post_attention_residual_rmsnorm | 0.036352 | 0.045056 |
| mlp_gate_up | 0.144384 | 0.145408 |
| mlp_silu_multiply | 0.028672 | 0.034816 |
| mlp_down | 0.08704 | 0.088064 |
| mlp_full | 0.251904 | 0.252928 |
| mlp_residual | 0.016384 | 0.019456 |
| final_rmsnorm | 0.026624 | 0.03072 |
| dense_o_proj_local | 0.038912 | 0.039936 |
| lm_head_local | 0.47104 | 0.472064 |
| greedy_local_argmax | 0.083968 | 0.084992 |

| Source rank | Fused Q/K/compact-V p50 ms |
|---:|---:|
| 32 | 0.050176 |
| 48 | 0.049152 |
| 64 | 0.049664 |
| 80 | 0.049152 |
| 96 | 0.049152 |
| 112 | 0.049152 |
| 128 | 0.050176 |

Decoder entries use the checkpoint's real compact decoder blocks. Attention, ragged AllGather, MLP AllReduce, and distributed-greedy communication are outside this single-GPU profile.
