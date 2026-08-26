# Qwen3-32B TP8 full-decode component profile

GPU: `NVIDIA A100 80GB PCIe`; batch: `512`; context: `1024`; dtype: `torch.bfloat16`.

| Unchanged component | p50 ms | p95 ms |
|:---|---:|---:|
| input_rmsnorm | 0.011776 | 0.013312 |
| qk_headnorm_rope | 0.118784 | 0.125952 |
| post_attention_residual_rmsnorm | 0.018944 | 0.019456 |
| mlp_gate_up | 0.136704 | 2.68902 |
| mlp_silu_multiply | 0.021504 | 0.022528 |
| mlp_down | 0.088064 | 0.091136 |
| mlp_full | 0.245248 | 2.79859 |
| mlp_residual | 0.009216 | 0.011264 |
| final_rmsnorm | 0.012288 | 0.012288 |
| dense_o_proj_local | 0.03584 | 0.036864 |
| lm_head_local | 0.407552 | 2.96653 |
| greedy_local_argmax | 0.104448 | 0.10752 |

| Source rank | Fused Q/K/compact-V p50 ms |
|---:|---:|
| 64 | 0.048128 |
| 128 | 0.048128 |

Decoder entries use the checkpoint's real compact decoder blocks. Attention, ragged AllGather, MLP AllReduce, and distributed-greedy communication are outside this single-GPU profile.
