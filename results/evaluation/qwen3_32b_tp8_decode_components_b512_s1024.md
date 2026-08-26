# Qwen3-32B TP8 full-decode component profile

GPU: `NVIDIA A100 80GB PCIe`; batch: `512`; context: `1024`; dtype: `torch.bfloat16`.

| Unchanged component | p50 ms | p95 ms |
|:---|---:|---:|
| input_rmsnorm | 0.011264 | 0.013312 |
| qk_headnorm_rope | 0.119808 | 0.123904 |
| post_attention_residual_rmsnorm | 0.019456 | 0.02048 |
| mlp_gate_up | 0.136192 | 2.63373 |
| mlp_silu_multiply | 0.021504 | 0.022528 |
| mlp_down | 0.08704 | 0.088064 |
| mlp_full | 0.239616 | 2.72691 |
| mlp_residual | 0.009216 | 0.011264 |
| final_rmsnorm | 0.012288 | 0.013312 |
| dense_o_proj_local | 0.034816 | 0.036864 |
| lm_head_local | 0.402432 | 2.90509 |
| greedy_local_argmax | 0.100352 | 0.1024 |

| Source rank | Fused Q/K/compact-V p50 ms |
|---:|---:|
| 32 | 0.047104 |
| 48 | 0.047104 |
| 64 | 0.047104 |
| 80 | 0.047104 |
| 96 | 0.047104 |
| 112 | 0.047104 |
| 128 | 0.047104 |

Decoder entries use the checkpoint's real compact decoder blocks. Attention, ragged AllGather, MLP AllReduce, and distributed-greedy communication are outside this single-GPU profile.
