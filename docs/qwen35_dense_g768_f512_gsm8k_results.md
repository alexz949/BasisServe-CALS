# Dense V + GDN Wo 768 / full-attention Wo 512：GSM8K

全量 1,319 题已完成并通过审计。thinking 关闭，5-shot，最多生成 1,024 tokens，seed 20260909。各组 prompts、targets 和 generation kwargs 一致；旧组复用此前已审计结果。

| Dense V 下的 Wo 配置 | Strict | Flexible | 数值等价 | 达到长度上限 | 含 closing think |
|---|---:|---:|---:|---:|---:|
| 无压缩（Dense） | 1230 / 93.25% | 1233 / 93.48% | 1239 / 93.93% | 6 | 0 |
| GDN 512 + full 512 | 915 / 69.37% | 566 / 42.91% | 571 / 43.29% | 765 | 1100 |
| 仅 GDN 512 | 1199 / 90.90% | 1175 / 89.08% | 1181 / 89.54% | 47 | 196 |
| 仅 full 512 | 1199 / 90.90% | 1216 / 92.19% | 1223 / 92.72% | 10 | 0 |
| **GDN 768 + full 512** | **1193 / 90.45%** | **1203 / 91.21%** | **1223 / 92.72%** | **15** | **0** |

数值等价评分只对既有 flexible 提取结果用 Decimal 比较，例如 26.00 与 26；没有人工判卷或重新选择答案。closing think 指响应包含 `</think>`。

将 GDN rank 从 512 放宽至 768，同时复用完全相同的 full-attention 512 因子，flexible 提高 48.29 个百分点。达到长度上限从 765 降至 15，含至少三个 closing think 的响应从 549 降至 0。这支持原组合中的 GDN 512 压缩过强，放宽后生成稳定性明显改善，之前的退化不能仅用答案提取解释。这里变化包括增加 rank 及其对应的重新拟合。

新配置仍比 Dense 低 2.27 个百分点（flexible）和 1.21 个百分点（数值等价）。数值等价得分与仅 full 512 相同，但逐题不同：双方各有 39 题仅自己答对，不能据此声称增加 GDN 压缩没有成本。

24 层 GDN 使用每 source rank 768，8 层 full attention 使用每 source rank 512；每层四个逻辑 source，每 source 输入维度 1024。V 保持原生 Dense。此次为 TP1 vLLM 质量评测，没有实测分布式通信、KV cache 压缩或加速收益，也没有新增此混合配置的 HF/vLLM 整模型对齐 smoke。

## 执行与产物

本机直接执行，无 Slurm；vLLM 使用此前授权的 lowrankarena 环境、GPU 6、CPU 2 线程，max-num-seqs 32。执行约 335.55 秒，退出码 0。退出时有 NCCL destroy_process_group 警告，未发生 OOM 或评测失败。CPU 审计使用 lowrank、2 线程，退出码 0。

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --bank results/q35_hybrid/banks/c1_uniform_v256.pt \
  --wo-bank results/q35_hybrid/wo_dense_g768_f512/wo_bank.pt --wo-scope all \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --max-new-tokens 1024 --max-model-len 8192 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/gsm8k_vllm/result_dense_g768_f512_wo.json \
  >> results/q35_hybrid/logs/gsm8k_vllm_result_dense_g768_f512_wo.log 2>&1

conda activate lowrank
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m evaluation.summarize_qwen35_gsm8k_v96 \
  --mixed-wo --output results/q35_hybrid/gsm8k_vllm/dense_g768_f512_wo_summary.json \
  > results/q35_hybrid/logs/gsm8k_g768_f512_wo_audit.log 2>&1
```

上述为已执行命令，输出已存在，不应直接覆盖重跑。模型路径为 results/q35_hybrid/model，版本 c202236235762e1c871ad0ccb60c8ee5ba337b9a；torch 2.10.0、vLLM 0.18.1、transformers 4.57.6、lm_eval 0.4.11。

审计结果 status 为 complete_and_audited，检查模型和 bank 身份、混合 rank 形状、full-attention 原因子精确复用、32 层覆盖、各组题目和生成参数一致，并重新计算各项分数与逐题配对差异。拟合细节见 [bank 报告](qwen35_dense_g768_f512_bank.md)。本次新增 summarizer 的 --mixed-wo 模式；未提交或上传 GitHub。
