# Qwen3.5-9B Two-sided V96 GSM8K

全量 1,319 题已完成并通过逐题审计。V96 不加 Wo；相同生成与评分协议下明显优于已有 Two-sided V64，但仍低于 Dense。

| 配置 | 官方严格匹配 | 官方宽松提取 | 已提取数字的数值等价比较 | 达到 1024-token 上限 |
|---|---:|---:|---:|---:|
| Dense | 93.25% (1230) | 93.48% (1233) | 93.93% (1239) | 6 |
| Two-sided V64 | 10.39% (137) | 65.50% (864) | 68.61% (905) | 162 |
| Two-sided V96 | 72.93% (962) | 87.26% (1151) | 88.63% (1169) | 49 |

括号为正确题数，分母均为 1319。数值等价比较仅将原 flexible-extract 提取的数字用 Decimal 比较，例如 26.00 与 26；没有重新选择正文中的答案，不是人工语义判分。原始官方结果不变。

## 配对比较与解释

- 相比 V64：V96 官方宽松提升 21.76 pp，数值等价提升 20.02 pp。宽松评分中 321 题从错变对、34 题从对变错；数值等价中分别为 298、34。
- 相比 Dense：V96 官方宽松低 6.22 pp，数值等价低 5.31 pp。
- 严格格式遵循也改善，严格分数从 10.39% 提升到 72.93%；但宽松和数值等价分数的提升说明改进不局限于 #### 格式。
- 截断从 V64 的 162/1319 降到 V96 的 49/1319；V96 生成回复没有 </think>。总生成 token 从 524617 降至 356803，Dense 为 230519。
- 结果支持当前 V64 点位过于激进、V96 更适合此生成任务的判断。两组使用各自已有拟合因子与 rank 分配，因此不是固定因子下只增加维度的严格消融，不能将所有差异唯一归因于 rank。
- 此次没有启用或评测任何 Wo 压缩，不能回答两类 Wo 各自的影响。原 V-ALS 有限迭代、未完全满足残差容差的限制仍适用。

Full-attention 层顺序为 3、7、11、15、19、23、27、31。V64 ranks 为 [32,48,32,48,96,48,128,80]；V96 ranks 为 [64,80,64,80,128,128,128,96]，均值分别为 64、96。原始 head V 维度为 256。

## 设置与执行

GSM8K main test 全量，5-shot，原生 chat template，thinking=False，greedy，seed 20260909；max_new_tokens=1024，max_model_len=8192，max_num_seqs=32，max_num_batched_tokens=4096。TP1，CUDA Graph 开启，prefix cache 关闭，固定 6 GiB KV cache。

使用获准的 lowrankarena 环境，本机 GPU 2 直接运行，无 Slurm；CPU 2 线程。vLLM 0.18.1、lm-eval 0.4.11、torch 2.10.0、Transformers 4.57.6。耗时约 650.7 秒，包含初始化、生成与评分。GPU 为共享使用，耗时不作受控吞吐比较。

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=2 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --bank results/q35_hybrid/banks/c1_twosided_v96.pt \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --max-new-tokens 1024 --max-model-len 8192 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/gsm8k_vllm/result_twosided96.json \
  >> results/q35_hybrid/logs/gsm8k_vllm_result_twosided96.log 2>&1
```

审计使用 lowrank、CPU 2 线程：

```bash
conda activate lowrank
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.summarize_qwen35_gsm8k_v96 \
  > results/q35_hybrid/logs/gsm8k_v96_audit.log 2>&1
```

审计验证全量题数、两种官方分数重算、三组逐题提示/目标/生成参数一致、没有提示截断、生成长度与结束原因、相同模型/分词器身份及 bank 哈希。Dense 与 V64 源文件哈希与之前已核对官方 test Arrow 的全量审计记录一致。两条执行命令均 exit 0。

## 输出及限制

- 原始回答：results/q35_hybrid/gsm8k_vllm/result_twosided96.json。
- 配对审计：results/q35_hybrid/gsm8k_vllm/v96_summary.json。
- 评测日志：results/q35_hybrid/logs/gsm8k_vllm_result_twosided96.log。
- 审计日志：results/q35_hybrid/logs/gsm8k_v96_audit.log。
- 运行有 FLA 短序列形状启发式提示及退出时 destroy_process_group 警告；没有 OOM 或评测失败，引擎正常关闭。
- vLLM 仍使用标准宽度补零 V cache，只验证质量，不代表已实现实际 KV cache 或通信压缩。

未运行先前准备的 V80 或分组 Wo 诊断。既有模型、bank、结果未覆盖；尚未提交或上传 GitHub。
