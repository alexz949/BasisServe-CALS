# Qwen3.5-9B Two-sided V96 + Wo GSM8K

V96 + 全部 Wo 的全量 1,319 题测试完成，逐题审计通过。Wo 覆盖 8 层 full attention 和 24 层 GDN/DeltaNet，每层保留一半输入维度。

| 配置 | 官方严格匹配 | 官方宽松提取 | 数值等价比较 | 达到 1024-token 上限 |
|---|---:|---:|---:|---:|
| Dense | 93.25% (1230) | 93.48% (1233) | 93.93% (1239) | 6 |
| Two-sided V96 | 72.93% (962) | 87.26% (1151) | 88.63% (1169) | 49 |
| Two-sided V96 + Wo | 31.08% (410) | 30.02% (396) | 30.63% (404) | 918 |
| Two-sided V64 + Wo | 15.31% (202) | 21.08% (278) | 21.76% (287) | 835 |

括号为正确题数，分母均为 1319。数值等价仅对已有 flexible-extract 的数字作 Decimal 比较，不重新选答案，不是人工语义评分。严格与宽松提取不是包含关系：前者寻找 #### 后的数字，后者取最后一个数字，后续续写可能导致宽松分数反而较低。

## 发现

- V96 加 Wo 后，宽松分数比 V96-only 低 57.24 pp，数值等价分数低 58.00 pp。数值等价配对统计为 778 题从对变错、13 题从错变对。
- 相比 V64+Wo，V96+Wo 宽松分数高 8.95 pp，数值等价高 8.87 pp，但截断和重复关闭 think 标签更严重，并非所有生成行为都改善。
- 918/1319 题（69.60%）达到输出上限；全部 1319 条回复含生成的 </think>，其中 778 条至少包含 3 次。V96-only 的对应计数为 49、0、0。
- 原始第 0、2 题（doc_id 从 0 开始）均反复生成 </think>，没有产生可提取答案。所有组提示均关闭 thinking；这些生成标签不能据此解释为开关被启用。
- 当前结果说明只提高 V rank 不能解决加入此 Wo bank 后的生成异常。尚不能区分 full-attention Wo、GDN Wo、拟合或共用实现各自的贡献；本次没有新做 V96+Wo 的 HF token 对照，也没有运行分组 Wo 消融，不能声称已排除全部实现问题。

## 参数与执行

GSM8K main test 全量、5-shot、原生 chat template、thinking=False、greedy、seed 20260909。max_new_tokens=1024、max_model_len=8192、max_num_seqs=32、max_num_batched_tokens=4096。TP1，CUDA Graph 开启、prefix cache 关闭，固定 6 GiB KV cache。

模型 results/q35_hybrid/model；V bank 为 c1_twosided_v96.pt，Wo bank 为 wo_twosided_v96/wo_bank.pt。审计验证 Wo upstream factor hash 与 V bank 一致、32 层完整覆盖、每层四个 source encoder 均保留一半维度。

lowrankarena 环境，本机 GPU 6，CPU 2 线程，直接执行，无 Slurm。vLLM 0.18.1、lm-eval 0.4.11、torch 2.10.0、Transformers 4.57.6。评测耗时约 657.9 秒，包括初始化、生成与评分；不作受控吞吐比较。

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --bank results/q35_hybrid/banks/c1_twosided_v96.pt \
  --wo-bank results/q35_hybrid/wo_twosided_v96/wo_bank.pt --wo-scope all \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --max-new-tokens 1024 --max-model-len 8192 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/gsm8k_vllm/result_twosided96_wo.json \
  >> results/q35_hybrid/logs/gsm8k_vllm_result_twosided96_wo.log 2>&1
```

审计在 lowrank 下使用 CPU 2 线程，验证五组逐题提示、目标与生成参数一致，题数、分数重算、生成长度、提示无截断、模型/分词器身份、V/Wo bank 哈希和 Wo 结构：

```bash
conda activate lowrank
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.summarize_qwen35_gsm8k_v96 --with-wo \
  --output results/q35_hybrid/gsm8k_vllm/v96_wo_summary.json \
  > results/q35_hybrid/logs/gsm8k_v96_wo_audit.log 2>&1
```

评测与审计均 exit 0。引擎正常关闭，有退出时 destroy_process_group 警告，没有 OOM 或评测失败。原始回答见 result_twosided96_wo.json，汇总含全部配对统计及执行参数。

vLLM 仍采用标准宽度补零 V cache，Wo 为 TP1 的四逻辑 source 等价实现；本次不测量真实 KV cache 或通信压缩。原模型、bank、旧结果保持不变。尚未提交或上传 GitHub。
