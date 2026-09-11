# Dense V：GDN Wo 与 full-attention Wo 分组 GSM8K

两个新实验均完成全量 1319 题并通过逐题审计。三个压缩组使用同一 Dense V bank、同一 Dense 激活上拟合的 Wo bank；没有重新拟合，唯一压缩配置变化是启用的层类型。

## 结果

| 配置（均为 Dense V） | 官方严格匹配 | 官方宽松提取 | 宽松提取的数值等价比较 | 达到 1024-token 上限 |
|---|---:|---:|---:|---:|
| 不压缩 Wo | 93.25% (1230) | 93.48% (1233) | 93.93% (1239) | 6 |
| 仅 8 层 full-attention Wo | 90.90% (1199) | 92.19% (1216) | 92.72% (1223) | 10 |
| 仅 24 层 GDN Wo | 90.90% (1199) | 89.08% (1175) | 89.54% (1181) | 47 |
| 全部 32 层 Wo | 69.37% (915) | 42.91% (566) | 43.29% (571) | 765 |

括号是正确题数，分母均为 1319。数值等价比较只对原 flexible-extract 提取出的数字做 Decimal 比较，不是重新选答案或人工语义判分。原官方结果不变。

## 生成行为与组合效应

| 配置 | 回复含生成的 </think> | 至少 3 次 </think> | 严格正确但宽松错误 | 生成 token 总数 |
|---|---:|---:|---:|---:|
| 不压缩 Wo | 0 | 0 | 1 | 230519 |
| 仅 full-attention Wo | 0 | 0 | 4 | 239096 |
| 仅 GDN Wo | 196 | 32 | 26 | 271846 |
| 全部 Wo | 1100 | 549 | 350 | 865925 |

在 Dense 和两个单独压缩组都正确、全部 Wo 组错误的题目数分别为：严格 274、宽松 592、数值等价 598。对应 doc_id 列表完整保存在汇总 JSON 的 split_interaction 字段。

例：第 6 题（羊的总数 260）与第 16 题（列车距离 230），两个单独压缩组均正确输出 #### 答案并结束；此前全部 Wo 组会重复解答，最终截断，导致最后数字提取错误。这里也仍有答案格式因素，不能把所有配对失败等同于数学推理失败。

结果支持：主要的大规模生成异常在两类 Wo 同时压缩时出现，组合后的影响明显超过各自单独启用时的观察值。GDN-only 已有少量结束标签和重复异常，值得继续定位，但不能将全部 Wo 的退化直接归给 GDN。

这不是某种精确的线性损失分解，也没有证明唯一机制。当前 bank 的 32 层均在上游 Wo 原生时做局部拟合，联合替换后输入分布变化、跨层误差累积与生成反馈是后续可检查的解释。两个单独组的压缩层数不同，不能由此推断每层 GDN 比每层 full attention 固有地更难压缩。

## 受控设置及核验

- Dense V 保留全部原生 256 维 V；bank 为 results/q35_hybrid/banks/c1_uniform_v256.pt，空因子映射，无 V 重建模块。
- 三个压缩组的 Wo bank 都是 results/q35_hybrid/wo_dense/wo_bank.pt，文件哈希、V factor 哈希一致。该 bank 已在 Dense 校准轨迹上用 FP64 拟合，BF16 导出；Wo 每 source 1024→512、四 source 合计 2048。
- full-attention-only 层为 [3,7,11,15,19,23,27,31]。
- GDN-only 层为 [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]。
- GSM8K main test 全量，5-shot，thinking=False，greedy，seed 20260909。max_new_tokens=1024，context=8192，max_num_seqs=32，max_num_batched_tokens=4096，固定 6 GiB KV cache，TP1，CUDA Graph 开启，prefix cache 关闭。
- 审计重算评分、校验四组逐题提示/目标/生成参数一致、无提示截断、题数 1319、评分记录 2638、模型和分词器身份、V/Wo bank 以及 Dense 校准 manifest。旧 Dense 与全部 Wo 结果哈希和此前审计记录一致。
- scope 开关的模块替换/保留行为已在此前单元测试中核验。本次没有再次运行两个新 scope 的 HF 全量或短前缀对照；既有全部 Wo 的 HF 对照不能当成这两组新实验的逐 token 等价证明。

## 实际命令与运行

环境 lowrankarena，本机直接运行，无 Slurm；GDN-only 使用 GPU 2，full-attention-only 使用 GPU 5，同时运行，每进程 CPU 2 线程。vLLM 0.18.1、lm-eval 0.4.11。

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=2 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --bank results/q35_hybrid/banks/c1_uniform_v256.pt \
  --wo-bank results/q35_hybrid/wo_dense/wo_bank.pt --wo-scope gdn \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --max-new-tokens 1024 --max-model-len 8192 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/gsm8k_vllm/result_dense_gdn_wo.json \
  >> results/q35_hybrid/logs/gsm8k_vllm_result_dense_gdn_wo.log 2>&1

CUDA_VISIBLE_DEVICES=5 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --bank results/q35_hybrid/banks/c1_uniform_v256.pt \
  --wo-bank results/q35_hybrid/wo_dense/wo_bank.pt --wo-scope full_attention \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --max-new-tokens 1024 --max-model-len 8192 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/gsm8k_vllm/result_dense_full_wo.json \
  >> results/q35_hybrid/logs/gsm8k_vllm_result_dense_full_wo.log 2>&1
```

两条命令实际由独立进程同时执行。GDN-only 约 416.3 秒，full-attention-only 约 350.8 秒，均含初始化和评分。共享 GPU、回答长度和启动开销不同，这些耗时不作为受控吞吐结果。

审计在 lowrank 下使用 CPU 2 线程：

```bash
conda activate lowrank
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.summarize_qwen35_gsm8k_v96 --split-wo \
  --output results/q35_hybrid/gsm8k_vllm/dense_split_wo_summary.json \
  > results/q35_hybrid/logs/gsm8k_dense_split_wo_audit.log 2>&1
```

两组评测和审计均 exit 0，无 OOM 或评测失败；vLLM 退出时有 destroy_process_group 清理警告，引擎正常关闭。Wo 为 TP1 四逻辑 source 等价计算，本次未测量真实通信收益。

原始回答和汇总在 results/q35_hybrid/gsm8k_vllm/；拟合 bank 和旧结果未覆盖。未提交或上传 GitHub。
