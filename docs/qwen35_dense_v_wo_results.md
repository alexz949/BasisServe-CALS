# Dense V + 重新拟合低秩 Wo：GSM8K 全量结果

全部流程已完成：原生 Dense V 校准、32 层 FP64 Wo 拟合、HF/vLLM 短前缀核验、GSM8K 全量 1319 题和逐题审计。

## 结果

| 配置 | 官方严格匹配 | 官方宽松提取 | 宽松提取的数值等价比较 | 达到 1024-token 上限 |
|---|---:|---:|---:|---:|
| Dense | 93.25% (1230) | 93.48% (1233) | 93.93% (1239) | 6 |
| V96-only | 72.93% (962) | 87.26% (1151) | 88.63% (1169) | 49 |
| V96 + Wo | 31.08% (410) | 30.02% (396) | 30.63% (404) | 918 |
| **Dense V + 新拟合 Wo** | **69.37% (915)** | **42.91% (566)** | **43.29% (571)** | **765** |

分母均为 1319，括号为正确题数。数值等价比较只修正已有 flexible-extract 数字的表示差异，例如 26.00 和 26；不重新选择答案，不是人工语义判分。原始官方评分不变。

## 为什么严格评分显著高于宽松评分

Dense V + Wo 有 350 题严格正确但宽松错误，只有 1 题宽松正确但严格错误。严格规则寻找 #### 后的数字；宽松规则取最后一个数字，两者不是包含关系。

定向抽查原始回答（doc_id 从 0 开始）：

- 第 6 题：正确算出总数并写出 #### 260，随后反复重写相同解答，最终截断在中间计算；宽松提取为 80。
- 第 14 题：正确写出 #### 60，反复重写步骤，最后截断在 0.25，宽松提取为 0.25。
- 第 16 题：正确写出 #### 230，随后重复同一解答与 </think>，末尾计算表达式被截断为 2，宽松提取为 2。

因此不能将 42.91% 直接解释为模型只会做对这么多数学题；这次有明显的答案后续写、重复、截断与最后数字提取的相互影响。严格 69.37% 也是协议评分，并非对全部回答的人工核验。这里仅定向抽查，不据此宣称全部失分原因已经分类。

## 生成异常与比较

- 765/1319 题（58.00%）达到输出上限。
- 1100 条生成回复含 </think>，549 条至少包含 3 次。所有提示仍关闭 thinking。
- 相比 Dense，严格准确率低 23.88 pp；配对统计为 335 题从对变错、20 题从错变对。
- 相比 V96+Wo，严格准确率高 38.29 pp，宽松高 12.89 pp，截断从 918 降至 765。
- 本次 Wo 在 Dense V 上重新校准和拟合，没有复用 V96 的 Wo 因子。因此与 V96+Wo 的差异同时包含 V 和 Wo 因子变化，不是固定 Wo 因子下只恢复 V 的消融。
- 原生 V 下单独压缩 Wo 仍会出现生成异常，说明问题不需要 V 压缩才能发生。尚不能区分两类 Wo、校准目标与实现各自的贡献，也不能把结果泛化为所有 Wo 低秩方法都会失败。

## 校准、拟合与验证

- 原模型 results/q35_hybrid/model，身份与之前评测一致。
- 原生 V bank：results/q35_hybrid/banks/c1_uniform_v256.pt。所有 8 层 schedule 为 256、layers 为空、encoder_sweeps=0，完全不安装 V 压缩或重建模块。
- 256 个 fit、64 个 heldout 窗口，每窗 2048 tokens，沿用 results/q35_hybrid/data/windows.pt。
- 校准轨迹标记 native_dense；目标为原生 post-gate 输入乘原始输出投影。
- Wo 覆盖 24 层 GDN + 8 层 full attention，每层 encoder [4,1024,512]、joint decoder [4096,2048]；保留一半维度。
- FP64 拟合，每层 6 sweeps，BF16 导出。全部层 encoder 线性求解最大相对残差为 1.7864e-12。该数值描述线性子问题，不构成 ALS 达到全局最优或生成质量良好的证明。
- HF/vLLM 四个短提示共 89 个 token，teacher-forced top-1 全部一致，最大 chosen-token logprob 差异 0.09539；通过预设短前缀门槛。本次没有再做全量 GSM8K 长回复的 HF 对照。
- 原生 V/Wo 接口相关 8 项单元测试已通过，核验了 QKV writer 和 attention 对象及参数不被 V 路径修改。
- 最终审计验证题数、官方评分重算、六组提示/目标/生成参数一致、无提示截断、模型/分词器身份、V/Wo bank 哈希、Dense 校准 manifest 和 Wo 层覆盖。审计 exit 0。

## 实际执行

本机 GPU 6，CPU 2 线程，无 Slurm。校准、拟合、HF 与审计用 lowrank；vLLM 用本会话获准的 lowrankarena。主流水线命令：

```bash
bash scripts/run_qwen35_dense_v_wo.sh 6 \
  >> results/q35_hybrid/logs/dense_v_wo_pipeline.log 2>&1
```

脚本保留全部 Python 命令、环境切换和参数。GSM8K 为 5-shot、thinking=False、greedy、seed 20260909、max_new_tokens=1024、context=8192、并发 32、batched tokens=4096、固定 KV cache 6 GiB、TP1、CUDA Graph 开启、prefix cache 关闭。评测使用 vLLM 0.18.1、lm-eval 0.4.11；全量评测阶段约 923 秒，包含初始化与评分，不含校准/拟合/短核验，也不是受控速度比较。

最终审计：

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.summarize_qwen35_gsm8k_v96 --with-wo --dense-wo \
  --output results/q35_hybrid/gsm8k_vllm/dense_wo_summary.json \
  > results/q35_hybrid/logs/gsm8k_dense_wo_audit.log 2>&1
```

## 文件和运行提示

- 校准：results/q35_hybrid/wo_dense_moments/。
- 因子：results/q35_hybrid/wo_dense/wo_bank.pt，逐层拟合记录同目录。
- 回答：results/q35_hybrid/gsm8k_vllm/result_dense_wo.json。
- 审计：results/q35_hybrid/gsm8k_vllm/dense_wo_summary.json，含配对计数、各层拟合摘要与身份校验。
- 短核验：同目录 smoke_dense_wo.json、reference_dense_wo.json。
- 日志：results/q35_hybrid/logs/ 下 dense_v_wo_pipeline.log、dense_v_bank.log、wo_dense_capture.log、wo_dense_fit.log、gsm8k_smoke_dense_wo.log、gsm8k_reference_dense_wo.log、gsm8k_vllm_result_dense_wo.log、gsm8k_dense_wo_audit.log。
- lowrank 缺少可选 FLA/卷积加速库，使用原生 Torch 路径；vLLM 退出时有 destroy_process_group 清理警告。流程正常退出，无 OOM 或评测失败。
- Dense V 保持原生 cache；Wo 是 TP1 的四逻辑 source 等价计算，没有测量实际分布式通信节省。

旧模型、banks 和结果未覆盖。没有提交或上传 GitHub。
