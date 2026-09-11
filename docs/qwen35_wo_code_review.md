# Wo 代码审查与局部核验

结论：本次没有发现可以解释当前 GSM8K 循环的 Wo 转置、source 分块、bias 或 gate/norm 顺序错误。确认并修正了一处 HF 诊断停止条件不一致；修正后两道异常题仍复现，不能据此将全部异常归因于该诊断问题。没有重跑全量 GSM8K、改动已保存因子或修改低秩拟合目标。

## 确认的问题与修正

evaluation/diagnose_qwen35_gsm8k_generation.py 原先调用 HF generate 时只设置 pad_token_id，EOS 继承模型配置 248044；本模型 tokenizer 的 chat EOS <|im_end|> 是 248046。vLLM 的任务停止列表包含 <|im_end|>，因此两种诊断自由生成的停止规则不完全一致。

已将 HF 诊断显式设置为 EOS [248044,248046]，并加入任务停止字符串 Question:、</s>、<|im_end|>。没有把 </think> 当作 EOS，也没有更改原 GSM8K vLLM 评分流程。新增 tests/test_qwen35_generation_stops.py 回归测试通过。

此问题会影响遇到 chat EOS 或任务停止文本后的 HF 对照严谨性，但不会使原 vLLM GSM8K 评测漏掉该停止字符串。旧原始结果和旧诊断输出均保留。

## 已核对的 Wo 数学和运行路径

- 校准通过输出投影的 forward-pre-hook 读取输入，因此 full attention 已乘 sigmoid gate，GDN 已通过原生 gated norm。没有重新手写或移动 gate。
- 目标使用原始输出投影权重，不压缩时的仿射 bias 在运行时只加一次。本模型相应原生投影无 bias；通用 bias 路径另有测试。
- source 分块是输入维度连续切成 4 段，E_s 形状 [1024,512]；decoder 每段 D_s 形状 [4096,512]。运行结果为 sum_s x_s E_s D_s^T。导出拼接顺序和这个定义一致。
- covariance_to_source_blocks 将完整二阶矩转为 [source,source,width,width]；拟合保留跨 source 协方差，decoder 为联合求解。encoder 每组仅对应一个 source，此设置下 two-sided 求解是对应的精确子问题。
- HF GDN 原生顺序是 norm(core,z) → reshape → out_proj；HF full attention 是 attention → sigmoid gate → o_proj。vLLM 顺序一致，替换的只有输出投影，返回 (output,None) 满足其 RowParallelLinear 调用约定。
- 实际运行 TP1；这里未验证真正 TP4 分布式 AllGather，不能将此结果当成分布式实现验证。

## 已保存因子的独立数值核验

脚本 evaluation/check_qwen35_wo_code.py 对 Dense-Wo 的层 0、3、15、31 执行：

1. 校准记录中的原始 Wo 与原模型 safetensors 权重逐元素完全一致。
2. 从 E_s、D_s 独立构造完整近似矩阵，与模块 FP64 前向比较。
3. 用保存的 heldout 二阶矩重算 trace((W_hat-W) C (W_hat-W)^T) / trace(W C W^T)，与导出指标比较。

| 层 | 类型 | 保存的 heldout 相对 MSE | 独立重算 | FP64 前向最大绝对差 |
|---|---|---:|---:|---:|
| 0 | GDN | 0.0007504081534296009 | 0.0007504081534298465 | 8.88e-14 |
| 3 | full attention | 0.004520692345474436 | 0.0045206923454746365 | 3.91e-14 |
| 15 | full attention | 0.0413076370897437 | 0.04130763708974279 | 2.58e-14 |
| 31 | full attention | 0.014089475230946044 | 0.014089475230944924 | 2.66e-14 |

另外在每层 17 个随机输入上，BF16 模块前向与 BF16 因子的 FP64 计算间相对 MSE 为 5.43e-6 至 5.55e-6。该随机输入检查不证明真实激活或任意长生成的数值误差界。

## 两道异常 GSM8K 题的 HF 对照

使用 Dense V + 新拟合 Wo 的既有完整提示，doc_id 6、16，分别 964、1003 个 prompt tokens，检查各前 256 个生成 token。

- teacher-forced top-1：两题均 100%，共 512 tokens。
- 修正停止条件后的 HF cached generation 文本与 vLLM 前缀完全相同。
- 修正前后这两题 HF 输出也相同，前缀中没有触发新增停止条件。
- 第 6 题重复给出 #### 260；第 16 题重复给出 #### 230 和 </think>。短前缀内分别包含 2、4 次 ####，2、6 次 </think>。

这支持所检查循环不是这两例中的 vLLM 特有问题；仍不能排除两后端共用的 Wo 拟合/实现问题，也不是全模型所有输入的等价证明。

## 诊断盲区与方法限制

- basisserve/core/tp_source_wo_fit.py 将 decoder_stationarity_override 设为恒等返回 0。这是 checkpoint 诊断的绕过，不改变 decoder 求解或前向；当前 heldout selector 也不以该值选择因子。不能用这个字段证明 decoder 已平稳。此前 1.79e-12 是 encoder 线性子问题残差，不是整个 ALS 的全局收敛证据。本次没有擅自更改该拟合路径。
- quantized_heldout_relative_mse 是 BF16 存储因子提升到 work dtype 后的二次型误差，不包含中间 codes 的 BF16 舍入和整网误差传播。独立核验确认它算对了，但它不是整个 BF16 模型的实测任务误差。
- 校准来自自然文本窗口，不含显式 chat-template/EOS 行为约束；目标是逐层输出 MSE，不是结束 token、KL 或生成损失。32 层 Wo 的输入在上游 Wo 仍原生时采集，之后一并替换，没有逐层重新采集上游 Wo 压缩后的轨迹。这是当前局部拟合方法的边界，不是已证明的 bug 或唯一退化原因。

若继续定位，应优先在 Dense V 条件下分开 GDN-only Wo 与 full-attention-only Wo，观察首次产生结束/重复异常的范围；之后才判断需修实现、提高 Wo rank，或调整校准目标。本次没有启动这些新实验。

## 命令、环境、输出

所有检查使用 lowrank、本机执行；GPU 局部检查和 HF 复核用 GPU 6，CPU 2 线程。未使用 Slurm。

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.check_qwen35_wo_code \
  --output results/q35_hybrid/wo_code_check.json \
  > results/q35_hybrid/logs/wo_code_check.log 2>&1

PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_qwen35_gdn_private_ag.py tests/test_qwen35_full_attention_private_ag.py \
  tests/test_qwen35_hybrid_wo_solver.py tests/test_tp_source_wo_c1.py \
  tests/test_qwen35_wo_scope.py tests/test_qwen35_vllm_hybrid.py \
  > results/q35_hybrid/logs/wo_review_tests.log 2>&1

PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_qwen35_generation_stops.py \
  > results/q35_hybrid/logs/wo_stop_settings_test.log 2>&1

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=results/q35_hybrid/deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.diagnose_qwen35_gsm8k_generation \
  --pilot results/q35_hybrid/gsm8k_vllm/result_dense_wo.json \
  --doc-ids 6,16 --max-new-tokens 256 \
  --output results/q35_hybrid/gsm8k_vllm/wo_review_hf_aligned.json \
  > results/q35_hybrid/logs/wo_review_hf_aligned.log 2>&1
```

相同 HF 命令在修复前使用输出 wo_review_hf_generation.json、日志 wo_review_hf_generation.log，保留用于对照。22 项既有测试与 1 项新增测试通过；全部局部检查正常退出。HF 日志有可选加速库缺失、使用 Torch 路径的提示，无核验失败。

仅新增核验脚本、停止设置回归测试和审查记录，并修正 HF 诊断停止设置；没有改 Wo 数学、已拟合 bank 或旧评测分数。未提交或上传 GitHub。
