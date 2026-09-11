# Qwen3.5-9B GSM8K：gated V + Wo 的 vLLM 评测

四组完整测试均已完成并通过逐题审计。当前 V64 checkpoint 的 GSM8K 表现明显低于 Dense，即使自然文本 PPL 变化相对有限。

## 设置

- GSM8K main test 全部 1,319 题，5-shot，相同示例与 seed 20260909。
- 原生 chat template，thinking 关闭，greedy，最多生成 1,024 tokens；不修改答案过滤规则。
- vLLM 0.18.1、lm-eval 0.4.11、lowrankarena；HF 对照和最终审计使用 lowrank。
- A100 本地执行，无 Slurm，每进程 2 CPU threads。vLLM 最多并发 32 条序列，4,096 batched tokens，context 上限 8,192，启用 CUDA Graph，关闭 prefix cache。
- Wo 覆盖全部 8 层 full attention 和 24 层 GDN/DeltaNet 的输出投影，均保留一半输入维度。GDN recurrence、卷积、原生 norm/gate 及状态表示保持原生实现。

## 完整结果

严格匹配使用官方 #### 答案过滤器；宽松提取使用官方数值过滤器。达到长度上限的题目仍按同一过滤规则计分，不剔除。

| 配置 | 严格匹配 | 宽松提取 | 宽松分数相对 Dense | 达到 1024-token 上限 |
|---|---:|---:|---:|---:|
| Dense | 93.25% (1230/1319) | 93.48% (1233/1319) | +0.00 pp | 6 (0.45%) |
| Uniform V64 | 3.49% (46/1319) | 56.03% (739/1319) | -37.45 pp | 185 (14.03%) |
| Two-sided V64 | 10.39% (137/1319) | 65.50% (864/1319) | -27.98 pp | 162 (12.28%) |
| Two-sided V64 + Wo | 15.31% (202/1319) | 21.08% (278/1319) | -72.40 pp | 835 (63.31%) |

## 与已有 PPL 对照

| 配置 | WikiText-2 PPL | C4 PPL | GSM8K 宽松提取 |
|---|---:|---:|---:|
| Dense | 8.6511 | 11.3291 | 93.48% |
| Uniform V64 | 9.5704 | 12.3024 | 56.03% |
| Two-sided V64 | 8.5679 | 12.1253 | 65.50% |
| Two-sided V64 + Wo | 8.9572 | 12.6917 | 21.08% |

## 生成异常

严格与宽松分数的差异包含答案格式变化；宽松分数也低于 Dense，因此不能仅用严格格式要求解释整个任务退化。以下是原始回复的简单统计，未据此改变评分或输出。

| 配置 | 含生成的 </think> | 至少重复 3 次 </think> | 宽松正确但严格错误 | 平均生成 tokens |
|---|---:|---:|---:|---:|
| Dense | 0 | 0 | 4 | 174.8 |
| Uniform V64 | 0 | 0 | 693 | 387.4 |
| Two-sided V64 | 0 | 0 | 732 | 397.7 |
| Two-sided V64 + Wo | 1319 | 370 | 87 | 684.9 |

## 接入验证及边界

- vLLM V writer 输出 latent，补零放入原生 256 维 V cache slots；attention 后仅恢复当前 query 输出，再施加原生 gate。没有跨 gate 折叠 decoder。
- Wo 使用原有四个 logical source encoder、拼接和 joint decoder。实际为 TP1 等价计算，没有执行分布式 AllGather。
- 本版本不提供真实 compact-cache 显存压缩；不能据此声称通信或 KV-cache 节省。GPU 为共享使用，端到端耗时还受输出长度与初始化影响，不作为吞吐对照实验。
- 四个短提示、四组模型共 438 个生成 token 的 HF teacher-forced top-1 与 vLLM 全部一致；最大 chosen-token log-probability 差异约 0.105。
- 针对异常表现，另外检查 two-sided V64 和 V64+Wo 的真实 GSM8K 第 0、2 题，各前 128 tokens；每组的首选一致率分别为 100% 和 99.21875%。HF cached generation 复现了 Wo 的重复 </think> 输出。
- 这些对照支持所观察异常并非这些案例中的 vLLM 接入错误，但不是对全部任意长度输出逐 token 等价的证明。
- 此轮没有重新拟合 V/Wo。原 V-ALS 的有限迭代、未完全达到残差容差的限制仍存在；不能仅凭此次 GSM8K 将退化归因于某一个原因。

## 运行修复与审计

- 补齐 text-only 插件的原生 hybrid-cache 以及 M-RoPE 接口，并用 spawn worker 避免继承父进程线程池。
- 修正 lm-eval 每题分别记录两种 filter 的计数检查。1319 题对应 2638 条评分记录，实际 generation request 为 1319 条。
- 共享 GPU 的显存变化曾使自动 KV-cache 估计失败。压缩组最终使用固定 6 GiB cache；Dense 已用原自动预算完成。并发数、提示和生成参数一致。
- 最终审计逐题核对缓存的官方 test Arrow、两个 filter 的评分重算、四组提示一致性、生成记录、长度上限、模型/分词器身份及 factor bank 哈希。
- 数学测试 4 项、hybrid-cache/M-RoPE 合同测试 1 项、题数审计测试 1 项通过；原始失败日志全部保留。

## 实际执行命令

环境前缀：
```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export PYTHONPATH=.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

Dense:

```bash
CUDA_VISIBLE_DEVICES=6 python -u /home/lz299/BasisServe-CALS/evaluation/eval_qwen35_hybrid_gsm8k_vllm.py --max-num-seqs 32 --max-num-batched-tokens 4096 --max-new-tokens 1024 --max-model-len 8192 --gpu-memory-utilization 0.40 --output results/q35_hybrid/gsm8k_vllm/result_dense.json > results/q35_hybrid/logs/gsm8k_vllm_result_dense.log 2>&1
```

Uniform V64:

```bash
CUDA_VISIBLE_DEVICES=5 python -u /home/lz299/BasisServe-CALS/evaluation/eval_qwen35_hybrid_gsm8k_vllm.py --max-num-seqs 32 --max-num-batched-tokens 4096 --max-new-tokens 1024 --max-model-len 8192 --gpu-memory-utilization 0.40 --kv-cache-gib 6 --output results/q35_hybrid/gsm8k_vllm/result_uniform64.json --bank results/q35_hybrid/banks/c1_uniform_v64.pt > results/q35_hybrid/logs/gsm8k_vllm_result_uniform64.log 2>&1
```

Two-sided V64:

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/lz299/BasisServe-CALS/evaluation/eval_qwen35_hybrid_gsm8k_vllm.py --max-num-seqs 32 --max-num-batched-tokens 4096 --max-new-tokens 1024 --max-model-len 8192 --gpu-memory-utilization 0.40 --kv-cache-gib 6 --output results/q35_hybrid/gsm8k_vllm/result_twosided64.json --bank results/q35_hybrid/banks/c1_twosided_v64.pt > results/q35_hybrid/logs/gsm8k_vllm_result_twosided64.log 2>&1
```

Two-sided V64 + Wo:

```bash
CUDA_VISIBLE_DEVICES=2 python -u /home/lz299/BasisServe-CALS/evaluation/eval_qwen35_hybrid_gsm8k_vllm.py --max-num-seqs 32 --max-num-batched-tokens 4096 --max-new-tokens 1024 --max-model-len 8192 --gpu-memory-utilization 0.40 --kv-cache-gib 6 --output results/q35_hybrid/gsm8k_vllm/result_twosided64_wo.json --bank results/q35_hybrid/banks/c1_twosided_v64.pt --wo-bank results/q35_hybrid/wo_twosided_v64/wo_bank.pt > results/q35_hybrid/logs/gsm8k_vllm_result_twosided64_wo.log 2>&1
```

结果目录：results/q35_hybrid/gsm8k_vllm/。summary.json 包含完整指标、逐题配对统计、运行参数和结果哈希；result_*.json 包含全部原始回答。日志位于 results/q35_hybrid/logs/gsm8k_vllm_*.log。

未执行 Git commit、push 或上传。
