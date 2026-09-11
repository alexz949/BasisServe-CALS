# Qwen3.5 GSM8K 回答审计与下一轮诊断

本次只读取既有四组 1,319 题回答，没有重新生成。自动标记用于筛选人工审阅案例；不将标准答案数字在正文中出现视为回答正确。全部语义错误尚未逐题人工判定。

## 数值表示造成的额外失分

官方 flexible-extract 取最后一个数字，再进行字符串 exact match；它不将 26.00 与 26 视为相等。下面仅将已经提取的数值用 Decimal 比较，未重新选择答案，未修改原始官方分数。

| 配置 | 官方宽松正确数 | 数值等价正确数 | 新增数值等价题 |
|---|---:|---:|---:|
| Dense | 1233 | 1239 | 6 |
| Uniform V64 | 739 | 757 | 18 |
| Two-sided V64 | 864 | 905 | 41 |
| Two-sided V64+Wo | 278 | 287 | 9 |

分母均为 1319。这仍是自动提取分数，不是人工确认的数学准确率。

## 人工核对案例（doc_id 从 0 开始，定向抽查，不是随机估计）

- Uniform V64 / 46：明确回答 163，最后追加验算以 23 结尾，官方提取 23。属于答案选择失误。
- Two-sided V64 / 24：明确回答 $26.00，标准答案 26。属于数值表示失分。
- Two-sided V64 / 27：明确回答 16.00，标准答案 16。属于数值表示失分。
- Two-sided V64 / 78：明确回答 $6.00，标准答案 6。属于数值表示失分。
- Uniform V64 / 19：将 12/4 算成 6，最终答 1.5 mph，标准答案 6 mph。属于实际计算错误，正文出现 6 不代表答对。
- Uniform V64 / 44：将 20×2 算成 20，最终答利润 0，标准答案 20。属于实际计算错误。
- Two-sided V64 / 4：将 0.75+1.25 算成 1.95，最终答 21，标准答案 20。属于实际计算错误。
- Two-sided V64 / 2：将上涨 150% 理解成变为原值的 150%，答 -10000，标准答案 70000。属于题意理解错误。
- Two-sided V64+Wo / 0：重复输出 </think> 达到输出上限，没有答案。属于生成异常。
- Two-sided V64+Wo / 34：明确答对 23，之后生成“Question 2”，提取器取到 2。同时涉及续写控制与答案选择。

## 自动筛选统计

| 配置 | 官方宽松错误 | 错误且截断 | 错误且出现重复标记 | 错误且两种标记都没有 | 错误但正文出现标准数值 |
|---|---:|---:|---:|---:|---:|
| Dense | 86 | 6 | 0 | 80 | 21 |
| Uniform V64 | 580 | 173 | 25 | 404 | 135 |
| Two-sided V64 | 455 | 145 | 19 | 308 | 139 |
| Two-sided V64+Wo | 1041 | 741 | 723 | 293 | 138 |

各标记可以重叠，不能相加。重复标记指至少 3 次相同非空长行或 </think>，正常复述也可能触发；不等同于已确认的死循环。没有标记的错误仍待语义核查，不能直接归为推理错误。正文数值命中只是候选筛选，不构成准确率上界或补分依据。

## 已执行命令与验证

环境 lowrank，本机 CPU，OMP_NUM_THREADS=2、MKL_NUM_THREADS=2、PYTHONPATH=.。

```bash
python -u -m evaluation.audit_qwen35_gsm8k_answers --output results/q35_hybrid/gsm8k_vllm/answer_numeric_audit.json > results/q35_hybrid/logs/gsm8k_answer_numeric_audit.log 2>&1
python -m pytest -q tests/test_qwen35_wo_scope.py tests/test_qwen35_vllm_hybrid.py > results/q35_hybrid/logs/gsm8k_wo_scope_tests.log 2>&1
```

审计完成，7 项测试通过。之前仅做候选标记的 answer_audit.json 和日志保留。新增 --wo-scope all/full_attention/gdn；未选中的 Wo 保持原生投影，默认仍为 all，原 bank 不变。

## 准备好的 GPU 诊断（尚未运行）

```bash
bash scripts/run_qwen35_gsm8k_diagnosis.sh 2
```

脚本使用 lowrankarena，在 GPU 2 顺序执行四组，各 128 题：Two-sided V64+full-attention-only Wo、Two-sided V64+GDN-only Wo、Two-sided V80、Two-sided V96。所有组 thinking 关闭、5-shot、seed 20260909、并发 32、最多生成 1024 tokens、context 8192、batched tokens 4096、固定 6 GiB KV cache、2 CPU threads。全部 Python 命令及 bank 路径在脚本中；日志和 JSON 位于 results/q35_hybrid/gsm8k_diagnosis/。

已有 Dense、V64、V64+全部 Wo 的全量结果可取同一前 128 题作配对对照，但需核对提示完全一致。选择固定前 128 题只用于诊断，不代表全量最终分数。

Wo 拆分复用在原始全层压缩流程中拟合的 bank，不重新拟合；结果说明当前 bank 的分组启用效果，并不独立估计某类 Wo 经重新校准后的最优效果。V80/V96 使用各自已有拟合 bank，差异同时包含 rank 分配和拟合因素。

尚未提交或上传 GitHub。
