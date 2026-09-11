# Dense V + low-rank Wo：执行方案

用户已确认并完成新校准、拟合、HF/vLLM 短核验与全量 GSM8K。结果见 [完整报告](qwen35_dense_v_wo_results.md)。本方案检验 Dense V 上重新校准的 Wo-only；与直接恢复 V96+Wo 的 V、保持 Wo 不变的消融不同。

## 具体设置

- 模型：results/q35_hybrid/model，与之前 GSM8K 相同的本地原始 checkpoint。
- V：全部 full-attention 层使用原生 256 维 V，不安装 V writer/decoder 包装。c1_uniform_v256.pt 只是空因子映射与原生 rank schedule、身份哈希的记录，不执行浮点恒等矩阵变换。
- 校准：results/q35_hybrid/data/windows.pt，沿用 256 个 fit 与 64 个 heldout 窗口，每窗 2048 tokens；全部 Wo 原生时收集输入二阶矩。
- Wo：8 层 full attention + 24 层 GDN，TP4 logical sources，每个 source 从 1024 压到 512，合计保留一半维度。FP64 拟合、6 sweeps，按既有 heldout 选择规则导出 BF16 因子。
- 验证：先做四提示 vLLM smoke 与 HF teacher-forced 对照，每提示 top-1 agreement 至少 95%、最大 chosen-token logprob 差异小于 0.3 才进入全量。该门槛只是短前缀回归检查，不证明所有长生成等价。
- GSM8K：全量 1319 题、5-shot、thinking=False、greedy、seed 20260909、并发 32、max_new_tokens 1024、context 8192、batched tokens 4096、固定 6 GiB KV cache。

## 已确认并执行的命令

```bash
bash scripts/run_qwen35_dense_v_wo.sh 6
```

全部 Python 命令和参数见脚本。拟合/HF 使用 lowrank，vLLM 使用本会话已获准的 lowrankarena。单张 GPU 6 顺序执行，CPU 2 线程，本机直接运行，无 Slurm。启动前需要再次检查共享 GPU 的即时可用性。

输出：

- 原生 V 身份 bank：results/q35_hybrid/banks/c1_uniform_v256.pt。
- 二阶矩：results/q35_hybrid/wo_dense_moments/。
- 拟合记录与 Wo bank：results/q35_hybrid/wo_dense/。
- 原始 GSM8K：results/q35_hybrid/gsm8k_vllm/result_dense_wo.json。
- Smoke/HF：同目录 smoke_dense_wo.json、reference_dense_wo.json。
- 日志：results/q35_hybrid/logs/dense_v_bank.log、wo_dense_capture.log、wo_dense_fit.log、gsm8k_smoke_dense_wo.log、gsm8k_reference_dense_wo.log、gsm8k_vllm_result_dense_wo.log。

旧模型、banks 和结果不覆盖。评测后需重新核对与 Dense/V96+Wo 的逐题提示及评分，再报告官方严格、宽松、数值等价评分与截断/重复计数；此准备文档没有预测或实测 GSM8K 分数。

## 已完成验证

lowrank 环境，本机 CPU 2 线程执行：

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_qwen35_wo_scope.py tests/test_qwen35_vllm_hybrid.py \
  > results/q35_hybrid/logs/dense_v_wo_unit_tests.log 2>&1
bash -n scripts/run_qwen35_dense_v_wo.sh
```

8 项测试通过，脚本语法检查通过。新增测试验证启用 Wo 后，原生 attention 对象、QKV writer 对象及其权重、偏置保持不变。

没有提交或上传 GitHub。
