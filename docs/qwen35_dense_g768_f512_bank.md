# Dense V + GDN Wo 768 / full-attention Wo 512

混合 bank 已完成并核验。本次先生成因子，没有启动此配置的全量 GSM8K 或新的整模型 HF/vLLM smoke。

## 产物与形状

- Wo bank：results/q35_hybrid/wo_dense_g768_f512/wo_bank.pt。
- 核验：同目录 audit.json；24 个 wo_lXX.pt 保留逐层 GDN 拟合记录。
- Dense V 身份 bank：results/q35_hybrid/banks/c1_uniform_v256.pt，原生 V，不装 V 压缩模块。
- 24 层 GDN：每 source E 为 [1024,768]，四 source 合计 3072；joint decoder [4096,3072]，保留 75%。
- 8 层 full attention：每 source E 为 [1024,512]，四 source 合计 2048；joint decoder [4096,2048]，保留 50%。
- 按 32 层相同原输入维度平均，Wo 保留率为 68.75%。这只是维度比例，不是实测通信或速度收益。

## 拟合方法和验证

复用 results/q35_hybrid/wo_dense_moments/ 的 Dense 校准数据：256 个 fit、64 个 heldout 窗口，每窗 2048 tokens。没有重新采集激活。

GDN 从相同二阶矩按 rank 768 独立拟合，不是将旧 rank 512 因子补零。采用已有 C1 求解器、FP64、每层 6 sweeps、heldout 选择、BF16 导出。选中 sweep 包含 0、1、4、6，sweep 0 表示 decoder-only 初始化候选；并非所有层都选最后一轮。

Full-attention 直接复用 results/q35_hybrid/wo_dense/wo_bank.pt 的原 rank 512 记录。导出后重新加载，全部 E 和 decoder tensor 与源 bank 逐元素一致。

- 24/24 层 GDN 的 BF16 因子 heldout 相对 MSE 比原 rank 512 降低。
- 逐层相对 MSE 的算术平均：0.05720128025 → 0.01428391807（5.72% → 1.43%）。这是逐层局部指标，不能当作整网误差、PPL 或 GSM8K 分数。
- 所有 GDN encoder 线性子问题最大相对残差：2.6648e-12。此项不证明整个 ALS 全局收敛。
- 校准 manifest、源 Wo、上游 V factor 身份核对通过；24 层 GDN + 8 层 full-attention 覆盖完整、形状及有限性检查通过。
- 8 项相关测试通过，其中求解测试覆盖 50% 和 75% 保留率，另含前向/接口测试。

## 实际执行命令

环境 lowrank，本机直接执行，无 Slurm。GPU 5、6 各负责 12 层，CPU 每进程 2 线程；两个 fit 命令实际并行运行。组装使用 CPU。

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.fit_qwen35_dense_mixed_wo fit --num-shards 2 --shard-index 0 \
  > results/q35_hybrid/logs/wo_g768_f512_fit_0.log 2>&1
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.fit_qwen35_dense_mixed_wo fit --num-shards 2 --shard-index 1 \
  > results/q35_hybrid/logs/wo_g768_f512_fit_1.log 2>&1
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.fit_qwen35_dense_mixed_wo assemble \
  > results/q35_hybrid/logs/wo_g768_f512_assemble.log 2>&1
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_qwen35_hybrid_wo_solver.py tests/test_qwen35_vllm_hybrid.py \
  > results/q35_hybrid/logs/wo_g768_f512_tests.log 2>&1
```

拟合、组装和测试均正常完成，无求解失败。原 rank 512 bank、校准数据和旧评测输出未覆盖。

## 下一步评测接入

已有 vLLM evaluator 支持这个混合形状：使用 --bank results/q35_hybrid/banks/c1_uniform_v256.pt、--wo-bank results/q35_hybrid/wo_dense_g768_f512/wo_bank.pt、--wo-scope all。沿用 lowrankarena 环境、thinking 关闭及此前 GSM8K 协议；本次尚未执行。

未提交或上传 GitHub。
