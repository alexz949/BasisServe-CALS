# TP8 外部基线结果摘要：ShadowKV 与 STAR-KV

实验均在 8 张 NVIDIA L40S 上以 TP8 / DP1 / PP1 运行，使用 `basis` conda 环境。这里是两组**不同模型、不同算法口径**的实验：ShadowKV 对比测量完整请求延迟；STAR-KV V-only 对比测量 GPU 显存容量。它们的结果不能互相当作同一工作负载的胜负关系。

| 实验 | 模型与对照 | 正式 trial | 主要结论 |
| --- | --- | ---: | --- |
| ShadowKV 请求与 steady decode | Llama-3.1-8B-Instruct；Dense、BasisKV Joint V96、ShadowKV | 108/108 成功 | 测试范围内无 OOM；约 128K/B8 完整请求为 BasisKV 95.624 s、ShadowKV 167.939 s |
| STAR-KV 上下文显存网格 | Qwen3-8B-Base；BasisKV V64、STAR-KV V-only adaptive | 35/36 成功 | 130048/B8 时 STAR-KV 在 V-cache 分配阶段 OOM，BasisKV 完成 |
| STAR-KV 16K batch sweep | 同上；16K prefill + 128 输出 token | 14/18 成功 | 最大成功**测试** batch：BasisKV 128、STAR-KV 32 |

## ShadowKV：请求延迟

固定配置为 65536/130048 输入 token、batch 1/4/8、三个真实文本 cohort、`request` 与独立 `steady` 两种模式。ShadowKV 使用官方 CPU V offload，K 路径包含在线 gather/SVD/分发；BasisKV Joint V96 的投影、编码和缓存写入计入 prefill 及完整请求时间。请求产生 128 个输出 token：prefill 预测第一个，随后执行 127 次 decode。下表每个方法/工作负载取**完整请求时间居中的 cohort**，构建时间是 ShadowKV 请求内部的组成部分，不应再加到总时间上。

| 输入 / batch | BasisKV 完整请求 | ShadowKV 完整请求 | ShadowKV 在线构建（其中 SVD） |
| --- | ---: | ---: | ---: |
| 64K / 1 | 7.630 s | 19.480 s | 9.866 s（3.739 s） |
| 64K / 8 | 43.392 s | 89.662 s | 39.020 s（29.863 s） |
| 约 128K / 1 | 13.562 s | 28.616 s | 12.417 s（5.932 s） |
| 约 128K / 8 | 95.624 s | 167.939 s | 59.188 s（47.724 s） |

独立 steady 窗口在新请求 prefill 后先运行 16 个 conditioning step，再测 128 个 decode step。约 128K/B8 时，各 cohort 的每步均值再取中位数，得到 BasisKV **27.912 ms/step**、ShadowKV **36.365 ms/step**、Dense **162.653 ms/step**；它不等于完整请求的平均每 token 延迟。约 128K/B8 的 ShadowKV 请求峰值 PyTorch allocated GPU 显存为 **39.764 GiB/卡**，decode-ready allocated 为 **14.055 GiB/卡**；此外 pinned CPU V 容量为八个 rank 合计 **63.5 GiB**。三种方法在此网格内均无 OOM，但本实验只测到 B8，不能据此断言 ShadowKV 在更高 batch 的容量上限。

详细数据：[请求分解 CSV](shadowkv/request_breakdown.csv)、[steady CSV](shadowkv/decode_summary.csv)、[显存 CSV](shadowkv/memory_summary.csv)、[请求分解图](plots/shadowkv_request_breakdown.png)、[协议与验证](shadowkv/README.md)。

## STAR-KV：上下文显存

Qwen3-8B-Base 的 BasisKV V64 与 STAR-KV V-adaptive 导出 checkpoint 均使用精确 dense K、完整注意力、BF16、无 CPU KV offload。STAR-KV 在这里仅比较 **V-only 放置**，不是完整 STAR-KV 系统。导出 checkpoint 的实际全局 V rank 保留率是 **54.0473%**，不能标作严格 50% 保留率。六种上下文长度（4K 至 130048）、B1/B4/B8、一个预选 cohort 组成 36 项网格；prefill chunk 为 4096 token，每项在完整 prompt 后运行一次 decode step。主显存指标是八个 TP rank 中最大的 **decode-ready NVML 进程显存**。

| 输入 token / batch | BasisKV V64 | STAR-KV V-only |
| --- | ---: | ---: |
| 130048 / 1 | 5.680 GiB/卡 | 9.324 GiB/卡 |
| 130048 / 4 | 11.574 GiB/卡 | 26.096 GiB/卡 |
| 98304 / 8 | 16.217 GiB/卡 | 38.102 GiB/卡 |
| 130048 / 8 | 19.451 GiB/卡 | V-cache 分配 OOM |

唯一失败项 130048/B8 的 STAR-KV 在 prefill **之前**分配 V-cache 时，八个 rank 均报告 CUDA OOM；没有给失败项填造 decode-ready 显存值。超过 Qwen3 原生 32768-token 上下文的点仅是**显存压力测试**，不是质量有效性证明。详细数据：[结果 CSV](starkv_v_only/summary.csv)、[OOM 边界](starkv_v_only/oom_frontier.csv)、[显存图](plots/starkv_tp8_memory.png)、[协议与验证](starkv_v_only/README.md)。

## STAR-KV：16K batch sweep

这一独立网格固定 16384-token prefill、256-token prefill chunk、实际生成 128 个输出 token，并预留 128 个 decode slot。两臂各测试 batch 1/2/4/8/16/32/64/128/256。原始 cohort 只有 8 条 prompt；B>8 时按顺序重复它们。因此这测量的是给定输入形状下的**容量边界**，不是请求多样性或质量。16512-token 总长度处于模型原生上下文内。下表同样使用最大 rank 的 decode-ready NVML 进程显存；OOM 无该数值。

| Batch | BasisKV V64 | STAR-KV V-only |
| ---: | ---: | ---: |
| 1 | 3.980 GiB/卡 | 4.482 GiB/卡 |
| 8 | 5.652 GiB/卡 | 9.322 GiB/卡 |
| 32 | 11.150 GiB/卡 | 25.867 GiB/卡 |
| 64 | 18.463 GiB/卡 | OOM |
| 128 | 33.160 GiB/卡 | OOM |
| 256 | OOM | OOM |

STAR-KV 的 B64/B128/B256 与 BasisKV 的 B256 均在 **cache-state 分配阶段、prefill 之前**失败；其余 14 项完成全部 128 个输出 token。最大成功**测试** batch 分别为 32 和 128，不代表对中间 batch 或其他实现的绝对上限。详细数据：[结果 CSV](starkv_v_only/batch_sweep_16k/summary.csv)、[失败记录](starkv_v_only/batch_sweep_16k/failures.csv)、[显存图](starkv_v_only/batch_sweep_16k/plots/batch_sweep_16k_memory.png)、[协议与验证](starkv_v_only/batch_sweep_16k/README.md)。

## 可比性与原始数据

- ShadowKV 使用 Llama-3.1-8B-Instruct 与 pinned CPU V；STAR-KV 网格使用 Qwen3-8B-Base 且 V 常驻 GPU。**不能直接比较两组的秒数、显存或 OOM 阈值。**
- ShadowKV 上述峰值是 PyTorch `allocated`，STAR-KV 表格是 NVML 进程显存；它们不是同一种显存口径。
- ShadowKV 请求表有三个 cohort，STAR-KV 两个显存网格各只有一个 cohort。所有数字都是所列配置的实测结果，没有跨配置统计外推。
- [HF 固定 revision](https://huggingface.co/alexz949/BasisServe-CALS/tree/26b66c9ad858cda646735ef28e1a6b6197e580fd/results/system_benchmarks/tp8_external_baselines) 保存全部 3337 个原始文件（rank JSON、日志、prompt、smoke、冻结源码）。[GitHub 发布提交](https://github.com/alexz949/BasisServe-CALS/commit/988a16369eea9c7122064983a64c55bd9e13b864)保存实现、测试和紧凑汇总。

## 正式运行命令

以下命令在 `/workspace/BasisServe-CALS-opt` 执行，环境均为 `basis`；各实验的汇总命令和失败日志见对应 README。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_llama31_8b_tp8_request_grid.py \
  --contexts 65536 130048 --batches 1 4 8 --cohorts 0 1 2 \
  --arms dense basis_joint shadowkv --modes request steady \
  --output-root results/system_benchmarks/tp8_external_baselines/shadowkv \
  > results/system_benchmarks/tp8_external_baselines/shadowkv/formal_grid.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_qwen3_8b_tp8_v_only_memory_grid.py \
  --contexts 4096 16384 32768 65536 98304 130048 \
  --batches 1 4 8 --cohorts 0 \
  --arms basis_v64 star_v_adaptive \
  --output-root results/system_benchmarks/tp8_external_baselines/starkv_v_only \
  > results/system_benchmarks/tp8_external_baselines/starkv_v_only/formal_grid.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_qwen3_8b_tp8_v_only_batch_sweep.py \
  --output-root results/system_benchmarks/tp8_external_baselines/starkv_v_only/batch_sweep_16k \
  > results/system_benchmarks/tp8_external_baselines/starkv_v_only/batch_sweep_16k/formal_grid.log 2>&1
```
