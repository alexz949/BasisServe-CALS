# 本机优化后 144-trial TP8 Full-Scan Decode 结果摘要

**本文件写的是第二次、优化后的 144-trial 网格**，结果目录为 `llama31_8b_tp8_full_scan/`，BasisKV 记录的路由模式是 `full_scan_b16r16_persistent_slots`。第一次 144-trial 网格保存在[旧版结果目录](../llama31_8b_tp8_combined/SUMMARY.md)，模式是 `two_stage_512_persistent_slots`。两次均在本机 8 张 NVIDIA L40S 上运行 Llama-3.1-8B-Instruct，并比较 `dense`、`als_full` 和 `basis_joint`；**均不包含 ShadowKV 方法**。ShadowKV 后续的 108-trial 请求实验是另一组结果，不能与这里的稳态 decode 延迟混为一谈。

| 两次 144-trial 网格 | 旧版两阶段路径 | 本文件：新版 full-scan 路径 |
| --- | ---: | ---: |
| 完成项 | 132/144 | 132/144 |
| BasisKV，64K/B8，每步均值 | 39.86 ms | 23.846 ms |
| BasisKV，130048/B8，每步均值 | 42.55 ms | 27.850 ms |

新版同时移除了两阶段粗筛及 512 候选页限制，改变了路由算法；它不只是同一算法的 kernel 加速。因此这两个数列只能说明两套**实际运行路径**的差别，不能作为纯 kernel 优化的独立加速比或质量等价证据。

## 实验口径

- TP8 / DP1 / PP1，BF16，`basis` conda 环境，直接在本机运行；GPU 间没有 NVLink。三个方法使用相同的模型、prompt cohort 和测量窗口。
- Prompt 长度为 4096、16384、65536、130048 token。4K 测 batch 1/8/32/128；其余长度测 batch 1/4/8/16。每个配置有三个冻结的真实文本 cohort，共 `16 x 3 x 3 = 144` 个 trial。
- 每项先 prefill、再运行 16 个 conditioning decode step，最后测量 128 个 decode step。结果是**完整模型的稳态 decode 每步延迟**，包含 MLP 和通信；不是完整请求时间、TTFT 或质量评估。
- `basis_joint` 使用 V96、全历史 B16R16/Page32 路由、62 个选中页加 recent64、2048 个 support slot。这里没有两阶段 512 候选页粗筛。Dense 和 ALS-full 将缓存放在 GPU；BasisKV 的历史 exact K 放在 pinned host memory，GPU 保留持久化 K slot。
- 下表只在相关方法的三个 cohort 全部完成时给出三-cohort 中位数。加速比为对照方法的 median mean-step 延迟除以 BasisKV 的对应值；不同实验或失败配置不外推。

## 关键结果

**132/144 trial 成功，12/144 发生 prefill GPU OOM。** 所有成功项的 8 个 TP rank 均通过结果校验，共验证 1056 个 rank 记录。完整的 48 个工作负载/方法组合见[全表](summary.csv)；下表摘录代表性配置，单位为 ms/step。

| Prompt / batch | Dense | ALS-full | BasisKV Joint | Basis vs Dense | Basis vs ALS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K / 1 | 29.399 | 20.439 | 24.205 | 1.215x | 0.844x |
| 4K / 128 | 39.326 | 33.776 | 51.364 | 0.766x | 0.658x |
| 16K / 16 | 32.096 | 27.945 | 24.517 | 1.309x | 1.140x |
| 64K / 8 | 87.412 | 75.546 | 23.846 | 3.666x | 3.168x |
| 64K / 16 | prefill OOM | 78.461 | 30.597 | - | 2.564x |
| 130048 / 1 | 160.567 | 138.067 | 23.869 | 6.727x | 5.784x |
| 130048 / 8 | 162.620 | 140.004 | 27.850 | 5.839x | 5.027x |
| 130048 / 16 | prefill OOM | prefill OOM | prefill OOM | - | - |

长上下文下，BasisKV 的全历史路由和稀疏 GPU slot 显著降低稳态 decode 时间：130048/B8 的三-cohort 中位数为 **27.850 ms/step**，对照 Dense **162.620 ms/step**、ALS-full **140.004 ms/step**。但这不是所有 batch 都更快：4K/B128 时 BasisKV **51.364 ms/step**，慢于两个对照。上述对照是本仓库匹配的 TP8 实现，不能据此声称胜过经过独立优化的 vLLM、SGLang 或 TensorRT-LLM Dense 服务。

在 130048/B8，BasisKV 的最大 rank **prefill 峰值 PyTorch allocated GPU 显存为 35.768 GiB**，decode-resident 为 **11.085 GiB**；Dense 对应为 **43.321 / 18.643 GiB**，ALS-full 为 **42.439 / 17.756 GiB**。BasisKV 同时使用八个 rank 合计 **63.570 GiB 的 host K 分配容量**；该值不是主机内存峰值 RSS，也不能从 GPU 数字中省略。

## 失败与限制

- 64K/B16 的 Dense 在三个 cohort 均于 prefill OOM：3 项。
- 130048/B16 的 Dense、ALS-full、BasisKV Joint 在三个 cohort 均于 prefill OOM：9 项。
- 这 12 项保留在[逐 trial 汇总](decode_trial_summary.csv)、`decode_grid_trials.json` 和 launcher 日志中。**Prefill OOM 不能表述为 decode 缓存容量上限**；失败配置没有加速比。
- GPU 显存指标是八个 rank 中最大的 PyTorch allocated 值；host K 是分配容量之和。容器无法执行严格 NUMA host-memory binding，但应用了 CPU affinity；NCCL barrier 有 current-device inference 警告。
- 成功 trial 的跨 rank token 和计时序列一致，不等于任务质量与 Dense 等价。实验未优化 MLP 或 Dense，亦未评估完整请求延迟。历史两阶段路由 pilot 不混入本次 full-scan 结果。

## 命令与数据

正式运行命令在 `/workspace/BasisServe-CALS-opt` 执行，conda 环境为 `basis`：

```bash
CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
/workspace/miniforge3/bin/conda run --no-capture-output -n basis \
python benchmarks/system/run_llama31_8b_tp8_decode_grid.py \
  --arms dense als_full basis_joint \
  --contexts 4096 16384 65536 130048 --cohorts 0 1 2 \
  --conditioning-steps 16 --measure-steps 128 \
  --tag full_scan_decode \
  --output-root results/system_benchmarks/llama31_8b_tp8_full_scan
```

汇总命令：`/workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/summarize_tp8_full_scan.py`。完整方法、每个配置及验证信息见[正式 SUMMARY](SUMMARY.md)和[ARTIFACTS](ARTIFACTS.md)。运行前冻结的 1629 个源码文件在结束后逐字节比对一致；无 SHA256 检查。[HF 固定 revision](https://huggingface.co/alexz949/BasisServe-CALS/tree/61eda8a41285730eb254a904813666cb38c94725/results/system_benchmarks/llama31_8b_tp8_full_scan)保存原始 rank JSON、日志、trial manifest 与源码归档。
