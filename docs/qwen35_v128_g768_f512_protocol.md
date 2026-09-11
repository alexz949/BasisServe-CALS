# Two-sided V128 + GDN Wo768 / full-attention Wo512

目标：Qwen3.5-9B，关闭 thinking，全量 GSM8K 1,319 题，使用 Two-sided 平均 V rank 128 与 GDN Wo 每 source rank 768、full-attention Wo 每 source rank 512。

候选 rank 为 32、48、64、80、96、112、128、160、192、224、256。复用已有候选因子；新增 8 层 × 160/192/224，共 24 组因子，沿用 Dense capture、6 次 encoder sweeps、separable 预条件、线性求解最多 200 次、heldout 选择和 BF16 导出。有限迭代求解不等于整体 ALS 已收敛，应保留并审查残差与选择记录。

KL 锚点为 Uniform 128，每层分别探测 96 和 160，加上锚点共 17 组测量。使用已有独立 profile 数据和 Dense teacher，exponent 1.25，分配总预算 8 × 128 = 1024。候选 256 为原生 Dense 端点。各层最终 rank 由分配决定，不预先指定，也不根据 GSM8K 答案选 rank。

冻结 V 后重新采集全部 32 层 Wo 的 fit/heldout 二阶矩，再分别以 GDN768、full512 拟合。Wo 不复用 Dense V 下的因子，采用 FP64、6 sweeps、BF16 导出，校验上游 V hash、校准身份、形状、有限性和覆盖。评测为 TP1 的逻辑四 source 等价实现，不测真实分布式通信或 KV cache 压缩收益。

## 命令与资源

仓库根目录执行：

```bash
bash scripts/run_qwen35_v128_g768_f512.sh \
  >> results/q35_hybrid/logs/v128_queue.log 2>&1
```

脚本完整列出每阶段命令：

- V 拟合：`python -u -m evaluation.run_qwen35_hybrid fit --layers <分片> --ranks 160,192,224 --chunk-rows 2048 --linear-max-iter 200 --encoder-preconditioner separable --output results/q35_hybrid/factors`。
- KL：`python -u -m evaluation.qwen35_hybrid_banks profile --anchor 128 --num-shards 2 --shard-index <0/1> --output results/q35_hybrid/kl`。
- 分配：`python -u -m evaluation.qwen35_hybrid_banks assemble --anchor 128 --target-average-rank 128 --output results/q35_hybrid/banks_v128`。
- Wo capture/fit/assemble 使用 `evaluation.qwen35_hybrid_wo`，bank 为 `results/q35_hybrid/banks_v128/c1_twosided_v128.pt`，moments 为 `results/q35_hybrid/wo_v128_moments`，输出 `results/q35_hybrid/wo_v128_g768_f512`，fit 参数 `--gdn-rank 768 --full-rank 512 --work-dtype float64 --num-shards 2`。
- GSM8K：`evaluation.eval_qwen35_hybrid_gsm8k_vllm`，上述 V/Wo bank，`--wo-scope all --max-num-seqs 32 --max-num-batched-tokens 4096 --max-new-tokens 1024 --max-model-len 8192 --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909`。
- 审计：`python -u -m evaluation.summarize_qwen35_gsm8k_v96 --v128-wo --output results/q35_hybrid/gsm8k_vllm/v128_g768_f512_wo_summary.json`。

本机直接执行，无 Slurm。启动时 GPU 2 已接近满负载，因此改为 GPU 5、6 分担 V 拟合、KL 和 Wo 拟合；GPU 6 负责 Wo capture 和 GSM8K。每进程 CPU 2 线程，最多两个进程共 4 线程。V 拟合的层分片为 3,11,19,27 和 7,15,23,31。

拟合和 HF 校准使用 lowrank，`PYTHONPATH=results/q35_hybrid/deps:.`；vLLM 使用此前授权的 lowrankarena，`PYTHONPATH=.`。不修改环境软件。每阶段日志均写入 results/q35_hybrid/logs/v128_* 或 gsm8k_vllm_result_twosided128_g768_f512_wo.log，使用追加模式保留历史记录。

评测输出为 results/q35_hybrid/gsm8k_vllm/result_twosided128_g768_f512_wo.json。审计对照 Dense、Dense V + GDN768/full512 等已有完整结果，复核模型身份、样本和生成参数一致性、strict/flexible/数值等价得分、长度上限及 closing-think 异常。

## 当前状态

用户确认后，扩展候选方案已于 2026-09-10 00:00 EDT 启动，脚本 PID 2703776，初始 V 拟合进程 PID 2703787、2703788。使用工具的持续运行会话 39048；两个 nohup 启动尝试未留下运行进程，仅在总日志留下时间戳，没有启动拟合。正式会话中两个拟合进程均已输出 lowrank 环境与完整参数。当前处于 V 候选拟合阶段，尚无本配置 GSM8K 分数。

早先以 V96 锚点和旧候选集生成的 results/q35_hybrid/banks/c1_twosided_v128.pt 保留为中间产物，不用于本方案。该次 Wo capture 在加载模型前因未设置本地依赖 PYTHONPATH 而失败，未产生 Wo bank 或 GSM8K 结果；日志保留。新方案使用 banks_v128 目录避免覆盖旧 bank。

准备检查为 14 项测试通过，且本地依赖的 Qwen3_5ForCausalLM 导入通过。未提交或上传 GitHub。

## 追加 GPU 2

用户要求使用 GPU 2 后，检查该卡约 47 GiB 空闲、5% 利用率，追加独立 lowrank 拟合进程，每进程 2 个 CPU 线程：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=results/q35_hybrid/deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.run_qwen35_hybrid fit --layers 31 --ranks 224,192,160 \
  --chunk-rows 2048 --linear-max-iter 200 --encoder-preconditioner separable \
  --output results/q35_hybrid/factors >> results/q35_hybrid/logs/v128_v_fit_gpu2.log 2>&1
```

GPU 2 提前计算原 GPU 6 队列最后一层，优先完成最晚使用的 rank；原 worker 到达该层时会跳过已存在的完整文件。当前 GPU 6 仍在第 7 层 rank160，后面还需完成第 7、15、23 层其余候选。监控时须关注两个 worker 的进度，避免它们同时计算尚未完成的第 31 层同一候选。旧进程不中断，已有结果不覆盖。追加任务工具会话为 42202。

监测约三小时后，GPU 2 已完成第 31 层 rank224 和 rank192，正在拟合 rank160；GPU 5 仍在第 3 层 rank224。为继续使用 GPU 2，追加等待当前 PID 2741378 结束后拟合第 27 层的任务，会话 97634：

```bash
while kill -0 2741378 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=results/q35_hybrid/deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m evaluation.run_qwen35_hybrid fit --layers 27 --ranks 224,192,160 \
  --chunk-rows 2048 --linear-max-iter 200 --encoder-preconditioner separable \
  --output results/q35_hybrid/factors >> results/q35_hybrid/logs/v128_v_fit_gpu2_l27.log 2>&1
```

第 27 层为 GPU 5 原队列最后一层，同样由原 worker 跳过已完成文件，监测时继续避免同一候选同时开算。模型、数据、拟合参数与原方案一致。
