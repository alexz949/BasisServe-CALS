# Nemotron-H-8B-Reasoning-128K · C1 V96 + Mamba Wo 75% · RULER 128K × 1100 · 六臂配对

模型 `nvidia/Nemotron-H-8B-Reasoning-128K`（52 层 = 24 Mamba2 + 24 MLP + 4 attention，attention 层 7/18/29/40，32q/8kv/128，无 RoPE），reasoning 关闭（system 消息 `{'reasoning': False}`，模板输出空的 `<think></think>`），chat 型 prompt 排版（user 轮只放 RULER 输入、生成头后补空行；把 completion 风格的 answer prefix 塞进 user 轮会让 reasoning 模型开始"思考"或答错）。1100 条冻结 prompt = 11 任务 × 100（RULER 官方生成器，seed 42，margin 128，Nemotron tokenizer），贪心、官方生成上限、BF16、单卡分片；每臂 1100 条预测均通过评估器审计。

## 运行时前提（本机没有 mamba_ssm）

- transformers 5.17 原生 `NemotronHForCausalLM`；Mamba2 chunk scan 用 `evaluation/nemotron_h_triton_mamba.py` 接入 vLLM 0.29 的 Triton `mamba_chunk_scan_combined_varlen`（conv1d 与逐 token 递推保留 torch 路径）。与 torch 参考、逐 token fp32 朴素递推在 3K–15K 上一致（rel 2e-3）；128K 单卡 prefill 约 10 s（`logits_to_keep=1`）。
- **transformers 5.17 的 bug**：`NemotronHMamba2Mixer.time_step_limit = (config.time_step_min, inf)` = (0.001, inf)，config.json 里的 `time_step_limit=[0, inf]` 被忽略；dt 小的头正是长记忆头，钳位后 8K 以外的信息全部丢失（dense 模型 128K 上 niah 全错，vLLM 正确）。`restore_dt_limit()` 把每个 mixer 复位为 (0, inf)，捕获、拟合、评估全部经过它，协议里记录 `dt_limit`。

## 校准与拟合

- 校准窗口：16 × 128K C4（seed 20260921）+ 16 × 128K 合成长程检索（seed 20260924 / haystack 20260925，8 multikey / 4 multivalue / 2 tracking / 2 aggregation，8 层 strata，96 问答/窗口）+ 16 × C4 验证窗口；Nemotron tokenizer。
- 128K 协方差（attention `o_proj` 输入、Mamba `out_proj` 输入）单卡一趟捕获（`capture_nemotron_h_covariances_128k.py`）。
- attention **uniform V96**：ALS 6 / CG 16 / damping 1e-7 / fp32 work / bf16 factors；heldout rel-MSE L7 0.017 / L18 0.031 / L29 0.049 / L40 0.037。
- Mamba Wo：TP4 source 布局，source_rank 1536 = 75% 保留（与已发表 8B/2048 协议同规则），24 层 heldout rel-MSE 均值 0.0123；运行时折叠为 dense 权重（质量等价，不跑真实 all-gather）。
- 路由：在 **V96 + Wo 部署模型**上回放，32 个拟合窗口、64 个拟合 query、无诊断窗口，Page-Fisher 残差，ALS 40 / PCG 100，page 32，sink 32 + recent 64 计入硬 B2048。总 rank 32 的三种分配（复用同一份 moments）：

| 分配 | L7 | L18 | L29 | L40 | 残差 NMSE（in-sample） |
|---|---|---|---|---|---|
| B16R16 | 0.096 | 0.053 | 0.058 | 0.057 | 固定 16/16 |
| **B8R24** | 0.062 | 0.041 | 0.039 | 0.031 | 采用 |
| B0R32 | 0.042 | 0.041 | 0.031 | 0.022 | 仅消融 |

- Loki：dense 模型 raw K（无 RoPE）的中心化 PCA32，32 × 128K 窗口，平均保留能量 0.733；运行时投影不减均值，top-856/query head，recent 0。
- LRQK：rank 32，top-k 按真实 128K prompt 扫描定为 832（并集 2032 = 0.992×2048），recent 64；全量实测每 KV group 物理 union **2015**（0.984×2048）。ShadowKV：rank 160，chunk 8，routed 2048 + 48 outlier chunks。Loki 实测物理 token 1661/KV group。

## 结果（1100 题，同批配对）

| 任务 | Full-K | B16R16 | **B8R24** | LRQK | ShadowKV | Loki |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_single_2 | 100.0 | 100.0 | 100.0 | 100.0 | 95.0 | 100.0 |
| niah_single_3 | 89.0 | 91.0 | 91.0 | 84.0 | 84.0 | 88.0 |
| niah_multikey_1 | 77.0 | 81.0 | 81.0 | 82.0 | 73.0 | 76.0 |
| niah_multikey_2 | 69.0 | 30.0 | 63.0 | 67.0 | 75.0 | 59.0 |
| niah_multiquery | 95.5 | 92.8 | 93.5 | 89.8 | 76.5 | 90.8 |
| niah_multivalue | 92.2 | 94.8 | 96.2 | 85.5 | 77.5 | 89.2 |
| vt | 18.0 | 17.6 | 17.6 | 17.4 | 14.8 | 17.6 |
| fwe | 84.7 | 99.0 | 98.7 | 89.3 | 94.3 | 93.3 |
| qa_1 | 51.0 | 54.0 | 55.0 | 55.0 | 47.0 | 55.0 |
| qa_2 | 44.0 | 41.0 | 39.0 | 41.0 | 39.0 | 38.0 |
| **RULER 均分** | 74.58 | 72.83 | **75.91** | 73.73 | 70.56 | 73.36 |

配对 bootstrap（10000 次）：

| 比较 | 均分差 | 95% CI | 胜/负 |
|---|---:|---|---|
| B8R24 − Full-K | +1.33 | [+0.00, +2.64] | 86/49 |
| B8R24 − LRQK | +2.18 | [+0.81, +3.57] | 117/41 |
| B8R24 − Loki | +2.55 | [+1.37, +3.75] | 85/24 |
| B8R24 − ShadowKV | +5.35 | [+3.67, +7.02] | 172/36 |
| B8R24 − B16R16 | +3.08 | [+1.78, +4.46] | 64/28 |
| B16R16 − Full-K | −1.76 | [−3.48, −0.09] | 87/85 |
| LRQK − Full-K | −0.86 | [−2.09, +0.36] | 60/98 |
| Loki − Full-K | −1.23 | [−2.44, +0.00] | 54/79 |
| ShadowKV − Full-K | −4.03 | [−5.68, −2.31] | 66/160 |

- **B8R24 是六臂第一**，高于 Full-K 1.3 分（fwe +14、multivalue +4、mk1 +4、single_3 +2；mk2 −6）。B16R16 与 B8R24 在 10 个任务上逐题几乎相同，差别全部来自 multikey_2（30 → 63）。
- Loki 在无 RoPE 的 Nemotron 上很强（73.36）：raw K 的 PCA 在没有旋转的坐标里保真；同一方法在 Llama 上受 RoPE 影响只有 50。
- vt 各臂 15–18：模型本身弱，与路由无关。

## multikey_2 的机制诊断（`evaluation/diagnose_conjunctive_pages.py`）

100 条 mk2，在生成第一个答案数字的那一步，逐层记录针行所在页（平均 1.7 页）在各路由器分数下的名次与是否选中：

| Router | ≥1 必需页命中（任一后层） | 全部命中（任一后层） | 三个后层同时全命中 | 正确率 |
|---|---|---|---|---|
| exact-K oracle（同 61 页预算） | 0.94 | 0.81 | 54 题 | 68% |
| B16R16 | 0.85 | 0.74 | 30 题 | 27% |
| LRQK（token 级） | 1.00 | 0.88 | 66 题 | 59% |
| B16R16 + 针页强制并入每层每步 | 1.00 | 1.00 | 100 题 | **70%** |

逐层"全部命中"B16R16 ≈ 0.5，exact ≈ 0.7，LRQK ≈ 0.77；漏掉的页名次中位数 260–310 / 4096。跨页（key 页 vs 数字页）不是原因（"只有 key 页"2/70）。三个后层同时命中时各路由都 ≥ 85% 正确；把针页强制并入后 B16R16 追平 Full-K——损失只来自"针页没有在需要的层/步被选中"。Llama 上同一诊断形状相同（层 13–27，强制并入 66 → 81，Full-K 83）。结论：离线低秩代理的逐层针页名次噪声，在"检索必须在每个 attention 层同时成功"的合取下被放大；这是集中于 near-duplicate 候选的 failure mode，不是路由设计、V96 或模型的问题。在 Nemotron 上把 8 维 Base 容量挪给残差即可消除大部分（B8R24），B0R32 在 4 任务子集上 mk2 = 69（= Full-K）、哨兵不变，但为保持与 Llama（Base 有用）一致的方法形式未采用。

## 预算与运行时（每题中位 / 峰值显存）

B8R24 11.3 s / 38.7 GiB；B16R16 11.8 / 38.6；Full-K 12.9 / 37.5；LRQK 11.7 / 38.5；ShadowKV 11.6 / 38.0；Loki 11.1 / 37.7。可部署 sidecar 存储 = 残差维数：B16R16 16、B8R24 24（128K 时 48 MiB/层，共 4 层）；LRQK 每 query head 全 token rank-32 K 码 256 MiB/层。

## 产物与脚本

本机 `/home/Ubuntu/nh8b_128k/`：`cov/`、`vfit/`、`ckpt_v96/` + `identity/v96.json`、`wo/` + `manifests/wo_audit.json`、`router/`（B16R16）、`router_b8r24/`、`router_b0r32/`、`loki/pca`、`ruler1100/`、`eval1100/`（五臂）、`eval1100_b8r24/`、`eval_ablation/`、`diag/`。脚本：`capture_nemotron_h_covariances_128k.py`、`package_c1_uniform_checkpoint.py`、`fit_k_routing_streaming.py`（nemotron_h + `--teacher deployed`）、`fit_nemotron_h_128k_loki.py`、`sweep_lrqk_topk.py`、`eval_k_routing_ruler.py`（Loki 臂、`--lrqk-topk`、`--prompt-layout`）、`diagnose_conjunctive_pages.py` / `summarize_conjunctive_pages.py` / `analyze_conjunctive_pages.py`、`compare_rank_ablation.py`、`compare_nemotron_arms.py`。
