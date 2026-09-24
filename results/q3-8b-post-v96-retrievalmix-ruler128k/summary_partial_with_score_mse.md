# Qwen3-8B (post-trained, non-thinking) · V96 · RULER 128K × 1100 · 部分结果快照

快照时间：2026-09-23 12:04 EDT。用户在 ShadowKV 臂 589/1100 时暂停，Loki 臂未跑；Full-K / B16R16 / B8R24 / LRQK 四臂完整（1100 题）。

模型 `Qwen/Qwen3-8B`（post-trained，36 层，32q/8kv/128，原生 40960）+ YaRN ×4（original 32768 → 131072）；prompt = chat 模板 + `enable_thinking=false` + 官方 answer prefix 作为 assistant 文本（Llama-Instruct 的对应写法）；1100 条冻结 prompt = 11 任务 × 100（RULER 官方生成器，seed 42，margin 128）；贪心、官方生成上限、BF16、单卡 8 分片。V96 uniform（ALS 6 / CG 16 / 1e-7，16 C4 + 16 合成检索 + 16 验证窗口，held-out rel-MSE 0.046）；路由在 V96 部署 teacher 上拟合（32 窗、64 query、ALS 40 / PCG 100，sink 32 + recent 64 计入硬 B2048）。LRQK rank 32 top-704 + recent 64（扫描：576→1668、640→1834、704→1999、768→2163、832→2324；取最接近 2048）。Loki post-RoPE PCA32（A/B：pre-RoPE single_3 1 / mk2 8，post-RoPE 84 / 38）。评估器 `eval_k_routing_ruler.py`（qwen3 走 sink+recent 页路由、分块 MLP、零填充 FlashAttention prefill）。

## 四个完整臂（1100 题，同题配对）

| 任务 | Full-K | B16R16 | **B8R24** | LRQK |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_single_2 | 100.0 | 98.0 | 98.0 | 98.0 |
| niah_single_3 | 100.0 | 98.0 | 100.0 | 100.0 |
| niah_multikey_1 | 85.0 | 84.0 | 86.0 | 86.0 |
| niah_multikey_2 | 61.0 | 44.0 | 56.0 | 57.0 |
| niah_multiquery | 93.5 | 94.8 | 93.8 | 93.8 |
| niah_multivalue | 96.5 | 92.5 | 95.8 | 96.5 |
| vt | 93.4 | 92.0 | 92.4 | 91.2 |
| fwe | 88.0 | 88.7 | 89.3 | 81.3 |
| qa_1 | 55.0 | 49.0 | 52.0 | 51.0 |
| qa_2 | 36.0 | 31.0 | 32.0 | 36.0 |
| **均分** | 82.58 | 79.27 | 81.38 | 80.98 |

配对 bootstrap（10000 次）：

| 比较 | 差 | 95% CI | 胜/负 |
|---|---:|---|---|
| B8R24 − B16R16 | +2.12 | [+1.01, +3.35] | 63/28 |
| B8R24 − Full-K | -1.20 | [-2.50, +0.04] | 57/65 |
| B8R24 − LRQK | +0.40 | [-0.84, +1.60] | 80/48 |
| B16R16 − Full-K | -3.32 | [-4.70, -1.97] | 48/93 |
| LRQK − Full-K | -1.60 | [-2.65, -0.57] | 28/67 |

## ShadowKV 部分臂（已完成 589 题，与其他臂在同一批题上配对）

| 任务 | n | ShadowKV | B8R24 | LRQK | Full-K |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_single_2 | 100 | 95.0 | 98.0 | 98.0 | 100.0 |
| niah_single_3 | 100 | 98.0 | 100.0 | 100.0 | 100.0 |
| niah_multikey_1 | 100 | 84.0 | 86.0 | 86.0 | 85.0 |
| niah_multikey_2 | 100 | 44.0 | 56.0 | 57.0 | 61.0 |
| niah_multiquery | 89 | 83.7 | 93.8 | 94.1 | 93.5 |
| 子集均分 | 589 | 84.13 | 88.88 | 89.09 | 89.86 |

## 预算与运行时（每题中位 / 峰值显存）

- Full-K: 26.4 s / 38.3 GiB
- B16R16: 23.1 s / 48.3 GiB
- B8R24: 23.1 s / 48.8 GiB
- LRQK: 40.5 s / 53.5 GiB
- ShadowKV: 40.7 s / 43.0 GiB（部分）

## 校准侧分配规则（不碰测试集）

`calibration_routing_quality.py` 在 16 个 C4 验证窗口（拟合未用）上对 exact-K 页路由的比较（36 层均值）：exact-K mass 0.881 / recall 1.000；B16R16 0.867 / 0.751 / KL 0.285；**B8R24 0.870 / 0.775 / KL 0.217**。规则事前选 B8R24，测试集证实（B8R24 − B16R16 +2.12 [+1.01, +3.35]，mk2 44 → 56，multivalue 92.5 → 95.8）。Base 秩扫描（V96 码 → K 的去均值 K 能量解释比例）：rank 16 33%，上限 46%（post-trained 与 Base 数值接近）。

## 待完成
- ShadowKV 剩余 511 题、Loki 1100 题（续跑：`cd /home/Ubuntu/q3_8b_post_128k && LOKI_COORD=post_rope TOPK=704 nohup ./stage_eval.sh evaluate`，评估器跳过已完成样本，会先审计 full/ours/b8r24/lrqk 再接 shadowkv、loki）。

## 产物
本机 `/home/Ubuntu/q3_8b_post_128k/`：`c4_128k`、`retrieval_16x128k`、`c4_retrieval_50_50`、`ruler1100`、`cov`、`vfit`、`ckpt_v96` + `identity/v96.json`、`router/ours_b16r16`、`router_b8r24/ours_b8r24`、`loki_pre_rope`、`loki_post_rope`、`manifests/lrqk_sweep.json`、`eval_loki_ab/`、`eval1100/{full,ours,lrqk,shadowkv}`、`eval1100_b8r24/ours`；诊断在 `/home/Ubuntu/q3_8b_128k/analysis/calib_quality_qwen3_8b_post.json`、`base_rank_sweep_qwen3_8b.json`。代码在 worktree `/home/Ubuntu/q3_8b_128k/repo`（未提交）。

## 残差目标消融：Score-MSE 对 Page-Fisher（追加于 2026-09-23 16:06 EDT）

`fit_k_routing_streaming.py --objective score_mse`：残差统计量改为 Base 残差 K − Base(V) 在可路由前缀（排除 sink 页与 recent 64）上的无权重 Gram（Section 4 的 Score-MSE），同一 ALS 40 / PCG 100 求解器；moments/Base 复用。校准侧（C4 验证窗口）：Score-MSE 页召回 0.755 / 0.779 对 Page-Fisher 0.751 / 0.775（B16R16 / B8R24），mass −0.2 pt，页 KL 明显更大（0.448 / 0.351 对 0.285 / 0.217）。

| 任务 | Full-K | B16R16-PF | B16R16-score | B8R24-PF | B8R24-score | LRQK |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_single_2 | 100.0 | 98.0 | 99.0 | 98.0 | 100.0 | 98.0 |
| niah_single_3 | 100.0 | 98.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_multikey_1 | 85.0 | 84.0 | 87.0 | 86.0 | 86.0 | 86.0 |
| niah_multikey_2 | 61.0 | 44.0 | 54.0 | 56.0 | 56.0 | 57.0 |
| niah_multiquery | 93.5 | 94.8 | 95.0 | 93.8 | 93.2 | 93.8 |
| niah_multivalue | 96.5 | 92.5 | 94.0 | 95.8 | 95.0 | 96.5 |
| vt | 93.4 | 92.0 | 92.4 | 92.4 | 91.4 | 91.2 |
| fwe | 88.0 | 88.7 | 92.0 | 89.3 | 89.0 | 81.3 |
| qa_1 | 55.0 | 49.0 | 51.0 | 52.0 | 51.0 | 51.0 |
| qa_2 | 36.0 | 31.0 | 32.0 | 32.0 | 35.0 | 36.0 |
| **均分** | 82.58 | 79.27 | 81.49 | 81.38 | 81.51 | 80.98 |

| 比较 | 差 | 95% CI | 胜/负 |
|---|---:|---|---|
| B16R16-score − B16R16-PF | +2.23 | [+1.12, +3.43] | 66/26 |
| B16R16-score − B8R24-PF | +0.11 | [-0.84, +1.05] | 41/34 |
| B8R24-score − B8R24-PF | +0.13 | [-0.77, +1.03] | 30/34 |
| B8R24-score − B16R16-score | +0.02 | [-0.88, +0.94] | 30/38 |
| B16R16-score − LRQK | +0.51 | [-0.73, +1.76] | 87/49 |
| B16R16-score − Full-K | -1.09 | [-2.32, +0.15] | 59/62 |

结论：只换残差目标（不加容量），B16R16 从 79.27 升到 81.49（+2.23，CI 不含 0），追平 B8R24-PF；B8R24-score 81.51 与 B8R24-PF 持平——Score-MSE 与更大的残差是替代关系而非叠加。覆盖型任务无代价（vt +0.4、fwe +3.3）。Score bank：`router_score_b16r16`、`router_score_b8r24`；评估：`eval1100_score_*`。
