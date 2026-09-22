# Llama-3.1-8B-Instruct · V96（C4+合成检索混合校准）· RULER 128K × 1100 · 五臂配对

所有臂共用同一个 V96 checkpoint 与同一批 1100 条冻结 prompt（11 任务 × 100，RULER 官方生成器 commit c3f5e3b，seed 42，margin 128，Llama chat 模板 + answer prefix），贪心解码、官方生成上限、BF16、单卡分片；5500 条预测全部通过评估器审计（协议哈希、输入 sha256、逐条重算分数）。

## 结果

| 任务 | **B16R16** | Full-K | LRQK | ShadowKV | Loki |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | 99.00 | 79.00 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | 99.00 | 96.00 |
| niah_single_3 | 97.00 | 100.00 | 100.00 | 98.00 | 0.00 |
| niah_multikey_1 | 100.00 | 99.00 | 99.00 | 98.00 | 97.00 |
| niah_multikey_2 | 66.00 | 83.00 | 76.00 | 61.00 | 18.00 |
| niah_multiquery | 98.75 | 99.00 | 99.00 | 97.00 | 55.50 |
| niah_multivalue | 93.25 | 95.00 | 93.25 | 91.75 | 53.25 |
| vt | 66.80 | 76.80 | 53.60 | 28.40 | 25.00 |
| fwe | 60.33 | 52.67 | 48.67 | 58.00 | 48.33 |
| qa_1 | 74.00 | 74.00 | 75.00 | 74.00 | 48.00 |
| qa_2 | 46.00 | 48.00 | 47.00 | 48.00 | 32.00 |
| **RULER 均分** | **82.01** | **84.32** | 81.05 | 77.47 | 50.19 |

配对 bootstrap（10000 次，同题配对）：

| 比较 | 均分差 | 95% CI | 胜/负 |
|---|---:|---|---|
| B16R16 − Full-K | −2.30 | [−3.62, −0.98] | 71/105 |
| B16R16 − LRQK | +0.97 | [−0.13, +2.06] | 100/52 |
| B16R16 − ShadowKV | +4.54 | [+3.18, +5.96] | 142/45 |
| B16R16 − Loki | +31.8 | — | 792/5 |
| LRQK − Full-K | −3.27 | [−4.40, −2.19] | 40/117 |
| ShadowKV − Full-K | −6.85 | [−8.42, −5.35] | 43/156 |

排名 Full-K > **B16R16** > LRQK > ShadowKV > Loki，与仓库正式的纯 C4 校准表（`results/l31-v96-ruler128k`，84.45 / 80.88 / 80.41 / 77.97 / 50.85）一致；本批 B16R16 对 LRQK 的领先由 +0.47 扩大到 +0.97，对 ShadowKV 由 +2.91 扩大到 +4.54。

## 逐任务解读

- **持平 Full-K**：四个 single/multikey_1 任务（100/100/97/100）、multiquery（−0.25）、qa_1（0）、qa_2（−2）。
- **优势任务**：vt（+13.2 vs LRQK，+38.4 vs ShadowKV）和 fwe（+11.7 vs LRQK，+2.3 vs ShadowKV，**高于 Full-K 7.7**）。两者都不是"找一根针"，而是要覆盖分散在全文的多跳赋值链 / 词频；page 级预算把 61 页铺开，比逐 query-head 的 token 级 top-k（LRQK）和 chunk-8（ShadowKV）覆盖得更全。均分领先 LRQK 的 0.97 分几乎全部来自这两项（+24.9）。
- **短板任务**：niah_multikey_2（66 vs Full-K 83 / LRQK 76）。该任务 128K 全部是同模板行 "One of the special magic numbers for `<adj>-<noun>` is: `<7位数>`."（约 7300 行，一页 32 token 不到 2 行），十几个共享前缀/后缀的干扰 key；page 级分数里 95% 的 token 相同，判别只靠 2–3 个词 token。离线在 32 个校准窗口上拟合的 16 维残差码抓不住这种 prompt 专属的词级方向，而 exact-K page 路由（64K 表：87.5）和 LRQK 的在线拟合（76）可以。B16R16 − Full-K 在 mk2 上是 −17，与正式表的 −16 一致：混合校准没有改变这项损失。
- **与正式纯 C4 表对比**：B16R16 − Full-K 的逐项差距几乎不变（mk2 −17 vs −16，vt −10 vs −10，multivalue −1.8 vs −1.2），本批题在 multivalue/vt 上更容易（Full-K 95.0/76.8 vs 89.8/66.8）、在 mk2 上更难（83 vs 91），fwe 反转（Full-K 52.7 vs 65.7）。混合校准的收益体现在 V96 本身（Full-K 这条线，见 `l31-v96-retrievalmix-ruler220`），而不是路由相对 Full-K 的损失曲线。

## 预算与运行时

| 臂 | 注意力预算 | 额外常驻状态 | 每题中位耗时 | 峰值显存 |
|---|---|---|---|---:|
| B16R16 | 硬 2048 token/KV group（sink 32 + recent 64 + 61 页×32） | 每 KV head 每 token 16 维残差码（≈32 MiB/层） | 33.6 s | 34.9 GiB |
| Full-K | 全部 | — | 33.6 s | 33.7 GiB |
| LRQK | top-832/query head + recent 64 → 实测并集 **2107**/KV group（1.03×2048，逐层 1826–2496） | 每 query head 全部 token 的 rank-32 K 码（256 MiB/层，≈8 GiB） | 35.1 s | 50.5 GiB |
| ShadowKV | rank 160，chunk 8，routed 2048 + 48 outlier chunk | 在线 SVD 状态 | 35.8 s | 40.0 GiB |
| Loki | PCA32，top-856/query head，recent 0 | 每 token 32 维 K 码 | 31.9 s | 37.7 GiB |

Prefill（128K）占每题耗时的绝大部分；8 卡 SM 利用率 96–99%，无双开空间。

## Loki 的两个版本

正式协议的 Loki PCA 是在 **dense 模型的 pre-RoPE K** 上做中心化 PCA（运行时投影 post-RoPE Q/K、不减均值），32 × 128K C4 拟合窗口，平均保留能量 0.758 → 均分 50.19（正式表 50.85）。本次最初用 post-RoPE K 拟合（保留能量 0.686）得到 18.51，作为消融保留：Llama 的 RoPE 让 post-RoPE 基在长距离上失去判别力。两套 bank 都在 HF。

## 校准与拟合

- V96：uniform rank 96，ALS 6 / CG 16 / damping 1e-7 / fp32 work / bf16 factors，校准 = 16 × 128K C4（seed 20260921）+ 16 × 128K 合成长程检索（seed 20260922 / haystack 20260923）+ 16 × C4 验证窗口。
- B16R16 路由：在 **V96 部署模型**上回放（压缩 V + 解码器已装入；dense v_proj 仅用于 Base 矩），32 个拟合窗口、64 个拟合 query（4 分层 × 16）、无诊断窗口，Page-Fisher 残差，ALS 40 / PCG 100，page 32，sink 32 + recent 64 计入 B2048。层 NMSE 均值 0.0425 / 中位 0.0228 / 最大 0.197（in-sample）。
- 评估器：`evaluation/eval_llama_cal128.py`（B16R16 与 Full-K；修正了此前 `FORMAL_ARMS=('full',)` 跳过 B16R16 的问题，并接受部署 teacher 的 v2 协议）与 `evaluation/eval_k_routing_ruler_v96.py`（LRQK / ShadowKV / Loki，读取同一份冻结 prompt）。运行时修复：`llama_b16r16_k_offload.py` 读取 `install()` 实际提供的路由因子键名（f9c43b0）。

## 产物

HF `alexz949/BasisServe-CALS`：
- `checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/`（V96 32 层 + fit_results + manifest）、`…/router_b16r16/`（32 层 bank，每层 json 内嵌协议）、`…/loki_pca32_dense_prerope/`（正式 Loki）、`…/loki_pca32_dense/`（post-RoPE 消融）
- `calibration/llama31-8b-instruct-128k/{c4-48x128k,retrieval-16x128k,c4-retrieval-50-50}/` + README
- `evaluation/llama31-8b-instruct-ruler128k-1100-seed42/{prompts.json,prompts.safetensors}`

本机：`/home/Ubuntu/l31_router_fit/{eval1100,eval1100_baselines,eval1100_loki_prerope}` 逐条记录（含协议、输入 sha256、路由统计、耗时、峰值显存），`eval1100_summary.txt` 为配对汇总输出。
