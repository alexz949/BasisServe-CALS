# Qwen3-8B：Residual 双侧 KL Rank Allocation 与 Attention Mass Recall 总结

日期：2026-09-04。本文汇总已经完成的实验，不新增实验、重新拟合或调整 schedule。层编号均从 0 开始；pp 表示百分点。

## 1. 结果概览

在固定 C1-V80、Base16、Page32 和 B2048 的情况下，使用 C4 64×32768 窗口的双侧 terminal KL probes，为 36 层分配 residual rank，保持平均 R8。

分配结果为：16 层 R4、12 层 R8、8 层 R16，总 layer rank 288。

| 对比指标 | Uniform R8 | Adaptive，平均 R8 | 变化 |
|---|---:|---:|---:|
| Confirmation terminal KL | 0.01884865 | 0.01242355 | -34.09% |
| Confirmation suffix PPL | 6.77600395 | 6.72728054 | -0.719% |
| Shared-teacher 平均 mass recall | 86.151771% | 86.234851% | +0.083080 pp |
| Shared-teacher 平均 non-sink recall | 83.567803% | 83.684502% | +0.116699 pp |
| Shared-teacher 总 mass P01 | 28.393322% | 27.299566% | -1.093756 pp |

Terminal KL 与 recall 是两个不同测量：前者运行各自的 sparse student suffix；后者在相同 full-attention teacher Q/K/latent 上比较 selector，避免不同 hidden states 干扰局部对照。

同预算下，平均 recall 的收益较小，低分位没有全面改善。Terminal KL 的平均改善又主要来自一个窗口，不能把 34.09% 的相对降幅解读成 routing 能力普遍大幅提升。

## 2. 固定模型与数据设置

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-8B-Base，BF16 |
| 层数 | 36 |
| Attention heads | 32 Q heads / 8 KV groups，每组共享 4 个 Q heads |
| Head dimension | 128 |
| Value payload | 冻结的 uniform C1-V80 |
| Base | 冻结的 per-group pre-RoPE affine Base16 |
| Residual 候选 ranks | R4、R8、R16 |
| 分配粒度 | 每层一个 rank，该层 8 个 KV groups 使用相同 rank |
| Page size | 32 tokens |
| Exact-K budget | B2048，即每个 query / KV group 选 64 个物理 pages |
| Pinned prefix | 固定 page0；它占用上述 64-page budget 中的 1 页 |
| KL profile | C4 64×32768，window indices 0–63 |
| Confirmation | C4 16×32768，window indices 64–79 |
| 每窗公共 prefix | 前 32640 tokens，full-attention C1 |
| 被测 suffix | 最后 128 tokens，causal block teacher forcing |
| Query block size | 8 |
| 正式 GPU / 环境 | L40S / `basis`，PyTorch 2.6.0+cu124 |

数据来源为 `qwen3_8b_c4_64f16h_s32768/windows.safetensors`。每个 32768-token 窗口由八个 4096-token C4 source windows 拼接而成，没有插入额外分隔符；不是原生 32K 文档。

Profile 窗口与 residual fitting 数据重叠。Confirmation 窗口没有参与 terminal rank allocation，但此前用于 factor diagnostics；它们不是从未查看过的最终测试集。后续 mass recall 使用同一批 confirmation 窗口，属于 allocation 后的诊断。

本轮没有修改 C1-V80 或 Base16，没有跨 KV groups 混合，也没有改变 attention budget。

## 3. 当前 Base + Residual 路由算法

设 resident C1 latent 为行向量 \(c_{t,g}\)，冻结的 Base16 产生 pre-RoPE Key 预测：

\[
\widehat k^{\mathrm{pre}}_{t,g}=(c_{t,g}L_g)D_g+b_g,
\qquad
\widehat k^{\mathrm{base}}_{t,g}
=\operatorname{RoPE}_t(\widehat k^{\mathrm{pre}}_{t,g}).
\]

Residual 编码的是真实 post-RoPE Key 与 Base 预测之差：

\[
\epsilon_{t,g}=k^{\mathrm{post}}_{t,g}-\widehat k^{\mathrm{base}}_{t,g},
\qquad z_{t,g}^{(r)}=\epsilon_{t,g}E_g^{(r)}.
\]

对属于 KV group \(g\) 的 Q head \(h\)，proxy score 为：

\[
\widehat s_{h,t}
=\frac{
q_h(\widehat k^{\mathrm{base}}_{t,g})^\top
+(q_hU_h^{(r)})(z_{t,g}^{(r)})^\top
}{\sqrt d}.
\]

后续处理为：proxy token scores → Page32 log-sum-exp → 每个 Q head 的 non-sink page mass 归一化 → 四个 Q heads 组内取 max → 固定预算选页并包含 page0 → selected exact-QK attention → resident C1-V80 payload。

这里是通过低维 proxy 选取 exact K，并非用低秩 K 直接替代最终 attention 的 exact K。

Residual bank 的 fitting 使用 Page32 non-sink Fisher BCD、40 sweeps、每个 fitting 文档的最后一个 query。Layers 0/13/33 复用匹配的已有 factors，其余 33 层完成拟合。总计 36 个 layer 文件、108 个 layer/rank 组合、324 个 FP32 tensors；全部 Base16 factors 与冻结来源逐项一致。

Fisher 用于 residual factors 的局部拟合；下面的 rank allocator 使用实测 terminal KL，不是把 Fisher 数值直接当作 layer cost。

## 4. 双侧 KL 分配的实际实现

Anchor 为所有层 uniform R8，记为 \(M_8\)。Teacher 为相同 C1-V80、full exact-K attention 的模型，而不是原始 dense-V128 模型。

对每个 profile 窗口 \(w\)，在最后 128 个位置计算 full-vocabulary teacher KL，记为 \(\mathcal K_w(M)\)。对每层分别执行 R4、R16 两个 intervention，其他层仍为 R8：

\[
C_\ell(r)=\frac1{64}\sum_{w=0}^{63}
\left[\mathcal K_w(M_{8;\ell\leftarrow r})-\mathcal K_w(M_8)\right],
\quad r\in\{4,16\},
\qquad C_\ell(8)=0.
\]

之后执行 exact-budget DP：

\[
\min_{r_\ell\in\{4,8,16\}}
\sum_{\ell=0}^{35}C_\ell(r_\ell),
\qquad
\sum_{\ell=0}^{35}r_\ell=288.
\]

这是双侧 probe + additive layer cost + DP 的结构。由于本轮只有 R4/R8/R16 三个候选点，实际直接使用各点的实测 signed KL cost，没有 local MSE 曲线插值、外推、alpha exponent 或负斜率裁剪。

每窗包含 36×2=72 个 intervention configurations，64 窗共 4608 个 interventions。它们并非各自重跑完整 32K prefix：prefix 复用，intervention 只重算被修改层及其下游层的 128-token suffix。整个 terminal 测量无需 backward。

DP 对实测 additive cost table 的最优性已独立复算；这不意味着它对真实 joint terminal KL 也全局最优。Schedule 在 confirmation 前冻结，之后未调整。

## 5. 完整 rank 分布

| Residual rank | Layers | 层数 |
|---|---|---:|
| R4 | 1、3、4、6、8、10、12、14、15、16、19、21、25、27、28、35 | 16 |
| R8 | 0、2、5、11、13、17、18、22、23、26、30、31 | 12 |
| R16 | 7、9、20、24、29、32、33、34 | 8 |

\[
16\times4+12\times8+8\times16=288=36\times8.
\]

R16 较多出现在后段，但分配不随层深单调增加：layer35 是 R4，layer7/9 则是 R16。没有预设这样的深度规律。

## 6. Terminal KL / Suffix NLL / Suffix PPL

### 6.1 Profile，64 个窗口

| 配置 | Mean teacher KL | Suffix NLL | Suffix PPL |
|---|---:|---:|---:|
| C1-V80 + full exact-K teacher | 0 | 1.85957542 | 6.42100994 |
| Uniform R8 | 0.01373872 | 1.86858038 | 6.47909201 |

DP 预测的 additive profile delta KL 为 **-0.0033297070**。没有另行测量 adaptive joint schedule 在这 64 个 profile 窗口上的真实 KL；该预测不能当作 joint 实测结果。

### 6.2 Confirmation，16 个窗口

| 配置 | Mean teacher KL | Suffix NLL | Suffix PPL |
|---|---:|---:|---:|
| C1-V80 + full exact-K teacher | 0 | 1.90025533 | 6.68760178 |
| Uniform R8 | 0.01884865 | 1.91338754 | 6.77600395 |
| Adaptive，平均 R8 | 0.01242355 | 1.90617098 | 6.72728054 |

Adaptive 相对 uniform：

- Mean KL：-0.00642510，即相对降低 34.09%。
- Mean NLL：-0.00721656。
- Suffix PPL：-0.04872341，即相对降低 0.719%。
- KL 改善 11/16 个窗口；NLL 改善 10/16 个窗口。
- Paired-window delta KL 的 standard error 为 0.00542688。

Window64 的 KL 从 0.10125574 降到 0.01382737，贡献了 **85.05% 的总净 KL 改善**。仅作为事后描述，去掉该窗后其余 15 窗的平均 delta KL 仍为 -0.00102488，但明显更小。主结果保留全部 16 窗，没有据此重新分配 rank。

每窗 KL 使用 128 个位置，NLL 使用具有下一 token 标签的 127 个位置。Confirmation 总计 2048 个 KL readout positions、2032 个 NLL labels；PPL 为 pooled mean NLL 的指数。这不是 full-corpus PPL，也不是整个 32K 序列都采用 sparse attention 的测试。

## 7. Attention Mass Recall：定义与测量口径

对 full-attention teacher 的相同 Q/K，定义真实 attention 概率：

\[
p_{h,i}=\operatorname{softmax}_i\left(q_hk_i^\top/\sqrt d\right),
\]

softmax 只覆盖 causally valid tokens。令 \(\mathcal S\) 为选中 pages 对应的有效 token 集合，\(\mathcal P_0\) 为 pinned page0：

\[
\mathrm{MassRecall}_h=\sum_{i\in\mathcal S}p_{h,i},
\qquad
\mathrm{NonSinkRecall}_h=
\frac{\sum_{i\in\mathcal S\setminus\mathcal P_0}p_{h,i}}
{\sum_{i\notin\mathcal P_0}p_{h,i}}.
\]

参考概率使用 cached BF16 Q/K 的 FP32 exact-QK 与 softmax。Non-sink 分布单独归一化以避免 sink mass 接近 1 时的消减误差。没有 non-sink support 的 query 会被排除，而不是计为零；实际全部 query 都有有效 non-sink support。

选页复用当前 native BF16 proxy contractions 和物理 GQA page selector，不使用 sparse attention 重新归一化后的概率作为 recall 参考。

总计每个 arm 有 16×36×32×128=**2,359,296** 个 layer/head/query observations。P01、P10 和 median 是 pooled observation 分位数，不是同等数量的独立文档样本。保持 R8 的 12 层在两种配置中复用同一份结果，其一致性是结构性的，不是独立重复实验。

## 8. Recall 总体结果

| 指标 | Uniform R8 | Adaptive | Delta，pp |
|---|---:|---:|---:|
| 平均总 mass recall | 86.151771% | 86.234851% | +0.083080 |
| 平均 non-sink recall | 83.567803% | 83.684502% | +0.116699 |
| 总 mass P01 | 28.393322% | 27.299566% | -1.093756 |
| 总 mass P10 | 63.811126% | 63.838464% | +0.027338 |
| Non-sink P01 | 18.610248% | 18.795598% | +0.185350 |
| Non-sink P10 | 58.963907% | 58.945721% | -0.018186 |

总 mass 的窗口均值在 13/16 窗改善，non-sink 在 14/16 窗改善。两者 paired-window mean delta 的 standard error 分别为 0.023122 pp 和 0.025003 pp。

Window64 占总净 mass 改善的 17.59%、non-sink 改善的 14.94%。事后描述性地去掉该窗后，其余 15 窗仍分别改善 0.073031 pp、0.105879 pp。因此 recall 的小幅平均收益并不像 terminal KL 那样主要集中于单个窗口。

## 9. R8 → R16 的变化究竟有多大

以下只统计 allocator 选中的 8 个 R16 层，不是全 36 层 uniform R16 的结果。

| 指标 | 这些层使用 R8 | 这些层使用 R16 | Delta，pp |
|---|---:|---:|---:|
| 平均总 mass recall | 87.221332% | 88.794078% | +1.572747 |
| 平均 non-sink recall | 82.480991% | 84.350037% | +1.869046 |
| 总 mass P01 | 39.128043% | 42.660198% | +3.532155 |
| Non-sink P01 | 11.298104% | 15.621477% | +4.323373 |

平均漏掉的总 attention mass 从 12.78% 降至 11.21%，相对减少约 12.3%。Residual 维度翻倍带来可见改善，但 recall 并未接近完全保留；维度翻倍也不等于总 KV 内存翻倍。

| Layer | R8 总 mass recall | R16 总 mass recall | Delta，pp |
|---|---:|---:|---:|
| 7 | 90.066% | 90.701% | +0.635 |
| 9 | 91.722% | 92.382% | +0.660 |
| 20 | 87.848% | 89.966% | +2.117 |
| 24 | 85.804% | 88.107% | +2.303 |
| 29 | 84.911% | 88.236% | +3.325 |
| 32 | 85.314% | 85.822% | +0.508 |
| 33 | 85.940% | 87.646% | +1.706 |
| 34 | 86.165% | 87.493% | +1.328 |

这些局部收益同时伴随降 rank 层的损失：

| Rank 变化组 | 层数 | 平均总 mass delta，pp | 平均 non-sink delta，pp |
|---|---:|---:|---:|
| R8 → R4 | 16 | -0.599444 | -0.671951 |
| R8 保持不变 | 12 | 0 | 0 |
| R8 → R16 | 8 | +1.572747 | +1.869046 |

按层数加权后，总 mass 的净收益仅为约 0.083 pp。局部 R8→R16 的改善与同预算 adaptive 的整体收益是两个不同问题。

## 10. 验证、硬件差异与运行记录

### 验证

- 6 项原 KL/replay CPU tests 与 5 项新 recall CPU tests 全部通过；未安装或修改环境依赖。
- Factor bank 的层覆盖、形状、finite values、文件 hashes 与冻结 Base16 一致性通过核对。
- KL profile costs 由原始 64 窗数据独立复算，另一个 DP 实现复现相同最优 cost。
- 所有 confirmation 文件的 frozen schedule hash 一致；NLL/KL/PPL 由逐窗结果复算一致。
- Recall GPU smoke 检查 layer0 R8、layer1 R8/R4、layer33 R8/R16：sidecar、page IDs、validity masks 与实际 native attention forward bitwise 一致。
- Teacher capture hooks 不改变 suffix hidden states；所有 16 个 recall 窗口精确复现此前 teacher NLL。
- Smoke 与正式 window64 的 raw recall arrays bitwise 一致。
- 独立 NumPy 汇总复现 pooled means/quantiles、逐层均值和 paired standard errors。
- 概率恒等式 `mass = sink_mass + (1 - sink_mass) × non_sink_recall` 的最大绝对误差为 7.1526e-7。
- 正式 GPU jobs 均成功，未出现 NaN、OOM 或失败检查。

### 硬件差异

早期相同输入在 A100 和 L40S 上产生不同的 terminal KL probe deltas，部分符号也不同；但各自设备内的 cached replay 正确性检查通过。该跨设备差异的底层原因未在本轮进一步确定，不能归因为已证实的 A100 cache bug。

正式 KL profile、confirmation 和 recall 均使用 L40S，不混合 A100 结果。A100 已完成的部分 profile windows 单独保留，未参与分配。

### Slurm 记录

| 阶段 | Job | Elapsed |
|---|---|---|
| Residual bank，4 shards | 8300163 | 17:49–23:17 |
| KL profile shard0 | 8300193 | 14:38 |
| KL profile shards1–3，包含 smoke | 8300185 | 15:32–15:36 |
| DP allocation | 8300194 | 00:26 |
| KL confirmation，4 shards | 8300198 | 00:52–00:53 |
| KL summary | 8300202 | 00:13 |
| Recall smoke | 8300221 | 00:26 |
| Recall evaluation，4 shards | 8300222 | 每 shard 00:48 |
| Recall summary | 8300223 | 00:17 |

上述 jobs 均完成，exit code 为 0:0。Recall smoke 核心计算 9.73 秒、峰值 allocated 显存 26.86 GiB；正式 recall 每窗计算 8.44–8.77 秒、最大 allocated 显存 23.20 GiB。这些是诊断流程时间，不是优化后的 sparse decode latency。

## 11. 结论与适用边界

1. 当前 adaptive residual rank allocation 相对 uniform R8 的平均 routing recall 收益较小，更接近小幅调优，不是 routing 能力的大幅跃升。
2. R8→R16 在被选中的 8 层上有局部收益，但需要用其他 16 层 R8→R4 的损失交换预算；低 recall 尾部没有全面改善。
3. Confirmation terminal KL 的平均降幅受单窗影响较大，本轮证据不足以宣称稳定的跨数据／任务收益。
4. Shared-teacher local recall 与不同 sparse student trajectories 上的 terminal KL 不是同一指标；不能由前者直接推出后者或任务 accuracy。
5. 这些结果不否定 Base+R8 residual 本身的价值。本轮对照回答的是 adaptive 与 uniform R8 的区别，不是 Base-only 与 Base+R8 的区别。
6. 本轮不是 full PPL、RULER 或真实 CPU offload 速度测试。Oracle 将 exact K 放在 GPU，并物化 Base128+R sidecar；不能据此报告部署形态的显存或 PCIe 性能。

## 12. 文件与复现入口

### 结果

- [冻结的 rank schedule](q8_residual_kl_64x32k/schedule.json)
- [Terminal KL / NLL / PPL 原始汇总](q8_residual_kl_64x32k/result.json)
- [Terminal KL 逐窗总结](q8_residual_kl_64x32k/summary.md)
- [Recall 原始汇总、逐层 quantiles 与 rank-group 数据](q8_residual_mass_16x32k/result.json)
- [Recall 全 36 层与逐窗结果](q8_residual_mass_16x32k/summary.md)

### 代码

- [Residual bank fitting](../../evaluation/fit_qwen3_8b_residual_kl_bank.py)
- [Terminal profile / allocate / confirm](../../evaluation/profile_qwen3_8b_residual_two_sided_kl.py)
- [Cached-prefix / layer-suffix replay](../../basisserve/core/residual_kl_replay.py)
- [Mass recall evaluator](../../evaluation/eval_qwen3_8b_residual_mass_recall.py)
- [Shared-teacher capture 与 mass recall primitives](../../basisserve/core/residual_mass_recall.py)

### 完整命令与协议

- [KL 实验设置、checkpoint 路径、运行命令和日志](../../docs/q8_residual_two_sided_kl_protocol.md)
- [Recall 实验设置、运行命令、smoke 与日志](../../docs/q8_residual_mass_recall_protocol.md)

本次仅新增这份总览，没有修改既有实验代码、factors、schedule 或原始结果，也没有进行 GitHub commit/push。
