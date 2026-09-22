# Qwen3-32B V96 · RULER 128K · B16R16（TP2）：尝试与结果

2026-09-21。状态：**主评测暂停在 579/1100**；路由诊断已完成。

## 结论速览

- **B16R16 在 Qwen3-32B 上明显落后于基线。** 已完成的 579 个样本：B16R16 **0.816**，LRQK 0.959，ShadowKV 0.889。其中 `niah_multikey_2` 只有 **0.16**（LRQK 0.81，ShadowKV 0.59）。
- **页机制本身没有问题。** 在 multikey_2 的 32 个样本上，只把打分换成精确 K、其余不变（每个 KV 组共用一套页、严格 2048、page 32），得分从 0.156 升到 **0.844**，和 LRQK 的 0.875 相当。GQA 8 头共用页、严格预算都不是瓶颈。
- **问题出在 B16R16 对 K 的近似上，且属于分布偏移，不是运行时 bug。** 在拟合用的 C4 窗口上，B16R16 选页比精确选页少捕获 2%–4% 的注意力质量；到了 needle 任务，后段（L44–62）的差距翻倍到 7.4%，而且在精确路由会选中 needle 的情况下，B16R16 有 42% 漏选。
- **page-Fisher NMSE 预测不了路由质量**：逐层"精确 − B16R16"的捕获质量差距与 NMSE 的相关系数为负（C4 上 −0.68）。NMSE 最好的后段，恰恰是检索时出问题的地方。
- 工程方面：prefill attention 从仓库的 Triton kernel 换成 FlashAttention 后，B16R16 prefill 从 **74.8 s 降到 48.7 s**，生成的 token 逐个不变；1100 个样本的预计总时长从 6.75 h 降到约 4.7 h。

---

## 1. 配置

| 项 | 值 |
|---|---|
| 模型 | Qwen3-32B（64 层，64 个 query head / 8 个 KV head，GQA 8，head_dim 128），YaRN factor 4.0（原生 40960 → 163840） |
| V 压缩 | HF `alexz949/BasisServe-CALS` `checkpoints/qwen3-32b-128k/uniform-v96-als6-cg16`，每个 KV 头秩 96 |
| 路由 | Base16（V96 → 16 → 128 的仿射降秩映射，预测 pre-RoPE K）+ Page-Fisher Residual16 |
| 预算 | 每个 KV 组**硬上限 2048**：sink 页 32 + recent 64 + 61 个路由页（1952）。每个 decode 步重新路由 |
| 数据 | `datasets/qwen3-32b-ruler128k-1100`，11 个任务 × 100 个样本 |
| 并行 | TP2，4 组副本占 8 张卡，batch 1 |
| 硬件 | 8 × RTX PRO 6000 Blackwell Server（sm_120，每卡 94.97 GiB 可用），KVM 虚拟机，**GPU 之间没有 P2P** |
| 环境 | conda `lowrank`（torch 2.13 + cu130，triton 3.7.1） |

### 拟合设置

32 个 × 128K 的 C4 窗口；每个窗口选 32 个分层采样的 query 位置；ALS 40 轮，PCG 100 步，`relative_damping=1e-5`；排除 sink 页和 recent 64；query 候选网格的起点为 96（`query_grid_prefix = 32 + 64`）。
产物：`b16r16_fit/bank/qwen3_32b_uniform96_b16r16_32x128k_recent64/`（64 层，已校验），统计量 `b16r16_fit/statistics/qwen3_32b_router_32x128k_recent64/`（131 GiB，可用来免 teacher 重拟合）。

---

## 2. 过程中遇到并修复的问题

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | 8 个拟合分片全部报 `unexpected keyword argument 'excluded_recent_tokens'`，但 grep 和单独导入都显示参数存在 | pkgroot 里的 `evaluation/` 大部分是指向仓库的符号链接。`eval_qwen3_8b_v80_conditional_residual_router.py` 会按 `__file__` 把**仓库根目录**插到 `sys.path[0]`；两边的 `evaluation/` 都没有 `__init__.py`，属于 namespace package，`__path__` 会被重算，于是加载了仓库里未打补丁的那份 | 给 pkgroot 的 `evaluation/` 加上 `__init__.py`；拟合脚本启动时打印实际加载的模块路径，运行时自证 |
| 2 | 部分层触发断言 `p + 1 - 64 > 32` | 候选网格最小位置 p=63，去掉 recent 64 后因果前缀为**空** | `candidate_positions(..., excluded_query_prefix=96)`，2048 个候选只去掉 1 个 |
| 3 | 重启拟合后 8 个分片立刻中止 | 续跑一致性断言：残留的 layer 0 用的是旧网格的 protocol | 经授权删除残留，重跑 |
| 4 | B16R16 prefill 比 dense 多 59% | 仓库的 Triton `compressed_v_prefill_attention` 写死了 sm_89/80 的参数（BLOCK_M=32、4 warps），在 sm_120 上只有 152 TFLOP/s，SDPA 是 353 | 改用 V 零填充到 128 + **FlashAttention**（固定后端）。cuDNN 能直接吃 V96 且更快，但在 `use_deterministic_algorithms(True)` 下被 PyTorch 拒绝 |
| 5 | 冒烟启动脚本拒绝启动（误报"有残留 worker"） | 守卫按命令行文本匹配，匹配到了启动它的那个 shell | 改为按可执行文件名匹配，并检查目标卡是否空闲 |
| 6 | 冒烟计时偏慢，11:15 出现双卡 OOM | 另一个会话的 Loki 作业排在拟合之后，拟合一结束就占满 8 卡，我的冒烟叠在了它上面 | 所有启动脚本都先检查目标卡占用；计时在干净的卡上重测 |

---

## 3. 拟合结果

64/64 层的 sha256 与状态全部通过，protocol 一致。

| page-Fisher NMSE | 均值 | 中位数 | 最大 |
|---|---|---|---|
| Qwen3-32B（本次，排除 recent 64） | 0.178 | 0.239 | 0.345（L36） |
| Qwen3-32B（旧版，recent 0） | 0.153 | 0.210 | 0.300 |
| Llama-3.1-8B-Instruct 128K V96（仓库，拟合） | 0.044 | 0.024 | 0.202 |

排除 recent 64 后，64 层的 NMSE **全部**上升，这在预期之内：最近的 token 最好预测，拿掉后剩下的路由问题更难。

**逐层存在断崖**：L7–43 为 0.24–0.35，L44–62 为 0.01–0.04，L43→L44 从 0.257 降到 0.036。

**数值条件**（回答"是否遇到条件数很大"）：

- PCG 全部收敛：最终相对残差中位数 0.0014，最大 0.0158（L32）；Llama 的中位数是 0.0031。
- 确实有条件数很大的层：L25 的聚合 Fisher Gram 约 1e7，`QᵀQ` 约 8.4e6；L15、L29 约 1e5；L45–L53 最高 1.45e6。
- 但条件数与拟合好坏**负相关**：拟合差的 L7–43 中位数只有 342，拟合好的 L44–62 为 3.5e4。
- Base 映射良态（σ1/σ16 为 3–8，L0 为 30），V 编码器为正交（条件数 1.0）。

**Base 与 residual 的分解**：

| 层段 | base 对 K 的相对重建误差 | residual NMSE | 乘积 |
|---|---|---|---|
| L0–6 | 0.263 | 0.082 | 0.021 |
| L7–43 | 0.336 | 0.263 | 0.097 |
| L44–62 | 0.394 | 0.025 | 0.009 |

后段 V 仍能解释 K 能量的约 60%；断崖完全出在 residual 那一步。聚合 Fisher Gram 的谱：中段前 16 个方向只覆盖 30%–36% 的能量（覆盖 90% 需要 88–98 个方向），后段覆盖 68%–84%（32–62 个方向）。**注意：第 6 节的结果表明，NMSE 高低和检索时的路由质量并不对应。**

---

## 4. 性能与显存

### Prefill（样本 0，130809 个 token）

| 路径 | 时间 | 峰值显存 |
|---|---|---|
| Dense SDPA，TP1 | 79.2 s | 73.35 GiB |
| Dense SDPA，TP2 | 47.0 s | 39.32 GiB/卡 |
| B16R16 TP2，Triton prefill kernel | 74.8 s | 64.51 GiB/卡 |
| **B16R16 TP2，FlashAttention（V 填充）** | **48.7 s** | 64.78 GiB/卡 |

Attention 本身（128K × 64 层）：Triton 51.1 s → FlashAttention 24.6 s（cuDNN 为 23.3 s，但确定性模式下不可用）。输出误差 1.2e-3，**生成的 128 个 token 与原路径逐个相同**。

### Decode

- B16R16 TP2：约 **144 ms/token**。
- Attention 侧拆分（每步 64 层）：打分时对 sidecar 整体 `.float()` 占 37.0 ms（改为 bf16 matmul 可降到 9.3 ms，但会改变路由打分的数值，未采用）；`page_support` 18.5 ms；query codes 2.1 ms；`split_indexed_attention` 2.4 ms。
- All-reduce 不是瓶颈：decode 大小的单次 all-reduce 0.019 ms（每步约 2.4 ms）；prefill 大小的单次 55.3 ms（每次 prefill 约 7.1 s）。`NCCL_PROTO=LL` 对大消息慢 4 倍。

### 并行方式的比较

| 方案 | 结论 |
|---|---|
| TP2，batch 1 | 实测每样本 65.6 s（4 组并发时没有互相拖慢）；1100 个样本约 4.7 h |
| TP2，batch 2 | **OOM**：实际分配 90.07 GiB，还要再申请 2.5 GiB，碎片只有 1.66 GiB。而且只有 66.5% 的样本能按等长配对 |
| TP1 + offload | 实体化 144 维 sidecar 时装不下（61.1 + 18.0 + 约 14 GiB 激活 ≈ 93 GiB）；需要先接入紧凑路由。估计比 TP2 快约 15%，未实现 |

### 紧凑路由 kernel（仓库已有 `compact_base_routing.py`，无调用方）

它存的是 [base 码 16 + residual 16] = 32 维/token，而不是实体化的 144 维；128K 下 TP1 为 4.0 GiB，而不是 18.0 GiB。原本按 base rank 4、每组 4 个头写死，已推广到 Qwen3-32B（改 4 行），并针对 sm_120 调参：

| 版本 | 每步路由（TP1，64 层） |
|---|---|
| 实体化 144 维 + bmm | 14.0 ms |
| 紧凑 kernel，原配置 | 25.5 ms |
| 只把 warps 降到 2 | 18.7 ms |
| **去掉 `tl.gather` + rotary 表按 64 宽读取 + warps 1** | **14.8 ms** |

与原 kernel 相比误差 2.9e-3。调参后的版本只放在 scratchpad，未进仓库。

---

## 5. RULER 128K 部分结果（579/1100，同一批样本配对比较）

| 任务 | n | B16R16 | LRQK | ShadowKV | 对 LRQK 胜/负 | 对 ShadowKV 胜/负 |
|---|---|---|---|---|---|---|
| niah_single_1 | 100 | 1.000 | 1.000 | 1.000 | 0/0 | 0/0 |
| niah_single_2 | 100 | 0.990 | 1.000 | 0.970 | 0/1 | 3/1 |
| niah_single_3 | 100 | 0.910 | 1.000 | 0.960 | 0/9 | 4/9 |
| niah_multikey_1 | 100 | 0.930 | 0.970 | 0.950 | 0/4 | 3/5 |
| **niah_multikey_2** | 100 | **0.160** | 0.810 | 0.590 | 2/67 | 4/47 |
| niah_multiquery | 79（部分） | 0.930 | 0.978 | 0.854 | 5/18 | 28/10 |
| **全部** | 579 | **0.816** | **0.959** | **0.889** | | |

说明：各方法每个 KV 组实际看到的 token 数不同（去重后的并集）：B16R16 严格 2048，ShadowKV 约 2514，LRQK 约 3437。

**参照**：同样是 128K、V96、严格 2048 的 Llama-3.1-8B-Instruct（`results/l31-v96-ruler128k`），B16R16 均分 80.88，是压缩方法里最高的（LRQK 80.41，ShadowKV 77.97）；multikey_2 为 75（Full-K 91），single_3 为 99。在 Qwen3-32B 上的退化要大得多。

---

## 6. 路由诊断

### 6.1 失败形态

失败基本都是"**差一点**"：先找对了 needle，拷贝到后面几位出错（例如 8869382 → 8869980；UUID 前 32–34 位正确、只错最后几位）。

先检验了"needle 值跨越页边界"的假设，结果**不成立**：

| 任务 | 值跨页时错误率 | 值不跨页时错误率 |
|---|---|---|
| niah_single_3（值长 25–35 个 token） | 0.08（8/97） | 0.33（1/3） |
| niah_multikey_2（值长 7 个 token） | 0.80（16/20） | 0.85（68/80） |

### 6.2 Oracle：只把打分换成精确 K，其余不变

在 niah_multikey_2 的 400–431（32 个样本）上：

| 路由 | 得分 |
|---|---|
| B16R16 | 0.156 |
| **精确 K + 共用页 + 严格 2048** | **0.844** |
| LRQK | 0.875 |

诊断副本的 B16R16 轮与主评测**逐样本完全一致**，说明诊断改动不影响正式路径。

### 6.3 每步埋点：精确路由会选中 needle 的情况下，B16R16 漏选的比例

| 层段 | residual NMSE | 漏选率 |
|---|---|---|
| L0–6 | 0.07–0.26 | 0.55 |
| L7–43 | 0.24–0.35 | 0.59 |
| **L44–62** | **0.01–0.04** | **0.42** |

拟合几乎完美的后段，检索时仍有 42% 漏选。

### 6.4 分布内与分布外的对比（teacher forcing，捕获的精确注意力质量）

用拟合时用过的 4 个 C4 窗口（强制喂最后 16 个真实 token），对比 8 个 multikey_2 样本（强制喂正确答案）：

| 层段 | C4：B16R16 / 精确 K / 差 | needle：B16R16 / 精确 K / 差 |
|---|---|---|
| L0–6 | 0.597 / 0.636 / 0.039 | 0.614 / 0.640 / 0.025 |
| L7–43 | 0.861 / 0.880 / 0.019 | 0.806 / 0.829 / 0.024 |
| **L44–62** | 0.850 / 0.894 / **0.043** | 0.709 / 0.783 / **0.074** |
| L63 | 0.804 / 0.822 / 0.018 | 0.817 / 0.835 / 0.018 |

逐层"精确 − B16R16"差距与 NMSE 的相关系数：C4 上 −0.68，needle 上 −0.54。

### 6.5 结论

1. **运行时路径没有明显 bug**：分布内（C4）B16R16 与精确选页只差 2%–4%，各层段一致。
2. **退化来自分布偏移，集中在后段**：检索和拷贝发生在后段，那里注意力尖锐，漏掉一个关键页就损失很多。平均差距只有 7%，但决定成败的是少数检索头在拷贝那几步的表现。
3. **NMSE 不适合作为选模型或调参的指标**：它在 C4 query 上以 base-only 误差做归一化，与检索时的路由质量不对应。
4. 已排除的解释：页边界切分、GQA 8 共用页、严格 2048 预算、条件数、拟合与运行时 residual 定义不一致、RoPE/YaRN 的 cos/sin 不一致、TP2 因子切分错误。

**局限**：C4 只用了 4 个窗口，且是拟合用过的样本内数据；needle 只有 8 个样本（oracle 为 32 个）；teacher forcing 分别喂 16 个和 8 个 token。

---

## 7. 待决事项与下一步

**主评测**：暂停在 579/1100（5 个 niah 任务全部完成，multiquery 完成 79 个）。重新运行 `b16r16_fit/run_b16r16_tp2.sh` 会审计已完成的样本并跳过，只补跑剩下的 521 个（约 2.2 h）。正式评测脚本未改动，续跑审计可以通过。

**候选方案**（都可以先在 multikey_2 的 400–431 上快速验证）：

| 方案 | 测什么 | 耗时 |
|---|---|---|
| A. 放宽每组预算（3072 / 4096） | B16R16 需要多少额外预算才能追上"精确 K @ 2048"的 0.844 | 每档约 12 分钟 |
| B. residual 秩提到 R32 | 用盘上的统计量免 teacher 重拟合（`refit_router_from_statistics.py`）；评测脚本需要支持 R32 | 约 25 分钟拟合 + 12 分钟评测 |
| C. 校准数据加入检索类 query | 直接针对分布偏移 | 数小时 |
| — recent 64 移到预算外（2112） | 只多 2 个路由页；Llama 上两种口径的均分只差 0.38 | 预计无明显效果 |

**尚未决定**：是否把 prefill 的 FlashAttention 修复推广到仓库里仍在调用 Triton kernel 的约 47 个文件（会带来约 1e-3 的数值差，已发布结果将无法逐位复现）；以及这批文件是否上传 GitHub。

---

## 附：文件与脚本

| 路径 | 内容 |
|---|---|
| `b16r16_fit/evaluation/calibrate_qwen3_32b_router_128k.py` | 拟合脚本（排除 recent 64、`query_grid_prefix=96`、启动时打印加载来源） |
| `b16r16_fit/evaluation/fit_qwen3_8b_q8_fisher_residual.py` | 增加 `excluded_recent_tokens` 的统计量构建 |
| `b16r16_fit/evaluation/__init__.py` | 修复 namespace 遮蔽 |
| `b16r16_fit/eval/eval_qwen3_32b_v96_b16r16_tp2.py` | 正式 TP2 评测脚本（FlashAttention prefill） |
| `b16r16_fit/eval/diag_router_tp2.py` | 诊断副本：`--router {b16r16,exact,exact_head,exact_head256}`、needle 埋点、捕获质量记录 |
| `b16r16_fit/eval/fidelity_tp2.py` | teacher forcing 保真度测试 |
| `b16r16_fit/out/evaluate/` | 主评测的 579 个样本 |
| `b16r16_fit/out_diag_b16r16/`、`out_diag_exact/`、`out_fidelity/` | 诊断输出 |
| `b16r16_fit/logs/`、`logs_diag/`、`logs_recent64/` | 日志 |
