# Query Selection 实现报告：显式 Gram 上的截断 CPQR 等价选点

日期：2026-09-10。本文描述当前仓库的实际实现，并引用已经完成的全层检查。本次撰写仅读取代码和既有结果，没有启动新实验，也没有修改 selector 或已训练的 router。

## 1. 结论与方法名称

当前方法是 **stratified, head-whitened, CPQR-based query-position selection**：

1. 用 fit windows 中的候选 Q 计算每个 head 的 whitening。
2. 按绝对位置划分四个 8K bins。
3. 为每个 bin **显式构造位置 Gram**。
4. 在 Gram 上做八步 diagonal-pivoted Cholesky 更新，选出八个位置。
5. 合并四个 bin，得到每层固定的 32 个拟合 query 位置。

因此，我们属于讨论中的“方案 B”，但必须保留两个实现事实：

- 正式 selector **没有执行完整 QR 分解**。
- 正式 selector **确实生成了 N×N Gram**；不能称为 Gram-free 或纯 feature-space CPQR。

精确算术、相同非零残差和并列处理下，这个贪心选点过程与对应 feature 转置上的前 k 步 CPQR 等价。全层实验进一步验证了本次数据与 Q32 配置下的实际数值一致性；这不是对任意输入、硬件或退化情形的逐 pivot 保证。

## 2. 采样单位与真实维度

以一个 transformer layer 为单位，输入张量为：

\[
Q\in\mathbb R^{W\times C\times H\times d_h}.
\]

| 符号 | 含义 | 本次全层验证 |
|---|---|---:|
| W | 参与选点的 fit windows | 64 |
| C | 每个 window 的候选位置数 | 512 |
| H | query heads | 32 |
| d_h | 每个 head 的 Q 维度 | 128 |
| L | window 长度 | 32768 |
| B | 位置 bins | 4 |
| N | 单个 bin 的候选位置数 | 128 |
| k | 每个 bin 的选点数 | 8 |
| Bk | 每层最终 query 位置数 | 32 |
| D=W H d_h | 等价拼接 feature 的维度 | 262144 |

**选中一个位置，意味着保留该位置在所有 fit windows、所有 heads 下的 Q。** 不是在所有独立 query 向量中总共只抽取 32 条。

选中的位置集合是 layer-specific，同一层跨 windows 共用。选点后，一层的 residual fitting 输入为 `[64,32,32,128]`：每个 head 有 64×32=2048 个 window/query observations。

这里的 Q32 表示离线拟合的 query-position 数量，既不是 residual rank，也不是在线 token/page 检索预算。

## 3. 候选 Q 的来源

当前全层验证使用本地固定 C4 windows 的 fit IDs 0–63，长度均为 32768。它们属于本地 `v96kl_64x32k` 数据，不能与缺失的历史远端 captures 宣称逐位相同。

候选位置由 `candidate_positions` 生成，采用 zero-based indexing：

\[
J=\{63,127,191,\ldots,32767\}.
\]

候选步长是 64；同时要求位置不小于默认 excluded-query-prefix 32。默认候选网格从 63 开始，已满足这个条件。

捕获使用原始 dense Qwen3-8B-Base 的 BF16 SDPA backbone forward，完整处理每个 window；保存的是实际 `q_proj → q_norm → RoPE` 后的 Q。不使用 C1 payload，不进行 Base 或 residual 拟合，也不使用 benchmark labels。

全层 capture 只保存候选 Q，每个 window 的文件包含：

```text
queries: BF16 [36, 512, 32, 128]
```

输入与结果：

- 固定窗口：[v96kl_64x32k manifest](../results/calibration/v96kl_64x32k/manifest.json)。
- 全层候选 Q：[cpqr_all_candidates manifest](../results/calibration/cpqr_all_candidates/manifest.json)。
- 捕获代码：[capture_query_cpqr.py](../evaluation/capture_query_cpqr.py)。

## 4. 每个 head 的 uncentered whitening

设原始 post-RoPE Q 为行向量 \(q_{w,i,h}\)。对每个 head 独立计算未中心化二阶矩：

\[
M_h=\frac{1}{WC}\sum_{w=1}^{W}\sum_{i=1}^{C}
q_{w,i,h}^{\top}q_{w,i,h}.
\]

**不减 query 均值。** Whitening 使用完整 512-position 候选网格，而不是为每个 bin 分别估计。

将对称化后的二阶矩做特征分解：

\[
M_h=V_h\operatorname{diag}(\lambda_h)V_h^\top.
\]

仅保留：

\[
\lambda_{h,j}>10^{-6}\lambda_{h,\max}.
\]

对应 inverse-square-root 系数为：

\[
a_{h,j}=\begin{cases}
\lambda_{h,j}^{-1/2},&\lambda_{h,j}>10^{-6}\lambda_{h,\max},\\
0,&\text{其他方向}.
\end{cases}
\]

于是：

\[
T_h=V_h\operatorname{diag}(a_h)V_h^\top,
\qquad \widetilde q_{w,i,h}=q_{w,i,h}T_h.
\]

这是带相对阈值的伪逆平方根，不是给所有特征值加 ridge。Whitening 不等于逐条 query 单位化；不同 whitened Q 的范数仍然可以不同。

二阶矩、特征分解、内部 whitening 与 Gram 累积均为 FP64。函数返回的 whitening 副本会转换成 FP32，但实际选点使用转换前的 FP64 张量。直接 CPQR 验证重新构造同样的 FP64 whitening，避免把返回的 FP32 副本当作原始选点输入。

## 5. 分层位置 Gram 与等价 feature

对绝对位置 p，bin 的定义是：

\[
b(p)=\left\lfloor\frac{4p}{32768}\right\rfloor.
\]

四个位置范围为 `[0,8191]`、`[8192,16383]`、`[16384,24575]`、`[24576,32767]`，每个范围各有 128 个候选位置。

对同一 bin 内的位置 i、j，代码计算：

\[
G_{ij}=\frac{1}{WH}\sum_{w=1}^{W}\sum_{h=1}^{H}
\langle\widetilde q_{w,i,h},\widetilde q_{w,j,h}\rangle.
\]

其等价的显式 feature 是：

\[
\phi_i=\frac{1}{\sqrt{WH}}
\operatorname{concat}_{w,h}\widetilde q_{w,i,h}
\in\mathbb R^{262144},
\qquad
\Phi\in\mathbb R^{128\times262144},
\qquad G=\Phi\Phi^\top.
\]

正式代码没有物化这个大拼接矩阵，而是逐 window、逐 head 累积内积。**平均内积不等于先平均 query 再求内积**；后者会引入不同的交叉项，改变选点几何。

Whitening 和 Gram 只使用 fit windows。校准代码随后在 fit 与 diagnostic windows 上读取相同的选中位置，但 diagnostic Q 不参与选点。

## 6. 八步 pivoted Cholesky 更新

实现入口：[select_positions_pivoted_gram](../basisserve/core/query_position_sampling.py)。

初始化残差对角线和低秩因子：

\[
d_i=G_{ii},\qquad L\in\mathbb R^{N\times k}=0.
\]

在第 t 步，从未选位置中取：

\[
p_t=\arg\max_{i\notin S_t}d_i.
\]

若 pivot 能量高于数值阈值，计算：

\[
L_{:,t}=\frac{G_{:,p_t}-L_{:,:t}L_{p_t,:t}^\top}{\sqrt{d_{p_t}}},
\qquad
d_i\leftarrow d_i-L_{i,t}^2.
\]

实际行为还包括：

- 输入 Gram 再次对称化，检查有限值及近似 PSD。
- 默认 `tolerance=1e-10`，能量阈值是 `tolerance * max(max(abs(diag(G))), 1)`。
- 位置预先按升序排列；完全并列时 `argmax` 选择第一个，即较小绝对位置。
- 已选位置在下一轮设为不可选。
- 数值误差产生的小负对角线在检查范围后截断为零。
- 即使遇到数值零 pivot，也继续填满固定 k 个位置，并标记 `numerically_zero_pivot`；不会自动减少 query 数量。

下面是非退化路径的说明性伪代码，省略了上述数值检查，不能替代实际源码：

```python
diag = diagonal(G).copy()
L = zeros((N, k))
selected = []
for t in range(k):
    p = argmax_over_unselected(diag)
    selected.append(p)
    L[:, t] = (G[:, p] - L[:, :t] @ L[p, :t]) / sqrt(diag[p])
    diag = maximum(diag - L[:, t] ** 2, 0)
```

每个 bin 保存原始 `pivot_order` 及每步 residual energy。供后续捕获/拟合使用的 `selected_positions` 则按绝对位置排序。**四个 bins 独立选八个位置，不是对整个 512-position pool 做一次 global CPQR 后取前 32 个。**

## 7. 为什么等价于前 k 步 CPQR

若对 \(A=\Phi^\top\) 做 CPQR，下一 pivot 的准则是最大未解释列范数：

\[
\arg\max_i\|(I-P_{S_t})\phi_i^\top\|_2.
\]

Gram 更新中的 \(d_i\) 对应这个范数的平方：

\[
d_i=\|(I-P_{S_t})\phi_i^\top\|_2^2.
\]

范数和平方范数具有相同排序，因此在精确算术、非退化与相同 tie-breaking 条件下，两者选取相同 pivot。

正式实现只执行 k=8 次 Gram 更新。有限精度下，Gram 构造、QR 的 norm updates、近似并列和零残差处理都可能导致不同路径，因此报告仍应区分数学等价性和实际数值验证。

## 8. 存储与计算成本

下面是主要张量大小，不是程序峰值内存测量：

| 张量 | 单层/单 bin 范围 | 大小 |
|---|---|---:|
| BF16 候选 Q | 单层 `[64,512,32,128]` | 256 MiB |
| FP64 head 二阶矩 | 单层 `[32,128,128]` | 4 MiB |
| FP64 whitening | 单层 `[32,128,128]` | 4 MiB |
| FP64 bin Gram | 单 bin `[128,128]` | 128 KiB |
| 四个 FP64 Grams | 单层 | 512 KiB |
| FP64 Cholesky 因子 L | 单 bin `[128,8]` | 8 KiB |
| FP64 显式拼接 feature | 单 bin `[128,262144]` | 256 MiB |
| 全层候选 Q 文件 payload | 36 层×64 windows | 9 GiB |

这里 **D≫N**，并非 N≫D。因此显式 Gram 在本任务中很小；“Gram 在大候选池下可能危险”的一般结论，并不表示当前的 128×128 Gram 存在同样的问题。

不计 whitening，单 bin Gram 构造约为 O(N²D)；构造完成后，当前 k 步更新约为 O(Nk²)，存储为 O(N²+Nk)。源码还执行 `eigvalsh` 等 PSD/诊断检查，包含 O(N³) 成本，不能把整个 selector 的成本都称为仅 O(Nk²)。

Feature-space truncated CPQR 可避免显式 Gram，主要工作约 O(DNk)，但需要安排 feature 存储或分块更新。当前直接 SciPy CPQR 验证则做完整 economic QR；在 D≫N 时，其分解成本约 O(DN²)，而非适用于 N≫D 的 O(ND²)。Economic QR 不生成 D×D 的完整 Q，但仍可能返回 D×N 的 Q 并产生输入副本。

本轮没有进行公平的端到端 selector 性能基准，因此这里只比较结构和张量规模，不声称测得某个普适加速比。

## 9. 与 Base / residual fitting 的衔接

本地拟合入口：[calibrate_v96kl_router.py](../evaluation/calibrate_v96kl_router.py)。实际调用关系为：

```text
dense teacher Q capture
  → fit-only stratified position selection
  → shared selected positions for fit/diagnostic windows
  → causal non-sink Page-Fisher statistics
  → fixed-rank residual E/U fitting
```

Query selector 只决定哪些 Q observations 进入 residual 拟合。当前闭式 affine Base16 使用全部 fit K/V token rows，Base 的 MSE-RRR 拟合不因选中 Q32 而仅使用 32 个 token。

冻结 Base 后，residual 为 post-RoPE exact K 减去 V latent 预测得到的 post-RoPE Base K。每个选中 query 仍使用它的完整 causal non-sink K prefix 构造 Page-Fisher Gram，并非只采样 32 条 K。

**位置 Query Gram 与 residual Page-Fisher Gram 是两种不同统计量。** 前者是本报告的候选位置相似性矩阵，用于选 Q；后者由每个 Q 的 exact attention page probabilities 和 residual features 构造，用于拟合 E/U。不能把前者的投影能量当作后者的 loss。

## 10. 全 36 层的实证验证

本次验证依次完成：全层 Q 重捕获、分片合并、显式 feature CPQR 对照、sRRQR 对照及最终结果复核。

| 检查 | 结果 |
|---|---|
| CPQR vs production Query-Gram | 144/144 个 layer/bin 的 pivot 顺序一致 |
| 每种方法的选点总数 | 36×4×8=1152 个位置槽位 |
| sRRQR f=2 | 144/144 区间不变，0 次交换 |
| sRRQR f=1.01 | 144/144 区间不变，0 次交换 |
| 两档 sRRQR 停止条件 | 288/288 组合满足 |
| 全局最大 initial/final rho | 1.004078783579815，layer 1、bin 0 |
| 最大 explicit-feature/production Gram 绝对差 | 6.536993e-12 |
| 最大 full-root Gram 绝对差 | 6.394885e-13 |
| full-root 被截断的负特征值数 | 0 |
| 所选 feature block 条件数范围 | 1.110502–1.525972 |
| 剩余 feature 投影能量比例 | 0.895644–0.923798 |
| 全层捕获与原三层缓存的重叠 Q | 192/192 个 window/layer 张量逐位一致 |

sRRQR 重用仓库的 Gu–Eisenstat fixed-rank 交换实现，使用每个 bin **完整谱**的 Gram 平方根表示列几何，不做 POD 截断。先核对 CPQR 初始化及几何与显式 feature 基线一致，再检查交换条件。

最大 rho 对应的最佳单次 volume 增益约为 0.407878%，低于 f=1.01 的 1% 门槛。它说明两档测试阈值下无需交换，不说明已找到全局 maximum-volume subset，也不排除更严格阈值下发生交换。

约 90% 的剩余投影能量是高维拼接 feature 空间的 Frobenius-energy 指标，不是 missed attention mass、K reconstruction error、Fisher NMSE 或下游任务错误率。

完整证据：

- [全层结果总结](../results/evaluation/srrqr_all_q32/summary.md)。
- [144 行逐 bin 指标](../results/evaluation/srrqr_all_q32/bins.csv)。
- [最终 audit 与 72 份逐层结果哈希](../results/evaluation/srrqr_all_q32/audit.json)。

## 11. 代码入口与职责

| 文件 | 职责 |
|---|---|
| [query_position_sampling.py](../basisserve/core/query_position_sampling.py) | 正式 candidate grid、whitening、Gram 构造和截断 pivoting |
| [calibrate_v96kl_router.py](../evaluation/calibrate_v96kl_router.py) | 本地校准中调用正式 selector，衔接 Base / residual fitting |
| [select_query_positions.py](../evaluation/select_query_positions.py) | 独立选点与 position manifest 接口 |
| [capture_query_cpqr.py](../evaluation/capture_query_cpqr.py) | 本次全层 Q capture 与分片合并 |
| [query_cpqr_audit.py](../basisserve/core/query_cpqr_audit.py) | 显式拼接 feature 上的 SciPy CPQR 验证 |
| [audit_query_cpqr.py](../evaluation/audit_query_cpqr.py) | CPQR 对照命令行入口与 provenance |
| [strong_rrqr.py](../basisserve/core/strong_rrqr.py) | 既有 Gu–Eisenstat swap solver |
| [query_srrqr_audit.py](../basisserve/core/query_srrqr_audit.py) | 全谱 Gram 平方根、sRRQR 与几何对照 |
| [audit_query_srrqr.py](../evaluation/audit_query_srrqr.py) | sRRQR 对照命令行入口与 provenance |

CPQR/sRRQR 对照文件属于验证路径；本次没有把它们安装为正式 selector，也没有改写现有 router bank。

## 12. 执行环境、命令与日志

全部实验使用 `lowrank`。本服务器经用户明确批准直接执行，不使用 Slurm。

- 全层 capture：GPU 5/6 两个独立 window shards，各 2 个 CPU/OMP/MKL/OpenBLAS 线程。实测每个 worker 峰值分配显存 20.157 GiB。
- CPQR/sRRQR：CPU 2 线程，无 GPU。
- 两个 capture workers、merge、CPQR、sRRQR、最终复核均退出码 0，无重试。
- 仅 capture 日志有 Transformers `torch_dtype` 弃用提示。没有 OOM、非有限值或停止条件检查失败。

以下是已执行的 CPU 检查命令；完整捕获、合并及资源记录见[全层执行方案](query_all_layers_protocol.md)。下面日志和输出已存在，不应直接覆盖重跑。

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=.
set -o noclobber
query_all_layers=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35

CUDA_VISIBLE_DEVICES= python -u -m evaluation.audit_query_cpqr \
  --candidate-capture results/calibration/cpqr_all_candidates \
  --layers "$query_all_layers" --queries-per-bin 8 \
  --output-dir results/evaluation/cpqr_all_q32 \
  > results/logs/query_cpqr/audit_cpqr_all_q32.log 2>&1

CUDA_VISIBLE_DEVICES= python -u -m evaluation.audit_query_srrqr \
  --candidate-capture results/calibration/cpqr_all_candidates \
  --cpqr-reference results/evaluation/cpqr_all_q32 \
  --layers "$query_all_layers" --bounds 2,1.01 --max-swaps 512 \
  --output-dir results/evaluation/srrqr_all_q32 \
  > results/logs/query_cpqr/audit_srrqr_all_q32.log 2>&1
```

日志：[CPQR](../results/logs/query_cpqr/audit_cpqr_all_q32.log)、[sRRQR](../results/logs/query_cpqr/audit_srrqr_all_q32.log)、[最终复核](../results/logs/query_cpqr/all_result_check.log)。

候选 manifest SHA256：`fbbc190c863dd5d8bd44079ee06e7c139fea360829828b47b509447bd382e05d`。

## 13. 可用于论文的方法描述

> We select calibration query positions independently for each layer using stratified, head-whitened pivoting. For each positional bin, we form the Gram matrix of query features concatenated across calibration windows and heads, and perform k steps of diagonal-pivoted Cholesky. In exact arithmetic, with consistent tie-breaking and nondegenerate pivots, this selects the same positions as the first k steps of column-pivoted QR on the transposed feature matrix. Our implementation explicitly stores the small position Gram and avoids a full feature-space QR factorization.

在本次 Q32 全层验证的结果描述中可以补充：

> Across all 36 layers and four positional bins per layer, the selected pivot sequences matched explicit feature-space CPQR. Strong-RRQR refinement with f=2 and f=1.01 required no swaps and retained the same selections.

这两段分别描述实现与已测结果，不应延伸为“所有输入都逐 pivot 等价”“sRRQR 永远不改变选点”或“routing/task quality 必然相同”。
