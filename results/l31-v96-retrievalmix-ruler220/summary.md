# Llama-3.1-8B-Instruct · V96 · 校准域实验：通用 C4 vs C4+合成长程检索（RULER 128K，220 条）

2026-09-22。第一阶段（V96 + Full-K，不含路由）完成。**结论：方向一致为正，但在 220 条上未达到统计显著，且效应量小于基线自身的"换一批 C4 窗口"方差；按第 15 节规则尚不足以触发第二阶段（B16R16 重拟）。**

## 结论速览

| | A′（32×C4） | B（16×C4 + 16×合成检索） | 差 |
|---|---|---|---|
| 全部 220 条均分 | 0.813 | **0.840** | **+0.027**，配对 bootstrap 95% CI [−0.001, +0.057] |
| macro 平均（11 任务） | 0.813 | 0.840 | +0.027，CI [−0.001, +0.056] |
| 配对胜/负/平 | | | **23 / 11 / 186**，符号检验 p = 0.058，Wilcoxon p = 0.060 |
| 任务方向 | | | B ≥ A′ 的任务 10/11，B > A′ 7/11，B < A′ 1/11（qa_1） |

- 三个重点任务全部为正，但每个 n=20、一条样本就是 5 分，CI 都跨零：**niah_multikey_2 +0.05** [−0.10, +0.20]，**vt +0.06** [−0.08, +0.21]，**fwe +0.05** [−0.08, +0.18]。
- 混入合成窗口**没有损害 C4 上的重建**：16 个 C4 验证窗口上的 heldout 相对 MSE，A′ 0.0512 vs B 0.0514（同一批验证窗口）。
- **基线自身的方差比效应量大**：本地重拟的 A′ 与 HF 上的正式 checkpoint 用的是同一算法、同一超参、都是 32×C4，只是窗口不同（seed 不同），在重叠的 118 条上 A′ 0.953 vs 正式 0.977；niah_multikey_2 上 **0.80 vs 0.95**。也就是说，"换一批 C4 窗口"在 multikey_2 上能差 3 条，而 B 相对 A′ 的改善是 1 条。

## 实验设计（严格按 spec 执行的部分）

| 项 | 值 |
|---|---|
| 模型 | meta-llama/Llama-3.1-8B-Instruct，rev `0e9e39f2`，config sha256 `29e4c210…`（与正式 128K V96 实验一致） |
| 唯一变量 | 校准数据分布。Condition A′：32 × 128K 通用 C4；Condition B：16 × 128K 通用 C4（**同一批的前 16 个**）+ 16 × 128K 合成检索。验证窗口两者共用同一批 16 × C4 |
| 不变项 | uniform rank 96；ALS 6；CG 固定 16；covariance damping 1e-7；fp32 工作精度；bf16 因子；不开 TF32；activation-weighted-SVD 初始化 seed 0；decoder full_layer；fit 32 / validation 16；token 总预算 32 × 131072 |
| 与正式 checkpoint 的 fit_config | 上述字段逐项一致（已比对） |
| 评测 | Full-K only（不装路由），greedy，官方 task cap，确定性算法；同一批 220 条 prompt，按 index 与 input_sha256 配对 |

### 与 spec 的偏离

1. **基线在本地重拟（A′），不是正式 checkpoint。** 正式实验的 32 个 C4 窗口 seed 未记录、无法复现，B 无法"复用基线前 16 个窗口"。改为本地新采 48 个窗口、同一代码同一机器同时拟 A′ 和 B，两者只差那 16 个窗口。正式 checkpoint 单独跑了 118 条作参照后取消（用户决定），上表已引用。
2. **RULER 用 220 条（11 任务 × 20），不是 spec 里的 600 条 SHA 匹配子集或 1100 条全集。** 用户决定：诊断用途，重点看 multikey_2。官方 RULER 生成器 commit `c3f5e3b`，seed 42，prompt margin 128，base 模板；与正式实验的 1100 条不是同一批，**绝对分数不可与已发表数字直接比**，A′ vs B 之间严格配对。
3. **合成窗口尾部 3.9K–7.9K token，低于 spec 的 9K–13K。** 96 条问答已到上限，答案已写成带键名/链路的完整句，未再用无意义填充凑长度。

## 合成检索校准数据

`data/retrieval_calibration/`（`windows.safetensors` 8 MB、`metadata.jsonl`、`manifest.json`、`summary.json`）。

- 生成器 `evaluation/prepare_retrieval_calibration_windows.py`，自写模板，**不导入、不复制任何评测生成器**。method、seed（20260922）、haystack seed（20260923）、每窗口 seed 规则、C4 revision、排除的文档哈希数（1536）全部写入 manifest。
- 16 窗口 × 131072 token（精确）；任务配比 8 multikey / 4 multivalue / 2 tracking / 2 aggregation；每窗口 96 组问答。
- 记录按 8 个位置层均匀分布（每层 127–143 条）；query→支撑记录距离：≥96K 1566 条，48K–96K 2541 条，16K–48K 1546 条，<16K 692 条。
- 校验 A–H 全部通过：长度精确、答案可从元数据完全复现、查询文本不含答案值、值只出现在预期位置、键/值无碰撞、haystack 与 A′ 的 48 个 C4 窗口 document-disjoint。

## 逐任务（220 条，n=20/任务）

| 任务 | A′ | B | B 胜/负 |
|---|---|---|---|
| niah_single_1 | 1.000 | 1.000 | 0/0 |
| niah_single_2 | 1.000 | 1.000 | 0/0 |
| niah_single_3 | 0.950 | 1.000 | 1/0 |
| niah_multikey_1 | 1.000 | 1.000 | 0/0 |
| **niah_multikey_2** | 0.800 | 0.850 | 2/1 |
| niah_multiquery | 0.975 | 0.988 | 2/1 |
| niah_multivalue | 0.863 | 0.938 | 5/0 |
| **vt** | 0.740 | 0.800 | 5/4 |
| **fwe** | 0.567 | 0.617 | 7/4 |
| qa_1 | 0.600 | 0.550 | 0/1 |
| qa_2 | 0.450 | 0.500 | 1/0 |

B 输掉的 11 条里，vt 4 条和 fwe 4 条与它赢的 5 条和 7 条是双向噪声（vt 5/4、fwe 7/4）；multivalue 5/0 是唯一单向的任务。qa_1 那一条（"north" → "Normandy"）是知识型 QA，方向与"检索校准可能牺牲通用性"的担忧一致，但只有 1 条。

## 第 15 节决策

规则："若在配对困难子集上**有实质改善**且无明显通用质量退化 → 进第二阶段；否则停止。"

- 方向一致（10/11 任务 ≥，23:11），但 p ≈ 0.06、CI 跨零、效应量 < 基线换窗口的方差。**不算"实质改善"**，不触发第二阶段。
- 也**不是**"Full-K 没改善、校准域不是主因"的否定结论——220 条的功效不够给出否定。
- 第 14 节守门（WikiText-2 / MCQ）因此**尚未执行**。现有 `eval_gqa_palu_m_wikitext.py` 只认 PaLU/ICLR 单文件格式，接我们的 32 层 uniform C1 manifest 需要改一版。

要把这个问题真正判定，最直接的是补跑同一 seed 下剩余的 880 条（凑齐 1100 条全集）：A′ 和 B 各约 80 分钟（8 卡），multikey_2 / vt / fwe 各到 n=100，一条样本从 5 分变 1 分。

## 产物

| 路径 | 内容 |
|---|---|
| `/home/Ubuntu/l31_retrieval_cal/c4_128k/` | 48 × 128K C4 窗口 + manifest（seed 20260921，每片段文档哈希） |
| `data/retrieval_calibration/`（仓库内） | 16 个合成窗口 + 元数据 + 校验报告 |
| `/home/Ubuntu/l31_retrieval_cal/c4_retrieval_50_50/` | Condition B 的 48 窗口拼装 + provenance |
| `/home/Ubuntu/l31_retrieval_cal/cov_{A,B}/` | 每层 o_proj 协方差快照 |
| `/home/Ubuntu/l31_retrieval_cal/fit_{A,B}/`、`ckpt_{A,B}/` | ALS 因子库与评测用 checkpoint（B 即 spec 的 `…_als6_retrievalmix`） |
| `/home/Ubuntu/l31_retrieval_cal/ruler220/`、`eval_{A,B,hfA}/` | RULER 数据与逐条预测（hfA 118/220，已取消） |
| `evaluation/prepare_llama31_8b_c4_128k_windows.py` | C4 128K 窗口构建 |
| `evaluation/prepare_retrieval_calibration_windows.py` | 合成检索窗口生成 + 校验 |
| `evaluation/package_llama31_8b_c1_checkpoint.py` | 因子库 → 评测 checkpoint（绕开 `build_llama31_8b_palu_m_checkpoint` 里钉死 base 模型 revision 的检查） |
| `evaluation/eval_llama_cal128_fullk.py` | `eval_llama_cal128.py` 的 Full-K-only 副本（原脚本的 smoke 审计强制要求所有路由 arm） |
| `/home/Ubuntu/l31_retrieval_cal/compare_conditions.py` | 配对比较 |

## 过程中修掉的问题

- 打包脚本沿用的共享 profile `llama31_8b` 钉死的是 **base 模型** `meta-llama/Llama-3.1-8B` 的 revision，与 Instruct 下载不匹配而中止；评测加载器实际只读 `manifest.layers[].{file,sha256,ranks}` 和每层的三个张量，改为直接按该契约打包（与正式 HF checkpoint 逐字段核对一致）。
- RULER 生成器需要 `punkt_tab`、Paul Graham essays、SQuAD/HotpotQA 数据；`--tasks all` 会多生成 `cwe`/`niah_multikey_3`，改为显式传项目的 11 个任务。
- 合成生成器的三处 bug（尾部长度断言、精确补齐、记录字段名与 `target` 撞名）在 4 窗口冒烟里逐个修掉。

## 与 B16R16 相关的备注

第二阶段若启动，Llama 的 B16R16 用 **ALS 40 / PCG 100**（用户决定 2026-09-22）。HF 上的 bank `…_als40_pcg100` 和 `eval_llama_cal128.py` 的断言均为 40/100；`results/l31-v96-ruler128k/README.md` 里的"ALS 60, PCG 50"与已提交的 `router/ours_b16r16/*.json` 元数据对应的是一次权重未提交的早期拟合，属过时记录。
