# Qwen3-8B Wo-only C1 严谨实验总览

更新日期：2026-08-29

这份文档汇总 Qwen3-8B-Base、TP4 下的 Wo-only C1 严谨实验主线。它覆盖表示能力、求解器、rank、source-subspace 机制、whole-model quality 和 L40S 系统实验，同时保留所有重要负结果与尚未完成的验证。

## 1. 范围与方法定义

这条主线只压缩 attention 后的 `W_o` 映射。所有方法都保留 dense V、dense K 和 dense KV cache，因此这里的结果不能用于声称 KV-cache compression、低维 AV 或 folded-V 收益。

Qwen3-8B 有 36 层、hidden size 4096、32 个 query heads、8 个 physical KV heads、head dimension 128。TP4 下每个 source 对应一个 TP rank，拥有 8 个 query heads；每个 source 的 post-attention 输入宽度为 1024。

方法定义为：

\[
Y_{\mathrm{dense}}=\sum_{p=1}^{4}Z_pW_p,
\]

\[
\widehat Y_{\mathrm{LR\text{-}AR}}
=\left(\sum_{p=1}^{4}Z_pF_p\right)D,
\]

\[
\widehat Y_{\mathrm{C1\text{-}AG}}
=\sum_{p=1}^{4}(Z_pE_p)D_p.
\]

`C1 local-AR` 使用与 C1-AllGather 完全相同的 factors 和近似函数，但先在每个 source 本地解码，再对 hidden-width output 做 AllReduce：

\[
\widehat Y_{\mathrm{C1\text{-}local\text{-}AR}}
=\operatorname{AllReduce}_p\left((Z_pE_p)D_p\right).
\]

主 operating point 为每个 C1 source rank 512，总 gathered rank 2048。BF16 ideal-ring 通信预算如下；单位为每层、每 activation row、每 rank 的字节数。

| Arm | Latent/decoder geometry | Collective | Ideal bytes | Reduction vs dense |
|---|---|---|---:|---:|
| Dense | local `K=1024` → hidden 4096 | hidden AllReduce | 12288 | 0% |
| C1-AllGather | 4 × rank-512；global decoder `K=2048` | rank-512/source AllGather | 3072 | 75% |
| Wire LR-AllReduce | shared rank 1024 | rank-1024 AllReduce | 3072 | 75% |
| Capacity LR-AllReduce | shared rank 2048 | rank-2048 AllReduce | 6144 | 50% |
| C1 local-AR | local decoder `K=512` | hidden AllReduce | 12288 | 0% |

Capacity LR 的 rank 等于 C1 aggregate rank，因此它的函数类包含 C1。它必须在充分优化后达到不差于 C1 的 calibration objective；该关系是 solver correctness kill gate，而不是 C1 应该获胜的 comparison。

实现和 tensor-ordering 审计见 [`docs/c1_rigorous_evaluation_audit.md`](../docs/c1_rigorous_evaluation_audit.md)。

## 2. 公共 calibration 与数值协议

- Calibration：256 个 C4 文档用于 fit，64 个不相交文档用于 heldout。
- 每个文档使用全部 2048 positions：fit 524288 rows，heldout 131072 rows。
- 保存 normalized covariance sufficient statistics，不保存原始 activation。
- C1 objective：完整 attention `o_proj` output MSE，包含 source 间 covariance。
- C1 初始化：activation-aware；随后 decoder-closed ALS。
- Encoder update：FP64 exact two-sided Cholesky solve。
- Covariance damping：`1e-5`；没有额外 encoder damping。
- Fit work dtype：FP64；部署 factor dtype：BF16。
- Whole-model execution：BF16；cross-entropy accumulation：FP32。
- 主要环境：`basis`，PyTorch 2.6.0+cu124；系统实验为 4×L40S TP4。

## 3. 实验完成状态

| Experiment | Status | Main artifact |
|---|---|---|
| Strong LR-AR、capacity inclusion、wire match | Complete | [`Phase-1`](checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100/summary.md) |
| Exact encoder solve 与 ALS convergence | Complete | [`s20`](checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s20/summary.md), [`s100`](checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100/summary.md) |
| Uniform rank Pareto | Complete | [`rank sweep`](evaluation/qwen3_8b_wo_c1_uniform_rank_sweep/summary.md) |
| Synthetic controlled angles | Complete | [`synthetic angles`](evaluation/qwen3_8b_wo_c1_lr_ar_synthetic_angles/summary.md) |
| Real source-subspace structure/correlation | Complete | [`source subspaces`](evaluation/qwen3_8b_wo_source_subspaces/summary.md) |
| Source permutation | Complete | [`permutations`](evaluation/qwen3_8b_wo_source_permutations/summary.md) |
| Latent-basis invariance | Complete | [`basis invariance`](evaluation/qwen3_8b_wo_latent_basis_invariance/summary.md) |
| Joint-final vs independent-local objective | Complete | [`joint vs local`](checkpoints/qwen3_8b_wo_c1_independent_local_r512_joint20/summary.md) |
| Cross-source covariance ablation | Complete | [`layer error`](checkpoints/qwen3_8b_wo_c1_cross_source_covariance_joint20/summary.md), [`PPL`](evaluation/qwen3_8b_wo_c1_cross_source_covariance_ppl_joint20/summary.md) |
| WikiText-2 and heldout C4 PPL | Complete | [`quality`](evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/summary.md) |
| Collective microbenchmark | Complete on L40S/TP4 | [`collective`](evaluation/qwen3_8b_wo_collective_microbench_tp4_l40s/summary.md) |
| Decoder GEMM microbenchmark | Complete on L40S | [`decoder GEMM`](evaluation/qwen3_8b_wo_decoder_gemm_l40s/summary.md) |
| Eager decode/prefill runtime | Complete on L40S/TP4 | [`eager runtime`](qwen3_8b_wo_tp4/full/summary.md) |
| Five-arm CUDA Graph comparison | Complete as appendix | [`CUDA Graph`](evaluation/qwen3_8b_wo_cuda_graph_l40s/comparison/summary.md) |
| High-pressure latency decomposition | Deferred | Hardware-specific; revisit on target TP8/A5000 system |
| Full batch×context phase map | Deferred | Hardware-specific |
| Iso-memory/max-throughput | Skipped | Wo-only keeps dense KV and target hardware differs |

## 4. Solver correctness、capacity inclusion 与 convergence

### 4.1 Capacity inclusion kill gate

36/36 layers 通过 capacity inclusion：rank-2048 shared LR-AllReduce 在每层 heldout objective 上都优于 rank-512/source C1。

| Method | Mean heldout relative MSE |
|---|---:|
| Capacity LR-AllReduce, rank 2048 | 0.01664296 |
| C1-AllGather, 4 × rank 512 | 0.04598326 |
| Wire LR-AllReduce, rank 1024 | 0.08651256 |

因此理论关系按预期成立。Capacity LR 相比 C1 使用约 2× ideal-ring traffic；它是表示能力上界控制，不是 equal-wire competitor。

在 equal-wire 条件下，C1 的平均 layer-output MSE 比 strong LR-AllReduce 低约 46.9%。这支持 private direct-sum representation 在当前模型中的 quality-per-byte 优势，但不构成普适函数类支配关系。

### 4.2 Exact encoder solve

早期 CG pilot 被 FP64 exact two-sided Cholesky update 取代。最终 100-sweep run 的最大 accepted encoder residual 为 `1.45e-12`，所有 layer 的最大 backtrack 数为 0。这移除了 CG iteration/tolerance 和 encoder damping 两类额外超参数。

### 4.3 ALS tail

- 100-sweep heldout selection：10/36 layers 选择 sweep 100；平均 selected sweep 为 63.86。
- 20-sweep mean heldout MSE：`0.04599003`。
- 100-sweep mean heldout MSE：`0.04598326`。
- 20→100 sweeps 的 aggregate relative improvement 只有 `0.0147%`。

因此部分层在 objective 上仍有可测的长尾下降，但 20 sweeps 后的整体收益已非常小。不能把“部分层命中 100-sweep cap”描述成严格数学收敛；更准确的结论是 aggregate quality 已基本饱和。

## 5. Uniform-rank quality Pareto

所有行使用相同 C4 covariance、uniform source rank、dense V/KV，并使 C1 与 wire LR 的 ideal-ring bytes 完全一致。

| C1 source rank | Retained source width | Wire LR rank | C1 PPL | Local-objective C1 PPL | Wire LR PPL | Capacity LR PPL |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 50.0% | 1024 | 7.281949 | 7.251400 | 7.546591 | 7.039705 |
| 640 | 62.5% | 1280 | 7.120592 | 7.119200 | 7.227802 | 7.013371 |
| 768 | 75.0% | 1536 | 7.048183 | 7.044131 | 7.115327 | 7.001656 |
| 896 | 87.5% | 1792 | 7.009468 | 7.010228 | 7.074950 | 7.002845 |
| 1024 | 100.0% | 2048 | 7.002026 | 7.002614 | 7.039705 | 7.002357 |

主要观察：

- C1 在五个 equal-wire ranks 上都优于 wire LR。
- C1 的优势随 rank 增大而缩小：相对 PPL 优势从 rank-512 的 3.63% 降至 rank-1024 的 0.54%。
- Capacity LR 基本沿 dense quality 上界运行，符合更大通信预算和函数类 inclusion。
- Joint layer-output objective 并不保证更好的 PPL；rank-512/640/768 上 independent-local PPL 反而略好。

## 6. Private-source mechanism

### 6.1 Synthetic controlled-angle sanity test

确定性 CPU/FP64 toy 使用 4 sources、source rank 4、C1 aggregate rank 16、equal-wire LR rank 8，并将 target source spaces 的角度控制为 `0°, 15°, 30°, 45°, 60°, 75°, 90°`。

- 0° 时 shared LR 与 C1 都达到数值零误差。
- 角度增大时 LR heldout MSE 单调增加。
- 90° 时 LR MSE 为 0.5，而 C1 保持约 `5e-31`。
- Diversity/C1-advantage：Pearson `r=0.9673`，Spearman `ρ=1.0`。

这验证 solver、budget accounting 和概念预测，但不替代真实模型证据。

### 6.2 Real Qwen3-8B source subspaces

每层对四个 `[512,4096]` source decoder row spaces 做 principal-angle 和 union-rank 分析。

| Metric across 36 layers | Mean | Min | Max |
|---|---:|---:|---:|
| Mean pairwise angle | 70.443° | 67.648° | 72.654° |
| Mean projection overlap | 0.1497 | 0.1183 | 0.1847 |
| Mean normalized chordal `d²` | 0.8503 | 0.8153 | 0.8817 |
| Union rank for 95% energy | 1586.6 | 1533 | 1648 |
| Union rank for 99% energy | 1882.8 | 1854 | 1920 |

四个 private spaces 的 aggregate row count 为 2048，而平均 95%-energy union rank 约 1587，显著大于 equal-wire LR rank 1024。这与 shared rank-1024 难以覆盖 private union 的观察一致。

Layerwise diversity 与 equal-wire C1 advantage 呈中等、统计显著的正相关：

| Diversity metric | Advantage metric | Pearson | Spearman |
|---|---|---:|---:|
| Mean angle | heldout relative MSE | 0.362 | 0.418 |
| Mean angle | heldout relative L2 | 0.377 | 0.520 |
| Chordal `d²` | heldout relative MSE | 0.370 | 0.425 |
| Union rank 95 | heldout relative MSE | 0.403 | 0.433 |

这支持 source diversity 是 C1 equal-wire 优势的一个机制，但相关性并不接近 1，不能声称它解释全部 layer variation。

### 6.3 Source permutation

对每层 exhaustive 测试 23 个 non-identity permutations，并将配对的 `(E_p,D_p)` 一起重分配到其他 logical sources，不做 refit。

- 828/828 wrong mappings 都比原始 mapping 差。
- 原始 mapping 在 36/36 layers 严格最优。
- Mean MSE：原始 `0.0459833`，permuted `1.30983`。
- Median permutation/original ratio：`29.78×`。

该结果说明 source assignment 是实质性的，但它只是 mechanism sanity check，不单独证明方法 novelty。

### 6.4 Latent-basis invariance

对每个 source 独立采样正交 `Q_p`，应用 `E'_p=E_pQ_p`、`D'_p=Q_p^TD_p`。

- 最大 FP32 product relative L2：`1.05e-6`。
- 最大 FP32 heldout-output relative L2：`1.28e-6`。
- BF16 factor requantization 后 probe relative L2：median `0.00409`，maximum `0.00455`。

表示在 FP32 下按预期对 latent rotation 不变；BF16 差异来自 requantization 和 GEMM operation ordering，不能解释为新的 representational error。

## 7. Joint-final objective 与 cross-source covariance

### 7.1 Joint-final versus independent-local

在 source rank 512、相同 covariance 和 BF16 factors 下：

| Method | Mean local heldout MSE | Mean final heldout MSE |
|---|---:|---:|
| Independent-local fit | 0.0525973 | 0.0485288 |
| Joint-final fit | 0.0692266 | 0.0459900 |

- Independent-local 在 36/36 layers 的 local objective 上更好。
- Joint-final 在 36/36 layers 的 summed final-output objective 上更好。
- Joint-final 将 mean final heldout MSE 相对降低约 5.23%。

但 layer objective 的改进没有稳定传递到 LM quality：rank-512 WikiText PPL 为 joint `7.28195`、independent-local `7.25140`。因此目前支持“joint objective 改善它所优化的 summed layer output”，不支持“joint objective 必然改善 PPL”。

### 7.2 Cross-source covariance

固定 joint-s20 encoders，只改变 decoder normal equations：full covariance 保留 `U_p^TU_q`，block-diagonal arm 将所有 `p≠q` blocks 置零。

| Decoder covariance | Mean final heldout MSE | WikiText-2 PPL |
|---|---:|---:|
| Full covariance | 0.0459879 | 7.282146 |
| Block diagonal | 0.0489504 | 7.286600 |

Full covariance 在 36/36 layers 获得更低 final heldout MSE，平均相对下降 6.45%；但 PPL 只改善 0.061%。这证明 joint decoder 确实利用了 source interactions，同时也显示 layer-MSE improvement 到 PPL 的 transfer 很弱。

## 8. Whole-model quality headline

最终 s100 factors 的质量结果：

| Arm | WikiText-2 PPL | Δ vs dense | Heldout C4 PPL | Δ vs dense |
|---|---:|---:|---:|---:|
| Dense | 7.002509 | 0% | 9.168594 | 0% |
| C1-AllGather | 7.279747 | +3.959% | 9.320145 | +1.653% |
| Wire LR-AllReduce | 7.548429 | +7.796% | 9.736591 | +6.195% |
| Capacity LR-AllReduce | 7.039787 | +0.532% | 9.209007 | +0.441% |

结论分为两部分：

1. Equal wire：C1 明显优于 strong LR-AllReduce，支持 private source specialization 的 quality-per-byte 价值。
2. Equal representation capacity：shared LR 优于 C1，符合理论 inclusion；C1 的优势来自预算如何被分配，而不是更大的函数类。

## 9. L40S/TP4 system experiments

### 9.1 Collective microbenchmark

BF16、TP4 下对 rank-512/source C1 AllGather 和 equal-wire rank-1024 LR AllReduce 测量：

| Rows | C1 packed ms | LR AllReduce ms | C1 packed vs LR |
|---:|---:|---:|---:|
| 1 | 0.016384 | 0.027648 | -40.74% |
| 8 | 0.016384 | 0.030720 | -46.67% |
| 32 | 0.035840 | 0.030720 | +16.67% |
| 64 | 0.040960 | 0.048656 | -15.82% |
| 512 | 0.097280 | 0.099840 | -2.56% |
| 2048 | 0.307200 | 0.295936 | +3.81% |
| 8192 | 1.176576 | 1.142320 | +3.00% |
| 32768 | 4.709888 | 4.500992 | +4.64% |

Packing overhead 从 very-small-message 的 23% 降至 large-message 的约 0.2–1.9%。Equal logical bytes 不等于 equal measured time；不同 rows 下 collective winner 会改变。

### 9.2 Decoder GEMM microbenchmark

主 rank 下 Dense/Wire-LR decoder `K=1024`，C1 global decoder `K=2048`。C1 decoder 的单 GPU latency 比 wire LR 高：

| Rows | C1 decoder ms | Wire-LR decoder ms | C1 overhead |
|---:|---:|---:|---:|
| 1 | 0.013312 | 0.010240 | +30.0% |
| 8 | 0.017408 | 0.013312 | +30.8% |
| 32 | 0.021504 | 0.013312 | +61.5% |
| 64 | 0.027648 | 0.013312 | +107.7% |
| 128 | 0.018432 | 0.013312 | +38.5% |
| 256 | 0.025600 | 0.015360 | +66.7% |

这证实 C1 用更大的 replicated decoder compute 换取更小的 wire；collective 节省不能脱离 decoder cost 单独解释端到端性能。

### 9.3 Eager 高压力 workload

相同 Wo-only factors、dense V/KV、4×L40S TP4：

- 连续 decode `B1/B8/B32/B64 × 128`、`B32×2048`、`B64×1024`：C1 throughput 比 dense 高 7.73–8.86%；wire LR 只高 0.58–0.94%。
- Prefill `B1×2048`、`B8×2048`、`B32×1024`、`B64×512`：C1 prefill throughput 高 20.81–24.97%；wire LR 高 22.49–27.37%，即 LR 在 isolated prefill 上比 C1 快 1.31–1.88%。
- Prefill + 32 output tokens：C1 相比 dense 降低 8.36–14.32% latency，相比 wire LR 降低 1.31–5.93%。
- C1 比 dense 多使用约 0.48–0.72 GiB peak allocated memory/rank，因为这条 Wo-only 路径保留 dense KV，同时保存 factor bank 和 packed workspace。

这些是 eager runtime 结果，不能重标为 CUDA Graph production evidence。

### 9.4 Five-arm CUDA Graph appendix

协议为 fixed context 512、单个 decode token、deterministic zero prefix cache、warmup 10、timed replay 50。Graph capture/build time不计入 latency。硬件为 4×L40S，GPU 间 topology 在 `nvidia-smi topo -m` 中均显示 `SYS`，不是 NVLink。

| Batch | Dense ms | Wire LR ms | Capacity LR ms | C1 AG ms | C1 local-AR ms | C1 AG speedup vs dense |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.217 | 9.371 | 9.909 | 9.612 | 9.266 | 0.959× |
| 8 | 10.536 | 10.633 | 11.201 | 11.004 | 10.584 | 0.957× |
| 32 | 14.443 | 13.543 | 15.025 | 14.672 | 14.444 | 0.984× |
| 64 | 16.521 | 16.085 | 16.951 | 16.536 | 16.552 | 0.999× |

`C1 AG` 和 `C1 local-AR` 使用完全相同的近似函数，因此它们之间只改变 collective boundary。结果显示，75% ideal communication reduction 没有在该 CUDA Graph 单步协议中形成端到端加速；global decoder 和 packed-collective 固定开销抵消了通信收益。

这与 eager 高压力结果并不等价。合理报告方式是：

- eager high-pressure 结果作为当前完整 request benchmark；
- fixed-step CUDA Graph 作为附录消融；
- 不从任一 L40S/TP4 结果外推 TP8/A5000 或其他 topology；
- 在 target hardware 上完成同 workload、同 graph settings 的 benchmark 之前，不声称 production-universal speedup。

## 10. 支持的结论

1. Capacity-matched LR 的函数类 inclusion 在 36/36 layers 得到验证，strong baseline 没有被人为削弱。
2. Equal ideal wire 下，C1 在 layer error 和 WikiText PPL 上持续优于 strong LR-AllReduce。
3. Qwen3-8B 的 TP sources 存在显著不同的 decoder subspaces；shared rank-1024 小于典型 95%-energy union rank。
4. Layerwise source diversity 与 C1 equal-wire advantage 存在中等正相关。
5. Source mapping 很重要；latent coordinate basis 本身不重要。
6. Joint-final objective 和 cross-source covariance 都改善其直接优化的 summed layer-output objective。
7. C1 的通信节省伴随更大的 replicated decoder cost，系统收益依赖 rows、runtime、graph 和硬件 topology。

## 11. 不支持或需要限制的结论

1. **不支持 C1 在 equal capacity 下优于 LR。** Capacity LR 明显更好，且理论上应如此。
2. **不支持 joint-final objective 必然改善 PPL。** Rank-512 上 independent-local PPL 更好。
3. **不支持 cross-source covariance 带来大幅 LM-quality 收益。** Layer MSE 改善 6.45%，PPL 只改善 0.061%。
4. **不支持 CUDA Graph 下已有稳定 speedup。** Fixed-step Graph 中 C1 为 `0.957–0.999×` dense。
5. **不支持普适硬件速度结论。** 当前 system data 只来自 L40S/TP4/SYS topology。
6. **不支持 KV-cache compression、低维 AV 或 folded-V claim。** Wo-only 分支明确保持 dense V/KV。
7. **不支持把 logical-byte reduction 当成 measured-time reduction。** Collective microbenchmark 存在 message-size-dependent crossover。

## 12. Deferred、skipped 与 unresolved

- High-pressure CUDA-event/Nsight latency decomposition：推迟到目标 TP8/A5000 系统。
- Batch×context phase map：推迟到目标硬件。
- TP8 sensitivity：未运行；TP8 必须重新定义 ownership 并重新拟合 factors，不能复用 TP4 conclusion。
- Calibration size、seed、domain robustness：本轮按决策不继续。
- Downstream tasks：本 Wo-only factor bank 尚未形成统一 downstream suite。
- Measured-time-matched rank selection：已经有 collective rank/row sweep，但尚未针对每个 measured-time-matched pair 重新拟合并比较 whole-model quality。
- Iso-memory/max-throughput：跳过；Wo-only 没有 KV memory advantage，而且当前硬件不是目标部署硬件。
- Eager 与 CUDA Graph 结果反转的 component-level 原因尚未用 target-hardware latency breakdown 闭合。

## 13. Reproducibility pointers

每个 linked `summary.md` 都记录了对应的准确命令；每个 `results.json` 保存 model/config hash、method、protocol、environment 和源 artifact。主要实现入口为：

- [`basisserve/core/tp_source_wo_c1.py`](../basisserve/core/tp_source_wo_c1.py)
- [`basisserve/core/tp_source_wo_fit.py`](../basisserve/core/tp_source_wo_fit.py)
- [`basisserve/core/qwen3_8b_wo_tp4.py`](../basisserve/core/qwen3_8b_wo_tp4.py)
- [`evaluation/benchmark_qwen3_8b_wo_tp4.py`](../evaluation/benchmark_qwen3_8b_wo_tp4.py)
- [`evaluation/benchmark_qwen3_8b_wo_cuda_graph.py`](../evaluation/benchmark_qwen3_8b_wo_cuda_graph.py)

关键 whole-model quality 命令为：

```bash
python evaluation/eval_qwen3_8b_wo_c1_lr_ar_quality.py \
  --model <Qwen3-8B-Base snapshot> \
  --phase1-dir results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 \
  --c4-windows results/calibration/qwen3_8b_c4_validation_128x2048/windows.safetensors \
  --output-dir results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality \
  --wikitext-batch-size 1 --c4-batch-size 1 \
  --model-dtype bfloat16 --attn-implementation sdpa \
  --device cuda:0 --torch-num-threads 4 --local-files-only
```

关键 eager runtime 命令族为：

```bash
torchrun --standalone --nproc-per-node=4 \
  evaluation/benchmark_qwen3_8b_wo_tp4.py \
  --workload <decode|prefill> --arm <dense|wo_c1_ag|wo_lr_ar_wire> \
  --model <Qwen3-8B-Base snapshot> \
  --phase1-dir results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 \
  --quality-results results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/results.json \
  <workload-specific configurations and output path>
```

## 14. 最终判断

这条实验主线已经较清楚地分离了三个问题：

- **Representation:** capacity LR 是更强的函数类；equal wire C1 在 Qwen3-8B 上获得更好的 quality-per-byte。
- **Mechanism:** source spaces 确实多样，diversity 与 C1 advantage 有中等相关；joint/cross-source objective 改善 layer target，但不能保证 PPL 同步改善。
- **System:** C1 在当前 eager L40S workload 中有加速，但 CUDA Graph 单步未复现；因此 speed claim 必须限制到已测协议，并等待目标 TP8/A5000 系统重新验证。

当前最稳健的核心贡献是 **equal-wire representation quality 和 source-private mechanism evidence**，而不是跨硬件的 production speedup claim。
