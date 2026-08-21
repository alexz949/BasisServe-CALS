# Llama-2-7B MHA 与 Qwen3-8B GQA：PaLU 与 C1 V/Wo Joint 对比

日期：2026-08-17

## 结论

在 Llama-2-7B MHA、K 保持 dense、V-cache 保留 75% 的 matched budget 下，C1 joint 同时优于 PaLU M-LRD 和 G-LRD4：

| 方法 | V latent 结构 | 每层 V-cache width | 每个 query head 的 V-attention width | 相对 V-attention FLOPs | WikiText2 PPL |
|---|---:|---:|---:|---:|---:|
| Dense | 32 × 128 | 4096 | 128 | 1.333× | 5.472056328 |
| PaLU M-LRD | 32 × 96 | 3072 | 96 | 1× | 6.946421792 |
| PaLU G-LRD4 | 8 × 384，四 heads 共享 | 3072 | 384 | 4× | 6.400272684 |
| **C1 joint** | **32 × 96** | **3072** | **96** | **1×** | **6.231372767** |

因此，在相同 V-cache width 3072、相同总 KV-cache reduction 12.5% 下：

- C1 比 PaLU M-LRD 改善 `0.715049025` PPL。
- C1 比 PaLU G-LRD4 改善 `0.168899917` PPL。
- C1 的理论 V-attention 计算量只有 G-LRD4 的 `1/4`。
- 这里的 `1/4` 只指每个 query head 的 Value-attention matmul，不代表整个 Transformer layer 或整个模型的 FLOPs 降为 `1/4`。

这个结果说明：在 MHA 上，与其通过 G-LRD4 让每个 query head 消费一个 384-wide 的共享 latent，不如保留 head-private V96，并把优化自由度放到 V writer 和新的 Wo decoder 的联合拟合中。

## Qwen3-8B Base：GQA 结果

Qwen3-8B Base 有 32 个 query heads、8 个 physical KV heads，head dimension 128。以下 compressed arms 同样保持 K dense，并将每层 V-cache 从

\[
8\times128=1024
\]

压缩到总 width 768，即 V 保留 75%、总 KV-cache reduction 12.5%。所有方法均使用 uniform rank。

### PaLU：M-LRD、G-LRD2 与 G-LRD4

PaLU checkpoints 使用 BF16 模型和 CPU FP64 activation-whitened SVD。WikiText2 test 使用完整 146 × 2048 windows。

| 方法 | Uniform V latent 结构 | 每层 V-cache width | PPL | 相对本协议 dense |
|---|---:|---:|---:|---:|
| Dense | 8 × 128 | 1024 | 6.998721290 | — |
| PaLU M-LRD | 8 × 96 | 768 | 9.211224584 | +2.212503294 |
| PaLU G-LRD2 | 4 × 192 | 768 | 9.124840272 | +2.126118982 |
| PaLU G-LRD4 | 2 × 384 | 768 | 9.126794526 | +2.128073236 |

G-LRD2 相比 M-LRD 只改善 `0.086384312` PPL；继续从 G2 扩大到 G4 反而退化 `0.001954254` PPL，基本完全没有额外 signal。

这与 Llama-2 MHA 形成鲜明对比：Llama 的 G-LRD4 比 M-LRD 改善 `0.546149108` PPL，而 Qwen3 GQA 的 G2/G4 几乎相同。一个合理解释是：Qwen3 已经通过 GQA 将 32 个 query heads 映射到仅 8 个 physical KV heads，KV redundancy 已经被结构性压缩；继续合并 physical V heads 的收益远小于 MHA。

### Output-aware routed C1 与 Pair C2

旧的 Qwen3 routed joint 实验使用同一个 8B Base checkpoint、BF16 forward、FP32 loss accumulation，以及 146 × 2048 WikiText2 test windows。该 evaluator 的 dense baseline 是 `7.003384583`，与上面的 PaLU evaluator 相差 `0.004663294`，因此这里单独相对自己的 dense 报告。

| 方法 | Equal-cache 结构 | 每层 V-cache width | PPL | 相对本协议 dense |
|---|---:|---:|---:|---:|
| Dense | 8 × 128 | 1024 | 7.003384583 | — |
| Routed C1 | 8 个独立 rank-96 physical V caches | 768 | 7.372747645 | +0.369363061 |
| Pair C2 | 4 个 rank-192 logical-pair caches；每 query head 有 rank-96 private wire | 768 | **7.343676206** | **+0.340291622** |

Qwen3 C1/C2 与 Llama C1 共享同一个核心思想：不要求先恢复 dense V，而是将 V coordinate encoder 与 consumer/output decoder 针对实际 attention-output reconstruction 联合拟合。

Qwen3 C2 在相同 cache width 下比 C1 再改善 `0.029071439` PPL。其 local report split 上：

- C1 mean relative output MSE：`0.063289564`。
- C2 mean relative output MSE：`0.059762716`。
- C2 relative improvement：`5.573%`，36 层中 34 层优于 C1。

C2 的 placement accounting 目前仍是 logical-only；尚未实现或验证真实 distributed runtime、HBM saving 和 pair-cache kernel。因此当前最坚实的 Qwen 结论是质量结果，而不是系统 speedup claim。

### 跨架构结论

综合 Llama-2 MHA 和 Qwen3 GQA：

1. **MHA 中跨 head redundancy 很强。** PaLU G4 显著优于 M-LRD，但代价是每个 query head 的 Value-attention width 从 96 增至 384。
2. **GQA 中继续合并 physical KV heads 的收益很弱。** Qwen3 PaLU G2 与 G4 几乎完全相同。
3. **V/Wo joint optimization 在两种结构上都比单纯 V reconstruction 更有效。** Qwen3 C1/C2 只比 dense 高约 0.34–0.37 PPL；Llama C1 也在同 cache budget 下超过 PaLU G4。
4. **最有吸引力的部署点仍是 private V96 + output-aware decoder。** 它保持最小的 per-query Value-attention width，并把模型质量恢复放到 consumer/output side，而不是扩大 runtime latent。

需要注意：Qwen3 PaLU 与 C1/C2 来自两个 evaluator protocol，dense baseline 相差约 `0.0047`；上面的算法比较主要依据各自相对 dense 的退化以及非常大的质量差距，不把两组 raw PPL 当作 bitwise matched evaluator 结果。

## Llama-2 matched-budget 设置

- 模型：`meta-llama/Llama-2-7b-hf`，32-layer MHA，32 heads，head dimension 128。
- K：所有 compressed arms 都保持 dense，width 4096/layer。
- V：所有 compressed arms 都保留总 width 3072/layer，即原始 V 的 75%。
- 总 KV：从 8192/layer 降到 7168/layer，retained ratio 87.5%，reduction 12.5%。
- TP accounting：TP=8 时每 rank 有 384 个 V latent elements。
- 模型 forward：FP16，SDPA。
- PPL loss accumulation：FP32。
- 数据集：WikiText2 test，sequence length 2048，166 chunks，339,802 evaluated tokens。
- 不运行 MCQ。

兼容的 dense FP32-loss baseline 来自：

`results/attention_o_proj_collective_ppl/wikitext2_tp8_r1536_fp32loss_20260809/llama2_7b_dense.json`

它使用同一个 Llama checkpoint、WikiText2 test、166 × 2048 chunks 和 FP32 loss accumulation。

## Llama-2 的三种压缩方法

### PaLU M-LRD

每个 V head 独立做 activation-whitened low-rank decomposition：

\[
32\times V128 \longrightarrow 32\times V96.
\]

每个 query head 只处理自己的 96-dimensional latent。PaLU factorization 使用 C4 activation whitening 和 CPU FP64 SVD。

### PaLU G-LRD4

每四个 V heads 形成一个 group：

\[
32\times V128 \longrightarrow 8\times V384.
\]

总 cache width 仍为 \(8\times384=3072\)，但一个 group 中的每个 query head 都需要 attend 384-dimensional group latent。因此，在相同 cache budget 下，Value-attention matmul 是 V96/head 的四倍。

### C1：V96 writer 与 Wo decoder 联合优化

对第 \(h\) 个 attention head，记 dense Value projection 为

\[
v_h=XW_{V,h}^{\top},
\]

attention 权重为 \(P_h\)，进入 dense `o_proj` 前的 head output 为

\[
u_h=P_hv_h\in\mathbb{R}^{T\times128}.
\]

将 \(W_o^\top\) 按 head 切成

\[
O_h\in\mathbb{R}^{128\times4096},
\]

dense attention-layer output 是

\[
y=\sum_{h=1}^{32}u_hO_h.
\]

C1 为每个 head 学习

\[
A_h\in\mathbb{R}^{128\times96},\qquad
D_h\in\mathbb{R}^{96\times4096},
\]

并直接拟合

\[
\hat y=\sum_h (u_hA_h)D_h.
\]

因为 attention 对 V 是线性的，且 Q/K 不依赖 V，下面的交换是精确的：

\[
(P_hv_h)A_h=P_h(v_hA_h).
\]

所以 \(A_h\) 可以在部署时折叠进 V projection：

\[
W'_{V,h}=A_h^\top W_{V,h}
\in\mathbb{R}^{96\times4096}.
\]

新的 Wo block 则直接是

\[
W'_{o,h}=D_h^\top
\in\mathbb{R}^{4096\times96}.
\]

因此实际 latent 路径为

\[
X
\xrightarrow{W'_{V,h}}
z_h\in\mathbb{R}^{T\times96}
\xrightarrow{P_h}
\bar z_h
\xrightarrow{D_h}
\mathbb{R}^{T\times4096},
\]

然后对 32 个 heads 的输出求和。

## C1 的 joint objective

C1 不最小化 V reconstruction，而是最小化完整 attention output reconstruction：

\[
\mathcal L(A,D)
=
\mathbb E_t
\left\|
\sum_h u_{h,t}O_h
-
\sum_h u_{h,t}A_hD_h
\right\|_2^2.
\]

定义跨 head covariance blocks：

\[
C_{hk}=\mathbb E_t[u_{h,t}^{\top}u_{k,t}]
\in\mathbb{R}^{128\times128}.
\]

目标展开为

\[
\mathcal L
=
\sum_{h,k}
\operatorname{tr}
\left[
(O_h-A_hD_h)^\top
C_{hk}
(O_k-A_kD_k)
\right].
\]

由于保留了 \(h\neq k\) 的 covariance，优化器可以利用真实 calibration distribution 上的跨 head error cancellation。

### 固定 A：闭式联合求 D

定义

\[
B_h=\sum_kC_{hk}O_k,
\qquad
H_{hk}=A_h^\top C_{hk}A_k.
\]

所有 decoder 满足 coupled normal equation：

\[
\sum_kH_{hk}D_k=A_h^\top B_h.
\]

将 32 个 heads 堆叠后，这是一个 \(3072\times3072\) 的 FP64 linear solve，右端宽度为 4096。实现使用 full-layer Cholesky solve，不使用 SGD 或 parameter backprop。

### 固定 D：更新 A

第 \(h\) 个 encoder 的 half-gradient 为

\[
G_h
=
\sum_kC_{hk}A_kD_kD_h^\top
-B_hD_h^\top.
\]

实现使用 FP64 Hessian-vector products、固定 16-step CG 和 quadratic backtracking 更新各个 \(A_h\)，然后重新闭式求解全部 \(D_h\)。最多运行五轮 alternating sweeps。

### 初始化与 held-out selection

- Snapshot：64 个 C4 contexts，每 context 2048 tokens，固定采样 128 个 attention-output positions。
- Fit：前 48 contexts，共 6144 positions。
- Held-out checkpoint selection：后 16 contexts，共 2048 positions。
- Snapshot 行严格按 window-major 排列，因此 48/16 split 是 context-disjoint。
- 每个 solver boundary 都在 held-out contexts 上评估，选择最早的最低 held-out output MSE checkpoint。
- 最终 factor 以 FP16 保存；FP64 只用于 covariance、linear solve 和 encoder optimization。

本次 32 层的 selection：

- 28 层选择 sweep 1，4 层选择 sweep 2。
- 31 层选择 `after_encoder`，1 层选择 `after_redecoder`。
- 平均 fit factor-dtype relative MSE：`0.0134582872`。
- 平均 held-out factor-dtype relative MSE：`0.176028554`。

fit/held-out gap 很大，说明 full-layer cross-head cancellation 存在明显 context overfit 风险。Held-out early selection 抑制了继续 sweep 的过拟合，但并没有消除 context-shift 风险。

## 为什么 C1 能超过 PaLU

PaLU 的 decoder 来自 V reconstruction factor 与原始 Wo 的组合，因此受到“先恢复 dense V，再通过原始 Wo”的结构约束。

C1 的 \(D_h\in\mathbb{R}^{96\times4096}\) 是自由的，直接把 compressed attention result 映射回 residual hidden space。它可以联合选择：

1. 每个 V head 应保留哪 96 个 coordinates；
2. 每个 latent coordinate 应如何直接写入 4096-dimensional output；
3. 不同 heads 的 output errors 如何在真实 activation distribution 上互相抵消。

这使 C1 能避免显式恢复原始 128-dimensional V head，也不需要像 G-LRD4 一样让每个 query head 消费 384-dimensional latent。

## Artifact 级审计

结果完成后进行了独立只读审计：

- 抽查 layer 0、1、15、31，snapshot 中的 Wo 与原始 Llama checkpoint bitwise identical。
- 所有抽查层的 \(A_h\) shape 为 `[32,128,96]`，每个 \(A_h\) 的实际 matrix rank 为 96。
- Folded V writer 的实际 rank 也是 96/head。
- Padded reference path 中每个 head 被丢弃的后 32 个 coordinates 严格全零。
- 独立构造的 unpadded latent path 与 padded-folded path 的 relative L2 error 约为 `5e-7`。
- 未发现 rank128 偷漏、head routing 错误、transpose 错误或 WikiText2 test leakage。

旧 Llama `head_ag` PPL `5.881128750` 不属于同一类 V-cache compression。它的 basis shape 是 `[32,4096,96]`，每个 logical group 都能读取完整 4096-dimensional post-attention vector；它混合所有 heads，不能折叠成 attention 前的 per-head V96 writer，因此不能作为本实验的 matched V-cache baseline。

## 运行时间与资源

Slurm partition：`athena-small`，node4，RTX A5000。

| 作业 | Job ID | 状态 | Slurm elapsed | Max RSS |
|---|---:|---|---:|---:|
| Whitening + PaLU M/G4 | 161323 | COMPLETED, exit 0 | 00:09:29 | 17,921,096 KiB |
| C1 fit + merge + PPL | 161324 | COMPLETED, exit 0 | 00:11:23 | 6,115,684 KiB |

分项时间：

- Whitening：`160.7531 s`，包含 32 个 4096×4096 FP32 Cholesky artifact 的写出。
- PaLU M-LRD installation：`143.8707 s`；该 arm model-load + installation + PPL 总计 `219.7281 s`。
- PaLU G-LRD4 installation：`289.4199 s`；该 arm model-load + installation + PPL 总计 `364.9715 s`。
- C1 installation：`2.4226 s`；model-load + installation + PPL 总计 `78.1321 s`，此前另有 C1 factor fitting。

PaLU M/G4 两路 evaluation 在 whitening 完成后并行运行；C1 的四个 layer shards 也在四张 GPU 上并行拟合。

## 限制与下一步验证

当前 PPL evaluation 使用 `use_cache=False` 和 stock Hugging Face attention。为了保持 stock kernel，C1 将每个 96-wide latent 写入 128-wide head 的前 96 个 coordinates，后 32 个 coordinates 置零；Wo 只读取前 96 个 coordinates。

这验证了 rank-96 latent function 的质量，但没有实际测量 96-wide KV-cache memory、decode latency 或真实 kernel FLOPs。当前表格中的 Value-attention FLOPs 是可部署 latent formulation 的理论计算量，不是本次 padded evaluator 的实测 runtime。

最关键的下一步是：

1. 实现真实 `[batch, heads, sequence, 96]` V cache；
2. 使用 `use_cache=True` 做 prefill + autoregressive decode；
3. 将真实 latent-cache logits 与当前 padded-folded reference 逐 token 对比；
4. 验证 cache bytes、Value-attention kernel time、端到端 tokens/s 和 peak memory。

另有两个非算法性复现警告：

- 当前 `capture_attention_o_proj_ppl_snapshots.py` 调用 `_OProjCapture(hidden_size=...)`，但当前 helper 构造函数参数名是 `input_width`；重新 capture 前需要修复该 API drift。已有 snapshot 来自旧 commit，因此不影响本次结果。
- Slurm shell 继承的 `CONDA_DEFAULT_ENV` 是 `base`，导致结果 JSON 的环境字段显示 `base`；实际运行解释器明确使用 `/home/lz299/miniconda3/envs/lowrankarena/bin/python` 和同环境下的 `torchrun`。

## Llama-2 代码与结果文件

- C1 fitter：`evaluation/fit_llama2_mha_c1_joint.py`
- Matched evaluator：`evaluation/eval_llama2_mha_v25_comparison.py`
- Focused tests：`tests/test_llama2_mha_v25_comparison.py`
- C1 factors：`results/llama2_mha_v25/c1_joint_v96_c4_48_16/`
- Whitening：`results/llama2_mha_v25/whitening_c4_n64_s2048/`
- PaLU M result：`results/llama2_mha_v25/ppl_palu_m_v96.json`
- PaLU G4 result：`results/llama2_mha_v25/ppl_palu_g4_v384.json`
- C1 result：`results/llama2_mha_v25/ppl_c1_joint.json`

## Qwen3 代码与结果文件

- PaLU dense result：`results/q3base_dense_bf16_full146.json`
- PaLU M-LRD V96 result：`results/q3base_palu_v96_bf16_cpufp64_full146.json`
- PaLU G-LRD2 rank-192 result：`results/q3base_palu_vg2_r192_bf16_cpufp64_full146.json`
- PaLU G-LRD4 rank-384 result：`results/q3base_palu_vg4_r384_bf16_cpufp64_full146.json`
- C1/C2 evaluator：`evaluation/eval_qwen3_base_pair_gld_ppl.py`
- C1/C2 factor analysis：`scripts/analyze_qwen3_base_pair_gld.py`
- C1/C2 factors and fit summary：`results/q3base_pairgld/c1v96_c2v192_w96_globalcv_all36_f192_v32_t16/`
- C1/C2 protocol dense result：`results/q3base_pairgld/ppl_all36_f192/dense.json`
- Routed C1 result：`results/q3base_pairgld/ppl_all36_f192/c1.json`
- Pair C2 result：`results/q3base_pairgld/ppl_all36_f192/c2.json`

Qwen3 C1/C2 的历史 evaluation 命令可由对应 JSON 的 `command` 字段直接复现；将 `ARM` 依次设为 `dense`、`c1`、`c2`，其中 compressed arms 使用上面的 factor directory。该组 factor analysis 历史记录使用 `lowrank` 环境；evaluation 使用 BF16 model forward 和 FP32 loss accumulation。

## Llama-2 主要命令

所有 Python 命令均使用 `lowrankarena` 环境中的解释器。

### Whitening

```bash
/home/lz299/miniconda3/envs/lowrankarena/bin/torchrun \
  --standalone --nproc_per_node=4 \
  evaluation/eval_llama2_mha_v25_comparison.py prepare-whitening \
  --model /home/lz299/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --windows results/cache/llama2_7b_c4_s2048_n64_seed20260901/windows.safetensors \
  --output-dir results/llama2_mha_v25/whitening_c4_n64_s2048 \
  --torch-num-threads 2
```

### C1 fit

下面的命令以 `INDEX=0,1,2,3` 在四张 GPU 上运行四个 layer shards：

```bash
/home/lz299/miniconda3/envs/lowrankarena/bin/python \
  evaluation/fit_llama2_mha_c1_joint.py fit-shard \
  --snapshot-dir results/llama2_7b_mha_o_proj_ppl_snapshots/c4_n64_p128_all \
  --output-dir results/llama2_mha_v25/c1_joint_v96_c4_48_16 \
  --layers all --fit-windows 48 --validation-windows 16 \
  --cache-rank 96 --work-dtype float64 --factor-dtype float16 \
  --covariance-damping 1e-7 --encoder-sweeps 5 \
  --minimum-encoder-sweeps 2 --encoder-relative-tolerance 1e-6 \
  --encoder-patience 2 --encoder-relative-damping 1e-8 \
  --encoder-cg-fixed-iterations 16 --maximum-backtracks 10 \
  --layer-shard-index INDEX --layer-shard-count 4 \
  --device cuda:0 --torch-num-threads 2 --resume
```

Merge：

```bash
/home/lz299/miniconda3/envs/lowrankarena/bin/python \
  evaluation/fit_llama2_mha_c1_joint.py merge \
  --output-dir results/llama2_mha_v25/c1_joint_v96_c4_48_16 \
  --layers all
```

### Evaluation

```bash
/home/lz299/miniconda3/envs/lowrankarena/bin/python \
  evaluation/eval_llama2_mha_v25_comparison.py evaluate \
  --model /home/lz299/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --arm ARM \
  --dataset wikitext2 --split test --seqlen 2048 --batch-size 1 \
  --device cuda:0 --torch-num-threads 4 \
  --output-json OUTPUT_JSON
```

其中：

- `ARM=palu_m` 或 `palu_g4` 时传入 `--whitening-dir results/llama2_mha_v25/whitening_c4_n64_s2048`。
- `ARM=c1_joint` 时传入 `--factor-dir results/llama2_mha_v25/c1_joint_v96_c4_48_16`。

### Tests

```bash
/home/lz299/miniconda3/envs/lowrankarena/bin/python -m pytest -q \
  tests/test_llama2_mha_v25_comparison.py \
  tests/test_palu_factorization_work_policy.py
```

结果：`5 passed`；只有两个 SWIG deprecation warnings，无测试失败。
