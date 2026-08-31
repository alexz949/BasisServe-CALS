# Qwen3-8B C1-V96 + KQ-SVD R32 Routing：实现、设置与当前结果

## 1. 文档范围

本文总结当前已经完成的 Qwen3-8B-Base `C1-V96 + KQ-SVD R32 routing + exact-K sparse attention` 实现，以及 32K/128K RULER correctness 与 decode benchmark 结果。

当前性能实验是一个 **GPU-resident exact-K oracle**：exact K 仍然完整驻留 HBM，代码记录逻辑上的 exact-K fetch bytes，但没有发生真实的 CPU-to-GPU PCIe K 传输。因此本文中的 decode 时间衡量 routing、page selection、exact sparse QK、C1-V attention 和 sidecar 维护，不是完整 K-offload 系统的端到端延迟。

## 2. 模型与缓存表示

### 2.1 模型

| 项目 | 设置 |
|:---|:---|
| 模型 | Qwen3-8B-Base |
| Transformer layers | 36 |
| Query heads | 32 |
| Physical KV heads | 8 |
| Query heads / KV head | 4 |
| Head dimension | 128 |
| 运行 dtype | BF16 |

### 2.2 C1-V96 payload

每个 physical V head 从 128 维压缩到 96 维：

\[
v_{g,i}\in\mathbb{R}^{128}
\quad\longrightarrow\quad
c_{g,i}=v_{g,i}E_g\in\mathbb{R}^{96}.
\]

每个 query head 使用对应的 output decoder，将 C1 attention latent 映射回 layer output：

\[
y_h^{\mathrm{C1}}
=
\left(\sum_i a_{h,i}c_{g(h),i}\right)D_h.
\]

C1 checkpoint 的具体设置：

| 项目 | 设置 |
|:---|:---|
| V rank | 96 / physical KV head |
| V retained scalar ratio | 75% |
| Dense-K + C1-V 总 KV ratio | 87.5% |
| Calibration source | C4 document windows |
| Fit | 256 windows × 2048 positions |
| Held-out | 64 windows × 2048 positions |
| Objective | full-layer attention-output MSE with cross-head covariance |
| Initialization | activation-weighted SVD |
| ALS sweeps | 5 |
| Factor dtype | BF16 |
| Mean fit relative MSE | 0.0433659 |
| Mean held-out relative MSE | 0.0553734 |

这套 C1 是 activation-aware 的：encoder/decoder 根据 C4 attention activation 的输出协方差拟合，而不是仅对静态 V weight 或未加权 V activation 做普通 SVD。

### 2.3 KQ-SVD R32 routing sidecar

对每层、每个 physical KV head、每个历史 token，保存一个 32 维 post-RoPE routing vector：

\[
z_{g,i}^{(\ell)}
=
k_{g,i}^{(\ell)}P_{K,g}^{(\ell)}
\in\mathbb{R}^{32}.
\]

每个 query head 同样映射到 routing space：

\[
u_h^{(\ell)}
=
q_h^{(\ell)}P_{Q,h}^{(\ell)}
\in\mathbb{R}^{32}.
\]

近似 routing score 为：

\[
\widehat{s}_{h,i}^{(\ell)}
=
\frac{\langle u_h^{(\ell)},z_{g(h),i}^{(\ell)}\rangle}{\sqrt{128}}.
\]

Routing checkpoint 的具体设置：

| 项目 | 设置 |
|:---|:---|
| Coordinate | post-RoPE，位于 Qwen3 q_norm/k_norm 之后 |
| Calibration source | C4 packed document windows |
| Calibration | 32 windows × 32768 tokens |
| Query sampling | 无抽样；使用全部 post-RoPE Q rows |
| Key rows / layer / KV head | 1,048,576 |
| Query rows / layer / GQA group | 4,194,304 |
| Offline pair rank | 128 / adjacent-layer pair |
| Runtime routing rank | 32 |
| Objective | unmasked pre-softmax QK-score Frobenius error |
| Softmax-aware | 否 |
| Value-aware | 否 |
| Held-out windows | 0 |

完整 pair-rank-128 checkpoint 的 weighted relative squared score error 为 `0.00352467`；独立 `rank64 + rank64` control 为 `0.00456498`，pairwise fit 在该 R128 calibration objective 上降低 `22.79%`。这些误差数字对应完整 R128 checkpoint，不直接等同于运行时截取后的 R32 routing error。

## 3. Page selection 与 exact sparse attention

### 3.1 Page score

当前 page size 为 64。对每个 query head，先计算全部可见 token 的 R32 proxy score，然后用 page 内 log-sum-exp 得到 page mass：

\[
\widehat{\ell}_{h,p}
=
\log\sum_{i\in p}\exp\left(\widehat{s}_{h,i}\right).
\]

`B=2048` 对应每个 query head 选择：

\[
2048/64=32\text{ pages}.
\]

### 3.2 GQA union

同一个 physical KV head 对应 4 个 query heads。当前实现先让每个 query head 独立选 32 pages，再对这 4 组 pages 取 union：

\[
\mathcal P_g
=
\bigcup_{h:g(h)=g}\operatorname{Top32Pages}
\left(\widehat{\ell}_{h,:}\right).
\]

因此 `B=2048` 不是 GQA union 后严格固定的 physical exact-K token 数。理论 union 上限为：

\[
4\times 32\times64=8192\text{ tokens / KV head}.
\]

当前运行未启用 adaptive budget，没有 forced-last-page，也没有额外 recent-exact window。

### 3.3 Sparse attention

Page selection 完成后，attention 使用被选中 pages 中的 exact post-RoPE K 重新计算 QK score；对应的 C1-V96 payload 直接从 GPU resident C1 cache 读取。未选中 token 不参与该 query 的 softmax denominator 或 numerator。

当前流程可以写成：

\[
Q
\rightarrow
\text{全长 R32 routing scan}
\rightarrow
\text{page log-mass Top-k}
\rightarrow
\text{GQA page union}
\rightarrow
\text{selected exact-QK}
\rightarrow
\text{selected C1-V96 attention}.
\]

## 4. Incremental routing sidecar 实现

### 4.1 Cache 语义

`RoutingDynamicCache` 为每层保存与 exact-K token axis 对齐的 R32 sidecar。Prefill 时 sidecar 随 exact post-RoPE K 分块建立；decode 时只投影并追加新 token：

\[
z_{L+1}=k_{L+1}P_K.
\]

它支持与 KV transaction 对齐的 crop、batch reorder、repeat、select 和 reset。Decode arm 结束后，exact KV cache 与 sidecar 一起 rollback 到相同 prompt boundary。

### 4.2 两个性能对照 arm

| Arm | Sidecar 使用方式 | 其余计算 |
|:---|:---|:---|
| `lazy_full_k_projection` | 每层、每个 decode step 从完整历史 exact K 重新计算 `K @ P_K` | 与 incremental arm 相同 |
| `incremental_cached_sidecar` | 直接读取 prefill 已缓存的 R32 sidecar，并只追加新 token | 与 lazy arm 相同 |

两个 arm 使用同一个 prompt cache、相同 R32 factors、相同 page scores、相同 selected pages、相同 exact K 和相同 C1-V96。这个 A/B 只隔离“是否重复计算历史 `K @ P_K`”。

### 4.3 Grouped-GQA score kernel

Routing score 使用 grouped GQA matmul：

\[
[H_{kv},G,R]\times[H_{kv},R,L]
\rightarrow
[H_{kv},G,L],
\]

其中 `Hkv=8`、`G=4`、`R=32`。实现不再把 8-head sidecar 显式扩张成 32 query heads，从而避免 4 倍 sidecar 临时展开。

### 4.4 Prefill 实现

Prefill 使用 mathematically dense causal C1 attention，chunk size 为 512。为避免 PyTorch GQA 路径显式展开完整 32-head KV workspace，attention 按 4 个 physical KV heads 一组执行 memory-bounded GQA SDPA。

Chunking 控制 workspace 峰值，但每个 query chunk 仍然 attention 到完整可见 prefix，因此总 prefill attention 仍为二次复杂度。

## 5. Evaluation 设置

### 5.1 128K correctness smoke

| 项目 | 设置 |
|:---|:---|
| Dataset | RULER v1 |
| Task | `niah_single_1` |
| Samples | 1 |
| Prompt tokens | 130,929 |
| Effective context | 131,072 |
| RoPE | YaRN factor 4.0；original max position 32,768 |
| Page size | 64 |
| Nominal token budget | 2,048 / query head before GQA union |
| Routing rank | 32 |
| Value rank | 96 |
| Layers using routing | 36 / 36 |
| Exact K placement | GPU HBM |

### 5.2 Decode performance benchmark

| 项目 | 32K smoke | 128K benchmark |
|:---|---:|---:|
| Task | `niah_single_1` | `niah_single_1` |
| Prompt tokens | 32,628 | 130,929 |
| Generated output tokens | 4 | 128 |
| Timed decode forwards | 3 | 127 |
| Warm-up output tokens | 2 | 4 |
| Prefill chunk size | 512 | 512 |
| Page size | 64 | 64 |
| Nominal budget | 2,048 | 2,048 |
| Routing rank | 32 | 32 |
| Pipeline devices | 2 × NVIDIA L40S | 2 × NVIDIA L40S |
| Layer placement | layers 0–17 / layers 18–35 | layers 0–17 / layers 18–35 |

首个 output token 在 prefill 末尾产生，因此 4 个 output tokens 对应 3 次 timed autoregressive forward，128 个 output tokens 对应 127 次 timed autoregressive forward。表中的每步延迟使用 timed forward 数作为分母。

## 6. 当前结果

### 6.1 C1 与 routing correctness

128K 单样本 `niah_single_1` 结果：

| Arm | Prediction | Score |
|:---|:---|---:|
| Dense K + Dense V | `2338687` | 1.0 |
| Dense K + C1-V96 | `2338687` | 1.0 |
| KQ-R32 B2048 exact-K + C1-V96 | `2338687` | 1.0 |

Dense-C1 与 KQ-R32 sparse-C1 的完整 generated token IDs 相同。Dense BF16 输出仅在 markdown/空白格式上不同，任务答案相同。

该结果只包含一个 RULER 样本，证明这一具体样本上的端到端一致性，不代表 128K RULER aggregate accuracy。

### 6.2 32K sidecar smoke

| 指标 | Lazy full-K projection | Incremental cached sidecar |
|:---|---:|---:|
| Raw elapsed，3 decode forwards | 0.642617 s | 0.625218 s |
| Latency / timed forward | 214.206 ms | 208.406 ms |
| Timed forwards / second | 4.668 | 4.798 |
| Peak memory / GPU | 11.181 GiB | 11.166 GiB |

对照结果：

- Decode speedup：`1.02783×`
- Decode time reduction：`2.7076%`
- Generated token IDs equal：`true`
- Routing logical statistics equal：`true`
- Prefill：`60.0587 s`
- Resident R32 sidecar：`601,399,296 bytes = 573.54 MiB`
- Physical selected-token fraction：`11.9231%`

### 6.3 128K sidecar benchmark

| 指标 | Lazy full-K projection | Incremental cached sidecar |
|:---|---:|---:|
| Raw elapsed，127 decode forwards | 87.2542 s | 84.7034 s |
| Latency / timed forward | 687.041 ms | 666.956 ms |
| Timed forwards / second | 1.4555 | 1.4993 |
| Peak memory / GPU | 20.266 GiB | 20.204 GiB |

对照结果：

- Decode speedup：`1.03012×`
- Decode time reduction：`2.9235%`
- Absolute saving：`20.085 ms / timed decode forward`
- Generated token IDs equal：`true`
- Routing logical statistics equal：`true`
- Prefill：`1014.010 s = 16 min 54.0 s`
- Resident R32 sidecar：`2,413,283,328 bytes = 2.248 GiB`

128K logical selection statistics：

| 指标 | 结果 |
|:---|---:|
| Timed layer-query instances | 4,572 = 127 × 36 |
| Physical selected-token fraction | 3.29932% |
| Average selected exact-K tokens / layer-step / KV head | 4,321.87 |
| Logical exact-K bytes | 37.957 GiB / complete arm |
| Logical exact-K bytes / timed decode forward | 306.05 MiB |

这里的 `4,321.87` 是 GQA union 后的平均 physical token 数。它大于 nominal `2,048`，但低于 4 个 query heads 完全无重叠时的上限 `8,192`。

## 7. 数据解释

### 7.1 Incremental sidecar 的数值正确性

两个 arm 的 generated IDs 与全部 routing logical statistics 完全一致，说明 cached sidecar、lazy full-K projection、cache crop 和 decode transaction 在当前 BF16 runtime 下产生相同的 selection 与最终输出。

因此 `1.03012×` 衡量的是 sidecar maintenance 优化本身，不包含 sparse attention 相对于 dense attention 的全部收益。

### 7.2 为什么 32K 与 128K 都只有约 3%

两种路径可以近似写为：

\[
T_{\mathrm{lazy}}(L)
=C+T_{K\rightarrow R32}(L)+T_{\mathrm{scan}}(L)+T_{\mathrm{select/attn}}(L),
\]

\[
T_{\mathrm{incremental}}(L)
=C+T_{\mathrm{append}}+T_{\mathrm{scan}}(L)+T_{\mathrm{select/attn}}(L).
\]

Incremental cache 消除了每步对完整历史 K 的 `K @ P_K`，但两个 arm 仍然对全部可见 token 的 R32 sidecar 做 query-dependent scan。被消除的 full-K projection 与保留下来的 routing scan 都随上下文长度线性增长，所以长度增加主要扩大绝对节省，而没有显著扩大相对节省。

实测 reduction 从 32K 的 `2.71%` 变为 128K 的 `2.92%`；128K 中被消除的投影只占完整 decode step 的约 3%。

### 7.3 B2048 与实际 selected tokens

`B=2048` 是每个 query head 在 GQA union 前的 page budget，而不是每个 physical KV head 的最终 fetch budget。当前 page-level union 使 128K 平均 physical selection 达到约 4,322 tokens/KV head，对应 3.30% 上下文。

因此 nominal ratio：

\[
2048/130929\approx1.56\%
\]

与实际 physical selected ratio `3.30%` 描述的是不同阶段的预算。

### 7.4 当前 GPU memory 口径

按每个 KV head、每个 token 的 BF16 scalars 计数：

\[
\underbrace{128}_{\text{exact K}}
+
\underbrace{96}_{\text{C1 V}}
+
\underbrace{32}_{\text{routing sidecar}}
=256.
\]

Dense BF16 KV 为：

\[
\underbrace{128}_{K}+\underbrace{128}_{V}=256.
\]

所以当前 GPU-resident oracle 的 per-token cache scalar count 与 dense BF16 KV 相同。C1-V96 单独将 V 缩减 25%，但 resident R32 sidecar 正好增加相当于 dense K 的 25%，同时 exact K 又完整保留。当前 benchmark 的目的因此是隔离 routing correctness 与 sidecar decode cost，而不是展示 GPU KV memory reduction。

Sidecar 从 32K 的 `573.54 MiB` 增长到 128K 的 `2.248 GiB`，与 token 数近似线性；prefill 从 `60.06 s` 增长到 `1014.01 s`，比例为 `16.884×`。上下文长度增加约 4 倍，而 dense causal prefill work 近似增加 16 倍，数据与二次 attention scaling 一致。

### 7.5 当前 PCIe 口径

`cpu_exact_key_bytes_fetched` 是根据 selected physical pages 计算的逻辑流量。128K arm 的 `37.957 GiB` 和 `306.05 MiB/decode forward` 没有形成真实 PCIe transaction，因为 page store 当前直接从 GPU-resident exact K 返回数据。

因此当前延迟结果不包含 CPU pinned-memory gather、PCIe copy、miss detection、GPU staging-buffer replacement 或 transfer/compute overlap。

