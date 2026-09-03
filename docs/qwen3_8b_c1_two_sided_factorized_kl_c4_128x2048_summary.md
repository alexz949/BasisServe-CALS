# Qwen3-8B C1 双侧 Factorized-KL：C4 128×2048 实验总结

## 1. 文档范围

本文档记录 Qwen3-8B-Base 上 C1 Value 压缩的双侧 Factorized-KL rank allocation 算法、固定因子来源、C4 `128×2048` profile、C4 `16×2048` confirmation、最终 layer-rank schedule，以及完整 WikiText-2 test PPL。

本文档不包含运行命令、数据解释或后续实验建议。

## 2. 模型与压缩对象

| 项目 | 设置 |
|---|---:|
| 模型 | Qwen3-8B-Base |
| Decoder layers | 36 |
| Query heads | 32 |
| Physical KV heads | 8 |
| 每个 KV source 对应的 query heads | 4 |
| Head dimension | 128 |
| Key cache | Dense，不压缩 |
| Value cache | C1 low-rank |
| Uniform anchor rank | 64 |
| Uniform V retained ratio | 0.5 |
| Dense-K + C1-V64 总 KV retained ratio | 0.75 |

每一层的8个 physical KV sources 使用相同的 Value rank。因此，layer rank 为 \(r_\ell\) 时，该层的 collective width 为 \(8r_\ell\)。

## 3. 固定 C1 factor banks

本次 Global-KL profile 没有重新拟合 C1 factors。使用的 factor banks 为：

\[
r\in\{32,48,64,80,96,112\}.
\]

Rank 128 是 exact dense-Value endpoint，对应 local reconstruction error 为零，不需要低秩 factor bank。

固定 factor banks 的共同配置如下：

| 项目 | 设置 |
|---|---:|
| C1 fit windows | 256 |
| Fit positions per window | 2048 |
| Fit rows | 524,288 |
| Held-out windows | 64 |
| Held-out positions per window | 2048 |
| Held-out rows | 131,072 |
| Encoder initialization | Activation-weighted SVD |
| Encoder sweeps | 5 |
| Work dtype | FP32 |
| Deployed factor dtype | BF16 |
| Covariance damping | \(10^{-5}\) |
| Decoder objective | Full-layer attention-output MSE with cross-head covariance |
| Selection boundary | Decoder-closed |
| Rank-bank selection metric | Held-out context aggregate-output MSE |

Factorized-KL 使用每层每个 rank 的 `factor_dtype_relative_mse` 作为 local error。记为

\[
e_{\ell,r}\ge 0,
\]

其中 \(\ell\) 是 decoder layer，\(r\) 是 Value rank，并定义

\[
e_{\ell,128}=0.
\]

## 4. 数据划分

### 4.1 C4 profile

| 项目 | 设置 |
|---|---:|
| Dataset | C4 train |
| Profile windows | 128 |
| Used sequence length | 2048 |
| Stored source-window length | 4096 |
| Used source indices | 320–447 |
| Batch size | 16 |
| Forward batches per evaluated schedule | 8 |

每个 source window 固定使用前2048个 token。Profile documents 与已有 C1 factor fit/held-out documents 在 document ID 上不重叠。

### 4.2 C4 confirmation

| 项目 | 设置 |
|---|---:|
| Dataset | C4 train |
| Confirmation windows | 16 |
| Sequence length | 2048 |
| Used source indices | 448–463 |
| Batch size | 16 |
| Forward batches per schedule | 1 |

Confirmation documents 与 C1 factor fit/held-out documents、128个 profile documents 均不重叠。

### 4.3 WikiText-2 PPL

| 项目 | 设置 |
|---|---:|
| Dataset | WikiText-2 |
| Split | Test |
| Evaluation sequence length | 2048 |
| Chunks | 146 |
| Evaluated tokens | 298,862 |
| Batch size | 1 |
| Loss accumulation dtype | FP32 |

## 5. Terminal teacher KL

令 \(P_T\) 为 dense BF16 teacher 的完整 vocabulary softmax，\(P_M\) 为给定 C1 schedule 下模型的完整 vocabulary softmax。Terminal KL 使用

\[
K_{\mathcal D}(M)
=
\frac{1}{N}
\sum_{(x,t)\in\mathcal D}
\operatorname{KL}
\left(
P_T(\cdot\mid x_{\le t})
\;\|\;
P_M(\cdot\mid x_{\le t})
\right).
\]

KL 使用完整 vocabulary；softmax probability 使用 FP32，KL accumulation 使用 FP64。Teacher logits 使用 BF16 保存。

Uniform anchor 模型记为 \(M_{64}\)：全部36层均使用 Value rank64。

## 6. 双侧 Factorized-KL 算法

### 6.1 每层两个直接 intervention

对每个 decoder layer \(\ell\)，直接测量两个 whole-layer interventions：

\[
M_{\ell\leftarrow32},
\qquad
M_{\ell\leftarrow96}.
\]

`Whole-layer` 表示该层全部8个 physical KV sources 同时改成相同 rank；其他35层保持 rank64。

测得的 terminal-KL deltas 为

\[
D^-_\ell
=
K_{\mathcal D}(M_{\ell\leftarrow32})
-K_{\mathcal D}(M_{64}),
\]

\[
D^+_\ell
=
K_{\mathcal D}(M_{\ell\leftarrow96})
-K_{\mathcal D}(M_{64}).
\]

本次 exponent 固定为

\[
\alpha=1.25.
\]

### 6.2 Compression-side sensitivity

Rank 小于64时使用 rank32 probe：

\[
s^-_\ell
=
\max\left(
0,
\frac{D^-_\ell}
{e_{\ell,32}^{\alpha}-e_{\ell,64}^{\alpha}}
\right).
\]

### 6.3 Expansion-side sensitivity

Rank 大于64时使用 rank96 probe：

\[
s^+_\ell
=
\max\left(
0,
\frac{D^+_\ell}
{e_{\ell,96}^{\alpha}-e_{\ell,64}^{\alpha}}
\right).
\]

### 6.4 Per-layer factorized cost curve

每层每个候选 rank 的预测 terminal cost 为

\[
\widehat C_\ell(r)
=
\begin{cases}
s^-_\ell
\left(e_{\ell,r}^{\alpha}-e_{\ell,64}^{\alpha}\right),
&r<64,\\[6pt]
0,
&r=64,\\[6pt]
s^+_\ell
\left(e_{\ell,r}^{\alpha}-e_{\ell,64}^{\alpha}\right),
&r>64.
\end{cases}
\]

候选 rank bank 为

\[
\mathcal R=\{32,48,64,80,96,112,128\}.
\]

Rank32与rank96的 terminal costs 来自直接 intervention。Rank48使用 compression-side curve；rank80、112和128使用 expansion-side curve。

### 6.5 Exact-budget allocation

最终 schedule 由 exact-budget dynamic programming 计算：

\[
\min_{r_0,\ldots,r_{35}\in\mathcal R}
\sum_{\ell=0}^{35}\widehat C_\ell(r_\ell),
\]

满足

\[
\sum_{\ell=0}^{35}r_\ell
=36\times64
=2304.
\]

对应全部 physical KV sources 的总 rank budget 为

\[
8\times2304=18{,}432.
\]

### 6.6 Selected-factor construction

对于非 anchor layer，选定 encoder factors 后执行 full-layer closed-form decoder refit。Decoder closure 使用 FP32；最终 deployed factors 使用 BF16。Rank64 layer 直接使用 uniform anchor factor bank；rank128 endpoint 使用 exact dense Value/O mapping。

## 7. 128-window per-layer measurements

Profile uniform-anchor terminal KL 为

\[
K_{\mathcal D}(M_{64})=0.123901240.
\]

下表记录每层直接测得的 rank32/rank96 terminal-KL delta、计算得到的双侧 sensitivity，以及最终选择的 rank。

| Layer | \(D^-_\ell\), rank32 | \(D^+_\ell\), rank96 | \(s^-_\ell\) | \(s^+_\ell\) | Selected rank |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.161484131 | -0.004655092 | 14.245136340 | 0.826091821 | 80 |
| 1 | 0.022116667 | -0.001820771 | 0.379909602 | 0.090774097 | 64 |
| 2 | 0.002256605 | -0.001112745 | 0.034834877 | 0.042773796 | 32 |
| 3 | 0.004627632 | -0.002068552 | 0.075234321 | 0.082056654 | 48 |
| 4 | 0.004073585 | -0.002033418 | 0.047216959 | 0.054310317 | 48 |
| 5 | 0.003673415 | -0.001651005 | 0.045155865 | 0.048170737 | 48 |
| 6 | 0.007456902 | -0.003662383 | 0.064184352 | 0.071137055 | 64 |
| 7 | 0.005692961 | -0.002828010 | 0.044985523 | 0.046948288 | 64 |
| 8 | 0.007704566 | -0.004524321 | 0.054007480 | 0.060934896 | 80 |
| 9 | 0.007242809 | -0.004146739 | 0.046087238 | 0.047803827 | 80 |
| 10 | 0.006842157 | -0.004386243 | 0.048209565 | 0.055801133 | 80 |
| 11 | 0.006678676 | -0.004032771 | 0.049740092 | 0.055433536 | 80 |
| 12 | 0.004967507 | -0.002846063 | 0.051814859 | 0.059892329 | 48 |
| 13 | 0.004012650 | -0.002286293 | 0.037579815 | 0.041613257 | 48 |
| 14 | 0.005644965 | -0.003407048 | 0.044760835 | 0.050850884 | 64 |
| 15 | 0.007012718 | -0.003578865 | 0.059330378 | 0.057337707 | 64 |
| 16 | 0.005893576 | -0.003439885 | 0.048482053 | 0.050661150 | 64 |
| 17 | 0.007186705 | -0.003792377 | 0.085007480 | 0.085816231 | 64 |
| 18 | 0.006910899 | -0.003850166 | 0.065815667 | 0.066847849 | 64 |
| 19 | 0.008195480 | -0.004308790 | 0.093560407 | 0.110274312 | 80 |
| 20 | 0.005775543 | -0.003346049 | 0.050441600 | 0.053030006 | 64 |
| 21 | 0.005401911 | -0.003095102 | 0.042861469 | 0.045256362 | 64 |
| 22 | 0.010229537 | -0.005618758 | 0.092345936 | 0.094671461 | 96 |
| 23 | 0.013856369 | -0.007385991 | 0.105538824 | 0.111970780 | 112 |
| 24 | 0.012961997 | -0.005543067 | 0.121930307 | 0.124447438 | 80 |
| 25 | 0.004720046 | -0.002879540 | 0.038213948 | 0.044726236 | 48 |
| 26 | 0.006106356 | -0.003652419 | 0.038659788 | 0.040925793 | 64 |
| 27 | 0.005175036 | -0.003283314 | 0.037206876 | 0.042435726 | 48 |
| 28 | 0.003524058 | -0.002603311 | 0.025960700 | 0.035944691 | 32 |
| 29 | 0.006030407 | -0.003378356 | 0.036045231 | 0.037067926 | 64 |
| 30 | 0.007267755 | -0.003896828 | 0.064956252 | 0.064165743 | 64 |
| 31 | 0.004610849 | -0.002966078 | 0.029312781 | 0.032735424 | 48 |
| 32 | 0.005114196 | -0.002967842 | 0.044812364 | 0.046119883 | 48 |
| 33 | 0.011167040 | -0.005652988 | 0.074392446 | 0.062172370 | 96 |
| 34 | 0.007766026 | -0.004457856 | 0.137039328 | 0.130538403 | 80 |
| 35 | 0.003634141 | -0.002350876 | 0.119059946 | 0.116912401 | 32 |

## 8. Final layer-rank schedule

按 decoder layer 0–35 排列的最终 ranks 为：

```text
[80, 64, 32, 48, 48, 48, 64, 64, 80, 80, 80, 80,
 48, 48, 64, 64, 64, 64, 64, 80, 64, 64, 96, 112,
 80, 48, 64, 48, 32, 64, 64, 48, 48, 96, 80, 32]
```

### 8.1 Layer-rank histogram

| Rank | Layers | Physical KV sources |
|---:|---:|---:|
| 32 | 3 | 24 |
| 48 | 9 | 72 |
| 64 | 13 | 104 |
| 80 | 8 | 64 |
| 96 | 2 | 16 |
| 112 | 1 | 8 |
| 128 | 0 | 0 |
| **Total** | **36** | **288** |

### 8.2 Budget accounting

| 项目 | 数值 |
|---|---:|
| Layer-rank sum | 2,304 |
| Source-rank sum | 18,432 |
| Uniform-rank64 source-rank sum | 18,432 |
| Dense-Value source-rank sum | 36,864 |
| Value retained ratio | 0.5 |
| Dense-Value reduction | 2.0× |
| Layers different from rank64 | 23 |
| Physical KV sources different from rank64 | 184 |
| Ragged-padding overhead | 0 |
| Predicted additive terminal cost | -0.015796433 |

## 9. C4 confirmation results

| Schedule | Terminal KL mean | Terminal KL paired SE | Terminal KL one-SE UCB | NLL mean | NLL paired SE | NLL one-SE UCB |
|---|---:|---:|---:|---:|---:|---:|
| Uniform V64 | 0.300654555 | 0.193973367 | 0.494627922 | 2.265406869 | 0.260495697 | 2.525902567 |
| Two-sided Factorized-KL | 0.249093473 | 0.156432508 | 0.405525981 | 2.214503236 | 0.230881478 | 2.445384714 |

16个 confirmation windows 的逐窗 terminal KL 如下：

| Window | Uniform V64 | Two-sided Factorized-KL |
|---:|---:|---:|
| 0 | 0.098381693 | 0.084310488 |
| 1 | 0.096756994 | 0.082238591 |
| 2 | 0.158597476 | 0.134348731 |
| 3 | 0.165501873 | 0.140777823 |
| 4 | 0.123311448 | 0.109571285 |
| 5 | 0.082934597 | 0.071618873 |
| 6 | 0.090810699 | 0.081705144 |
| 7 | 0.093820966 | 0.084883227 |
| 8 | 0.127414314 | 0.106978135 |
| 9 | 0.074245792 | 0.061613654 |
| 10 | 0.098376783 | 0.087909884 |
| 11 | 0.093140942 | 0.083968003 |
| 12 | 0.108376389 | 0.096148032 |
| 13 | 0.096720493 | 0.082548950 |
| 14 | 3.208709090 | 2.594257677 |
| 15 | 0.093373329 | 0.082617074 |

## 10. Full WikiText-2 test PPL

| Schedule | C4 profile windows | Candidate ranks | PPL |
|---|---:|---|---:|
| Uniform C1-V64 | — | \(\{64\}\) | 8.416738065 |
| C4 64-window two-sided allocation | 64 | \(\{32,48,64,80,96\}\) | 8.419701128 |
| C4 128-window two-sided allocation | 128 | \(\{32,48,64,80,96,112,128\}\) | **8.245895888** |

本次128-window schedule 的完整 PPL accounting：

| 项目 | 数值 |
|---|---:|
| NLL sum | 630,513.826171875 |
| Evaluated tokens | 298,862 |
| Chunks | 146 |
| Sequence length | 2048 |
| Batch size | 1 |
| PPL | 8.245895888 |

## 11. Numerical and execution record

| 项目 | 数值 |
|---|---:|
| Hardware | NVIDIA L40S |
| Model dtype | BF16 |
| Deployed factor dtype | BF16 |
| Decoder-closure dtype | FP32 |
| Terminal probability dtype | FP32 |
| Terminal-KL accumulation dtype | FP64 |
| Peak CUDA allocation during finalize | 28,835,655,168 bytes |
| Profile shard 0 assigned layers | 0, 2, 4, …, 34 |
| Profile shard 0 elapsed time | 1,805.866 s |
| Profile shard 0 peak CPU RSS | 80,481,624 KiB |
| Profile shard 1 assigned layers | 1, 3, 5, …, 35 |
| Profile shard 1 elapsed time | 1,802.795 s |
| Profile shard 1 peak CPU RSS | 79,690,624 KiB |
| Finalize elapsed time | 36.367 s |
| Full WikiText PPL elapsed time | 33.730 s |
| Python | 3.11.15 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.15.1 |

## 12. Validation record

| 项目 | 结果 |
|---|---:|
| Factorized/global-KL related tests | 27 passed |
| Whitespace/error-marker validation | Clean |
| Successful profile shards | 2/2 |
| Completed layer interventions | 72/72 |
| Non-finite terminal-KL measurements | 0 |

## 13. Result artifacts

- Allocator result: `results/checkpoints/qwen3_8b_c1_factorized_c4_128measure_twosided_avg64_a1p25/result.json`
- Selected factor artifacts: `results/checkpoints/qwen3_8b_c1_factorized_c4_128measure_twosided_avg64_a1p25/selected_factors/`
- Profile shard 0: `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_profile/shard_00.json`
- Profile shard 1: `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_profile/shard_01.json`
- Full WikiText-2 PPL: `results/evaluation/qwen3_8b_c1_factorized_c4_128measure_twosided_full_wikitext_ppl.json`
- Uniform V64 WikiText-2 PPL: `results/evaluation/qwen3_8b_c1_uniform_r64_als5_wikitext2_ppl.json`
