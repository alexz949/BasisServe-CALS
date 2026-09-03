# Qwen3-8B TP4 Exact-K Offload on Five Hard RULER Samples

All arms use BF16 Qwen3-8B-Base, C1-V80, TP4, Page32, group-max fixed physical B4096 per KV head, the same selected-page CUDA exact-QK/online-softmax/V80 kernel, and the same five 64K YaRN4 samples.

| Arm | Hard-5 accuracy | Prefill tok/s | Decode tok/s | E2E model tok/s | K read/step | Total K read | GPU cache/rank | Host K/rank |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 Base16+R8 GPU oracle | 70.00% | 9402.46 | 3.373 | 3570.87 | 0.28125 GiB | 53.719 GiB | 2.461 GiB | 0.000 GiB |
| C1 Base16+R8 mapped host | 70.00% | 9339.47 | 3.332 | 3534.96 | 0.28125 GiB | 53.719 GiB | 1.336 GiB | 1.125 GiB |
| QUEST mapped host | 50.00% | 9346.55 | 25.971 | 7729.41 | 0.28125 GiB | 53.438 GiB | 0.773 GiB | 1.125 GiB |
| ShadowKV-style mapped host | 48.33% | 9343.00 | 25.952 | 7768.42 | 0.28125 GiB | 51.750 GiB | 0.738 GiB | 1.125 GiB |

| Sample | C1 Base16+R8 GPU oracle | C1 Base16+R8 mapped host | QUEST mapped host | ShadowKV-style mapped host |
|:---|---:|---:|---:|---:|
| niah_multikey_2:7 | 0.00% | 0.00% | 0.00% | 0.00% |
| niah_multivalue:3 | 50.00% | 50.00% | 50.00% | 75.00% |
| niah_single_2:3 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3:3 | 100.00% | 100.00% | 0.00% | 0.00% |
| fwe:2 | 100.00% | 100.00% | 100.00% | 66.67% |

## Exact-K placement equivalence

C1 mapped-host versus the same-kernel GPU-resident oracle generated identical token sequences on `5/5` samples.

## Measurement definitions

- `Group-max` first normalizes routable page scores independently for every Query head, then takes the maximum mass over the four Query heads sharing one physical KV head.
- `K read/step` is the requested Page32 BF16 exact-K payload across all layers, physical KV heads, and ranks for one model decode step. `Total K read` also depends on when each generation reaches EOS. Neither value is a PCIe hardware-counter measurement.
- `GPU cache/rank` includes C1-V80 plus router metadata; the GPU oracle additionally includes dense exact K.
- C1 Base16+R8 full-token reconstruction, RoPE, and page scoring are currently chunked PyTorch operations, not a fused routing kernel; the selected-page exact-QK/online-softmax/V80 attention is fused CUDA.
- `ShadowKV-style` isolates post-RoPE Page32 mean-landmark routing. It does not include ShadowKV's chunk8 outlier/local caches or online-SVD Key payload.
- End-to-end model throughput counts prompt tokens and decode forward steps and excludes model loading.
