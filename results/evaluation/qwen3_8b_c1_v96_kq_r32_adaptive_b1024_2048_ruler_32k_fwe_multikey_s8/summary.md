# Qwen3-8B-Base adaptive KQ-routing RULER-v1 32K

Adaptive policies start from the base page budget and expand one Query head to the maximum budget when the proxy mass of the next page band exceeds the configured fraction of the base-band mass.

| Task | Samples | BF16 dense | C1-V96 exact-K | R32/B1024 | R32/B1024->2048 tail>=0.5 | R32/B1024->2048 tail>=0.25 | R32/B1024->2048 tail>=0.1 | R32/B2048 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| fwe | 8 | 87.50% | 83.33% | 70.83% | 79.17% | 79.17% | 79.17% | 79.17% |
| niah_multikey_1 | 8 | 87.50% | 87.50% | 87.50% | 87.50% | 87.50% | 87.50% | 87.50% |
| niah_multikey_2 | 8 | 100.00% | 75.00% | 62.50% | 75.00% | 75.00% | 75.00% | 75.00% |
| **Task-balanced mean** | 24 | **91.67%** | **81.94%** | **73.61%** | **80.56%** | **80.56%** | **80.56%** | **80.56%** |

| Sparse policy | Accuracy | Selected exact-K tokens | Adaptive refinements | Logical exact-K MiB/decode token |
|:---|---:|---:|---:|---:|
| R32/B1024 | 73.61% | 5.8591% | -- | 132.741 |
| R32/B1024->2048 tail>=0.5 | 80.56% | 6.2808% | 5.09% | 142.188 |
| R32/B1024->2048 tail>=0.25 | 80.56% | 7.7453% | 22.04% | 172.558 |
| R32/B1024->2048 tail>=0.1 | 80.56% | 9.3435% | 46.12% | 207.735 |
| R32/B2048 | 80.56% | 12.2231% | -- | 271.098 |

Prompt prefill is shared within each sample and uses dense C1-V96 attention. Exact K remains GPU-resident, so exact-K traffic is logical rather than measured PCIe.
