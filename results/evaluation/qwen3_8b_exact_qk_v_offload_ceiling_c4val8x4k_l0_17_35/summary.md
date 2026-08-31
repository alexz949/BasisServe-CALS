# Qwen3-8B exact-QK Value-offload ceiling

Exact post-RoPE K is treated as GPU-resident. Selection is unioned across the four query heads in each GQA group, and all heads attend over the fetched union with exact QK scores. Reported traffic excludes K.

## Aggregate

| selector | nominal tokens/head | mass | dense-V decoded rel-L2 | dense cosine | C1-V decoded rel-L2 | C1 sparse→dense rel-L2 | unique-token fraction | page-rounded fraction | token union amp | page union amp |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| page_mass | 64 | 0.688919 | 2.132134e-01 | 0.960483 | 2.170433e-01 | 3.199859e-01 | 0.039403 | 0.039403 | 1.576122 | 1.576122 |
| page_mass | 256 | 0.849634 | 8.579018e-02 | 0.993343 | 8.685053e-02 | 2.598640e-01 | 0.160116 | 0.160116 | 1.601162 | 1.601162 |
| page_mass | 512 | 0.915898 | 4.889327e-02 | 0.997966 | 4.971845e-02 | 2.513323e-01 | 0.313612 | 0.313612 | 1.568059 | 1.568059 |
| page_mass | 1024 | 0.968297 | 2.183900e-02 | 0.999621 | 2.239809e-02 | 2.479043e-01 | 0.566737 | 0.566737 | 1.416842 | 1.416842 |
| token_topk | 64 | 0.847529 | 9.748765e-02 | 0.994995 | 1.003748e-01 | 2.646900e-01 | 0.055820 | 0.529968 | 2.232804 | 21.198718 |
| token_topk | 256 | 0.941327 | 4.310277e-02 | 0.999277 | 4.501137e-02 | 2.506515e-01 | 0.203626 | 0.876532 | 2.036256 | 8.765325 |
| token_topk | 512 | 0.973494 | 2.290294e-02 | 0.999818 | 2.420954e-02 | 2.480385e-01 | 0.363627 | 0.967528 | 1.818135 | 4.837640 |
| token_topk | 1024 | 0.992300 | 8.850927e-03 | 0.999975 | 9.548615e-03 | 2.471662e-01 | 0.603704 | 0.996835 | 1.509259 | 2.492087 |

## Full-attention payload control

C1-V64 full attention versus dense-V full attention after decoding: relative L2 `2.470499e-01`, mean cosine `0.965690`.

`token_topk` is the scatter-gather quality ceiling. `page_mass` ranks each page by exact QK log-sum-exp and is the page-granular offload ceiling. Logical bytes assume BF16 Value vectors and do not measure PCIe latency.
