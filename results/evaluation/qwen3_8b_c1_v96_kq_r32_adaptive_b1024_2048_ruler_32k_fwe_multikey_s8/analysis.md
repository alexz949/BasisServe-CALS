# Adaptive B1024-to-B2048 K-routing result

## Setup

- Model: Qwen3-8B-Base with the 256 x 2K C1-V96 checkpoint.
- Routing: independent per-layer KQ-SVD R32, page size 64.
- Tasks: `fwe`, `niah_multikey_1`, and `niah_multikey_2`; 8 fixed 32K samples per task.
- Base budget: 16 pages/query head (B1024).
- Maximum budget: 32 pages/query head (B2048).
- Adaptive confidence statistic:

  \[
  \rho(q,\ell,h)=
  \frac{\sum_{p=17}^{32}\exp(\hat\ell_p)}
       {\sum_{p=1}^{16}\exp(\hat\ell_p)}.
  \]

  A layer/query-head decision expands from 16 to 32 pages when
  \(\rho\ge\tau\). Physical pages are still unioned within each GQA group.

## Accuracy

| Arm | fwe | multikey-1 | multikey-2 | Task mean |
|:---|---:|---:|---:|---:|
| BF16 dense | 87.50% | 87.50% | 100.00% | 91.67% |
| Exact-K + C1-V96 | 83.33% | 87.50% | 75.00% | 81.94% |
| Fixed B1024 | 70.83% | 87.50% | 62.50% | 73.61% |
| Adaptive, tau=0.5 | 79.17% | 87.50% | 75.00% | 80.56% |
| Adaptive, tau=0.25 | 79.17% | 87.50% | 75.00% | 80.56% |
| Adaptive, tau=0.1 | 79.17% | 87.50% | 75.00% | 80.56% |
| Fixed B2048 | 79.17% | 87.50% | 75.00% | 80.56% |

All adaptive thresholds recover the B1024 routing-only failures on `fwe:2`
and `niah_multikey_2:7`. The remaining `fwe:6` regression is not repaired by
fixed B2048 either. The `multikey_2:0` and `multikey_2:6` failures already
occur with exact-K C1-V96 and are payload failures rather than routing errors.

## Traffic and refinement

| Sparse policy | Selected-K fraction | Refinement rate | Logical exact-K traffic | Relative to B1024 |
|:---|---:|---:|---:|---:|
| Fixed B1024 | 5.8591% | -- | 132.741 MiB/token | 1.000x |
| Adaptive, tau=0.5 | 6.2808% | 5.09% | 142.188 MiB/token | 1.071x |
| Adaptive, tau=0.25 | 7.7453% | 22.04% | 172.558 MiB/token | 1.300x |
| Adaptive, tau=0.1 | 9.3435% | 46.12% | 207.735 MiB/token | 1.565x |
| Fixed B2048 | 12.2231% | -- | 271.098 MiB/token | 2.042x |

The conservative `tau=0.5` policy recovers 83.3% of the task-mean gap between
fixed B1024 and exact-K C1, while adding only 7.1% logical exact-K traffic. It
matches fixed B2048 accuracy with 52.5% of B2048's per-decode-token traffic.
It does not add persistent per-token metadata; resident V96 plus R32 remains at
50% of the dense BF16 KV scalar count.

## Interpretation

`tau=0.5` is the Pareto winner in this targeted 24-sample diagnostic. The
proxy tail-mass statistic successfully identifies both observed routing-only
failures, while more aggressive thresholds fetch substantially more K pages
without changing task accuracy.

This is not yet a final threshold selection: the three tasks were deliberately
chosen because fixed B1024 was difficult, and only eight samples per task were
used. The next controlled check should freeze `tau=0.5` and evaluate all 11
tasks on the same 88-sample 32K suite. The fixed B1024 and B2048 endpoints
should remain in that run to detect regressions and confirm the traffic curve.

## Run

- Slurm job: `8288167`.
- Hardware: 2 x NVIDIA A100-PCIE-40GB.
- Wall time: 14 minutes 18 seconds.
- Peak allocated GPU memory: about 28.32 GiB per process.
- Status: complete, 24/24 records, no NaN or OOM.
- Exact K remained GPU-resident, so traffic is logical rather than measured PCIe traffic.
