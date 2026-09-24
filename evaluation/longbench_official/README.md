# Official LongBench-v1 files (verbatim)

Verbatim copies from the upstream LongBench repository, used by
`evaluation/longbench_metrics.py` and `evaluation/prepare_longbench_shadowkv9.py`.
Do not edit them; scores are only comparable to published numbers while these
files match upstream byte for byte.

- Upstream: https://github.com/THUDM/LongBench
- Revision: `2e00731f8d0bff23dc4325161044d0ed8af94c1e`
- Source paths at that revision: `LongBench/config/dataset2prompt.json`,
  `LongBench/config/dataset2maxlen.json`, `LongBench/metrics.py`

| File | sha256 |
| --- | --- |
| `dataset2prompt.json` | `56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30` |
| `dataset2maxlen.json` | `72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022` |
| `metrics.py` | `e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323` |

The dataset-to-metric table and the per-sample post-processing from upstream
`LongBench/eval.py` (`scorer`) are reproduced in `evaluation/longbench_metrics.py`.

Dataset: `THUDM/LongBench` `data.zip`, sha256
`cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`.
