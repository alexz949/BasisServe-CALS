# C1 K-only Reverse ShadowKV oracle

`teacher_exact` and `teacher_mass` consult full exact K and are not serving policies. 
Logical byte/FLOP counts are not measured CPU-offload speedups.

Each observation is one held-out window at one layer. The tail location therefore identifies both the layer and the window.

| policy | landmarks/page | page | sparse-layer budget | recent | obs. | mass mean | mass min | decoded rel-L2 mean | decoded rel-L2 p95 | decoded rel-L2 max (layer/example) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| quest_minmax | 1 | 16 | 256 | 0 | 576 | 0.754409 | 0.313374 | 3.539569e-01 | 7.494704e-01 | 2.300912e+00 (33/8) |
| quest_minmax | 1 | 16 | 512 | 0 | 576 | 0.826419 | 0.460887 | 2.379178e-01 | 5.367533e-01 | 1.574445e+00 (33/8) |
| quest_minmax | 1 | 16 | 1024 | 0 | 576 | 0.895180 | 0.595039 | 1.386527e-01 | 3.274690e-01 | 7.293618e-01 (15/7) |
