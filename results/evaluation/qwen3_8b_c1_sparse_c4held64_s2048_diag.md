# C1 K-only Reverse ShadowKV oracle

`teacher_exact` and `teacher_mass` consult full exact K and are not serving policies. 
Logical byte/FLOP counts are not measured CPU-offload speedups.

Each observation is one held-out window at one layer. The tail location therefore identifies both the layer and the window.

| policy | head aggregation | landmarks/page | page | sparse-layer budget | recent | obs. | mass mean | mass min | decoded rel-L2 mean | decoded rel-L2 p95 | decoded rel-L2 max (layer/example) | energy rel-L2 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| quest_c1_latent | logsumexp_head | 1 | 64 | 64 | 0 | 320 | 0.318747 | 0.018014 | 1.262927e+00 | 2.662471e+00 | 6.784163e+00 (33/1) | 1.185122e+00 |
| quest_c1_latent | logsumexp_head | 1 | 64 | 128 | 0 | 320 | 0.536829 | 0.104698 | 8.373515e-01 | 2.141858e+00 | 6.390036e+00 (33/1) | 7.166253e-01 |
| quest_c1_latent | logsumexp_head | 1 | 64 | 256 | 0 | 320 | 0.688072 | 0.283885 | 5.344492e-01 | 1.564281e+00 | 5.872122e+00 (33/1) | 4.941544e-01 |
| quest_c1_latent | logsumexp_head | 1 | 64 | 512 | 0 | 320 | 0.803985 | 0.394322 | 3.003878e-01 | 8.636677e-01 | 4.622151e+00 (33/1) | 3.009901e-01 |
| quest_c1_latent | logsumexp_head | 1 | 64 | 1024 | 0 | 320 | 0.905908 | 0.622348 | 1.358890e-01 | 4.730236e-01 | 1.447315e+00 (15/62) | 1.513781e-01 |
| quest_c1_output | logsumexp_head | 1 | 64 | 64 | 0 | 320 | 0.320251 | 0.018014 | 1.270365e+00 | 2.662471e+00 | 6.784163e+00 (33/1) | 1.196958e+00 |
| quest_c1_output | logsumexp_head | 1 | 64 | 128 | 0 | 320 | 0.534900 | 0.104698 | 8.553016e-01 | 2.136972e+00 | 6.390036e+00 (33/1) | 7.553989e-01 |
| quest_c1_output | logsumexp_head | 1 | 64 | 256 | 0 | 320 | 0.687061 | 0.286581 | 5.393886e-01 | 1.553845e+00 | 5.872122e+00 (33/1) | 4.967578e-01 |
| quest_c1_output | logsumexp_head | 1 | 64 | 512 | 0 | 320 | 0.803274 | 0.401831 | 3.044562e-01 | 8.951198e-01 | 4.619988e+00 (33/1) | 3.020457e-01 |
| quest_c1_output | logsumexp_head | 1 | 64 | 1024 | 0 | 320 | 0.905440 | 0.622348 | 1.393982e-01 | 5.123232e-01 | 1.446972e+00 (15/62) | 1.537525e-01 |
| quest_k | logsumexp_head | 1 | 64 | 64 | 0 | 320 | 0.322248 | 0.018014 | 1.215510e+00 | 2.662471e+00 | 6.784163e+00 (33/1) | 1.180774e+00 |
| quest_k | logsumexp_head | 1 | 64 | 128 | 0 | 320 | 0.547340 | 0.160669 | 7.798750e-01 | 2.029389e+00 | 6.390217e+00 (33/1) | 7.024247e-01 |
| quest_k | logsumexp_head | 1 | 64 | 256 | 0 | 320 | 0.694178 | 0.281305 | 5.009709e-01 | 1.518716e+00 | 6.146120e+00 (33/1) | 4.823307e-01 |
| quest_k | logsumexp_head | 1 | 64 | 512 | 0 | 320 | 0.809517 | 0.397506 | 2.741115e-01 | 7.875456e-01 | 3.663338e+00 (33/1) | 2.966183e-01 |
| quest_k | logsumexp_head | 1 | 64 | 1024 | 0 | 320 | 0.907141 | 0.617888 | 1.362509e-01 | 4.673241e-01 | 1.449737e+00 (15/62) | 1.537710e-01 |
| quest_minmax | max_head | 1 | 64 | 64 | 0 | 320 | 0.319087 | 0.018014 | 1.227900e+00 | 2.656509e+00 | 9.174604e+00 (33/1) | 1.194921e+00 |
| quest_minmax | max_head | 1 | 64 | 128 | 0 | 320 | 0.540226 | 0.150435 | 7.881070e-01 | 2.037427e+00 | 6.539193e+00 (33/1) | 7.128257e-01 |
| quest_minmax | max_head | 1 | 64 | 256 | 0 | 320 | 0.686816 | 0.274106 | 5.078725e-01 | 1.436057e+00 | 5.804181e+00 (33/1) | 4.912973e-01 |
| quest_minmax | max_head | 1 | 64 | 512 | 0 | 320 | 0.806069 | 0.378808 | 2.775500e-01 | 7.807340e-01 | 3.437133e+00 (33/1) | 3.040810e-01 |
| quest_minmax | max_head | 1 | 64 | 1024 | 0 | 320 | 0.904948 | 0.601577 | 1.377423e-01 | 4.711477e-01 | 1.450655e+00 (15/62) | 1.559847e-01 |
