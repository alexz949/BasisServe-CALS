# C1 K-only Reverse ShadowKV oracle

All `teacher_*` selectors consult full exact K and are not serving policies. 
Logical byte/FLOP counts are not measured CPU-offload speedups.

Each observation is one held-out window at one layer. The tail location therefore identifies both the layer and the window.

| policy | head aggregation | landmarks/page | page | sparse-layer budget | recent | obs. | mass mean | mass min | decoded rel-L2 mean | decoded rel-L2 p95 | decoded rel-L2 max (layer/example) | energy rel-L2 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| quest_k | logsumexp_head | 1 | 64 | 64 | 0 | 320 | 0.322248 | 0.018014 | 1.215510e+00 | 2.662471e+00 | 6.784163e+00 (33/1) | 1.180774e+00 |
| quest_k | logsumexp_head | 1 | 64 | 128 | 0 | 320 | 0.547340 | 0.160669 | 7.798750e-01 | 2.029389e+00 | 6.390217e+00 (33/1) | 7.024247e-01 |
| quest_k | logsumexp_head | 1 | 64 | 256 | 0 | 320 | 0.694178 | 0.281305 | 5.009709e-01 | 1.518716e+00 | 6.146120e+00 (33/1) | 4.823307e-01 |
| quest_k | logsumexp_head | 1 | 64 | 512 | 0 | 320 | 0.809517 | 0.397506 | 2.741115e-01 | 7.875456e-01 | 3.663338e+00 (33/1) | 2.966183e-01 |
| quest_k | logsumexp_head | 1 | 64 | 1024 | 0 | 320 | 0.907141 | 0.617888 | 1.362509e-01 | 4.673241e-01 | 1.449737e+00 (15/62) | 1.537710e-01 |
| teacher_influence | not_applicable | 1 | 64 | 64 | 0 | 320 | 0.535883 | 0.249444 | 7.345747e-01 | 1.327952e+00 | 3.062148e+00 (33/54) | 9.613127e-01 |
| teacher_influence | not_applicable | 1 | 64 | 128 | 0 | 320 | 0.722340 | 0.385122 | 3.193217e-01 | 7.638792e-01 | 1.574927e+00 (33/54) | 3.356291e-01 |
| teacher_influence | not_applicable | 1 | 64 | 256 | 0 | 320 | 0.809292 | 0.528909 | 1.926057e-01 | 4.866229e-01 | 7.112099e-01 (33/21) | 2.129750e-01 |
| teacher_influence | not_applicable | 1 | 64 | 512 | 0 | 320 | 0.879970 | 0.668777 | 1.104235e-01 | 3.034583e-01 | 3.899801e-01 (33/5) | 1.290864e-01 |
| teacher_influence | not_applicable | 1 | 64 | 1024 | 0 | 320 | 0.946850 | 0.814962 | 4.569639e-02 | 1.289018e-01 | 2.007905e-01 (33/24) | 5.645643e-02 |
| teacher_mass | not_applicable | 1 | 64 | 64 | 0 | 320 | 0.555180 | 0.285077 | 6.307440e-01 | 9.898923e-01 | 1.638744e+00 (17/57) | 6.945671e-01 |
| teacher_mass | not_applicable | 1 | 64 | 128 | 0 | 320 | 0.737082 | 0.391636 | 3.278375e-01 | 7.732460e-01 | 9.476231e-01 (33/10) | 3.260887e-01 |
| teacher_mass | not_applicable | 1 | 64 | 256 | 0 | 320 | 0.820915 | 0.542795 | 1.988967e-01 | 5.159189e-01 | 7.019328e-01 (33/24) | 2.230116e-01 |
| teacher_mass | not_applicable | 1 | 64 | 512 | 0 | 320 | 0.887464 | 0.698874 | 1.150758e-01 | 3.133548e-01 | 5.360149e-01 (33/24) | 1.354890e-01 |
| teacher_mass | not_applicable | 1 | 64 | 1024 | 0 | 320 | 0.950428 | 0.835043 | 4.997360e-02 | 1.405066e-01 | 2.279975e-01 (33/24) | 5.937437e-02 |
| teacher_output | not_applicable | 1 | 64 | 64 | 0 | 320 | 0.429863 | 0.064488 | 1.085021e+00 | 3.412843e+00 | 8.619426e+00 (33/43) | 1.268889e+00 |
| teacher_output | not_applicable | 1 | 64 | 128 | 0 | 320 | 0.659478 | 0.211555 | 6.118415e-01 | 2.337525e+00 | 5.498071e+00 (33/22) | 8.674413e-01 |
| teacher_output | not_applicable | 1 | 64 | 256 | 0 | 320 | 0.768889 | 0.295521 | 3.615667e-01 | 1.551745e+00 | 3.477603e+00 (33/22) | 5.627894e-01 |
| teacher_output | not_applicable | 1 | 64 | 512 | 0 | 320 | 0.857060 | 0.443046 | 1.804921e-01 | 8.326691e-01 | 1.790026e+00 (33/38) | 3.335829e-01 |
| teacher_output | not_applicable | 1 | 64 | 1024 | 0 | 320 | 0.938008 | 0.595338 | 6.099665e-02 | 1.490938e-01 | 6.711299e-01 (33/16) | 1.262351e-01 |
