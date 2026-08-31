# Qwen3-8B 32K K-only offload benchmark

## Geometry

- GPU: `NVIDIA A100-PCIE-40GB`; layers `36`; sequence `32768`; dtype `bfloat16`.
- Q/K: `32` Query heads / `8` KV heads / D`128`; routing R`32`.
- Pages: size `64`, B`1024` per Query head.

## Decode latency

| Path (all layers) | Median ms | Mean ms | P95 ms | Per-layer median ms |
|---|---:|---:|---:|---:|
| GPU-resident BF16 exact-K full scan | 112.897 | 112.910 | 113.003 | 3.1360 |
| Pinned CPU -> GPU full exact-K copy | 94.675 | 94.681 | 94.768 | 2.6299 |
| Full exact-K onload + full scan | 207.880 | 208.061 | 208.965 | 5.7745 |
| GPU R32 routing scan + page Top-k | 11.895 | 11.905 | 11.960 | 0.3304 |
| Selected exact-K CPU gather + PCIe + pack | 49.742 | 49.807 | 50.094 | 1.3817 |
| Prefetched sparse exact-K/C1-V attention | 20.024 | 20.023 | 20.031 | 0.5562 |
| R32 route + exact-K offload + sparse attention | 78.158 | 78.443 | 80.573 | 2.1711 |

## Traffic and ratios

- Exact K cache: `2.250 GiB` in pinned CPU memory; dense baseline keeps the same amount on GPU.
- Resident R32 sidecar: `0.562 GiB`; resident C1-V: `1.688 GiB`.
- Selected exact-K traffic: `272.047 MiB/token` = `11.808%` of a full K onload.
- Full-scan / R32 routing latency ratio: `9.49x`.
- Offloaded end-to-end / resident full-scan latency ratio: `0.69x`.
- Measured full-copy effective bandwidth: `23.77 GiB/s`.

The benchmark uses synthetic activations. It measures the real pinned-host gather and H2D path, but it is not an end-to-end model tokens/s result.
