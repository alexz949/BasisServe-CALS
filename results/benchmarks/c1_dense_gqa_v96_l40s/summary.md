# Dense shared-GQA C1-V96 decode attention

The CUDA arm scans the complete valid exact-K128/C1-V96 cache. It does not perform page selection, packing, routing, or sparse attention.

| Context | BF16 SDPA ms | Old C1 Triton ms | Shared C1 ms | Split | C1 / BF16 speedup |
|---:|---:|---:|---:|---:|---:|
| 32768 | 0.1915 | 2.3895 | 0.1997 | 128 | 0.959x |
| 131072 | 0.7332 | 9.5437 | 0.7844 | 64 | 0.935x |

The traffic-only upper-bound comparison is `1.143x` for `(K128+V128)/(K128+V96)`.
