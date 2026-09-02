# Fused R32 page-routing microbenchmark

| Context | Pages | Selected | Eager ms | Page-LSE ms | Top-k/union ms | Fused ms | Speedup | Page rel. L2 | Same selection |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 32768 | 512 | 259 | 0.2730 | 0.0559 | 0.0478 | 0.1038 | 2.629x | 0.00261012 | False |
| 131072 | 2048 | 279 | 0.2997 | 0.1950 | 0.1513 | 0.3483 | 0.860x | 0.00262726 | False |
