# Compact Decoder Microbenchmark

Phase: smoke. Environment: basis.
Real adaptive C1 decoder weights; fixed KV4; BF16 encoder. Single GPU, no communication.
CUDA event median timings. W8A8 decoder excludes activation quantization; combined timing includes it.

| Rank | Layer | Rows | K | BF16 ms | A8/BF16 ms | W8A8 ms | Quantize + W8A8 ms | Combined speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 18 | 1 | 2048 | 0.020064 | 0.040624 | 0.027552 | 0.069200 | 0.290 |
| 64 | 18 | 16 | 2048 | 0.019536 | 0.040960 | 0.026800 | 0.069440 | 0.281 |
| 96 | 18 | 1 | 3584 | 0.052800 | 0.074704 | 0.027440 | 0.071312 | 0.740 |
| 96 | 18 | 16 | 3584 | 0.028880 | 0.050880 | 0.026704 | 0.069264 | 0.417 |

Not a serving benchmark: repeated train-sample latents, warm weights, no TP8 collectives or scheduling.
Wire payload counts are analytical and exclude protocol overhead; no measured communication speedup is claimed.
PPL uses padded HF projections for consistency with the earlier quality baseline; these timings remove inactive padding.
