# LRQK Runtime Supplement

TP1/B1, Llama-3.1-8B-Instruct BF16, one L40S, basis environment.

| Prompt | GPU-local request (s) | Request-tail decode (ms/token) | Complete |
|---:|---:|---:|---:|
| 32768 | 28.658504265360534 | 187.5007277866406 | 3/3 |
| 65536 | - | - | 0/3 |
| 130048 | - | - | 0/3 |

At 32K, CPU-offload request/tail values were 55.603 s / 379.791 ms; GPU-local build profiling measured 0.910 s in a separate, non-additive pass. 64K failed during warmup prefill MLP allocation; 130,048 failed during online fitting. These are properties of the evaluated implementation/runtime, not proven algorithmic capacity limits. Graph-break/recompilation warnings remain in the original logs. LRQK is not a main-figure series.
