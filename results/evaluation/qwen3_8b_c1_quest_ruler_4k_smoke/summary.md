# Qwen3-8B-Base C1-V64 + QUEST RULER-v1 4K

Dense-K and physical-shared QUEST use the same C1-V64 ALS5 Value checkpoint and identical greedy prompts.

| Task | Samples | Dense-K + C1-V64 | Physical-shared-1024 | Delta | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 1 | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 1 | **100.00%** | **100.00%** | **+0.00 pp** | **0** | **0** |

Protocol: official RULER-v1 base completion prompts and substring scorers; 100 examples per task; greedy decoding; dense prompt prefill; QUEST enabled for decode after the first generated token; page size 16; fixed 1024-token budget; layers 0 and 1 exact.

This is a quality oracle. Exact K remains GPU resident and QUEST metadata is rebuilt in Python, so elapsed time is not an offload or serving-throughput result.
