# Llama-3.1-8B-Instruct RULER 128K: original V128

This directory contains the audited V128 results for 11 RULER tasks with 100 frozen prompts per task and arm.

| Arm | 11-task mean | 10-task mean excluding niah_single_3 |
|---|---:|---:|
| Dense Full-K | 86.8061 | 85.5867 |
| B16R16 | 84.9924 | 83.4917 |
| ShadowKV | 84.2076 | 82.6283 |
| LRQK top832 + recent64 | 83.7227 | 82.0950 |
| Loki top856, recent0 | 52.1742 | 57.3917 |

Dense used TP2. The four routing arms ran independently on four GPUs. The native single-GPU Full-K audit checked all 27 observed first-token differences against TP2 plus two controls. Twenty-one of those 27 samples retained the same final score; six differed. This means the Dense number is specifically the audited TP2 protocol.

The predictions directory contains one compact JSONL record per prompt. It retains the complete sample, generated IDs and text, score, routing statistics, kernel counts, runtime, first-token argmax, and SHA256 of the original per-sample file. Repeated protocol and bank metadata is stored once per arm in run_metadata.json.

The 572 MB token tensor file and model/checkpoint weights are intentionally excluded. prompts.json preserves every prompt row, input-token count, input SHA256, answer, task, and the SHA256 of the omitted token tensor.

Reproduction environment: basis.

Formal entry point:

    python -m evaluation.run_llama_ruler100 evaluate

The original terminal ended during Loki, so Loki resumed through four direct evaluator workers using the same evaluator, protocol, frozen prompt bank, and resume audit. V96 evaluation was not started.
