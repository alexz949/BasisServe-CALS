# Qwen3.5-9B MATH500, MBPP+, and IFEval

Prepared comparison, not yet launched. User selected Dense, Dense V + GDN Wo768 / Full Wo512, and Two-sided V128 + GDN Wo768 / Full Wo512. Reuse the authenticated existing banks; no fitting required. Thinking is disabled for every arm, with the same chat template, deterministic generation, and seed 20260909.

| Task | Questions | Few-shot | Generation cap | Metrics |
|---|---:|---:|---:|---|
| minerva_math500 | 500 | 4 | 4096 | math_verify and exact_match |
| mbpp_plus_full | 378 | 0 | 2048 | greedy base/plus pass@1 |
| ifeval | 541 | 0 | 2048 | prompt/instruction strict/loose accuracy |

MBPP+ uses the repository's official EvalPlus 0.3.1 scorer with v0.2.0 base and augmented tests, isolated CPU subprocesses, and existing resource limits. It is not the sampled pass@3 protocol. MATH500 uses the harness's fixed four examples and `Problem:` stop boundary. IFEval stops at EOS. All task outputs retain raw responses, generation records, truncation counts, package versions, and model/bank hashes. IFEval's cap is explicitly increased from the harness default 1280 to 2048 for all arms.

Use the authorized vLLM environment `lowrankarena`, with `PYTHONPATH=.`, OMP/MKL threads 2, TP1, max model length 8192, max sequences 32, batched tokens 4096, fixed KV cache 6 GiB, GPU memory utilization 0.40. GPU 2 runs Dense, GPU 5 Dense V + Wo, GPU 6 V128 + Wo; each GPU processes three tasks sequentially. Direct execution is used because this machine has no Slurm.

Launch after confirmation:

```bash
bash scripts/run_qwen35_hard_tasks.sh smoke && bash scripts/run_qwen35_hard_tasks.sh full
```

Smoke first runs `python -u -m evaluation.validate_mbpp_plus_full` on CPU to check all 378 canonical solutions, then evaluates two questions per arm/task. Full evaluation begins only if all checks and smoke jobs exit successfully. Each expanded evaluation uses:

```bash
CUDA_VISIBLE_DEVICES=<2|5|6> python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --task <minerva_math500|mbpp_plus_full|ifeval> \
  --max-num-seqs 32 --max-num-batched-tokens 4096 --max-model-len 8192 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/hard_tasks/full/<arm>_<task>.json
```

Dense has no bank arguments. Dense V + Wo adds `--bank results/q35_hybrid/banks/c1_uniform_v256.pt --wo-bank results/q35_hybrid/wo_dense_g768_f512/wo_bank.pt`. V128 + Wo adds `--bank results/q35_hybrid/banks_v128/c1_twosided_v128.pt --wo-bank results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt`. Smoke adds `--limit 2` and writes to `smoke/` instead of `full/`.

Logs: `results/q35_hybrid/hard_tasks/logs/`, one log per stage/arm/task. Dependency preparation: `results/q35_hybrid/logs/hard_tasks_dependencies.log`. Local tests: `results/q35_hybrid/logs/hard_tasks_prepare_tests.log`. Existing results are never overwritten. Results must be compared within this matched protocol; these are quality tests with padded vLLM V cache, not physical cache-memory or communication benchmarks.
