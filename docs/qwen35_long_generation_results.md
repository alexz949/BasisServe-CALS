# Qwen3.5-9B V128 + Wo: full longer-generation results

Completed 2026-09-10. Two-sided V128 + GDN Wo768 / full-attention Wo512, thinking disabled. Increasing output limits gives a modest MATH500 gain and no observed MBPP+ gain in these full reruns. Both jobs exited 0; original outputs are preserved.

## Full benchmark results

| Task / metric | Original cap | New cap | Original score | New score | Change |
|---|---:|---:|---:|---:|---:|
| MATH500 math_verify | 4096 | 8192 | 75.60% (378/500) | 77.40% (387/500) | +1.80 pp |
| MATH500 exact_match | 4096 | 8192 | 59.80% (299/500) | 61.00% (305/500) | +1.20 pp |
| MBPP+ augmented pass@1 | 2048 | 4096 | 51.59% (195/378) | 51.06% (193/378) | -0.53 pp |
| MBPP base pass@1 | 2048 | 4096 | 61.90% (234/378) | 62.17% (235/378) | +0.26 pp |

| Task | Original / new capped responses | Original / new generated tokens | Original / new elapsed seconds |
|---|---:|---:|---:|
| MATH500 | 147 / 125 | 840395 / 1383973 | 586.60 / 941.53 |
| MBPP+ | 165 / 159 | 442763 / 766276 | 1397.30 / 1474.16 |

Generated token counts increased approximately 64.7% and 73.1%, respectively. Runtime includes scoring and is affected by shared GPU load; it is not a controlled throughput comparison. MBPP+ generation took approximately 413 seconds, followed by sequential isolated CPU scoring. MATH500 generation took approximately 857 seconds. Capped answers may already contain a correct answer; these counts are not failure counts.

## Paired changes and interpretation

MATH500 has 31 wrong-to-correct and 22 correct-to-wrong transitions. Of the 31 improvements, 27 were originally capped, but only two of those preserve the original response as an exact full text prefix. MBPP+ has 15 wrong-to-correct and 17 correct-to-wrong transitions; 12 improvements were originally capped, none retaining the complete original text prefix.

These are fresh full reruns with identical rendered prompts, target documents, model factors, package versions, seed, and greedy generation settings. They are not resumed original generations. MATH500 also raises maximum total context from 8192 to 12288. Different batching/scheduling under longer limits can change response trajectories, so the full score difference should not be attributed entirely to completing truncated text. No repeated-seed uncertainty estimate was performed.

The results support a modest observed MATH500 gain from the larger-budget configuration, but do not support increasing the MBPP+ budget as an effective fix. Repeated final-answer text was observed in long MATH500 responses, and many answers still reach their larger caps. The runs configured thinking off, but literal closing `</think>` strings occurred in 23 MATH500 and 8 MBPP+ responses (originally 2 and 6); these are output artifacts, not evidence that thinking was enabled in the request.

## Execution

Use the previously configured vLLM `lowrankarena` environment; GPU 6 for MATH500, GPU 5 for MBPP+, concurrent direct execution without Slurm. Each job uses two OMP/MKL threads, TP1, max sequences 32, batched tokens 4096, KV cache 6 GiB, seed 20260909, and the same V/Wo banks. MATH500's KV cache approached capacity during generation, affecting scheduling without an OOM failure.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --task minerva_math500 \
  --bank results/q35_hybrid/banks_v128/c1_twosided_v128.pt \
  --wo-bank results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt \
  --max-new-tokens 8192 --max-model-len 12288 \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/hard_tasks/full/v128_wo_minerva_math500_8192.json \
  > results/q35_hybrid/hard_tasks/logs/full_v128_wo_minerva_math500_8192.log 2>&1

CUDA_VISIBLE_DEVICES=5 HF_ALLOW_CODE_EVAL=1 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --task mbpp_plus_full \
  --bank results/q35_hybrid/banks_v128/c1_twosided_v128.pt \
  --wo-bank results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt \
  --max-new-tokens 4096 --max-model-len 8192 \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.40 --kv-cache-gib 6 --seed 20260909 \
  --output results/q35_hybrid/hard_tasks/full/v128_wo_mbpp_plus_full_4096.json \
  > results/q35_hybrid/hard_tasks/logs/full_v128_wo_mbpp_plus_full_4096.log 2>&1
```

The commands above are shown separately; actual jobs ran concurrently. Both emitted the previously observed FLA short-sequence format warning and NCCL process-group cleanup warning. Both engines shut down and the jobs exited successfully. Math scoring logs include verbose responses that the exact-match parser could not extract; both exact-match and math_verify scores are retained.

## Audit and artifacts

`evaluation/summarize_qwen35_long_generation.py` verified old-source hashes against the previous audited summary, identical provenance and versions, matching prompts/targets for all 878 questions, no input truncation, complete unique sample IDs, unchanged non-length arguments, and metric/count recomputation from per-question records. It records all paired transitions and text-prefix identity. Syntax compilation and `git diff --check` passed.

```bash
conda activate lowrank
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
python -u -m evaluation.summarize_qwen35_long_generation \
  > results/q35_hybrid/hard_tasks/logs/long_generation_audit.log 2>&1
```

Audited summary: `results/q35_hybrid/hard_tasks/long_generation_summary.json`. Raw results and execution logs are at the paths in the commands above. The audit ran on CPU in `lowrank` and exited 0.
