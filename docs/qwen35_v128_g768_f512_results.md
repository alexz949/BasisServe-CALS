# Qwen3.5-9B Two-sided V128 + GDN Wo768 / Full Wo512

Completed on 2026-09-10; PPL finished at 10:25:50 EDT. GSM8K audit status is `complete_and_audited`, with all 1,319 questions and identical prompts, targets, and generation kwargs across comparison arms. Thinking was disabled.

| Configuration | Strict correct | Flexible correct | Numeric-equivalent correct | Length capped |
|---|---:|---:|---:|---:|
| Dense | 1230 (93.25%) | 1233 (93.48%) | 1239 (93.93%) | 6 |
| Dense V + GDN Wo768 / Full Wo512 | 1193 (90.45%) | 1203 (91.21%) | 1223 (92.72%) | 15 |
| Two-sided V128 + GDN Wo768 / Full Wo512 | 1145 (86.81%) | 1143 (86.66%) | 1184 (89.76%) | 33 |

Numeric-equivalent scoring normalizes the existing flexible extraction; it is not manual answer grading. The new arm loses 39 numeric-equivalent correct answers (2.96 percentage points) relative to Dense V with the same Wo ranks. None of these three arms emitted closing-think tags. These measurements do not isolate the effect of V alone, because Wo was refitted on the frozen compressed-V trajectory.

Final full-attention V ranks, in layer order 3, 7, 11, 15, 19, 23, 27, 31, are `[80, 96, 96, 96, 192, 160, 192, 112]`, summing to 1024 (average 128). Selection used anchor 128 and expanded candidates through 224 and 256. GDN Wo uses rank 768 per source and full-attention Wo rank 512 per source.

| PPL dataset | PPL | Windows | Predicted tokens |
|---|---:|---:|---:|
| WikiText | 8.2226337184 | 146 | 297047 |
| C4 evaluation | 11.6606417144 | 128 | 262016 |

PPL model identity was verified by the evaluator. The recorded PPL values are finite and agree with `exp(nll_sum / predicted_tokens)`.

## Execution and reproducibility

Direct execution on this machine (no Slurm), with two OMP/MKL threads per worker. Main fitting used GPUs 5 and 6, with GPU 2 helping on layers 31 and 27. Final GSM8K and PPL used GPU 6. Fitting, audit, and PPL used conda `lowrank`; vLLM GSM8K used the previously authorized `lowrankarena` environment.

Pipeline commands:

```bash
bash scripts/run_qwen35_v128_g768_f512.sh
bash scripts/run_qwen35_v128_ppl.sh 2703776
```

The scripts contain all fitting and evaluation arguments and environment activation. Final evaluation commands were:

```bash
CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
  --bank results/q35_hybrid/banks_v128/c1_twosided_v128.pt \
  --wo-bank results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt --wo-scope all \
  --max-num-seqs 32 --max-num-batched-tokens 4096 --max-new-tokens 1024 \
  --max-model-len 8192 --gpu-memory-utilization 0.40 --kv-cache-gib 6 \
  --seed 20260909 \
  --output results/q35_hybrid/gsm8k_vllm/result_twosided128_g768_f512_wo.json

CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.run_qwen35_hybrid evaluate \
  --bank results/q35_hybrid/banks_v128/c1_twosided_v128.pt \
  --wo-bank results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt \
  --output results/q35_hybrid/ppl/twosided128_g768_f512.json
```

Use `PYTHONPATH=.` for vLLM; use `PYTHONPATH=results/q35_hybrid/deps:.` for HF PPL in `lowrank`.

## Artifacts and limitations

- GSM8K result: `results/q35_hybrid/gsm8k_vllm/result_twosided128_g768_f512_wo.json`
- Audited comparisons: `results/q35_hybrid/gsm8k_vllm/v128_g768_f512_wo_summary.json`
- PPL result: `results/q35_hybrid/ppl/twosided128_g768_f512.json`
- GSM8K log: `results/q35_hybrid/logs/gsm8k_vllm_result_twosided128_g768_f512_wo.log`
- Audit log: `results/q35_hybrid/logs/v128_gsm8k_audit.log`
- PPL log: `results/q35_hybrid/logs/v128_g768_f512_ppl.log`
- Stage commands and fitting logs are documented in `docs/qwen35_v128_g768_f512_protocol.md` and `docs/qwen35_v128_ppl_protocol.md`.

V-ALS includes linear solves that reached the 200-iteration cap without meeting residual tolerance. Completion is not proof of ALS convergence. Wo was fitted in FP64 for six sweeps and passed the existing bank audit. vLLM emitted a process-group cleanup warning at exit; the complete outputs and downstream audit were produced successfully. The vLLM adapter evaluates quality using native-width padding at TP1; these results do not measure actual KV-cache savings or collective communication reductions.
