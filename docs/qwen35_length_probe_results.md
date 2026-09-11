# V128 capped-response inspection and longer-output probe

Completed 2026-09-10. This is targeted inspection of selected capped answers and a matched short/long rerun of eight cases, not an exhaustive semantic classification or a new full-benchmark score. All use Two-sided V128 + GDN Wo768 / Full Wo512, with thinking disabled.

Original capped answers include 52/147 MATH500 answers already scored correct by math_verify and 38/165 MBPP+ answers already passing augmented tests. A length cap therefore does not imply an unfinished or incorrect answer.

## Inspected examples

- MATH500 doc 239: repeats an incorrect parallelogram-area formula and correction phrase in the original response.
- MATH500 doc 296: continues comparing equation options, with no completed final answer at the cap.
- MATH500 doc 445: cut off during shoelace arithmetic, before finalizing the area.
- MBPP+ doc 4: repeats definitions of `find_char_long`.
- MBPP+ doc 54: identifies `arr[k-1]` in the original response, but is cut off in the implementation docstring.
- MBPP+ doc 192: already provided code, then repeats the same Final Answer explanation.
- MBPP+ doc 374: code is followed by excessive checking that degenerates into counting hundreds of integers.

## Matched short/long results

IDs below are zero-based original dataset doc IDs. Scores use math_verify for MATH500 and augmented pass@1 for MBPP+.

| Task | Doc | Original full score | Short score | Long score | Short / long generated tokens | Long finish | Long contains exact short text prefix |
|---|---:|---:|---:|---:|---|---|---|
| minerva_math500 | 1 | 0 | 1 | 1 | 1533 / 1533 | stop | True |
| minerva_math500 | 239 | 0 | 0 | 0 | 4096 / 8192 | length | True |
| minerva_math500 | 296 | 0 | 0 | 1 | 4096 / 8192 | length | True |
| minerva_math500 | 445 | 0 | 0 | 1 | 4096 / 8192 | length | True |
| mbpp_plus_full | 4 | 0 | 0 | 0 | 2048 / 4096 | length | False |
| mbpp_plus_full | 54 | 0 | 0 | 0 | 2048 / 668 | stop | False |
| mbpp_plus_full | 192 | 0 | 0 | 0 | 1360 / 1107 | stop | False |
| mbpp_plus_full | 374 | 0 | 0 | 0 | 2048 / 1116 | stop | False |

MATH500 increases from 1/4 to 3/4 correct in this selected matched probe. Docs 296 and 445 become correct when increasing the cap from 4096 to 8192; their longer responses preserve the entire shorter text prefix. Doc 445 reaches the correct area 15, then repeats the solution and still hits the 8192 cap. Doc 239 remains wrong and capped. Doc 1 was already corrected by the small-batch short rerun, so its improvement over the original full run is not attributed to the longer cap.

MBPP+ stays at 0/4 augmented pass@1 when increasing 2048 to 4096. Early output prefixes differ across the two runs, so this is not an exact continuation experiment. Doc 54 finishes but incorrectly sorts the array instead of selecting the original kth element; doc 192 still has an incorrect sorting implementation. Doc 374 passes base tests in the long run but fails augmented tests, despite complete code. Length is not the only failure source.

All eight prompts and target documents match the original full run and the corresponding short/long runs; no input truncation occurred. Model/bank provenance and package versions match. Short/long CLI settings differ only in output path and generation cap within each task. Text-prefix identity is explicitly reported because a repeated greedy run need not reproduce the old full-batch trajectory.

## Execution

Both jobs ran directly in `lowrankarena`, with OMP/MKL 2 threads, seed 20260909, max sequences 32, batched tokens 4096, and KV cache 6 GiB. GPU 2 ran MATH500 at caps 4096 and 8192 sequentially with max model length 12288; GPU 6 ran MBPP+ at caps 2048 and 4096 with max model length 8192. Both job sessions exited 0. Model bank arguments were:

```text
--bank results/q35_hybrid/banks_v128/c1_twosided_v128.pt
--wo-bank results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt
```

Entry point: `python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm`. MATH500 adds `--task minerva_math500 --doc-ids 1 239 296 445`; MBPP+ adds `--task mbpp_plus_full --doc-ids 4 54 192 374`. Each result records its exact command.

Artifacts under `results/q35_hybrid/hard_tasks/length_probe/`: `math_4096.json`, `math_8192.json`, `mbpp_2048.json`, `mbpp_4096.json`, same-named `.log` files, `inspection.json`, and audited `comparison.json`. Audit ran in `lowrank`. Raw original full results are unchanged. vLLM emitted the same process-group cleanup warning at normal shutdown.

The probe supports increasing the MATH500 output budget for some unfinished answers, but these deliberately selected cases cannot estimate the full-test accuracy gain. It provides no observed augmented-test improvement for the four selected MBPP+ cases.
