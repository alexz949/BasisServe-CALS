# Terminal Query-Gram Q32 residual experiment

## Fixed configuration

Qwen3-8B-Base, frozen uniform C1-V96 and bitwise-reused Base16 from results/checkpoints/c1_v96_b16r16_qgram. Refit offline Page-Fisher R16 only: 40 BCD sweeps, PCG100, tolerance and damping 1e-5, spectral initialization. Page32, B2048, pinned/excluded first32 tokens.

64 C4 fit windows of32768 tokens. Select32 query positions per layer in [24576,32767] from original gram3, retaining the original full-window fit-only whitening and candidate grid. First8 pivots must equal the original terminal-bin8. No model recapture. Diagnostic remains the original16 windows and full-window Q32; diagnostics do not choose factors.

Fitting uses V100 FP32 with existing BF16 captures. LongBench uses V100 FP16, C1-V96 prefill/decode and the same192 prompts. Reference full-window Q32 mean38.81467579869914; reference factors were fitted on L40S, so fit hardware is not identical. This is an accuracy oracle, not a CPU-offload benchmark.

## Jobs and exact commands

Environment: /home/zhangal/.conda/envs/basis/bin/python. All stages use Slurm with success dependencies. GPU stages request one V100 per shard,2 CPUs and64GiB host RAM; formal arrays have4 shards.

- 8301616: select

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/select_qgram_terminal32.py
```

- 8301617: extract

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/extract_qgram_terminal32_fit.py
```

- 8301618: fit_smoke

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_c1_v96_r16_terminal32.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 0,35
```

- 8301619: fit

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_c1_v96_r16_terminal32.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --shard-index $SLURM_ARRAY_TASK_ID
```

- 8301620: lb_smoke

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_terminal32.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
```

- 8301621: lb

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_terminal32.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index $SLURM_ARRAY_TASK_ID
```

- 8301622: sum

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_terminal32.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```

## Artifacts

- Selection: results/evaluation/qgram_terminal32
- Extracted fit queries: results/calibration/qgram_terminal32_fit
- Bank: results/checkpoints/c1_v96_b16r16_terminal32
- Evaluation: results/evaluation/longbench_r16_terminal32_fp16
- Logs: tq32_<stage>_<job>_<shard>.log

## Completed results

All36 checkpoint hashes, terminal query positions and40/100 settings verified. Base16 tensors are bitwise equal to the reference bank at every layer. All GPU jobs and score verification completed successfully. Evaluation shard durations:13:03,11:08,15:16,17:34. The192-prediction audit verifies official scores, first tokens, cache/dispatch and shard coverage. Result SHA256:e9b64365de9377011d4056175f3f016ca4a48ab17f57b9491cc64e7c7de19de1.

| Task | Full-window Q32 | Terminal Q32 |
|---|---:|---:|
| qasper |31.8998|32.5766|
| multifieldqa_en |42.6738|43.2112|
| hotpotqa |56.4416|56.4416|
| 2wikimqa |43.8690|42.7530|
| gov_report |29.6477|30.0178|
| qmsum |28.3561|27.7730|
| Mean |38.81467579869914|38.79553559385474|

Mean difference:-0.0191402048444 points. Paired scores:33 improvements,28 regressions,131 ties. Terminal42 predictions reached the generation cap without EOS; peak allocated memory25.45413827896118GiB. Both compared generations use V100 FP16.

On identical full-window diagnostic Q32, layer-average Page-Fisher NMSE increases0.13163288754990252 to0.24668148598822193; all36 layers increase. Fit NMSE0.07500868598340754 versus0.06953947100868414 uses different query positions and is not a same-target comparison. Every layer has a final query PCG solve reaching100 iterations; maximum final relative residual0.013804352842271328. Full numerical convergence is not asserted.

The LongBench means are nearly identical despite the higher full-window diagnostic loss. This does not establish statistical equivalence or improvement; the192 prompts are reused development examples.
