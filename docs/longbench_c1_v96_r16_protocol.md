# Offline Page-Fisher R16: L40S queue

## Objective and fixed settings

Test whether offline residual rank16 improves the existing C1-V96/Base16 LongBench result. This is distinct from the concurrent V100 prompt-specific spectral R8 run. Qwen3-8B-Base, BF16, basis, L40S; all36 layers; same192 frozen prompts and six-task scoring. Full causal C1-V96 prefill, C1-V96 sparse decode, exact selected K, Page32/B2048, pinned page0; no adaptive budget or forced current page.

The frozen payload remains C4 32x32K fit /4x32K diagnostic. Base and residual use the same existing C4 captures: 64x32K fit /16x32K diagnostic, Query-Gram32 positions (eight per8K bin). That is 2,097,152 fit tokens, 2,048 fit query positions per layer (each with32 query heads); diagnostic contains524,288 tokens and512 query positions. No additional capture or dataset expansion is included in the primary rank comparison.

Base16 is deterministically refitted with the unchanged closed-form affine MSE RRR procedure on the same inputs. Residual Page-Fisher is refitted at rank16 rather than padded from R8. BCD40 sweeps, relative damping1e-5, PCG tolerance1e-5 and maximum100 iterations. Initial E is the top16 eigenvectors of the group Page-Fisher Gram summed over fit queries and associated query heads; initial U repeats E for those heads. This is deterministic spectral initialization, not zero or random initialization. Final BCD endpoint is exported; diagnostic data do not choose factors.

Keep sweeps and initialization unchanged for the primary experiment so rank is isolated. Sweeps refer to optimization rounds over already-built statistics, not additional samples. Solver diagnostics record each sweep and final query residual/iteration counts. Any later extension of iterations, samples or initialization must be documented separately rather than silently changing this comparison.

## Queue and verification

- 8301254: full-protocol fitting smoke, layers0/35, outputs reused.
- 8301255_0–3: remaining all36-layer fit, four independent workers.
- 8301256: shortest/longest LongBench smoke, repeated logits and full-prefill comparison.
- 8301257_0–3: 192 LongBench prompts,48 per worker.
- 8301258: score and coverage verification, summary.

Each stage depends on successful completion of the preceding stage. Invalid dependencies are cancelled. Runtime failures require inspecting logs and resubmitting the affected stage and dependencies; an error is not bypassed automatically. The active user goal remains open until results are verified. At submission all four L40S were occupied by8301240, whose scheduled limit is2026-09-09 02:55:50 local time; earlier availability is possible, not guaranteed.

Bank: results/checkpoints/c1_v96_b16r16_qgram. Evaluation: results/evaluation/longbench_c1_v96_b16r16. Log prefixes: r16_fit_smoke, r16_fit, r16_lb_smoke, r16_lb, r16_sum. No existing checkpoints/results overwritten.

Summary compares the matching BF16 offline R8 reference (37.0550009215), full-K/C1-V96 (38.6171237069), and new R16. It validates reference hash, prompt identity, scoring, EOS/caps, cache/dispatch, first-token agreement and shard coverage. These are accuracy-oracle runs with resident exact K, not PCIe throughput tests.

## Commands

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_c1_v96_r16_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --layers 0,35
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_c1_v96_r16_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --shard-index 0
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_r16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage smoke
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_r16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index 0
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_longbench_c1_v96_r16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage summarize
```

Formal arrays use shard-index0/1/2/3. Every job uses two CPUs; GPU workers request one L40S each. Slurm uses wrap submission, so no temporary sbatch files remain.

## Completed result and final verification

All five stages completed successfully without resubmission. Resources became available earlier than the scheduler estimate: fitting smoke started September7 at03:50:58 and the final summary completed at04:21:30 local time. Formal fitting shards took14:46–16:49; LongBench shards took5:47–8:39.

Six-task mean: offline R16 37.9562908396, offline R8 37.0550009215, same-payload full-K38.6171237069. R16 improves over R8 by0.9012899181 points and trails full-K by0.6608328673 points. Four task means increase, hotpotqa is unchanged, and2wikimqa decreases. Paired against R8:41 improved,33 regressed,118 tied.

Final audit rechecked36 checkpoint hashes, factor dimensions/finite values,40 recorded sweeps per layer,192 sample records/protocols, cache and dispatch invariants, generation lengths and four-shard coverage. Base16 factors are bitwise identical to the old R8 bank (maximum absolute difference0). The summary audit verifies official scores, EOS/caps and192/192 first-token agreement with full-K. There are45 generation-cap exits without EOS; peak allocated memory25.454 GiB. Result SHA256:6c0aa3d9549189510dd8334975c5af8d987ec0d617cec6889273156036d0b02a.

Solver warning: all36 layers have at least one final query-factor PCG solve reaching the100-iteration cap, with maximum relative residual above1e-5; the largest recorded value is0.02002023533. Thus the run completed with finite factors and valid predictions, but does not establish converged Page-Fisher optima. The reported result retains the same40-sweep/100-PCG-limit protocol as the R8 control; no extra data, alternative initialization or expanded solver iteration setting was tested in this primary comparison.

Detailed per-task scores and paired changes: results/evaluation/longbench_c1_v96_b16r16/summary.md. Machine-readable result and audit are in the same directory.
