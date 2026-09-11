# Uniform V96: Full-K, LRQK, Base16/R16 and ShadowKV

Models: Qwen3-8B-Base and Llama-3.1-8B Base, using the HF uniform C1 V96
checkpoints at revision `f1a6253b5d5c747a2475cbf9e704a67d97930b31`.
All layers/physical KV heads use rank96. Exact model revisions, manifests and
factor hashes are authenticated by `evaluation/uniform96_common.py`.

The uploaded uniform V checkpoints themselves were fitted on256x2048 positions
with64x2048 validation (`R96-S6/results.json`). The old remote L40S Qwen V96
protocol instead records32x32768 fit and4x32768 diagnostics. These are different
V-factor fitting settings; uniform rank does not establish checkpoint identity.
The64x32K router calibration below is separate and does not refit the V payload.

Reuse each model's completed uniform ShadowKV88 records and their exact prompt
token IDs, answers, task names, sample indices and generation caps. Generate all88
for every new arm; primary87 mean excludes global index86, and full88 is also
reported. Each task has8 examples; per-task tables use all8. Different model
tokenizers mean the Qwen and Llama prompts are not cross-model paired.

All arms use BF16, greedy generation, saved official caps/model EOS, seed0 and
the same full causal C1 Triton prefill. First generated tokens must match the
completed uniform ShadowKV reference for every prompt. Llama has identity Q/K
normalization and its native RoPE; Qwen retains its Q/K RMSNorm. A small native
Llama causal prefill/decode test authenticates the shared attention interface.

## Calibration

Separate banks are fitted against the respective uniform value encoders. Dense
teacher hidden states, normalized Qwen K (unnormalized Llama K) and Q use original
BF16 SDPA model execution. Qwen reuses `results/calibration/v96kl_64x32k`;
Llama windows are prepared in `results/calibration/llama31_64x32k` with its tokenizer
using the same C4 revision, seed and sampling recipe. These are packed windows:
each32K window contains eight document-disjoint4096-token excerpts, no separators.

64x32768 fit +16x32768 diagnostic validation; validation does not select factors.
Fit-only Query-Gram32, four8K bins with eight pivots each. Base16 is affine
pre-RoPE K MSE reduced-rank regression. Residual16 uses causal non-sink Page32
Fisher statistics,40 BCD sweeps, PCG tolerance1e-5/max100, damping1e-5. Fitting
permits TF32 as in the original calibration; evaluation disables TF32.
Current-layer operands and hidden states remain in host RAM; no large raw
activation files are written. Two workers per model split layer fitting by parity;
each propagates the complete dense teacher to preserve hidden-state provenance.

```bash
python -u evaluation/prepare_v96kl_data.py windows --model /home/lz299/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --calibration results/calibration/llama31_64x32k
python -u evaluation/calibrate_uniform96_router.py --model-family qwen3 --num-shards 2 --shard-index 0
python -u evaluation/calibrate_uniform96_router.py --model-family qwen3 --num-shards 2 --shard-index 1
python -u evaluation/calibrate_uniform96_router.py --model-family llama31 --num-shards 2 --shard-index 0
python -u evaluation/calibrate_uniform96_router.py --model-family llama31 --num-shards 2 --shard-index 1
```

## Evaluation

- Full-K: exact full K and uniform C1 V96.
- LRQK: rank32, per-query topk1152, exact recent64, FP32 routing state,2/2
  iterations, seed0. The GQA union is measured and is not a hard2048 budget.
- `recent_extra`: Base16/R16, Page32,2048 selected tokens including sink32,
  union sliding exact recent64; maximum2112 tokens/group.
- `recent_fixed`: Base16/R16, sink32 +61 disjoint complete historical pages
  +sliding exact recent64, hard2048 tokens/group; shorter contexts retain all.
- ShadowKV: reuse completed rank160/chunk8/routed2048 plus48 outlier chunks,
  local tail and generated tokens. See `docs/uniform96_shadow_protocol.md`.

```bash
python -u evaluation/eval_uniform96_compare.py --model-family qwen3 --arm ARM
python -u evaluation/eval_uniform96_compare.py --model-family llama31 --arm ARM
python -u evaluation/eval_uniform96_compare.py --model-family qwen3 --stage summarize
python -u evaluation/eval_uniform96_compare.py --model-family llama31 --stage summarize
```

`ARM` is `full`, `lrqk`, `recent_extra` or `recent_fixed`. Every arm first performs
two four-token repeatability smoke examples (64 and0) before formal generation.
The final audit checks all440 records/model including the reused ShadowKV arm,
first-token agreement, prompt identity, scoring, caps/EOS and bank hashes.

Environment: `lowrank`, direct execution without Slurm, OMP/MKL threads2 per
process, tokenizer parallelism disabled. Initial fitting GPUs: Qwen3/6 and
Llama4/7. Initial baseline queues: Qwen GPU0, Llama GPU2, Full-K then LRQK.
Ours arms use the fitting GPUs when their respective banks are complete.

Banks: `results/checkpoints/{qwen3,llama31}_uniform96_b16r16`.
Results: `results/evaluation/{qwen3,llama31}_uniform96_compare`.
Logs: `results/logs/uniform96_compare/`.

## Execution notes

Six tests passed: native Llama attention equivalence and existing recent-budget
tests. Initial baseline smoke failed before any result was written because the
new script passed DynamicCache config positionally. It was corrected to the
keyword argument `config`; the same settings were rerun with appended logs.
This issue did not affect fitting or the previously completed ShadowKV runs.

GPU3 acquired additional external load during fitting. After layer10 was saved,
the Qwen even-layer worker was restarted on GPU6 with identical settings and
appended logs. Completed factors are verified and reused; preceding dense
teacher states are replayed. GPU6 temporarily hosts both Qwen fit shards.
The replacement queue waits for Qwen LRQK before running recent_extra on GPU0
and recent_fixed on GPU6. Operator events are recorded in `operator.log`.

Before the Qwen ours launch, GPU2 became idle. A dispatch watcher redirects
recent_extra from the queued GPU0 launch to GPU2, keeping the identical script,
settings and output paths, with appended logs and immutable-record resume.
Recent_fixed remains on GPU6. The dispatch watcher runs the Qwen and combined
audits after both arms finish; an intentional termination of the superseded
GPU0 dispatch is not an evaluation failure.
