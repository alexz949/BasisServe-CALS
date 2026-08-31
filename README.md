# CALS

CALS is a compact, runnable extraction of the algorithms and experiments that
support C1 / Communication-Aware Alternating Least Squares. It is a code
capsule, not a separate framework: run the preserved Python entrypoints
directly from this directory.

## Core objective

For TP-owned attention sources \(Z_s\), C1 learns a private encoder and a joint
output decoder:

\[
\min_{\{A_s,D_s\}}
\left\|Y-\sum_s Z_s A_sD_s\right\|_F^2 .
\]

The encoder is block diagonal with respect to TP ownership. With encoders
fixed, all decoder blocks are solved in one closed-form least-squares problem.
With decoders fixed, the coupled encoder normal equations are solved by CG.
Alternating these two exact/quadratic subproblems is structured ALS.

The default initialization is activation-aware output regression, followed by
a joint decoder closure. Weight-only SVD and random orthogonal initialization
are retained as ablations. Encoder sweeps are optional; zero sweeps means
“initial encoder + closed-form joint decoder refit”.

## What is included

| Area | Canonical entrypoints or implementation |
|---|---|
| C1 objective, decoder closure, encoder CG/ALS | basisserve/core/gqa_routed_ov_joint.py |
| Activation-aware V/Wo geometry | basisserve/core/gqa_vo_svdllm.py |
| Llama-2 C1 fitting | evaluation/fit_llama2_mha_c1_joint.py |
| Calibration capture/finalization | evaluation/capture_attention_o_proj_ppl_snapshots.py, evaluation/finalize_attention_o_proj_ppl_snapshot.py |
| WikiText-2 PPL and PaLU comparison | evaluation/eval_llama2_mha_v25_comparison.py |
| Layer-wise Global-KL | evaluation/allocate_llama2_mha_c1_global_kl.py |
| Per-TP-source/head-group Global-KL | evaluation/allocate_llama2_mha_c1_tp_source_global_kl.py |
| CKA TP ownership | evaluation/analyze_llama2_mha_c1_cka_head_allocation.py |
| PaLU paper reproduction | evaluation/reproduce_palu_paper_llama2_distributed.py |
| Dense/low-rank AllReduce | basisserve/core/tp_output.py, evaluation/run_low_rank_allreduce.py |
| Top-K AllGather | basisserve/core/topk_all_gather.py, scripts/benchmark_topk_all_gather.py |
| Ragged / prepared uniform AllGather | basisserve/kernels/ragged_allgather.py, basisserve/kernels/feature_ragged_allgather.py, basisserve/kernels/csrc/ |
| Qwen3.5 hybrid private-AG runtime/PPL | basisserve/core/qwen35_*, evaluation/eval_qwen35_hybrid_private_ag_ppl.py |
| Qwen3.5 factor builder | scripts/build_qwen35_gdn_private_ag_joint_factors.py |
| lm-eval | evaluation/eval_llama2_c1_lm_eval.py |

Support modules imported by these entrypoints are preserved under their
original relative paths. See [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md) for
provenance and [RESULTS.md](RESULTS.md) for the promoted results.

## Environment

Use the existing environment:

~~~bash
conda activate lowrankarena
cd /home/lz299/BasisServe/CALS
export PYTHONPATH=.
~~~

requirements.txt lists the direct Python dependencies. The ragged kernel is
JIT-compiled only when called and additionally requires CUDA, a CUDA compiler,
and NCCL development headers/libraries.

## Reproduction flow

First inspect the exact CLI for the preserved revision:

~~~bash
python evaluation/capture_attention_o_proj_ppl_snapshots.py --help
python evaluation/fit_llama2_mha_c1_joint.py --help
python evaluation/eval_llama2_mha_v25_comparison.py --help
~~~

A full C1 fit is sharded by layer. One shard has the following form:

~~~bash
python evaluation/fit_llama2_mha_c1_joint.py fit-shard \
  --snapshot-dir SNAPSHOTS \
  --validation-snapshot-dir VALIDATION_SNAPSHOTS \
  --output-dir FACTORS \
  --fit-windows 128 --validation-windows 64 \
  --cache-rank 96 \
  --work-dtype float64 --factor-dtype bfloat16 \
  --encoder-initialization activation-weighted-svd \
  --decoder-objective full_layer \
  --encoder-sweeps 0 --minimum-encoder-sweeps 0 \
  --selection-boundaries decoder-closed \
  --layer-shard-index 0 --layer-shard-count 4 \
  --device cuda:0
~~~

Run shard indices 0 through 3, then merge:

~~~bash
python evaluation/fit_llama2_mha_c1_joint.py merge \
  --output-dir FACTORS --layers all
~~~

Evaluate the same folded C1 weights on WikiText-2:

~~~bash
python evaluation/eval_llama2_mha_v25_comparison.py evaluate \
  --model LLAMA2_7B \
  --arm c1_joint --factor-dir FACTORS \
  --dataset wikitext2 --split test --seqlen 2048 \
  --batch-size 1 --max-samples 146 \
  --device cuda:0 --output-json OUTPUT.json
~~~

Run the six-task PaLU-style zero-shot suite:

~~~bash
python evaluation/eval_llama2_c1_lm_eval.py \
  --model LLAMA2_7B \
  --arm c1_joint --factor-dir FACTORS \
  --tasks openbookqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande \
  --batch-size 8 --dtype bfloat16 --device cuda:0 \
  --local-files-only --output-json LM_EVAL.json
~~~

Global-KL and exact wire-budget allocation:

~~~bash
python evaluation/allocate_llama2_mha_c1_global_kl.py --help
python evaluation/allocate_llama2_mha_c1_tp_source_global_kl.py --help
~~~

Collective microbenchmarks:

~~~bash
torchrun --standalone --nproc-per-node=8 scripts/benchmark_ragged_allgather.py --help
torchrun --standalone --nproc-per-node=8 scripts/benchmark_ragged_allgather_algorithms.py --help
torchrun --standalone --nproc-per-node=8 scripts/benchmark_topk_all_gather.py --help
torchrun --standalone --nproc-per-node=8 evaluation/run_low_rank_allreduce.py --help
torchrun --standalone --nproc-per-node=4 benchmarks/bench_uniform_allgather.py --help
~~~

## Cheap validation

These checks do not load a model or compile the CUDA extension:

~~~bash
python -m compileall -q basisserve evaluation scripts palu tests
python -m pytest -q tests/test_capsule_imports.py tests/test_gqa_routed_ov_joint.py \
  tests/test_metric_rank_allocation.py tests/test_topk_all_gather.py
~~~

Large calibration, fitting, PPL, lm-eval, and distributed benchmarks still need
the original model/dataset artifacts and suitable GPU resources.


## Experimental feature-major one-sided ragged transport

An opt-in prototype is documented in
[`docs/feature_ragged_one_sided.md`](docs/feature_ragged_one_sided.md). It adds a
feature-major receive arena, a two-sided direct control, and an NCCL 2.29+
one-sided `PutSignal`/`WaitSignal` path followed by one decoder GEMM. The
existing ragged collective remains available as a benchmark control. RMA
requires a coherent PyTorch/NCCL 2.29+ environment; the current `lowrank`
environment is NCCL 2.28.9 and can validate only `feature_direct`.


## Uniform compressed-V AllGather optimization

The fixed-width C1 serving path now has a prepared NCCL fast path and an opt-in
single-node CUDA-IPC prototype with direct fanout, recursive-doubling, and ring
algorithms. Qwen3-32B uniform-V64 TP4 decode additionally fuses its two local
KV sources into one CUDA launch and writes the resulting 1,024 coordinates
directly into the collective source slot. The wire layout and exact AllGather
semantics are unchanged. Build, benchmark, stress-test, and safety details are in
[`docs/uniform_feature_allgather.md`](docs/uniform_feature_allgather.md).
