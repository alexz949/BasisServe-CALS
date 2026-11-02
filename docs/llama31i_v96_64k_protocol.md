# Llama-3.1-8B-Instruct: uniform C1-V96, Q64 Base16/R8, RULER64K

Run directory: `results/llama31i_v96_64k`.
Model: `meta-llama/Llama-3.1-8B-Instruct`, revision
`0e9e39f249a16976918f6564b8830bc894c89659`.
Environment: `basis`, Python3.10, PyTorch2.6.0+cu118, Transformers4.51.3.
Slurm: node1 / athena-mini, four NVIDIA RTX A5000 GPUs. No login-node experiments.

## Calibration and fitting

- 64 fit windows ×65536 tokens, plus16 document-disjoint diagnostic windows.
- C4 English train revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`.
  Cached source shards shuffled with seed20260914; each window packs16 unique
  4096-token excerpts, without separators or chat wrapping. All1280 source
  document hashes are distinct. Source, excerpt, tokenizer/model and token-bank
  hashes are recorded in the window manifest.
- Original dense Instruct BF16 teacher, complete causal attention, all token
  positions. Tokenwise MLP execution is chunked at1024 rows for24GiB GPUs.
- C1 uses full `o_proj` input covariance, including cross-head blocks.
  Covariance products are FP32, accumulation across windows is FP64 on CPU,
  merged normalized covariance is FP32. No raw activation corpus is saved.
- Existing Llama GQA C1 solver: uniform rank96 for all8 KV groups/all32 layers;
  activation-weighted SVD initialization, full-layer joint decoder objective,
  six encoder sweeps, fixed16 CG iterations, final decoder refit.
  FP32 factor files are folded and cast to BF16 for evaluation.
- Recovery of layer0 adds finite-factor/finite-solution checks to the existing
  Cholesky jitter search. CUDA returned `info=0` with NaN factors for its
  indefinite reduced matrix (CPU minimum eigenvalue -6.07035e-8). Relative
  jitter1e-3, already in the solver's search range, gave a finite FP32 solution
  with linear-system relative residual1.27499e-6. No ranks, sweep counts,
  objective or requested damping settings changed. Existing completed layers
  are reused; the repaired solver handles the missing shard0 layers.
- After C1 completes, the original dense teacher supplies candidate post-RoPE Q
  at stride64. Fit-only head-whitened Query-Gram selection uses8 bins of8192
  tokens and8 pivots per bin:64 shared absolute positions per layer/window.
- Closed-form affine rank16 pre-RoPE K-MSE Base, fitted with all fit K/V rows.
  Freeze Base, then fit post-RoPE residual R8 using causal non-sink Page32
  Fisher statistics from Q64. BCD40, PCG100, tolerance1e-5 and damping1e-5.
  Diagnostic windows do not choose positions, ranks or factor endpoints.

## RULER and serving path

Official NVIDIA/RULER revision `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`,
`meta-llama3` Instruct template, seed42,30 prompts per task:
`niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2`.
The generator cap is65528, reserving8 tokens for tokenization overhead; every
final prompt plus official generation cap is checked against65536. Template
already contains BOS, so evaluation uses `add_special_tokens=False`.
All330 prompts are included; no historical sample exclusions.

Paired full-K/C1-V96 and sparse-K/C1-V96 arms share one complete causal Triton
C1 prefill. Greedy generation uses official task caps and model EOS/EOT IDs.
Each worker uses2 GPUs, with two prompt shards across the4-GPU allocation.

Sparse support is **sink32 + the last two physical Page32 blocks (including
the current token) +61 complete historical Page32 blocks**, at most2048 unique
physical tokens per KV group. Recent coverage is33–64 tokens depending on the
current page offset; unused capacity is not filled with another historical
page. Sink and recent pages are excluded from historical selection. Short
contexts use all available tokens. This follows the latest user clarification.

Runtime calls `compressed_v_prefill_attention`, `gqa_page32_log_mass_triton`
and `gqa_indexed_sparse_decode_attention_triton` directly. Full decode uses
`compressed_v_decode_attention_triton`. These are resident accuracy runs;
exact K, compressed V and Base128/R8 sidecars remain GPU-resident. Every sample
records kernel invocation counts and the maximum selected-token count.

Before formal evaluation, GPU tests compare V96 kernels with dense selected-
support arithmetic and an identity-factor Llama adapter with native HF Llama.
Two actual64K prompts then run short smoke generations. Any failing test stops
the evaluation script before formal generation.

## Commands and jobs

Activate `basis` and run from `/home/lz299/BasisServe-CALS`.
CPU/OMP/MKL/OpenBLAS thread counts are2 per worker.

```bash
python -u evaluation/llama31_instruct_c1_64k.py prepare
python -u evaluation/llama31_instruct_c1_64k.py capture --shard S
python -u evaluation/llama31_instruct_c1_64k.py merge
python -u evaluation/fit_llama31_8b_c1_joint.py fit-shard \
  --snapshot-dir results/llama31i_v96_64k/covariance \
  --output-dir results/llama31i_v96_64k/c1 \
  --fit-windows 64 --validation-windows 16 --cache-rank 96 \
  --work-dtype float32 --factor-dtype float32 \
  --layer-shard-index S --layer-shard-count 4 --torch-num-threads 2 --resume
python -u evaluation/fit_llama31_8b_c1_joint.py merge \
  --output-dir results/llama31i_v96_64k/c1
python -u evaluation/llama31_instruct_router_64k.py --shard S
python -u evaluation/llama31_instruct_router_64k.py --stage merge
python -u evaluation/llama31_instruct_ruler_data.py
python -m pytest -q tests/test_c1_strict_recent_decode.py
python -m pytest -q tests/test_llama_c1_two_gpu.py
python -m pytest -q tests/test_decoder_cholesky_finiteness.py tests/test_gqa_routed_ov_joint.py
python -u evaluation/llama31_instruct_ruler_64k.py smoke
python -u evaluation/llama31_instruct_ruler_64k.py evaluate --shard T
python -u evaluation/llama31_instruct_ruler_64k.py summarize
```

`S=0,1,2,3`; `T=0,1`. Shell orchestration is preserved in
`scripts/run_llama31i_{c1,router,ruler}_64k.sh`.

- C1 prepare/capture/fit:202077,4 GPUs/8 CPUs/128GiB host memory; failed in
  layer0's first decoder solve with a non-finite objective. Capture/covariance
  and24 fitted layers from shards1–3 completed and are preserved.
- RULER data:202084,2 CPUs/12GiB, completed exit0.
- Q64/Base16/R8:202092,4 GPUs/8 CPUs/416GiB; cancelled after dependency failure.
- RULER tests/smoke/evaluation/summary:202094,4 GPUs/4 CPUs/96GiB,
  cancelled after dependency failure. Neither router nor RULER has run.
- Decoder diagnostics:202098/202099,1 GPU/2 CPUs/12GiB, completed; command
  `python -u evaluation/diagnose_llama31_c1_decoder.py`, logged to
  `logs/decoder_diagnostic.log` (second run appends extended jitter checks).
- C1 continuation:202100,2 GPUs/4 CPUs/24GiB; resumes only shard0 using the
  same fit command above, merges all32 layers, then runs physical-page kernel
  and two-GPU dispatch tests. Completed exit0 in1m54s. Original `fit_0.log` is
  preserved by appending. All32 layers are complete; mean held-out factor-dtype
  attention-output relative MSE is0.0522933917 (layer0:0.0100225589).
- Restored Q64/Base16/R8 job:202101,4 GPUs/8 CPUs/416GiB, afterok202100;
  pending resources because node1 currently has only2 unallocated GPUs.
- Restored RULER job:202102,4 GPUs/4 CPUs/96GiB, afterok202101:202084.

All logs are in `results/llama31i_v96_64k/logs`. Temporary sbatch files were
deleted after submission. The A5000 kernel/adapter/support test run passed
all40 tests in21.03 seconds (`logs/kernel_tests.log`) under the previous strict
recent64 policy. The new physical-page policy passed all37 CPU boundary
tests in2.24 seconds, then all41 support/kernel/adapter/two-GPU tests in14.40
seconds on A5000 (`logs/physical_page_gpu_tests.log`). The numerical solver
regression suite passed24 tests in4.50 seconds (`logs/decoder_tests.log`).
The earlier separate two-GPU test was terminated when its parent allocation
failed; the complete rerun in202100 subsequently passed. C1 capture completed
all80 windows, at18.769GiB peak allocated memory per worker. RULER data job
202084 completed all330 length-validated prompts in16m36s, exit0. C1 fitting
is complete. Router fitting and final RULER accuracy remain pending.
No GitHub write is authorized.
