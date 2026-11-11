<!-- Agent-generated read-only audit notes (2026-11-10). Key claims re-verified in ../COMPATIBILITY.md; treat the rest as leads, not verified facts. -->
# TP1 / 130,048-token / L40S compatibility audit (Dense, BasisKV, LRQK, ShadowKV)

Audit date 2026-11-10. Read-only: nothing under `/workspace/BasisServe-CALS` was changed, nothing ran on a GPU, no
pytest, no torch import. `nvidia-smi --query-gpu` was used only to identify the node. Upstream repos were cloned
read-only into `scratchpad/upstream/{LRQK,ShadowKV}` at the pinned commits. Nothing in them was executed.

Tags used below: **[V]** = verified directly in a file, code or hash. **[I]** = inference from code or arithmetic, not
measured. "repo:" paths are relative to `/workspace/BasisServe-CALS`. "up-LRQK:" means
`scratchpad/upstream/LRQK/lrqk_attention.py` at `caf16293`. "up-SKV:" means
`scratchpad/upstream/ShadowKV/models/kv_cache.py` at `e51904cd`.

Not fetched: the raw per-trial logs and JSON on the user's Hugging Face artifact repo. These include the
LRQK OOM tracebacks and compile warnings, the frozen `dependencies.tar.gz` that holds the dirty
`external/LRQK` and `external/ShadowKV` trees, and the `prompts.safetensors` cohorts. They are named in the
repo but are not in this checkout. Section 8 lists their exact paths.

---

## 0. Current node (for reference)

- **[V]** 4x NVIDIA L40S, 46,068 MiB each, driver 570.211.01, PCIe Gen4 x16. UUIDs match
  `results/rebuttal/v2q8_palu_g4_tp4/logs/gpu_check_l40s_node2.csv`, so this is "L40S node 2".
- **[V]** Each GPU is currently about 27.6 GB in use by another job.
- **[V]** CPU: Xeon Platinum 8558, 192 threads, 4 NUMA nodes, about 2 TB RAM.
- **[V]** Single-GPU pinned host-to-device bandwidth is 27.0 GB/s (`results/interconnect_l40s_node2.log:122`).
- **[V]** Environment `/workspace/basis_artifacts/envs/vllm/basis` has torch 2.13.0+cu129, transformers 5.22.2,
  triton 3.7.1, vLLM 0.30.0. The vLLM FA2 varlen entry point that `paper_faithful_compat.py` calls exists with a
  compatible positional order (`vllm/vllm_flash_attn/flash_attn_interface.py:174-206`).
- **[V]** Not present on this node:
  - `external/LRQK` and `external/ShadowKV` (never tracked in git: `git ls-files | grep ^external/` returns 0).
  - `results/system_benchmarks/tp1_sparse_local/frozen_source`, which `audit_tp1_request.py:17` needs.
  - `/workspace/runs/...`, which holds the model cache paths, the V128 router bank and the calibration prompts.
  - Consequence: the existing TP1 harness **cannot run from this checkout as is**.
- **[V]** Present locally:
  - the V96 checkpoint `/workspace/basis_artifacts/checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix`
    (`value_coordinate_encoders [8,128,96]`, `head_output_decoders [32,96,4096]`, BF16);
  - calibration windows `c4-48x128k` (48x131072, seed 20260921);
  - the model at `/workspace/.hf_home/...` (`basis_artifacts/bench_model_path`).
- **[V]** Not present locally: the V96 router banks (`router_b16r16`, `router_page4_b16r16`, ...). They exist only
  on HF, according to the cached tree listing
  `basis_artifacts/hf_stage/.cache/huggingface/trees/57d90e28....json`.

---

## 1. Which arms already have a working TP1 runtime (Q1)

### Main runtime

The main runtime is the "TP1 offline" harness:

- Entry point: `benchmarks/system/run_tp1_offline_grid.py`. It launches `benchmarks/system/bench_tp1_offline_request.py`.
- Runtime factories: `benchmarks/system/audit_tp1_request.py`.
- Arm names: `dense`, `basis`, `shadowkv`, `lrqk` (`run_tp1_offline_grid.py:13`; `bench_tp1_offline_request.py:161`).

| Arm | Factory (file:line) | Last formal run | Contexts | 130,048 completed? |
|---|---|---|---|---|
| dense | `audit_tp1_request.py:55-93` (frozen `bench_tp1_dense_full_v6.py`) | tp1_offline, 2026-09-24, 1x L40S | 16K (steady), 32K, 64K, 130,048 | **Yes, 3/3** (request n128; steady) |
| basis | `audit_tp1_request.py:55-93` (frozen `bench_tp1_sparse_full_v6.py` + `bench_tp1_router_config.build("w16u1")`) | same | same | **Yes, 3/3** (request n32/n128/n512; steady) |
| shadowkv | `audit_tp1_request.py:96-154` (upstream native `models.llama.Llama`, `attn_mode="shadowkv_cpu"`) | same | 32K, 64K, 130,048 | **Yes, 3/3** (n32/n128/n512). No steady mode. |
| lrqk | `audit_tp1_request.py:157-197` (upstream `lrqk_attention` from `external/LRQK`) | same | 32K, 64K, 130,048 | **No.** 2/3 at n128 (cohort 0 OOM). 0/3 at n32 and n512. All failures were OOM in `same_shape_runtime_warmup` (`results/system_benchmarks/tp1_offline/formal_outcomes.json`; `formal_summary.csv`). |

Hardware:

- **[V]** `results/system_benchmarks/tp1_offline/SUMMARY.md:6` and `freeze/provenance.json`: "one NVIDIA L40S (GPU 0)",
  repository HEAD `bab891d3`, LRQK head `caf16293`, ShadowKV head `e51904cd`. The provenance file notes "Dirty source trees".
- **[V]** `results/rebuttal/table3_k_offload_128k/README.md:19-20` says the paper's TP1 runs were on "RunPod L40S,
  torch 2.13.0+cu130".
- **[V]** The same-harness rerun `results/system_benchmarks/tp1_paper_scaling/formal/manifest.json` (2026-09-25)
  records an 8x L40S node, driver 580.178.04. Its result JSON reports torch `2.13.0+cu130`.

### Other TP1 runtimes and supplements

| Runtime | Arms | Hardware | Contexts / outcome |
|---|---|---|---|
| `bench_tp1_lrqk_local.py` / `run_tp1_lrqk_local.py` (LRQK with exact K/V on GPU) | lrqk | L40S, same environment | 32K 3/3. 64K 0/3 and 130,048 0/3 (warmup OOM) (`tp1_lrqk_local/SUMMARY.md:11-32`) |
| `run_tp1_paper_scaling.py` (byte-exact copy of the frozen harness) | dense, basis (steady only) | 8x L40S, driver 580.178.04, torch 2.13.0+cu130 | 16K-130,048 x3 cohorts, 42/42 complete |
| `bench_table3_k_offload.py` / `run_table3_k_offload.py` (rebuttal) | dense_resident, basis_resident, basis_resident_nobase, basis_k_offload_nobase, plus `_p32`/`_p4` | "node6" L40S, driver 550.54.15, torch 2.13.0+cu126 (README:19-21) | 130,048 x3, steady only, 2026-11-09 |
| `run_tp1_capacity.py` | dense-local, dense-K-offload, basis-K-offload | L40S | 65,536 and 131,072, B1-B4, decode only |
| `bench_tp1_sparse_full_v6.py`, `bench_tp1_dense_full_v6.py`, `bench_tp1_router_config.py`, `tp1_decode_v7/v8` | basis, dense (+ShadowKV/LRQK "default" in v8) | L40S | 16K-131,072, one teacher-forced prompt. v8 LRQK at 128K "OOM on first decode" (`tp1_decode_v8/E2E_AND_BREAKDOWN.md:55-61`) |
| `benchmarks/a100/tp1_paper_scaling.py` | dense, basis | 8x A100-80GB | smoke only recorded (`benchmarks/a100/README.md`) |

No GH200 or A5000 TP1 runtime was found. A5000 appears only for TP4 PaLU in `INTERCONNECT_L40S.md`.

---

## 2. Per-runtime configuration, timing boundary and memory method (Q2)

### 2.1 Dense (tp1_offline / tp1_paper_scaling / table3 `dense_resident`)

- **Value** [V]: dense V128, original W_O. Preallocated GPU K and V caches
  (`bench_tp1_dense_full_v6.py:82-85`). Capacity is prompt + output tokens (`audit_tp1_request.py:69`).
- **Keys** [V]: GPU-resident.
- **Attention** [V]: flash SDPA for prefill and decode, `enable_gqa=True` (`bench_tp1_dense_full_v6.py:233-262`).
- **Memory-saving wrappers** [V]: `ChunkedTokenwise` MLP (1024 tokens) and norm wrappers (2048 tokens) (`:316-322`).
- **Sanity check** [I]: 23.85 ms of dense attention per step at 130,048 tokens is about 83% of the L40S HBM
  bandwidth bound. That bound is 17.0 GB of K+V divided by about 864 GB/s, or 19.7 ms. Dense is not artificially slow.

### 2.2 BasisKV (`basis`)

**Value** [V]
- Dense V128 (`bench_tp1_sparse_full_v6.py:158`).
- Kernels are compiled with `-DBASIS_VALUE_DIM=128` (`bench_tp1_router_config.py:47`; `audit_tp1_request.py:64`).

**Keys** [V]
- GPU-resident in tp1_offline (`storage="local"`, `audit_tp1_request.py:69`; `bench_tp1_sparse_full_v6.py:161-166`).
- Mapped pinned-host Keys plus persistent GPU K slots exist in the offload variants: tp1_capacity, tp1_router_config,
  and table3 `basis_k_offload_nobase` (`:167-170`, `:192-211`).

**Router** [V]
- B16R16. The Base code is `V128 @ base_left [8,128,16]`. The residual is `(K - RoPE(predicted K)) @ residual_encoder`
  (`:269-290`).
- Bank: `/workspace/runs/l31-router-source/v128-router/ours_b16r16` (`:47`).
- Page32, full scan of every historical page (`:380-385`), GQA-max aggregation.
- 16-warp register router, `w16u1` (`bench_tp1_router_config.py:19,32-34`).

**Budget** [V]
- `ROUTED_PAGES=62`, `RECENT_TOKENS=64`, so `SUPPORT_TOKENS=2048` (`:49-52`).
- The fused selector pins prefix page 0: `select_fixed_group_max_pages_impl(..., kRoutedPages=62, pinned_prefix_pages=1, ...)`
  (`basisserve/kernels/csrc/conditional_router_page32.cu:1307-1330`; pinned-page semantics at `:714-780`).
- The effective policy is therefore **a 32-token sink plus 61 routed Page32 pages plus 64 exact recent tokens,
  2048 tokens hard**. That equals the quality protocol "sink32 + recent64 inside B2048". It is **not** "2048 + 64".

**Prefill** [V]: exact flash SDPA. Codes are written in 2048-token blocks inside prefill (`:609-630`).

### 2.3 LRQK (`lrqk`, upstream imported)

Parameters [V]:
- `DynamicLRQKCache(num_key_value_groups=4, r=32, num_active_tokens=2048, lite_tokens=64, max_iter=(2,2), tol=1e-8,
  max_sequence_length=L+N+64, lwattn_factory=LightAttentionIndicesOffloadPrefill, init=randn)`
  (`audit_tp1_request.py:180-185`).

Placement [V]:
- **V128 and exact K/V on host**: pinned `KVcpu` (up-LRQK:1089-1095).
- Routing state `A_K`, `B_Q`, `B_K` on GPU.
- Selected `Kgpu`/`Vgpu` of shape `(1, 32 query heads, 2048+64, 128)` on GPU (up-LRQK:1418-1427). These buffers are
  expanded per query head, four times the KV-head size.

Prefill state movement [V]:
- During prefill, each layer's `Kgpu`, `Vgpu` and `A_K` are copied asynchronously to CPU (up-LRQK:1617-1653).
- They are restored lazily on that layer's first decode (up-LRQK:1655-1681).

Budget [V]:
- Per query head: top-2048 historical tokens plus the 64-token lite ring.
- The physical GQA union is not capped: up to 4 x 2048 + 64 = 8,256 tokens per KV group.

Adapters [V]:
- vLLM FA2 adapter for `flash_attn` (`paper_faithful_compat.py:40-136`; B1-only assert at `:57`).
- Model loaded as eager and then switched to `flash_attention_2` (`audit_tp1_request.py:167-170`).
- Prefill MLP chunked to 1024 tokens only when the length is at least 130,048 (`:171-174`).
- Forward runs under BF16 autocast (`:189-192`).

### 2.4 ShadowKV (`shadowkv`, upstream native engine)

Engine [V]:
- `models.llama.Llama(attn_mode="shadowkv_cpu", sparse_budget=2048, rank=160, chunk_size=8, minference=False)`
  (`audit_tp1_request.py:119-122`). This is ShadowKV's own Llama runner, not the HF model.
- Import adapters (`paper_faithful_compat.py:139-169`): MInference stubs, vLLM `silu_and_mul` alias, FA2 adapter.
- Compatibility patches: `rope_theta` property, shared `rotary_emb` (`audit_tp1_request.py:102-112`).

Placement [V]:
- **V128 on host**: pinned `v_cache_cpu` (up-SKV:400-407).
- Keys are reconstructed on GPU from a rank-160 global pre-RoPE SVD. The FP32 `torch.svd` runs per layer inside
  prefill (up-SKV:480-513).
- U, SV and landmarks are first built on CPU, then moved to GPU by `H2D()` (up-SKV:668-688). In the harness this is
  the `place()` phase.

Support [V]:
- 2048 routed (256 chunks of 8), plus 48 outlier chunks (384 tokens; up-SKV:398), plus the local tail.
- At 130,048 tokens the chunk count is aligned to a multiple of 8 (up-SKV:548), so the local tail is 64 tokens.
- Generated tokens are added on top: **2,496 + generated tokens per KV head**.

Harness change [V]:
- The generation buffers are enlarged to the actual support plus the output count (`audit_tp1_request.py:123-137`).
- The default buffer is 2,592 rows. 128 outputs need 2,624.

Kernel patch [V]: the separate `external/ShadowKV` checkout carries a CUDA current-stream patch to
`batch_gemm_softmax` (`RESULTS_AND_KERNELS.md:84-109`). The patch changes stream semantics only.

### 2.5 Timing boundaries (all arms, `bench_tp1_offline_request.py`)

**Request** [V] (`:40-80`)
- Wall-clock `perf_counter`, with `torch.cuda.synchronize()` only at phase boundaries (`:21-23`).
- The boundaries are:
  1. start;
  2. `prefill()`, then sync;
  3. first argmax, then `place()` (ShadowKV `H2D`, a no-op for the others), then sync;
  4. N-1 decode calls with greedy argmax and no per-step sync, then sync.
- `request_seconds = stop - start`.
- Construction happens inside these phases:
  - ShadowKV SVD and landmarks run inside prefill. Placement is a separate phase.
  - The LRQK fit runs inside prefill. The state restore happens inside the first decode, so the "ready" delay includes
    that decode work (`:51-63`; `SUMMARY.md:34`).
- Excluded:
  - model loading, tokenization and input transfer (the prompt is already `.cuda()`);
  - upfront cache allocation (`:209`).
  - A same-shape warmup request runs first, then the state is reset (`:189-195`).
- "Steady tail" is the wall-clock mean over decode calls 17 to N-1, including greedy selection (`:64-77`). It is
  **not** a CUDA median.

**Steady** [V] (`:130-155`)
- Prefill and placement, then 16 conditioning plus 128 measured decode calls.
- CUDA events and synchronized wall time per step. Argmax is outside the timed interval.
- Allowed **only for dense and basis** (`assert args.method in ("dense","basis")`, `:175`).
- The attention block comes from a separate 32-step profiled pass with per-layer CUDA events (`:220-232`). The block
  spans from after QKV/RoPE to before `o_proj`.

**Separate build profile** [V] (ShadowKV, LRQK; `:83-127`): synchronized wrappers. They are explicitly not additive
to the request time.

### 2.6 Memory method

**[V] tp1_offline**
- Only `torch.cuda.reset_peak_memory_stats()` before the measured run, then `max_memory_allocated()` after it
  (`:196, :204`).
- That gives one peak over prefill and decode. It does not report reserved memory, decode-resident memory, host bytes,
  staging, or a tensor inventory.
- The steady-mode peak also includes prefill.

**[V] table3** (`bench_table3_k_offload.py:65-75, 160-206`, `SUMMARY.md` "Memory by phase" and "State-level bytes")
- Phase snapshots: allocated, reserved, peak allocated, peak reserved, `mem_get_info` device used, process RSS.
- A per-tensor inventory by state category and device, with allocated and active bytes.
- Only dense and basis arms.

**[V] TP8 ShadowKV**: `cache_state_bytes` (`basisserve/core/shadowkv_tp8_attention.py:123-133`) groups U, SV,
landmarks, CPU V, selected buffers and workspace.

### Measured values at 130,048 tokens (tp1_offline, three-cohort medians)

| Arm | Request s (n128) | Prefill-incl. s | Post-prefill s | Decode s | Tail ms/token | Steady CUDA ms | Attn block ms | Peak alloc GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 43.418 | 37.424 | 0.0002 | 5.994 | 47.179 | 47.370 | 23.857 | 38.36 |
| basis | 42.898 | 38.128 | 0.0002 | 4.770 | 37.522 | 37.817 | 14.649 | 40.48 |
| shadowkv | 58.805 | 53.526 | 1.021 | 4.133 | 32.594 | n/a | n/a | 20.61 |
| lrqk (c1/c2 only) | 104.01 / 104.39 | 52.53 / 51.75 | ~0.0005 (ready 1.38/1.36) | 51.48 / 52.64 | 399.3 / 410.2 | n/a | n/a | 29.48 |

Source: `aggregate.csv`, `formal_summary.csv`. The ShadowKV build profile is 20.35 s. The LRQK build profiles are
17.71 s and 16.80 s.

---

## 3. LRQK deep dive (Q3)

### 3.1 Evidence of compile warnings and incomplete trials

**[V] Statements in the repo**
- `tp1_offline/SUMMARY.md:79`: "The LRQK environment emitted PyTorch graph-break and recompile-limit warnings during
  screening. Raw logs retain these warnings."
- `tp1_lrqk_local/SUMMARY.md:36` and `COMPARISON.md:67` repeat this.
- `paper_figures/tp1/lrqk_appendix.md:11`, and `paper_figures/tp1/inputs/Pasted text.txt:458-462`
  ("compiler graph-break / recompilation warnings ... incomplete longer-context results").

**[V] What the verifiers checked**
- `verify_tp1_offline.py:52-56` and `summarize_tp1_lrqk_local.py:37-39` scan only the *measured* section of each log
  for the strings `"AUTOTUNE"` and `"recompile_limit"`.
- Both found none: `verification.json` has `primary_timing_compile_warning_logs: []`.
- So the warnings were in warmup or screening sections. The verifier notes this "is not a profiler proof".

**[V] Incomplete 130,048-token trials**
- `formal_outcomes.json` has 7 LRQK OOMs, all with phase `same_shape_runtime_warmup`:
  - c0/n128, c0/n32, c0/n512, c1/n32, c1/n512, c2/n32, c2/n512.
- The successful c1/c2 n128 trials measured a peak of only **29.48 GiB** on a 44.99 GiB card.
- The capacity smoke at 130,048/n4/c0 completed (`capacity_summary.csv`).
- GPU-local supplement: 64K failed at "prefill MLP temporary allocation" and about 128K failed "inside online
  fitting" (`tp1_lrqk_local/COMPARISON.md:65`).
- The earlier v8 run reports 128K "OOMed while restoring its official per-layer low-rank decode state"
  (`tp1_decode_v8/E2E_AND_BREAKDOWN.md:60-61`).

### 3.2 Likely causes (code reading; logs not inspected)

**Graph breaks** (up-LRQK, all five `@torch.compile(fullgraph=False, options={"max_autotune": True, ...})` functions
at :72, :212, :313, :427, :523)

- [V] Data-dependent Python control flow inside the compiled solvers: `delta = max(tensor, tensor)` and
  `if delta < tol: break` (:291/:299, :401/:413, :500, :609). Each forces a tensor-to-bool conversion, which is a
  graph break once per iteration.
- [V] `_take_along_dim_with_mask_python_indices` (:79-99) uses `indices_mask.nonzero()` (:92) and boolean-mask
  indexing (:95). Both have data-dependent output shapes. This function runs on every decode step for every layer.
  It sees a different hit/miss count each call. [I] That is a natural source of resume-frame recompiles.
- [V] `max_autotune` makes the first call per shape very expensive. The 130K FP32 prefill graph is one example.
- [I] The exact frame that hit `recompile_limit` cannot be identified without the logs.

**OOM**

[V] Memory facts:
- Upstream `AutoIncreaseTensor` sizes the first allocation at `max(capacity, int(1.5*L))` (:850, :881). At 130,048
  tokens that is 195,072 rows for both `A_K` on GPU and `KVcpu` on host.
- `A_K` alone is 11.91 GiB on GPU after the first decode.

[V] Code facts:
- `torch.cuda.empty_cache()` runs on every layer of every decode step (:2023), and again on growth (:919).
- `only_indices_in_cpu` builds a `(1,32,2048,2048)` boolean match matrix per layer per step (:1276-1316).

[I] Assessment:
- A measured peak of 29.5 GiB with about 15 GiB of headroom, together with every failure occurring only in warmup,
  points to allocations that happen only on the first (warmup) request.
- The most likely candidates are Inductor `max_autotune` benchmarking of the 130K FP32 solver graphs and the first
  cuSOLVER and cuBLAS workspaces, near the first-decode restore of 11.9 GiB of `A_K`.
- Alternative: host pinned-memory exhaustion reported as a CUDA OOM. `KVcpu` is 23.8 GiB pinned, plus about
  13 GiB of pageable restore staging. The runner labels any log containing "CUDA out of memory" as `gpu_oom`
  (`run_tp1_offline_grid.py:124-125`).
- Fragmentation driven by `empty_cache` is less likely.
- GPU-local OOM is **structural**: 14.96 + 11.91 + 23.81 + 1.03 = 51.7 GiB of GPU state, more than the 44.99 GiB card.

**[V/I] TF32 leak**
- [V] up-LRQK:33 calls `torch.set_float32_matmul_precision('high')` at import time.
- [V] The harness disables TF32 at `bench_tp1_offline_request.py:182`, then imports LRQK inside the factory (`:186`).
- [I] So the LRQK FP32 fit and solves ran with TF32 *enabled*, even though `SUMMARY.md:6` says "TF32 off".
- [V] `l40s/SUMMARY.md:101` records the same effect for an older run (`torch_matmul_tf32=true`).

**[I] Why decode is slow (about 400 ms/token)**

This is upstream implementation overhead, not the compile issue:
- a device-to-host sync and a CPU gather of misses (`d0_index_select`) per layer (:1589);
- 32 `empty_cache` calls per step;
- the O(k^2) hit matching;
- 4x per-query-head K/V buffers;
- TopK over about 130K tokens for 32 heads per layer.

Moving storage to GPU (`tp1_lrqk_local`) halved decode time at 32K (379.8 to 187.5 ms). The existing numbers are
therefore not an optimized lower bound for LRQK.

### 3.3 What the TP1 LRQK runtime does

- **Rank** [V]: r = 32 per *query* head (32 heads). `B_Q` and `B_K` are `(1,32,32,128)`. `A_K` is `(1,32,T,32)` BF16.
  That is 64 KiB per token over all layers, the same size as exact K.
- **Prefill factor construction** [V]:
  - Per layer, over the full prompt, `cast_lrqk_prefill` (:773-801) runs `_lrqk_prefill_inv_w1` (:320-425).
  - Inputs: FP32 Q (32 heads) and FP32 repeated K.
  - Two alternating iterations, `tol=1e-8`, random-normal initialization from the global RNG.
  - The harness reseeds it with `manual_seed(20260924)` per request (`bench_tp1_offline_request.py:36`).
  - Initial selection: the top-2048 scores of `A_Q[-1] @ A_K[:L-64]`, plus the last 64 tokens in the lite area
    (:1468-1505).
- **Decode** [V] (:1507-1615):
  - Append the token to the lite ring. Every 64 tokens the ring is flushed to `KVcpu` (:1175-1197).
  - Run the analytic solver `_lrqk_decode_inv_w1` (:530-630), including the B_Q/B_K gradient step, over the
    previously hit keys plus the lite ring.
  - Append `hat_k_t` to `A_K`.
  - Compute top-2048 over `A_K[:lite_indices[lite_len]-1]` (:1564-1570). The historical range is cut at the oldest
    lite token minus 1, which leaves a one-token gap.
  - Fetch only missed rows from CPU (:1572-1613).
- **Placement** [V]: exact K/V on host for the whole request, with prefill state round-tripped through host memory.
  The GPU-local supplement keeps `KVcpu` on GPU and gathers with `torch.index_select`
  (`bench_tp1_lrqk_local.py:23-39`). That swap was validated bit-exact on 70 synthetic steps.

### 3.4 Provenance: vendored or imported?

- [V] **Imported, not vendored.** `sys.path.insert(0, ROOT/"external/LRQK")` at `audit_tp1_request.py:160-162`.
  `external/` is absent and was never tracked.
- [V] Upstream is `https://github.com/tenghuilee/LRQK.git` at `caf16293db2e4423a84ab2e895bacf64479f1eb7`
  (`docs/experiment_snapshots/2026-09-14/README.md:27`; `docs/c1_lrqk_integration.md:19-23`; `tp1_offline/freeze/provenance.json`).
- [V] My clone's `lrqk_attention.py` SHA256 is `2bfbc6df...fab5`, which matches `docs/c1_lrqk_integration.md:23`.
- [V] The tp1_offline provenance says "Dirty source trees: actual imported source is preserved in
  dependencies.tar.gz". Any dirty diff against upstream is **not verifiable locally**.
- The cpp extension is built unchanged by `evaluation/build_lrqk_cpp.py`.

### 3.5 Upstream comparison and algorithmic deviations

There are three distinct LRQK code paths in the repo.

**(a) TP1 runtime arm (upstream imported)**
- No algorithmic change found in the harness.
- Non-algorithmic changes:
  - FA2 adapter (bottom-right causal semantics preserved);
  - eager-to-FA2 attention switch;
  - exact tokenwise MLP chunking at 130,048;
  - timestamp monkey-patch on the measured run (`:51-63`).
- Parameter choices that differ from upstream's own 128K RULER configuration (`eval_configs_lrqk/eval_seq_ruler128k_offload.py`,
  r32/act2048/lite64/offload, with `tol` defaulting to 1e-2 in `lrqk/wrap/chat_lrqk.py:33`):
  - `tol=1e-8` (as in `demo_lrqk.py`);
  - TF32 left on by the upstream import.

**(b) GPU-local supplement**: a storage and gather swap only, validated bit-exact on synthetic steps.

**(c) In-repo quality adapter `basisserve/core/c1_lrqk.py` (+ `basisserve/checkpoint/c1_lrqk_qwen3.py`)**

This adapter produced **every** Llama LRQK RULER quality number:
- the V96 and V128 sources list `basisserve/core/c1_lrqk.py`, hash `fdd6d795`, which still matches the current file;
- neither lists `external/LRQK`.

Same as upstream [V]:
- prefill equations (`c1_lrqk.py:52-71` vs up-LRQK:320-425);
- decode equations and gradient step (`:80-99` vs :530-630).

Deviations [V] (`docs/c1_lrqk_integration.md:41-53`, `c1_lrqk.py:23-25,112-158`):
1. Exact K and V stay resident on GPU. There is no CPU hit/miss pool or custom transfer.
2. The decode-solver active set is the *previous* selected IDs, used for both K and `A_K`.
   - Upstream has written the current token into `Kgpu` before the solve, while its `A_K` row is still the
     zero-initialized slot.
3. "Recent" is the exact latest 64 tokens with no one-token gap.
4. Initialization uses a per-layer seeded generator instead of the global RNG.
5. Execution is eager with TF32 disabled.
6. Defaults: `tol=0.01`, and `topk` set to **832** in both quality protocols (`eval_k_routing_ruler.py:271`;
   `eval_llama_dense_v_routing128.py:30`).
7. `A_K` grows by `torch.cat` on every decode step (`c1_lrqk.py:150`). That is an O(T) copy per layer per step:
   fine for a quality run, wrong for timing.

[I] Trajectories therefore differ from upstream token by token. Equation parity holds; official cache-trajectory
parity is not claimed.

---

## 4. ShadowKV at TP1 (Q4)

There are three ShadowKV paths in the repo, in addition to the TP8 adapter.

1. **[V] Native TP1 system (used in tp1_offline).** Upstream engine with an unchanged cache. See §2.4.
2. **[V] `evaluation/official_shadowkv_cpu.py`.**
   - HF Llama plus the unchanged upstream `ShadowKVCache_CPU`, extracted by AST from `models/kv_cache.py` and
     `models/tensor_op.py` (`:20-40`).
   - V128 pinned on CPU, with an assert for width 128 (`:113`). TP1. The repo Triton kernel does decode attention
     (`:72`).
   - This path produced the V128 RULER ShadowKV quality, 84.21 over 11 tasks.
   - [V] The recorded upstream sources in `results/l31-v128-ruler128k/run_metadata.json`
     (`kv_cache.py 62f790d3...`, `tensor_op.py 48ef0a07...`) **match my clean clone at `e51904cd` byte for byte**.
3. **[V] `basisserve/core/c1_shadowkv.py`.**
   - An adaptation of upstream's GPU *accuracy* class `ShadowKVCache`:
     - same FP32 global pre-RoPE SVD, rank 160;
     - same post-RoPE chunk-mean landmarks, 48 outlier chunks, softmax-then-GQA-max chunk selection, top-256 chunks.
   - Differences from the CPU class:
     - the chunk count is **not aligned**, the default `landmark_alignment=1` (`:29-33`). At 130,048 tokens the local
       tail is 32 tokens instead of 64;
     - V96 stays **resident on GPU**;
     - there is no hit reuse;
     - Triton attention.
   - This path produced the V96 RULER ShadowKV quality: 77.47 for retrievalmix and 77.97 for C4-only.

**TP8 adapter at world size 1: no, not as written** [V]
- `shadowkv_tp8.py:10` hard-codes `TP_SIZE = 8`, and `:44` asserts `dist.get_world_size() == TP_SIZE`.
- The cache is built with a per-rank configuration of 1 KV head, 4 query heads and hidden size 512 (`:17-21`).
- `shadowkv_tp8_attention.py` hard-codes `view(batch,1,4,128)` and `reshape(batch,1,512)` (`:63, :77, :80-82, :95`).
- It uses the TP decoder layer `decoder_layer_forward` from `llama31_8b_tp8_combined` (`:15, :117`).
- [I] At TP1 the owner gather, broadcast and shard steps collapse into upstream `get_svd`. The TP8 smoke test already
  showed reconstruction agreement with `get_svd` (`tp8_external_baselines/shadowkv/README.md:49-56`).
- [I] So generalizing the TP8 adapter is unnecessary. The decode path of `official_shadowkv_cpu.attention_forward`
  matches the TP8 module, except for the attention kernel.

**Upstream revision** [V]: `https://github.com/ByteDance-Seed/ShadowKV.git` at `e51904cdeab7d4d34013370f09f2cf5fcd655e15`
(`tp8_external_baselines/shadowkv/README.md:36-38`; `RESULTS_AND_KERNELS.md:33`; `docs/experiment_snapshots/2026-09-14/README.md:29`).

---

## 5. Compatibility of prior TP1 timing and memory results with the new protocol (Q5, strict)

The new protocol requires all of the following:

| Requirement | Value |
|---|---|
| Hardware | L40S node 2 (driver 570.211.01, torch 2.13.0+cu129) |
| Context | 130,048 |
| Batch | B1 |
| Output | 128 greedy tokens |
| Prompts | 3 frozen cohorts |
| Request boundary | prefill + construction + placement + decode |
| Steady boundary | 16 + 128 steps, full-model and attention, for **every** arm |
| Memory | full breakdown |
| Value | a common V96 checkpoint wherever an arm compresses V |

| Prior result | HW / software | Ctx / out / cohorts | V | Budget / sink | Boundary | Memory | Verdict |
|---|---|---|---|---|---|---|---|
| tp1_offline dense | RunPod L40S, cu130 | 130,048 / 128 / 3 | V128 (correct for Dense) | n/a | Request and steady match | Single peak only | **Not a matched point** (different machine and software, no memory breakdown). Use as a cross-check; table3 reproduced it within about 1%. |
| tp1_offline basis | same | same | **V128**, V128 bank | sink32 + 61 pages + 64 = 2048 | match | peak only | **Not reusable** for V96. Could pair only with V128 quality (§7). |
| tp1_offline shadowkv | same | same | **V128 CPU** (native) | 2048 + 384 + 64 + gen | request yes; **no steady**; tail is a wall mean | peak only | **Not reusable** as V96-matched. Only a separately labeled native historical reference, and the machine differs. |
| tp1_offline lrqk | same | 130,048 **2/3** | V128 CPU | **top-2048**/q-head + 64 (quality uses 832) | request; no steady | peak only | **Not reusable.** Incomplete, wrong budget, `tol` 1e-8 vs 0.01, TF32 on, compile warnings. |
| tp1_lrqk_local | same | 32K only | V128 GPU | 2048 + 64 | request | peak | **Not reusable** at 130,048. |
| tp1_paper_scaling | 8x L40S, cu130 | 130,048 / steady / 3 | V128 | as basis | steady only | peak | **Not reusable** (V128, different node). Cross-check only. |
| table3_k_offload_128k arms and pages | node6 L40S, cu126 | 130,048 / steady / 3 | **V128** | sink32 + 1952 + 64 = 2048 (p32). p4: 32-token sink + 1952 + 64 = 2048. **Not** the 2048 + 64 = 2112 no-sink Page4 quality point. | steady only, no request | **Full phase memory and inventory** | **Not matched** (node, V128, budget). The memory *methodology* and the state-byte inventory are reusable for Concern B. |
| tp1_capacity | L40S | 65,536 / 131,072, decode only | V128 | 1984 + 64 | decode only | peak | Not reusable |
| tp1_sparse/dense_full_v6, tp1_router_config | L40S | 131,072, one teacher-forced prompt | V128 | 2048 | CUDA median, teacher-forced | peak | Not reusable |
| tp1_decode_v7/v8 | L40S | 131,072 | V128 | **two-stage** router (excluded) | mixed | peak | Not reusable |
| RESULTS_AND_KERNELS "native comparison" | L40S, torch 2.6 | 64K / 128K | Llama-3.1-8B **base** | various | 100-step wall | — | Not reusable |
| l40s/SUMMARY router-only | L40S | 64K | base model | — | router microbenchmark | — | Not reusable |
| TP8 ShadowKV grid | 8x L40S | TP8 | V128 CPU | 2496 | TP8 | per-rank | Not TP1 |
| benchmarks/a100 | A100 | — | V128 | — | — | — | Hardware differs |

**[I] Bottom line**

- No prior TP1 timing or memory measurement is a matched point for the new protocol.
- All four arms must be re-measured on node 2.
- The prior Dense and BasisKV-V128 numbers are useful only as same-SKU cross-checks: expect about 47.3 and 37.8 ms
  steady, and about 43 s per request.

---

## 6. Missing pieces and estimated GPU memory at 130,048 tokens (Q6)

### 6.1 Concrete adapters and changes needed

**Common infrastructure** (one new harness; do not edit the frozen one)

1. Restore the runtime, or port it to node 2.
   - Options:
     - restore `dependencies.tar.gz` (frozen `tp1_sparse_local/frozen_source`, dirty `external/LRQK`, `external/ShadowKV`);
     - or clone upstream at the pins and re-apply only the documented patches (ShadowKV stream patch).
   - Rebuild the ShadowKV CUDA extension for SM89 and the LRQK cpp extension.
   - Check `torch.ops._C.silu_and_mul` in vLLM 0.30, which the native ShadowKV engine needs.
   - Relocate the model, factor and prompt paths, as table3's `runtime_manifest.json` did.
2. Prompts: fetch the frozen HF `system_benchmarks/tp1_offline/inputs/prompts.safetensors`, or prove that local
   `c4-48x128k` rows 0-2 equal it. Equality is not verified.
3. Add a **steady mode for LRQK and ShadowKV**. Today only dense and basis have one (`bench_tp1_offline_request.py:175`).
   Use the same CUDA-event and synchronized-wall boundary, and record mean, median and p95.
4. Add a **per-arm attention-block profiler** with the same span definition for every arm. It needs hooks into
   upstream LRQK `update` and FA, and into ShadowKV's native layer forward.
5. **Memory instrumentation** as in table3, for every arm:
   - phase snapshots (after load, prefill peak, decode-ready, decode peak, reserved, `mem_get_info`, RSS);
   - a tensor inventory, allocated vs active, GPU vs host. For ShadowKV reuse `cache_state_bytes`. For LRQK, add
     `_A_K.data` capacity vs `current_len`, and `KVcpu` pinned capacity vs used.
6. Record environment and versions per trial, and the H2D bandwidth of the node.

**Dense**

- Reuse the frozen dense module: GPU V128 K/V, flash SDPA. Change only paths and instrumentation.
- Optional: a "Full-K V96" arm as the quality anchor. It needs the V96 encoder and decoder plus V96 prefill and decode
  attention.

**BasisKV V96 (paper-aligned)**

- A new attention module ported from `TP1SparseAttention`:
  - a V96 latent cache using `value_coordinate_encoders [8,128,96]`;
  - `head_output_decoders [32,96,4096]` replacing `o_proj`;
  - V96 prefill attention (zero-padded V SDPA, or the repo's C1 Triton prefill);
  - V96 decode attention (check that `slot_indexed_attention` handles a 96-wide V);
  - `attention_workspace` and output widths changed from 128 to 96 (`bench_tp1_sparse_full_v6.py:186-219`);
  - router and paged extensions rebuilt with `-DBASIS_VALUE_DIM=96`.
- Use the **V96 B16R16 bank**. `router_page4_b16r16` (Page4, 2048 + 64 = 2112, no sink) or `router_b16r16` (Page32
  sink32) exist only on HF.
- Page4 needs a selector with `kPinnedPages=0` and `kRoutedPages=512`. The current `table3_page_select.cu:23-25`
  hard-codes `1984/kPageSize` and a 32-token sink. 130,193/4 = 32,549 pages, which is close to `kMaxGroups*8 = 32,768`.
- Base: either a separate base cache (+0.99 GiB), or the table3 "nobase" rotation (`bench_table3_k_offload.py:79-90`)
  adapted to the 96-dim latent.
- Key placement: the quality protocol says "exact K in pinned CPU, selected rows fetched" (`b16r16_page4_fisher.json`
  `memory_policy`). Use the K-offload plus slots path, and optionally a resident control.
- Validate against `evaluation/eval_k_routing_ruler.py` and `evaluation/llama_b16r16_k_offload_p1.py` on short
  prompts: selected pages, logits, tokens.

**LRQK**

- Decide which computation is the reference. The V96 quality (81.05 and 80.41) came from **c1_lrqk with top-832 + 64
  and `tol` 0.01**, not upstream.
- (i) **Paired V96 arm.** c1_lrqk semantics with:
  - exact K on host and gathered selected K (same selection; placement change only);
  - V96 resident;
  - a preallocated `A_K` instead of `torch.cat`;
  - head-blocked prefill fit (`evaluation/lrqk_chunked_prefill.py`, the same equations);
  - TF32 off.
  - Validate selected IDs and logits against c1_lrqk on short prompts.
- (ii) **Native-system arm.** Upstream `DynamicLRQKCache` with CPU offload.
  - Parameters: top-832 + lite 64 to match quality, or top-2048 as upstream's own 128K configuration. Disclose which.
  - V96 needs an adapter: `AutoIncreaseTensorSharedKV` concatenates `[k,v]`, and the decode path slices the result
    assuming V width equals K width (up-LRQK:960-965, :1605-1606).
    - [I] Simplest correct option: keep V128 for the native arm and label it native.
  - Fix or label the compile issues:
    - run once with `TORCH_LOGS=graph_breaks,recompiles`;
    - pre-compile, or raise `recompile_limit`;
    - report an eager control (`TORCHDYNAMO_DISABLE=1`).
  - Record the TF32 state explicitly.
  - Keep the 1.5x allocation but report allocated vs active bytes.

**ShadowKV**

- **Native-system arm**: the existing upstream engine with V128 CPU, plus steady mode and memory instrumentation.
- **Paired V96 arm.** Either:
  - c1_shadowkv semantics with V96 resident on GPU (exact match to the quality reference; deviates from native CPU-V
    placement, so label it);
  - or the official CPU cache with V96 on host. That needs `v_cache_cpu` and `v_cache_buffer` 96 wide, and decoupled
    K/V strides in `gather_copy_with_offsets`, which currently uses `head_dim` for both. Validate against c1_shadowkv,
    and align or disclose the chunk-alignment difference (32 vs 64 local tail).
- The TP8 adapter is not needed at TP1.

### 6.2 Memory arithmetic

Model constants: Llama-3.1-8B, 32 layers x 8 KV heads x head_dim 128, BF16 (2 bytes). T = 130,048 = 127 x 1024.
Steady capacity is 130,193; at that capacity K = 7.9464 GiB, which matches the table3 inventory.

| State | Bytes per token (all 32 layers) | At T = 130,048 |
|---|---|---:|
| Exact K | 32 x 8 x 128 x 2 = 65,536 B (64 KiB) | **7.9375 GiB** |
| V128 | 65,536 B | 7.9375 GiB |
| V96 | 32 x 8 x 96 x 2 = 49,152 B | **5.9531 GiB** |
| B16 Base | 32 x 8 x 16 x 2 = 8,192 B | 0.9922 GiB |
| R16 Residual | 8,192 B | 0.9922 GiB |
| LRQK A_K (32 query heads x r32) | 32 x 32 x 32 x 2 = 65,536 B | 7.9375 GiB exact; **11.906 GiB** with upstream 1.5x capacity (195,072 rows) |
| ShadowKV U (rank 160) | 32 x 160 x 2 = 10,240 B | 1.2402 GiB |
| ShadowKV landmarks (16,200 chunks x 8 heads x 128 x 2 x 32) | — | 0.9888 GiB |

Other constants:
- Weights: 14.9575 GiB (measured inventory).
- L40S: 46,068 MiB = 44.99 GiB.
- Non-PyTorch overhead is about 0.76 GiB (table3: 31.62 GiB device used vs 30.86 GiB allocated). Usable is therefore
  about 44.2 GiB.
- The HF-harness prefill transient is about 7.5 GiB (measured: dense 38.36 - 30.86; basis 40.48 - 32.98).

| Arm (130,048 tokens) | GPU decode-resident (GiB) | GPU prefill peak (GiB) | Host (GiB) | Fits on L40S? |
|---|---|---|---|---|
| Dense V128, GPU K/V | 14.96 + 7.95 + 7.95 = **30.85** (measured 30.86) | **38.36** measured | 0 | Yes (about 6 GiB headroom) |
| Full-K V96 (optional anchor) | 14.96 + 7.95 + 5.96 = 28.9 | about 36.4 [I] | 0 | Yes |
| BasisKV V96, resident K, Page4/2112, Base as view | 14.96 + 7.95 + 5.96 + 0.99 + about 0.25 (page scores 0.13, RoPE and index 0.06, workspace) = **about 30.1** | about 37.6 [I] | 0 | Yes |
| BasisKV V96, separate Base cache | about 31.1 | about 38.6 | 0 | Yes |
| BasisKV V96, K offload + 2112 slots | 14.96 + 5.96 + 0.99 + 0.13 slots (32 x 8 x 2112 x 128 x 2) + 0.13 maps + about 0.25 = **about 22.4** | about 29.9 (table3 V128 analog 31.79 - 1.99) | **7.95** K | Yes |
| LRQK upstream offload, V128, top-2048 (existing) | 14.96 + 11.91 + 1.03 (Kgpu+Vgpu: 32 x 32 x 2112 x 128 x 2 x 2) = **about 27.9** (measured request peak 29.48) | about 25-27 [I]: per-layer fit uses FP32 Q 1.98 + FP32 K 1.98 + repeated K 0.99 + A/new-A 1.98 + qkv/hidden | 23.81 pinned KVcpu (32 x 8 x 195,072 x 256 x 2), plus up to about 13 pageable restore staging | Yes on paper. Observed 7/9 warmup OOM at 130,048. |
| LRQK upstream offload, V96, top-832 | 14.96 + 11.91 + 0.38 (K 0.22 + V96 0.16) = 27.3. With exact capacity: 23.3 | about 25-27 | 20.84 (1.5x) / 13.90 exact (width 224) | Yes |
| LRQK GPU-local, upstream 1.5x (existing supplement) | 14.96 + 11.91 + 23.81 + 1.03 = **51.7** | above 52 | 0 | **No** (structural) |
| LRQK c1_lrqk (quality ref.), V96, exact K on GPU | 14.96 + 7.95 + 5.96 + 7.95 = **36.8**, plus a 0.25 per-step `cat` transient | about 44-51 [I]. Measured 50.5 on a 96 GB card (`l31-v96-retrievalmix-ruler1100/summary.md`). | 0 | **No / marginal**. With head-blocked fit, about 40-42 (marginal). |
| LRQK c1_lrqk, V96, exact K on host | 14.96 + 5.96 + 7.95 = **about 28.9** | about 36-37 | 7.95 | Yes |
| ShadowKV native, V128 CPU (existing) | 14.96 + 1.24 U + 0.99 landmarks + 0.03 index + 0.32 k/v buffers (2 x 32 x 8 x 2624 x 128 x 2) + about 0.03 = **about 17.6** | **20.61** measured (request peak) | 7.95 V (32 x 8 x 16,273 x 1024 x 2) | Yes |
| ShadowKV, V96 on CPU (adapter) | about 17.5 | about 20.5 | 5.96 | Yes |
| ShadowKV, V96 resident on GPU (c1_shadowkv semantics) | 14.96 + 1.24 + 0.99 + 5.96 + about 0.05 = **about 23.2** | about 30.7 (HF runner) [I] | 0 | Yes |

**[I] Summary.** At 130,048 tokens on a 46 GB L40S, every arm fits *if* each method keeps its native host placement:

| Arm | GPU decode-resident (GiB) |
|---|---:|
| ShadowKV | about 18 |
| BasisKV, K offload | about 22 |
| LRQK | about 23-28 |
| BasisKV, resident K | about 30 |
| Dense | about 31 |

The infeasible variants are LRQK with K/V resident on GPU, both the upstream 1.5x version and the unmodified c1_lrqk
quality adapter. LRQK's routing state (`A_K`) alone equals the full exact-K size: 7.94 GiB, or 11.9 GiB with the 1.5x
policy.

---

## 7. Quality pairing

[V] None of the existing TP1 runtimes is the deployed computation behind a V96 quality number.

| Quality set | Arms and settings (implementation) | 11-task mean |
|---|---|---|
| `l31-v96-retrievalmix-ruler1100` (V96 retrievalmix, the checkpoint present locally; RTX PRO 6000) | Full-K | 84.32 |
| | B16R16 p32 sink32/2048 | 82.01 |
| | **B16R16 p4 no-sink 2048+64** | 82.02 |
| | LRQK top-832+64 (c1_lrqk) | 81.05 |
| | ShadowKV (c1_shadowkv, V96 on GPU) | 77.47 |
| `l31-v96-ruler128k` (C4-only V96 `..._als6`, not local) | Full-K / B16R16 p32 sink32 / LRQK 832 / ShadowKV | 84.45 / 80.88 / 80.41 / 77.97 |
| `l31-v128-ruler128k` (V128) | Dense | 86.81 (TP2) |
| | B16R16 p32 sink32/2048 (same budget and sink as the TP1 basis runtime; bank identity not verified) | 84.99 |
| | ShadowKV, unchanged upstream `ShadowKVCache_CPU` (source hashes match upstream) | 84.21 |
| | LRQK c1_lrqk top-832+64 | 83.72 |

[I] The existing V128 TP1 runtimes could pair with V128 quality only after these changes:
- re-measure on node 2;
- LRQK changed to top-832;
- the BasisKV bank confirmed to be the bank used for V128 quality.

[I] `runs/q3-32b-t2/summary.md:6-17` describes the paper's "Table 2 recipe" as V96 mixed-calibration plus Page4
no-sink 2048 + 64 = 2112, with a 10-task mean. That suggests the Page4 point is the paper-aligned one. The paper PDF
could not be rendered here (no poppler), so this is not verified against the paper itself.

---

## 8. Open uncertainties and read-only follow-ups

1. **Exact LRQK warning and OOM text.** It is in HF raw logs not fetched here:
   - `system_benchmarks/tp1_offline/formal/request_lrqk_t130048_c{0,1,2}_n{32,128,512}/run.log`
   - `system_benchmarks/tp1_offline/formal.log`
   - `system_benchmarks/tp1_lrqk_local/t130048_n128_c*/run.log`
   - The OOM messages include "Tried to allocate / free / reserved", which would separate the autotune, fragmentation
     and host-pinned hypotheses. Revision `86c03092c663dc6654584143130b5e5abf2fedaa`.
2. **Dirty `external/LRQK` and `external/ShadowKV` diffs.** They are in `system_benchmarks/tp1_offline/freeze/dependencies.tar.gz`
   (same revision). Upstream equivalence of the TP1 runtime is verified only for the clean pins.
3. Whether the tp1_offline prompts (`/workspace/runs/l31-cal128/calibration/windows.safetensors` rows 0-2) equal
   local `c4-48x128k` rows 0-2.
4. Whether the V128 TP1 bank (`l31-router-source/v128-router/ours_b16r16`) is the bank used for `l31-v128-ruler128k`.
5. The machine and driver for tp1_offline itself. The table3 README says RunPod; tp1_paper_scaling records driver
   580.178.04. That they are the same machine is inferred.
6. Whether `slot_indexed_attention` and the router kernels support a 96-wide V without code changes.
   `_load_extension` accepts `value_dim` up to 256, but `TP1SparseAttention` hard-codes 128.
7. Whether vLLM 0.30 still registers `torch.ops._C.silu_and_mul`, needed by the native ShadowKV engine through
   `paper_faithful_compat.py:145-146`.
8. All memory figures marked [I] are arithmetic estimates. Prefill peaks for the new V96 paths should be measured with
   a smoke run before the formal grid.
9. The protocol choice for the LRQK budget (832 vs 2048) and for BasisKV (Page4/2112 vs Page32 sink32/2048) is a
   decision for the authors. This report only shows which quality number each choice pairs with.
