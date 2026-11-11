# V2Q8 L40S sparse-baseline rebuttal: compatibility and provenance audit

Date: 2026-11-10. Node: Vast.ai 4x L40S "node 2" (details in section 7). Scope: Dense, BasisKV, LRQK, ShadowKV at
TP1/130048/B1 and TP4/65536/large batch, plus Value-reuse state accounting. PaLU is out of scope.

How this was produced: three read-only audits of the repository, the paper PDF and the Hugging Face artifact
repo `alexz949/BasisServe-CALS`, with no GPU use. Their notes are kept in `working_notes/`. Every claim marked
**[V]** below was re-checked directly against the cited file. Unmarked claims come from the audit notes and
have not been re-checked.

## 1. Bottom line

- **No existing timing or memory result is a matched point for the new protocol.** Every prior TP1 compressed
  arm uses dense V128, and every prior run is on another node or software stack. All four arms need fresh runs on
  this node at both TP1 and TP4.
- **No arm runs as true TP4 today.**
  - Dense, BasisKV and ShadowKV have TP8 implementations with world size 8 hard-coded. They need ports.
  - LRQK has no multi-GPU Llama implementation.
- **Two quality protocols exist with all four methods evaluated on the same prompts.** The cleanest pairing is the
  paper's Table 2 (V96). Neither protocol is implemented by the current runtime as-is (section 2).
- **There is no 64K quality result at the Table 2 configuration, and no TP4 quality result of any kind.**
- **Value-reuse accounting (Concern B) can be built entirely from existing Table 3 records** (section 6).

## 2. Quality protocols

| | P-V96 (paper Table 2, primary) | P-V128 (paper Table 11, dense Values) |
|---|---|---|
| Value | compressed V96, `checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix`, shared by every arm including Full-K **[V]** paper §5, App. C.1 | original V128 and Wo **[V]** `results/l31-v128-ruler128k/run_metadata.json` |
| BasisKV router | B16R16, **Page4**, 2048 routed + 64 recent, **no sink**; bank `router_page4_b16r16` (layer-0 sha256 `e71777a2`, on HF) **[V]** | B16R16 refit on V128, **Page32, sink 32, recent 64, group budget 2048** **[V]**; bank `l31-dense128/router/ours_b16r16` (layer-0 `1f542b46`) is **not on HF** **[V]** |
| LRQK | rank 32, top-832 per query head + 64 recent; union ≈ 2107 tokens/group | rank 32, top-832 + recent 64, 2+2 iterations, tol 0.01 **[V]** |
| ShadowKV | native chunk/outlier/local configuration on the shared V96 checkpoint | upstream `ShadowKVCache_CPU` + compiled upstream kernels, rank 160, chunk 8, 2048 routed, 48 outlier chunks; V on pinned CPU **[V]** |
| Prompts | RULER-128K, seed 42, 100/task; paper reports 10 tasks (excl. niah_single_3) **[V]** paper App. C.1 | the same 11 tasks × 100, prompt-set sha `d1dbe3be…` **[V]** |
| Scores (10-task / 11-task) | Full-K 82.75/84.32, BasisKV 80.32/82.02, LRQK 79.15/81.05, ShadowKV 75.41/77.47 (reproduced from `results/l31-v96-retrievalmix-ruler1100/predictions`) | 11-task: Dense 86.81, BasisKV 84.99, ShadowKV 84.21, LRQK 83.72 **[V]** `results/l31-v128-ruler128k/README.md` |
| Hardware | RTX PRO 6000 Blackwell | L40S **[V]** |

Caveats that block a direct pairing:

- **P-V96 was not computed by a runtime router.** BasisKV quality used a Python page selector with a 144-dim
  "quality oracle" sidecar, not the CUDA runtime router (audit note, not re-checked). A runtime implementing
  P-V96 therefore has to show that it selects the same pages, or RULER has to be re-run through it.
- **P-V128 does not match the existing runtime's router bank.** The Table 3 / TP1 runtime loads V128 bank
  `e8a79558` (`system_benchmarks/table3_k_offload_128k/factors/ours_b16r16` on HF) **[V]**. That bank has no
  quality record. The Table 11 bank `1f542b46` could not be found on HF **[V]**.
- **LRQK quality came from the in-repo `basisserve/core/c1_lrqk.py`, not from upstream.** The audit notes list
  semantic deviations: K stays resident, the decode solve uses the previous selected set, a different random
  initialisation, eager execution. The existing TP1 runtime instead runs **upstream** LRQK at **top-2048** + 64,
  so it does not match either quality protocol.
- **64K quality:**
  - `results/l31-v96-ruler64k` uses a different, 64K-calibrated V96 checkpoint, 30 prompts per task, and Page32
    with a sink. Its LRQK used top-2048, giving a union of about 4466 tokens, so support is not matched.
  - No TP4 routing-quality result exists.

## 3. TP1 runtime status (130048 tokens, B1)

The existing harness is `benchmarks/system/run_tp1_offline_grid.py` → `bench_tp1_offline_request.py`, with
factories in `audit_tp1_request.py`. It was last run formally on 2026-09-24 on one RunPod L40S with torch
2.13.0+cu130 (`results/system_benchmarks/tp1_offline/SUMMARY.md`).

| Arm | Status at 130048 | Configuration | Gaps for the new protocol |
|---|---|---|---|
| Dense | complete, 3/3 cohorts | V128, GPU-resident | Fine as the native dense baseline. Needs phase-level memory and a tensor inventory. |
| BasisKV | complete, 3/3 | V128, K on GPU, Page32 full scan; page 0 pinned, so 32 sink + 61 pages + 64 recent = 2048 | P-V96 needs a V96 decoder, the Page4 bank and **no sink**. |
| LRQK | **2/3 cohorts at 128 output tokens; 0/3 at 32 and 512**; all failures were OOM during warmup | upstream `tenghuilee/LRQK@caf16293`, top-2048, K/V on CPU, `tol=1e-8` | See below. |
| ShadowKV | complete, 3/3 | upstream engine at `ByteDance-Seed/ShadowKV@e51904cd`, V128, V on CPU | Native V128 only; no V96 path. |

Measurement gaps:

- Steady decode (16 + 128 CUDA-event steps) is implemented for dense and BasisKV only. LRQK and ShadowKV report
  a wall-clock mean over the request tail.
- Memory is a single `max_memory_allocated`. Phase-level memory and tensor inventories exist only in the Table 3
  runtime, and only for dense and BasisKV.

LRQK findings (audit notes; they are the reason the old LRQK number cannot stand as an optimized reference):

- All five upstream `@torch.compile` functions contain data-dependent breaks (`if delta < tol: break`,
  `max(tensor, tensor)`), and the per-step gather uses `nonzero()` and boolean masking. These are the sources of
  the graph breaks and recompiles.
- Memory and time are inflated by a 1.5× over-allocation (routing state `A_K` is 11.9 GiB on GPU, CPU K/V is
  23.8 GiB), by `torch.cuda.empty_cache()` on every layer of every step, and by a CPU gather. Together they give
  roughly 400 ms/token.
- Upstream calls `set_float32_matmul_precision('high')` at import, after the harness disables TF32. The LRQK fit
  therefore probably ran with TF32 on.

## 4. TP4 runtime status (65536 tokens)

Sharding at TP4: rank r owns Q heads 8r..8r+7 and KV heads 2r and 2r+1, so the 4:1 GQA grouping stays local.
The existing TP8 code is launched with torchrun plus Transformers' tensor parallelism.

| Arm | Existing code | Needed for TP4 |
|---|---|---|
| Dense | TP8 Instruct path decodes with a custom paged kernel: 85.6 ms/step at 64K vs 30.7 ms with FlashAttention. A TP4 runner (`llama_tp4_k_offload.py`) has run only on the base model. | TP4 with FlashAttention-2 `flash_attn_with_kvcache`. Using the slow paged kernel would handicap Dense. |
| BasisKV | `llama31_8b_tp8_combined.py`: full-scan B16R16, K pinned on host with GPU slots, plain als6 V96; TP8 only. Older TP4 paths use the base model. | Port to world size 4. |
| ShadowKV | TP8 adapter: gathers pre-RoPE K to an owner rank, runs **one global FP32 SVD**, truncates to rank 160, broadcasts U (replicated), scatters SV slices. V on pinned CPU. | Port: 2 KV heads/rank, owner = layer % 4. The method is preserved. |
| LRQK | none for Llama. Upstream patches attention with the global head count. | New TP4 adapter (largest item). |

- **Collectives.** Dense, ShadowKV and LRQK issue 2 AllReduce of [B,1,4096] per layer (64 per step) plus one
  logits AllGather. BasisKV replaces the o_proj AllReduce with an AllGather of [768, B].
- **World size 8 is hard-coded** in the adapters, runner, timing aggregator, summarizer and tests. The runner
  only accepts B ∈ {1,4,8}. The kernels take the KV-head count at runtime.

Capacity estimates at 65536 tokens (audit inference, ±1–2 GiB; about 44 GiB usable per L40S):

| Arm | GPU state / request / rank | B8 prefill peak | B16 prefill peak | Pinned host, all ranks |
|---|---|---|---|---|
| Dense | 2.00 GiB KV | ~33 GiB | ~62 GiB, OOM | 0 |
| BasisKV | 0.93 GiB | ~26 GiB | ~46 GiB, likely OOM | 4 GiB/request |
| ShadowKV | 0.84 GiB | ~25 GiB | ~44–46 GiB, likely OOM | 4 GiB/request |
| LRQK | 1.76 GiB | ~38–41 GiB, at risk | ~75 GiB, OOM | 12 GiB/request |

- **B8 is the likely largest joint batch**, provided LRQK fits its prefill one request at a time (otherwise B4).
- **B16 would require per-request prefill in every arm.** That is a runner change, not an algorithm change, and
  must be disclosed if used.

## 5. Reuse decisions

- **Not reusable as matched points:** every prior TP1, TP4 and TP8 timing or memory result, because hardware,
  software, V rank, budget or timing boundary differs.
  - Prior TP1 dense (~47.3 ms) and BasisKV (~37.8 ms) steady decode at 128K are kept as cross-checks only.
  - Prior 64K TP8 and base-model TP4 runs are kept as references only.
- **Reusable exactly:**
  - P-V96 and P-V128 quality scores, as quality references, labelled with their exact configuration.
  - The Table 3 records, for Value-reuse accounting.

## 6. Value-reuse accounting (Concern B), from existing records

Source: `results/rebuttal/table3_k_offload_128k/arms/formal/summary.csv`. All arms use dense V128 on an L40S at
130048 tokens with B1. Every value below is **[V]**.

| Arm | Decode-resident GPU | Host pinned K | Router scan | Attention block | Full step (wall) |
|---|---:|---:|---:|---:|---:|
| `dense_resident` | 30.859 GiB | 0 | — | 23.821 ms | 47.347 ms |
| `basis_resident` (separate Base cache) | 32.975 GiB | 0 | 11.750 ms | 14.177 ms | 37.400 ms |
| `basis_resident_nobase` (Base read from rotated V) | 31.975 GiB | 0 | 11.896 ms | 14.411 ms | 37.830 ms |
| `basis_k_offload_nobase` | 24.289 GiB | 7.946 GiB | 11.905 ms | 16.301 ms | 39.481 ms |

- **Base saving.** Removing the separate Base cache saves 1.000 GiB as measured. The tensor itself is
  `[1, 8, 130193, 16]` BF16 × 32 layers = 1,066,541,056 B = 0.993 GiB; the allocator rounds each layer to 32 MiB.
  This is a real reduction in persistent routing state:

  | Router | Persistent state |
  |---|---|
  | B16R16 with separate Base | 32 dims/token/KV head |
  | B16R16 with Value reuse | 16 dims |
  | Independent R32 Key router (analytical, no latency) | 32 dims |

- **Scan width is unchanged.** Every arm still scans 32 dims per cached token. The router scan does not get faster
  (11.750 → 11.896 ms, +1.2%), and the full step is 1.15% slower. **No speedup from Value reuse is claimed.**
- **K offload moves bytes; it does not eliminate them.**
  - GPU drops 7.686 GiB, while host pinned memory holds 7.946 GiB of exact Keys.
  - This comparison is no-Base vs no-Base, so it is separate from the 1.000 GiB Base saving.
  - Offload output is bitwise identical to its resident control.
- **Numerical caveat.** The no-Base arms apply an FP64 QR rotation of `v_proj`/`o_proj` and re-round the weights to
  BF16. The placement check shows argmax agreement 1.0, logits max |Δ| 0.672, and identical support on 0 of 16
  steps (audit note).
- **BasisKV at V128 uses more GPU memory than dense** (32.975 vs 30.859 GiB). Its memory advantage comes from V96
  compression and K offload, not from the router.
- **R32 quality is available.** Paper Table 9 (30 prompts/task, Page4, V96) compares Base-only, R16, B16R16 and
  R32 quality. It can be cited next to the state table; there is still no R32 latency result.

## 7. This node

- **Hardware.** 4x L40S 46 GB. 2x Xeon Platinum 8558: GPU0/1 on NUMA 0, GPU2/3 on NUMA 2, no NVLink. 2 TB RAM.
- **Driver.** 570.211.01, max CUDA 12.8, so cu130 builds are unusable. The prior TP1 run used cu130.
- **Interconnect.**
  - 4-GPU AllReduce of 256 KiB takes 76 µs, and 64 MiB reaches 13.9 GB/s busbw (NCCL 2.29.7).
  - Cross-socket hops use SHM.
  - Logs: `results/rebuttal/v2q8_palu_g4_tp4/logs/interconnect_l40s_node2.log`.
- **Missing here:** `flash_attn`, upstream LRQK and ShadowKV checkouts and builds (ShadowKV needs CUTLASS),
  the B16R16 banks, and the prompt cohorts. Transformers 5.22.2 is installed; the TP8 path was validated on
  5.17.0.
- **`/workspace` is not a persistent volume.** Results must be uploaded before the instance is released.

## 8. Required work, in dependency order

1. **Environment and assets.**
   - A separate `lowrank` env (torch 2.13 cu129, Transformers 5.17.0, flash-attn, Triton) that leaves `basis`
     untouched.
   - `external/LRQK@caf16293` and `external/ShadowKV@e51904cd` with their kernels built.
   - Banks: V96 Page4 `e71777a2` and V128 runtime `e8a79558`.
   - Frozen prompt cohorts: C4 timing windows at 130048 and 65536, plus a RULER subset for runtime-vs-quality
     checks.
2. **TP1 harness.**
   - Steady-decode and attention timing for LRQK and ShadowKV.
   - Phase-level memory and tensor inventories for all arms.
   - One timing boundary for every arm.
3. **TP1 P-V96 arms.**
   - BasisKV: V96 decoder plus the Page4, no-sink selector.
   - LRQK: `c1_lrqk` semantics at top-832 + 64 with K/V placed on host, preallocated state, no per-step
     `empty_cache`, and TF32 pinned off.
   - ShadowKV: on the shared V96 checkpoint.
4. **TP1 native-system arms, separately labelled.** Upstream LRQK and ShadowKV at V128.
5. **Selection-equivalence checks.** On RULER prompts, compare each runtime arm's selected sets and attention
   outputs with its quality reference.
6. **TP4 ports.**
   - Generalise world size and batch in the runner, timing and summary code.
   - Dense with FlashAttention-2, then BasisKV, then ShadowKV (global SVD preserved), then a new LRQK TP4 adapter.
   - Per-collective logical bytes and latency.
7. **TP4 quality.** A small, predefined 64K RULER evaluation run through the deployed TP4 runtime. Otherwise, the
   128K scores carry an explicit "not measured at 64K" label.
8. **Smoke and capacity probes**, then the formal launch matrix for approval.

## 9. Decisions for the authors

1. **Primary protocol.** P-V96 (paper Table 2), as the task prefers. P-V128 native-system arms are reported
   separately and never mixed with it.
2. **P-V128 router bank.** Either locate bank `1f542b46` used for Table 11, or accept the runtime bank
   `e8a79558`. With `e8a79558`, BasisKV's V128 quality has to be re-measured or labelled non-identical.
3. **LRQK operating point.** Top-832 + 64, matching both quality protocols, for the matched comparison. The old
   top-2048 runtime becomes a "native default" reference only.
4. **TP4 quality.** A small 64K RULER run through the TP4 runtime (prompt count to be fixed in the launch matrix),
   or 128K quality as a labelled reference.
5. **B16 at TP4.** Allow per-request prefill (a disclosed runner change), or cap at B8.

## 10. Further verification during implementation (2026-11-10) [all V]

- **Quality-code identity.** The repository's `basisserve/core/c1_lrqk.py`, `basisserve/core/c1_shadowkv.py`,
  `basisserve/checkpoint/c1_lrqk_qwen3.py` and `basisserve/kernels/compressed_v_decode_attention.py` match, byte for
  byte, the `source_sha256` recorded in the Table 2 protocol (`results/l31-v96-retrievalmix-ruler1100/protocols/lrqk.json`).
- **ShadowKV adapter.** `basisserve/checkpoint/c1_shadowkv_qwen3.py` differs from the recorded hash. The only change
  since then (commit `31d6179`) makes the routed budget a module setting, defaulting to 2048, Table 2's value. RULER
  semantics are unchanged.
- **Frozen kernel tree.** The frozen Table 3 tree (`tp1_sparse_local/frozen_source`, from HF
  `system_benchmarks/tp1_offline/freeze/dependencies.tar.gz`) carries older `compressed_v_decode_attention.py` and
  `slot_indexed_attention.py`. The new runners load that tree only for the BasisKV arms (Table 3 kernels). The
  baselines import the repository versions, the Table 2 code.
- **Upstream sources.** The frozen upstream LRQK and ShadowKV sources equal clean `caf16293` and `e51904cd` (only build
  products were added). CUTLASS was not recorded. v3.5.1 (`f7b19de`) is pinned here and is a recorded deviation.
- **TF32.** Upstream LRQK calls `torch.set_float32_matmul_precision('high')` at import (`lrqk_attention.py:33`).
  The new TP1 runner records the precision actually in force after the runtime is built.
- **RULER prompts.** HF `evaluation/llama31-8b-instruct-ruler128k-1100-seed42/prompts.json` has sha256 `1f6389dd…`,
  equal to Table 2's `prompts_sha256`, with identity `f6fbef8b`. Its `prompts.safetensors` holds the token tensors.
  A runtime-paired RULER evaluation and selection checks on real RULER samples are therefore possible.
- **Banks.**
  - Table 2 V96 Page4 bank `router_page4_b16r16` (layer 0 `e71777a2`): `base_left` is `[8,96,16]`, acting on V96
    coordinates.
  - Table 3 runtime V128 bank `e8a79558`: identical to `tp1_offline/freeze/factors.tar.gz` (64/64 files).
- **Kernel genericity.**
  - The mapped-host paged-attention extension is compiled per `BASIS_VALUE_DIM` (up to 256).
  - `slot_indexed_attention` is Triton-generic in the Value width.
  - The page selector and slot refresh derive rows from `size(0)*size(1)` (batch × KV heads).
  - The selector's sink was hard-coded (`kPinnedPages = 32/kPageSize`). It is now a compile-time parameter, and the
    Table 3 callers pass the archived 1984 routed / 32 sink.
- **TP1 dense.** The archived TP1 dense runtime decodes with PyTorch FlashAttention SDPA, a reasonable optimized
  dense baseline.
- **TP4 inputs.**
  - The 64K cohorts are frozen in `tp4_64k/inputs/prompts.safetensors`: rows 0..15 of cohort c are C4 windows
    16c..16c+15 of `c4-48x128k`, first 65536 tokens.
  - These are timing inputs, distinct from the TP1 cohorts (`l31-cal128` windows 0/1/2).

## 11. Smoke findings and fixes before the formal V128 runs (2026-11-10) [all V]

Two smokes ran. The first (`smoke/runner_20261110T104457Z.json`) found the problems below. The second
(`smoke/runner_*` after 10:54 UTC, log `logs/exp_chain2.log`) ran after all fixes. Every 8192-token smoke cell
completed in it, for all four arms at TP1 and TP4.

- **ShadowKV build.** The redistributable CUDA 12.9 subset has no cuSPARSE/cuBLAS/cuSOLVER headers, which
  `ATen/cuda/CUDAContextLight.h` includes. The build now takes them from the env's matching `nvidia-*` wheels
  (`CPATH`, `run_v2q8_exp_smoke.environment`). Before this fix every ShadowKV cell failed at import.
- **LRQK B layout.**
  - Upstream keeps `B_Q`/`B_K` in the column-major layout that `torch.linalg.solve_ex` returns. The TP factory had
    them row-major. Values were equal, but cuBLAS ran other GEMMs, so rounding differed, and the rounding difference
    grew over decode steps through near-tied top-k scores.
  - Now allocated in upstream's layout. With it, the W=1 factory equals upstream **bitwise** over 72 decode steps:
    hit sets, GPU hit cache and B factors.
- **LRQK at W=4, synthetic tensors.** The rank's batched GEMMs cover 8 heads instead of 32, so cuBLAS picks other
  kernels. The same divergence appears (minimum hit overlap 0.954 after 72 steps; the first B difference is
  4.9e-4).
  - Judged instead on real text (`benchmarks/system/check_v2q8_tp_consistency.py`): one 8192-token C4 prompt, 128
    greedy tokens, W=1 vs W=4.
  - LRQK: identical, 128/128. ShadowKV: first divergence at token 25. Dense: first divergence at token 22, its own TP
    noise. Both adapters pass the criterion "agree at least as long as dense".
- **ShadowKV model smoke, logits criterion.**
  - The smoke's own criterion was max relative logits error < 0.02. It failed at 0.0278 (one of nine positions).
    Generated tokens were identical, and selection overlap was at least 0.969.
  - Dense HF TP4 vs one GPU on the same positions measures 0.010–0.018
    (`tests/distributed_dense_tp_noise_smoke.py`), so the original criterion was below dense TP's own noise.
  - The formal gate now requires: identical tokens, overlap >= 0.95, and logits error <= 2x dense TP noise (0.036).
    This criterion was changed after seeing the result, and is recorded as such.
- **Harness fixes, no effect on any method.**
  - The memory inventory crashed on DTensor parameters and torch.compile FakeTensors. It now takes a DTensor's local
    shard and skips storage-less wrappers.
  - BasisKV value reuse at TP>1 with V128 now rotates the row-parallel local `o_proj` with the local KV heads' bases.
- **Allocator.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set for every arm.
- **LRQK memory, exact savings.** Neither changes any value; the W=1 bitwise check still passes after both.
  - The FP32 repeated Keys are built without keeping upstream's BF16 repeated copy alive (1 GiB at 128K).
  - The ALS early-exit MSE is computed in the dead old factors instead of a new 0.5 GiB temporary.
- **LRQK capacity under the requested placement** (Keys on host; Values, `A_K` and the hit cache on GPU). `A_K` at
  rank 32 per query head is 4 x 32 = 128 dims per KV head, the size of the Keys themselves. GPU-resident LRQK state
  is therefore the dense KV cache plus the hit cache.
  - TP1, 130048 tokens: **OOM, about 50 MiB short** on the 44.4 GiB L40S. The failure is inside the ALS fit
    (`new_A_K = K @ solve(...)`), whose FP32 working set (Q, K, four factor tensors) upstream needs too.
  - TP4, B16, 65536 tokens: **OOM** at the first prefill. About 38.5 GiB per rank is resident before any transient.
  - The formal runner records these as failed runs, without changing settings.
- **LRQK torch.compile.** Steady decode at TP1, 8192 tokens, with the formal step counts (16 + 128, GPU 3 alone):
  eager 188.7 ms/step median, upstream torch.compile 241.8 ms/step. Eager is the faster setting. The 4-step smoke
  numbers (460 / 850 ms) were taken with four TP1 smoke jobs sharing the node and are not used as timings.
  - These two runs overlapped the formal Exp 1 BasisKV cells on GPU 0. They used the other socket, root port and
    NUMA node.

## 12. Round 2: baselines reproduced from their upstream code (2026-11-10) [all V]

Requester's rule after round 1: do not handicap the baselines, and do not optimize them either; reproduce their
upstream code honestly. Round 1 results stay in `formal/` and `smoke/`; round 2 writes `smoke_v2/` and `formal_v2/`.

- **Why round 1 was replaced.**
  - The round-1 ShadowKV and LRQK adapters reimplemented parts of the upstream code:
    - ShadowKV: the SVD without upstream's sync, `gc.collect()` and `empty_cache()`; U and SV placed on GPU
      directly; the HF model around the attention.
    - LRQK: the ALS and decode solvers, exact-size storage, one-request prefill, no prefill parking, torch.compile
      off by default.
  - Some of these favoured the baselines. Others did not: keeping LRQK's `A_K` for every layer on GPU during
    prefill raised its prefill peak by up to about 8 GiB at 130K, where it was 50 MiB short.
- **ShadowKV (`shadowkv_upstream_v128`).**
  - TP1: the upstream `models.llama.Llama` + `ShadowKVCache_CPU`, imported unedited, as `system_benchmarks/tp1_offline` ran it:
    - flash_attn served by vLLM's FA2 build;
    - MInference disabled;
    - generation buffers sized for the requested output.
  - TP4: `basisserve/core/shadowkv_upstream_tp.py` subclasses the upstream `Llama` and changes only:
    - weight shards: local heads and MLP slice;
    - AllReduce after `wo` and `down_proj`;
    - local head counts;
    - the global SVD: the layer owner gathers the Keys, then U and SV are scattered.

    The SVD stays inline in the layer, as upstream. Embedding, norms and lm_head are replicated, as upstream holds them.
  - Prefill is upstream `batch_prefill`: request groups of 4, or 8 for 120K–200K prompts. TTFT of a request is the
    end of its group.
- **LRQK (`lrqk_upstream_koff_v128`).** The upstream runtime as `tp1_offline` ran it:
  - HF model + `load_model_hack` + `DynamicLRQKCache` + `LightAttentionIndicesOffloadPrefill`;
  - torch.compile as upstream;
  - per-decode-layer `empty_cache`, 1.5x pool growth;
  - the whole batch prefilled in one forward;
  - `ChunkedTokenwise` MLP, as in tp1_offline.

  The one deviation is the requested placement: Values on GPU (`basisserve/core/lrqk_upstream.py`).
  - `SplitKV` takes the place of the shared CPU K/V pool: Keys go to an upstream pinned host `AutoIncreaseTensor`,
    Values to an upstream GPU `AutoIncreaseTensor`.
  - The decode miss fetch takes Keys through upstream `take_along_dim_with_mask_python`, and Values with the same
    upstream indices by GPU `index_select`.

  TP4 runs the HF TP model with local head counts on each attention module, which upstream's forward reads first.
  The random `A_Q`/`A_K` initialisation and the early-exit mean then cover each rank's local heads.
- **Environment.**
  - The baselines run with PyTorch's default CUDA allocator, as upstream does. Measured at TP1, 8K, upstream
    ShadowKV: `expandable_segments` added about 1 s per request to the housekeeping inside `get_svd` (5.6 s against
    4.6 s).
  - BasisKV keeps `expandable_segments:True`.
  - One informational smoke run measures LRQK at 130K under `expandable_segments`.
- **Shim.** The flash_attn shim (`paper_faithful_compat.py`) no longer asserts B1. Its varlen call with uniform
  `cu_seqlens` is exact for equal-length rows; B1 is unchanged.
- **BasisKV fix (our method).** `v2q8_basis.py` wrote prompt Keys to host with a plain `copy_` between non-contiguous
  tensors. PyTorch staged that copy through a pageable buffer: 226 ms per layer at 130K.
  - It now uses `router.append`, as the archived Table 3 runtime did: `cudaMemcpy2DAsync`, 10.2 ms per layer.
  - The result is bit-identical: the same tokens at 8K.
  - 8K prefill: 1.24 s before, 0.83 s after.
- **Checks.** `check_v2q8_tp_consistency` compares each TP4 port with its upstream TP1 runtime: 8192 tokens, 128
  greedy tokens. The reference is dense attention's own W=1 vs W=4 agreement. This gates the Exp 2 baselines.

### 12.1 Fixes after the round-2 review (2026-11-11) [all V]

An independent review covered the ShadowKV port, the LRQK port, the harness, and cross-arm fairness. Each finding
was checked again by a separate skeptic; the confirmed findings are below. `smoke_v2/` was run before these fixes.
`smoke_v3/` and `formal_v2/` were run after them.

- **ShadowKV TP port bug, found by smoke_v2.** At TP4 B16 64K the job failed when it entered upstream's chunked
  post-attention path (batch x length > 64K).
  - Cause: upstream shapes the output buffer like the attention output. On one GPU that has the model width; under
    TP it is narrower.
  - Fix: `TPLlama.layer_compute` is the upstream body with that one line changed, to shape the buffer like the
    residual. On one GPU the two shapes are identical.
- **flash_attn shim at batch > 1 (both baselines).** Flattening the per-row decode buffers made a copy of every
  layer's K and V at every step, which upstream's `flash_attn_with_kvcache` does not do. At Exp 2 that is about
  1.3 GB per step and per rank.
  - Batched decode now reads each row's buffer in place as one paged-KV block: vLLM FA2 with `block_table`, the
    path the round-1 adapter used.
  - B1 and prefill are unchanged, since they were already views.
  - ShadowKV generation buffers are rounded up to 16 tokens to fit the paged path. This changes capacity only.
- **LRQK GPU Value store.** It used upstream's host-pool container, whose first allocation is 1.5x the prompt. On
  the GPU that reserved 50% more than the Values need: about 4 GiB at TP1 130K, 8 GiB per rank at TP4.
  - It now uses upstream's `PreAllocatedTensor`, sized to the cache capacity (prompt + output + 64).
  - The host Key pool keeps upstream's 1.5x policy.
- **LRQK miss fetch.** The linear indices were computed twice per layer and step, once for K and once for V.
  - They are now computed once, with upstream's compiled helper, as upstream does for its shared pool.
  - They are remapped to the Key pool's capacity. Keys then go through `cpp_kernel.d0_index_select` on the host and
    H2D; Values are gathered on the GPU with `index_select`.
- **TP consistency gate.** The first-divergence gate was one-sided: it checked only the baselines, and judged them
  on a single near-tie in free-running greedy decoding.
  - In smoke_v2, first divergence on three prompts was:

    | Arm | Prompt 0 | Prompt 1 | Prompt 2 |
    |---|---|---|---|
    | dense | 22 | 128 | 75 |
    | ShadowKV port | 20 | 12 | 24 |

  - The review checked the port on CPU against upstream: the SVD factors are bitwise equal, and the sharded layer
    matches to 3e-13 in fp64. The early divergence is therefore top-k selection amplifying BF16 rounding.
  - Replacement (`check_v2q8_tp_consistency.py`): every arm, BasisKV included. The W=4 run is teacher-forced on the
    W=1 greedy tokens over three prompts; it reports top-1 agreement and |Δ logprob| next to dense.
  - Criterion, fixed before those runs: pooled top-1 agreement >= 0.90. It catches a broken port and does not rank
    arms.
  - Teacher forcing reproduces free-running tokens and log-probabilities exactly when an arm is fed its own tokens
    (checked at TP1).
- **Harness.**
  - A relaunch deletes the earlier `result.json` and `rank_*.json` first. A run counts as complete only with exit
    code 0.
  - The CPU binding slot is the physical GPU index, so concurrent TP1 smoke jobs on GPU 0 and GPU 1 no longer share
    one CPU half. Formal Exp 1 was serial on GPU 0 and is unaffected.
  - `result.json` records `PYTORCH_CUDA_ALLOC_CONF`.
  - BasisKV's `attention_block` remainder is reported as `attention_block_rest`, like the baselines' parent
    remainders.
  - With grouped prefill, `prefill_seconds_per_request` was 0 for every row after the first in a group. It now
    splits each prefill call's time evenly over its rows; `prefill_call_rows_and_seconds` records the calls.
  - The request mode records the log-probability of every fed token.
  - The ShadowKV breakdown label is now `factor_redistribution_incl_owner_svd_wait`: non-owners wait in the U
    broadcast until the owner's SVD is done.
- **Disclosure corrections to section 12.**
  - LRQK uses tol (1e-2, 1e-2). That is upstream's `LRQKChatBot` default, which its RULER-128K configs use;
    `tp1_offline` used 1e-8, the README example. With max_iter 2, tol only controls the early exit.
  - `ChunkedTokenwise(1024)` MLP is applied at every length for LRQK. tp1_offline applied it from 130K up.
    Numerically it is exact.
  - The ShadowKV TP port replicates embedding and lm_head on every rank, as upstream holds them. The HF-TP arms shard
    lm_head. The cost is about 0.75 GiB more per rank, plus the full lm_head GEMM per step.
  - Prefill schedules differ by arm:
    - BasisKV: one row at a time;
    - ShadowKV: upstream groups of 4;
    - LRQK: the whole batch in one forward.

    `batch_ready_seconds` has the same meaning for every arm and is the primary prefill metric. Mean TTFT depends on
    the schedule.
  - Host memory from the inventory is taken after decode. ShadowKV's pageable host U and SV during prefill appear only
    in `process_rss` after prefill.

### 12.2 BasisKV implementation used in round 2 (the authors' method) (2026-11-11) [all V]

The round-2 BasisKV arms are `basis_koff_v128_opt` and `basis_koff_v128_reuse_opt` (formal grid: `basis_koff_v128_reuse_opt` only, the requester's choice; `_opt` was smoke-tested): the same method and placement as
`basis_koff_v128[_reuse]`, with two implementation changes. The paper's Table 3 used the archived runtime. These arms
select the same pages; only the speed differs.

- **Lean forward** (`basisserve/core/lean_llama.py`). The non-attention computation is that of upstream ShadowKV's
  model and its TP port:
  - replicated embedding and lm_head, flashinfer RMSNorm;
  - fused gate_up + vLLM `silu_and_mul`, plain local weights, AllReduce after down_proj;
  - 8192-token MLP chunks.

  It replaces HF's DTensor tensor-parallel path, which cost about 12 ms per decode step at TP4 B4. The two arms now
  differ only in attention.
  - At TP1 the generated tokens are identical to the HF path.
  - Teacher-forced on the HF path's 128 tokens: top-1 agreement 0.992, mean |Δ logprob| 0.014 (W=1); 1.000 and
    0.013 (W=4). Dense W=1 vs W=4 noise is 0.992 and 0.013.
  - The replicated embedding and lm_head add 0.74 GiB per rank at TP4, as in the ShadowKV port.
- **Faster router** (`benchmarks/system/table3_router_fast.cu`, built by `table3_page_router.build_fast`). Page scores
  are bit-identical to the archived register router.
  - Bit equality was checked on seven shapes, including TP1 130K, TP4 B16 64K, the strided reuse view and ragged
    lengths. The tokens are identical to the lean arm with the archived router.
  - Unchanged: MMA, thread-to-feature mapping, accumulation order, reductions.
  - Changed:
    - prefetched RoPE values and fully unrolled feature loop;
    - vector code loads, with each residual code read once for two query heads;
    - BF16-operand roundings done with packed BF16 FMA. These are equal by exact products and by Figueroa's
      double-rounding bound.
  - Per layer: 343 → 220 µs at TP1 130K, 737 → 506 µs at TP4 B16 64K.
  - Per decode step at TP1 130K (CUDA profile): router 11.6 → 7.4 ms; total kernel time 36.5 → 32.3 ms (lean
    forward plus fast router).
- **Formal_v2 contents.** It keeps the HF-path BasisKV runs made before this change (`basis_koff_v128[_reuse]_*`,
  Exp 1 and part of Exp 2) and the baseline runs. The final comparison uses the `_opt` arms.

### 12.3 LRQK in its upstream placement (2026-11-11) [all V]

With Values kept on GPU (`lrqk_upstream_koff_v128`), LRQK ran out of GPU memory at TP1 130K in every attempt. The
failure is in upstream's ALS prefill fit, `_lrqk_prefill_inv_w1`. The requester then asked for LRQK with K and V both
offloaded.

- New arm `lrqk_upstream_kvoff_v128`: the upstream classes themselves: `LightAttentionIndicesOffloadPrefill` with
  upstream's shared pinned host K/V pool and its single CPU gather for missed K and V. None of
  `basisserve/core/lrqk_upstream.py`'s cache code is used; only `install` (load_model_hack, local head counts under
  TP) and `new_cache` (configuration) are.
- This is upstream's own placement, as `tp1_offline` ran it; here tol is (1e-2, 1e-2), and MLP chunking is applied
  at every length.
- It runs after the three-arm formal grid: smoke, teacher-forced TP check, then formal Exp 1 and Exp 2 into
  `formal_v2/`.
