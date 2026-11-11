<!-- Agent-generated read-only audit notes (2026-11-10). Key claims re-verified in ../COMPATIBILITY.md; treat the rest as leads, not verified facts. -->
# TP4 compatibility audit: Llama-3.1-8B-Instruct, 4x L40S, 64K prompt, dense / basiskv / lrqk / shadowkv

Audit date 2026-11-10. Repo `/workspace/BasisServe-CALS` at `abf8550` (read-only; nothing in the repo was modified, nothing was run on GPU, no pytest).
Upstream clones (read-only, never executed) are under `/tmp/claude-0/-workspace-BasisServe-CALS/933b9de8-a7b5-4403-a350-5f7c8680deb1/scratchpad/upstream/`:
- `ShadowKV/`: https://github.com/ByteDance-Seed/ShadowKV, checked out at `e51904cdeab7d4d34013370f09f2cf5fcd655e15` ("ICML Spotlight", 2025-05-01).
- `LRQK/`: https://github.com/tenghuilee/LRQK, checked out at `caf16293db2e4423a84ab2e895bacf64479f1eb7`. `sha256(lrqk_attention.py)` = `2bfbc6df…fab5`, which matches the hash recorded in `docs/c1_lrqk_integration.md:23`. The directory already had content when I cloned (`git clone` reported "not empty", probably from an earlier agent in this session). The checkout at the pinned commit worked.

Tags: **[V]** means verified from code, results or the environment. **[I]** means inference or estimate.

---

## 0. Target node and environment facts (this instance) [V]

| Item | Value | Evidence |
|---|---|---|
| GPUs | 4x L40S, 46,068 MiB each, PCIe Gen4 x16. Currently ~27.6 GB used per GPU by another job | `nvidia-smi` query |
| Topology | GPU0–GPU1 `PXB`, GPU2–GPU3 `PIX`, all cross pairs `SYS`, no NVLink. GPU0/1 are on NUMA 0 (CPUs 0-23,96-119), GPU2/3 on NUMA 2 (48-71,144-167) | `nvidia-smi topo -m` |
| CPU / RAM | 2x Xeon Platinum 8558, 4 NUMA nodes (SNC), 2.0 TiB RAM (1.9 TB available) | `lscpu`, `/proc/meminfo` |
| Python env | `/workspace/basis_artifacts/envs/vllm/basis`: torch 2.13.0+cu129, transformers **5.22.2**, vllm 0.30.0, flashinfer 0.6.18, triton 3.7.1, NCCL 2.29.7. **No `flash_attn` package** (only `vllm/vllm_flash_attn`) | site-packages listing |
| CUDA toolkit | `/workspace/basis_artifacts/cuda/12.9`, `/usr/local/cuda` (nvcc present) | ls |
| Model | `/workspace/.hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f2…` | `/workspace/basis_artifacts/bench_model_path` |
| Local BasisKV factors | `checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix` (V96, `value_coordinate_encoders [8,128,96]`, `head_output_decoders [32,96,4096]`). The V64 bank is also present. **No B16R16 router bank locally** (no `router_b16r16/`, no `ours_b16r16/`) | safetensors headers |
| Missing externals | `/workspace/BasisServe-CALS/external/{ShadowKV,LRQK}` do not exist, so the ShadowKV `.so` and the LRQK `cpp_kernel` are not built. The TP8 prompt cohorts `/workspace/runs/l31-cal128/tp8-benchmark-prompts/p65536_c*.safetensors` and `/workspace/runs/...` are absent | ls |
| HF transformers TP plan (5.22.2) | q/k/v `colwise`, o `rowwise`, gate/up `colwise`, down `rowwise` (`models/llama/configuration_llama.py:49-57`). lm_head is `colwise_gather_output` (`modeling_llama.py:412`). Rowwise inference path is `dist.all_reduce(output)` (`transformers/distributed/tensor_parallel.py:323`). Embedding is not in the plan, so it is replicated | site-packages |

Note: the TP8 formal runs used Transformers **5.17.0**, torch 2.13.0+**cu130** and a source-built flash_attn 2.8.3.post1 on an 8x L40S AMD EPYC 9354 host (`docs/hardware/l40s_tp8_topology.md:10-16`; `tp8_external_baselines/shadowkv/SUMMARY.md` provenance). This node has a different TF version and no flash_attn. The adapters use `hasattr(w,"to_local")` defensively (`llama31_8b_tp8_combined.py:275-286`), but HF-TP internals changed between 5.17 and 5.22, so a TP4 smoke run is mandatory.

`results/rebuttal/v2q8_palu_g4_tp4/INTERCONNECT_L40S.md` describes a *different* 4x L40S node ("node 1", AMD EPYC, all GPUs on NUMA 0, Gen4 x8 uplink). On that node a 256 KiB 4-GPU all-reduce took 292 µs with NCCL's LL protocol, versus 171 µs with `NCCL_PROTO=Simple` (lines 21-26, 147-161). Run `benchmarks/system/check_interconnect.sh` on this node before any formal TP4 timing.

---

## 1. Which arms already run as true multi-GPU TP

### 1a. Summary table [V]

| Arm | TP8 | TP4 | Launcher | Notes |
|---|---|---|---|---|
| Dense | yes: `benchmarks/system/bench_llama31_8b_tp8_request.py` (request/steady) and `bench_llama31_8b_tp8_combined.py` (decode grid), both using `DenseTP8Attention` | yes: `benchmarks/system/bench_e2e_tp4.py --mode dense` (`LlamaTP4KCacheAttention`, Llama **base**). Also vLLM `bench_llama_vllm.py --tp 4 --arm dense` (4K prompts only in the PaLU grid) | torchrun `--nproc-per-node=8` (`run_llama31_8b_tp8_request_grid.py:106`); `torch.distributed.run --nproc-per-node=4` (`run_e2e_tp4.py:32`); vLLM mp executor | HF TP via `DistributedConfig(tp_size=8)` (`bench_llama31_8b_tp8_request.py:109-111`) or `tp_size=4` (`bench_e2e_tp4.py:53-54`) |
| BasisKV | yes: `basis_joint` full-scan B16R16 in `basisserve/core/llama31_8b_tp8_combined.py` (JointALSTP8Attention). Also vLLM TP8 two-stage path (results `results/tp8_joint_l40s_20260923/`; its runner `bench_vllm_tp8_joint.py` is **no longer in the repo**) | partially: (i) `basisserve/core/llama_tp4_c1_system.py` modes `sparse_local`/`offload` (Llama **base**, KL-allocated variable V ranks averaging 96, mapped host K, no slots); (ii) `benchmarks/system/tp4_joint_sparse.py` (TP4/8-generic, two-stage-512 router, uniform V96, **no recorded results**); (iii) vLLM `basisserve/vllm/llama_joint_als.py` + `llama_joint_backend.py` (`world_size in (4, 8)`, two-stage router; TP4 run only as `als_full` at 4K) | torchrun / torch.distributed.run / vLLM | see §4 |
| ShadowKV | yes: `basisserve/core/shadowkv_tp8.py` + `shadowkv_tp8_attention.py` via `bench_llama31_8b_tp8_request.py --arm shadowkv` | **no** | torchrun `--nproc-per-node=8` | hard-wired to 1 KV head per rank (see §2) |
| LRQK | **no** | **no** | none | Only multi-GPU LRQK in the repo is `evaluation/eval_qwen3_32b_v96_lrqk_tp2.py` (Qwen3-32B, TP2, repo-adapted equations, quality evaluation). See §3 |
| (Quest) | TP8 native CUDA (`results/system_benchmarks/quest_tp8/`). Its integration files (`basisserve/core/quest_native_tp8.py`, `basisserve/kernels/quest_native.py`) live in the `-opt` worktree and are **not in this repo** (`quest_tp8/native/README.md`) | no | | out of scope |

### 1b. Head and KV-head sharding [V]
HF colwise sharding gives rank r Q heads `[8r, 8r+8)` and KV heads `{2r, 2r+1}` at TP4, and Q heads `[4r,4r+4)` with KV head `r` at TP8. GQA mapping is h → h//4, so each rank's groups are complete and the local GQA ratio stays 4.
- TP8 code hard-codes this: `TP_SIZE = 8; QUERY_HEADS_PER_RANK = 4; KV_HEADS_PER_RANK = 1` (`llama31_8b_tp8_combined.py:32-34`). The factor encoder slice is `encoder[rank]` (`:123`). Router tensors are narrowed with width 1 per KV head, or 4 for `residual_query` (`:97-106`).
- The TP4 code hard-codes 2 KV heads and 8 Q heads: `value_coordinate_encoders[rank*2:(rank+1)*2]` (`llama_tp4_c1_system.py:28`), router slice `rank*(8 if residual_query else 2)` (`:45`), `.view(b,…,8,128)` (`:97`), `.reshape(b,…,1024)` (`llama_tp4_k_offload.py:85`).
- Generic code: `local_kv = 8 // world_size` (`tp4_joint_sparse.py:31`; `llama_tp4_uniform_als.py:17-20`). vLLM uses `count = 8 // group.world_size` (`llama_joint_als.py:125`).

### 1c. Collectives [V]
| Arm | Decode per layer | Prefill | Evidence |
|---|---|---|---|
| Dense (HF TP) | AllReduce of o_proj output `[B,1,4096]` BF16 plus AllReduce of down_proj output `[B,1,4096]`, so 64 AR per step. One logits AllGather (lm_head `colwise_gather_output`, `[B,1,vocab/TP]` input) | Same ARs per 2048-query chunk `[B,2048,4096]` and per 1024-token MLP block `[B,1024,4096]` | `tensor_parallel.py:323`; `llama31_8b_tp8_combined.py:198-233,734-736` |
| BasisKV | **AllGather** of local V-coordinate outputs (feature-major `[Hq_local·96, B]`: 384 rows at TP8, 768 at TP4) through the custom `FeatureRaggedCommunicator` `uniform_nccl`, then the replicated decoder `[32·96, 4096]` (no o_proj AR). The MLP AR is kept | AG per 2048 chunk plus MLP AR | `llama31_8b_tp8_combined.py:361-369, 409-423`; `llama_tp4_c1_system.py:49-50,58-67` |
| ShadowKV | Same as dense (2 AR/layer). No K gather in decode (README check 4) | Extra per layer per request: P2P `batch_isend_irecv` of `[T,128]` pre-RoPE K from every non-owner to owner `layer%8`, `dist.broadcast` of U `[T,160]` BF16, P2P SV shards `[128,160]` | `shadowkv_tp8.py:58-98` |
| LRQK (TP2 Qwen precedent) | Row-parallel local decoder then `dist.all_reduce(hidden)`, plus scalar AllReduce for convergence deltas | same | `eval_qwen3_32b_v96_lrqk_tp2.py:121-125,178,208,392` |

Greedy token choice: `_choose` does `dist.all_gather` only if the logits are vocab-sharded (`bench_llama31_8b_tp8_combined.py:122-134`). Under 5.22 `colwise_gather_output` the logits are already full, so this collective is skipped.

### 1d. World-size-8 assumptions to remove for TP4 [V]
- `bench_llama31_8b_tp8_request.py`: `DistributedConfig(tp_size=8)` `:111`; `args.batch in (1, 4, 8)` `:147` (B16 not allowed); `WORLD_SIZE == 8` `:157`; `get_world_size() == 8` `:165`; ShadowKV `svd_calls == args.batch * 4` `:326` (32 layers / 8 owners; this becomes `batch*8` at TP4); `[None]*8`, `set(range(8))`, `successful_ranks: 8`, `range(8)` `:380-401`.
- `run_llama31_8b_tp8_request_grid.py`: `--nproc-per-node=8` `:106`; `range(8)` `:33,42`; `batches choices=(1,4,8)` `:65`; `FULL_GRID` `:19-25`.
- `tp8_request_timing.py`: owner test `layer % 8 == rank` `:43`; `len(ranks) == 8` `:63-64`.
- `summarize_tp8_shadowkv_request.py:35-37,80` (8 rank files).
- `llama31_8b_tp8_combined.py:32-34` constants. Also the derived shapes: `(1, VALUE_RANK, BASE_RANK)` assert `:101`, single-head fold `encoder.float().T @ dense_value_weight` with `(HEAD_DIM,4096)` assert `:274-280`, `coordinate_base_left` `(1,96,16)` `:326`, all `(batch, 1, …)` buffers `:334-359`, plan `(4·96,)*8` `:361-363`, `.view(batch,…,1,HEAD_DIM)` in prefill `:612-631`.
- `bench_llama31_8b_tp8_combined.py`: `WORLD_SIZE == TP_SIZE` `:197`; `_bind_cpu` uses `local_peers = TP_SIZE // 2` `:115`. This happens to equal 2 at TP4, which matches 2 GPUs per NUMA node here.
- `shadowkv_tp8.py:10,17-18,26,31,44-45,50,54,60-62,72,83,89`; `shadowkv_tp8_attention.py:43,48-49,63,69,77,80-82,95`.
- Tests: `tests/distributed_shadowkv_tp8_{factors,cache,model}_smoke.py` assert `world_size == 8` (`factors:32`, `cache:62`, `model:61`).

The CUDA/Triton kernels are already generic in the KV-head count. `conditional_router_page32.cu` indexes `kv_heads` at runtime (`:107-148, 933-1016`, compile-time constant only `BASIS_GQA=4`). `fused_decode_postprocess.cu` takes `kv_heads = page_log_mass.size(1)` (`:439`). `slot_indexed_attention` uses `HEADS_PER_KV=heads//k.shape[1]` (`slot_indexed_attention.py:157-171`). `mapped_host_paged_attention.py` derives `kv_heads` from shapes (`:103-175, 221-287`). `StaticRaggedPlan`/`FeatureRaggedCommunicator` take any world size up to 32 (`ragged_allgather.py:144-159`, `feature_ragged_allgather.py:373,422-425`). Upstream ShadowKV kernels take `batch_size*heads` blocks at runtime (`kernels/gather_copy.cu:52,123,198,249`).

---

## 2. ShadowKV TP8 adapter vs upstream

### 2a. Global SVD: confirmed **one global rank-160 SVD, then sharded factors** [V]
- Pre-RoPE K is projected per rank (`shadowkv_tp8_attention.py:43-49`, `k_proj` only). Factors are built **before** RoPE is applied (`:56-62`).
- `build_global_factors` (`shadowkv_tp8.py:35-101`) loops over requests one at a time (`:58`). For each request:
  1. Non-owners P2P-send their `[T,128]` head to owner `layer % TP_SIZE` (`:54,59-74`).
  2. The owner forms `global_key = gathered.permute(1,0,2).reshape(1, T, 1024)` (`:72`), which is head-major across all 8 KV heads, the same layout as upstream `get_svd` (`models/kv_cache.py:484`).
  3. The owner runs `torch.svd(global_key.float())` (`:79`), the same as upstream (`kv_cache.py:499`).
  4. U `[T,160]` is copied into a replicated BF16 buffer and broadcast (`:80,87`). `SV = diag(s[:160]) @ Vᵀ[:160]` is cast to BF16 and cut into per-head shards `sv.view(160, 8, 128).permute(1,2,0)` (`:81-83`). The upstream equivalent is `kv_cache.py:506-511`. Each rank receives only its `[128,160]` shard (`:88-95`).
- README states the same: "Concatenate all eight pre-RoPE local KV heads on owner `layer % 8`; apply the official FP32 `torch.svd` … truncate to rank 160" (`tp8_external_baselines/shadowkv/README.md:39-40`). Smoke check: "Factor reconstruction relative errors were zero" against unchanged upstream `get_svd` (`:56-58`).
- Per-GPU independent SVDs are **not** used.

### 2b. Representation details (unchanged official `ShadowKVCache_CPU`, loaded by AST extraction in `evaluation/official_shadowkv_cpu.py:20-40`) [V]
| Component | Behaviour | Upstream line |
|---|---|---|
| Chunk size | 8 tokens; `sparse_budget=2048` gives `select_sets=256` chunks | `make_tp8_cache` `shadowkv_tp8.py:19-21`; `kv_cache.py:441` |
| Local window | `local_chunk=4`. `chunks = T//8 - 4`, rounded down to a multiple of 8. `prefill_local = T - chunks*8` (64 tokens at T=65536). Exact post-RoPE K/V kept on GPU | `kv_cache.py:397, 546-553` |
| Outliers | `outlier_chunk = (2048//1024)*24 = 48` chunks (384 tokens). Chosen as the chunks with the lowest min cosine similarity between each token and the chunk mean. Exact post-RoPE K and V kept on GPU | `:398, 556-577` |
| Landmarks | Chunk-mean of post-RoPE K for non-outlier chunks (8184−48 = 8136 per KV head at 64K). Stored on CPU during prefill and moved to GPU in `H2D()` | `:556, 579-588, 668-688` |
| Initial selection | `softmax(q_last · landmarks / √128)`, max over the 4 Q heads of a group, top-256 chunks | `:590-603` |
| Decode selection | CUTLASS `batch_gemm_softmax`, max over group, `topk(256)`. `reorder_keys_and_compute_offsets` computes hit/miss against the previous step | `:616-645` |
| Values | **Full prompt V on pinned CPU** (`v_cache_cpu [L,B,Hkv,T/8, 8·128]`, `pin_memory=True`). Selected **misses** are fetched by `shadowkv.gather_copy_with_offsets`, a GPU kernel that reads pinned host memory zero-copy into `v_cache_buffer`. Hits are compacted on GPU. The fetch runs on `copy_stream`, overlapped with K reconstruction (`shadowkv_tp8_attention.py:86-91`) | `:400-409, 647-653` |
| Keys | Misses are reconstructed as `U[pos] @ SV_hᵀ` plus RoPE by CUTLASS `batch_gather_gemm`. Hits are moved with `gather_copy_d2d_with_offsets`. Generated tokens' exact K/V are appended to GPU buffers | `:655-666, 690-702`; `tensor_op.py:199-230` |
| Decode attention | `flash_attn_with_kvcache(…, causal=True)` over local+outlier+selected+generated tokens (2496 + generated at 64K) | adapter `:92-94`; upstream `base.py:161` |

### 2c. Algorithmic and implementation deviations, adapter vs upstream [V unless marked]
1. **TP distribution** (new): one owner per layer (`layer % 8`) holds global K. U is **replicated on every rank** (`cache.U` `[L,B,T,160]` on GPU, `shadowkv_tp8.py:30`). SV is sharded per KV head. Upstream is single-GPU with one U.
2. **Placement and timing of U/SV**: upstream allocates U/SV on **CPU** during prefill (`kv_cache.py:489-492`) and moves them to GPU in `H2D()` after prefill. The adapter allocates U/SV directly on GPU before prefill (`shadowkv_tp8.py:29-31`), so U is GPU-resident during prefill. This raises the prefill peak and removes the U H2D. Values are bitwise the same BF16 casts.
3. **Batching of the SVD**: upstream prefills micro-batches of `T=4` requests (`T=8` for >120K), each with one batched `torch.svd` over `[bsz,T,1024]` (`base.py:266-289`, `:278-280`). The adapter runs the whole batch in one model forward and does a **sequential per-request** SVD on the owner, with every other rank waiting (`shadowkv_tp8.py:58`). The math per request is identical, but concurrency and timing differ.
4. **Model execution**: upstream uses its own Llama runner with vLLM rotary op and flashinfer norm (`llama.py:143`). Its prefill attention is one `flash_attn_with_kvcache` over the whole prompt (`base.py:138`) and its MLP runs in 8K-token chunks when `bsz·q_len > 64K` (`:171-183`). The adapter uses HF modules under HF TP, SDPA FLASH backend in 2048-query chunks with `causal_lower_right` (`shadowkv_tp8_attention.py:67-77`), `ChunkedTokenwise` norms and 1024-token MLP blocks (`llama31_8b_tp8_combined.py:705-738`).
5. **Prefill trigger**: upstream calls `get_svd` only when `q_len > 4*1024` (`base.py:129`). The adapter always does it on the first forward (`shadowkv_tp8_attention.py:41`). This only matters for the 4K smoke.
6. **Buffer capacity**: upstream's GPU buffer is `2048+128+(48+4)*8 = 2592`. The adapter uses `local + 384 + 2048 + decode_steps` (`shadowkv_tp8.py:22-28`; README `:43`). At 64K upstream would overflow after 96 generated tokens.
7. **Cache config trick**: each rank instantiates the official cache as a 1-KV-head model (`hidden_size=512, heads=4, kv=1`, `shadowkv_tp8.py:17-18`). Landmarks, outliers and selection are per KV head, so they are identical to upstream for that head. Only the SVD is cross-head (deviation 1).
8. `H2D()` is timed inside "representation build" (`bench_llama31_8b_tp8_request.py:241-247`). Upstream's `batch_generate` calls it before a warmup (`base.py:320`).
9. [I] RoPE application differs in kernel (HF `apply_rotary_pos_emb` vs vLLM custom op) but the math is the same. The `cos_sin` table is built from HF `rotary_emb` (`shadowkv_tp8_attention.py:109-111`), like upstream `llama.py:156-160`.
10. Build: upstream needs `3rdparty/cutlass` from an **unpinned** `git clone` of NVIDIA/cutlass (`README.md:56-61`; `setup.py` include dirs). The CUTLASS commit used for the TP8 build is not recorded in the files I read.

### 2d. Changes for TP4 (2 KV heads per rank) [V where code is cited, I for the design]
- `shadowkv_tp8.py`:
  - `TP_SIZE = world_size` (4) and `LOCAL_KV = 8//TP`.
  - Cache config `hidden_size = 8·128 = 1024, num_attention_heads = 8, num_key_value_heads = 2`. The official class derives `head_dim = hidden/heads = 128` (`kv_cache.py:390`).
  - Buffer and SV shapes `(L,B,2,cap,128)` and `(L,B,2,128,160)`.
  - Assert `local_key.shape[1] == 2`.
  - Gather `[4, 2, T, 128]`, then global `permute(2,0,1,3).reshape(1,T,1024)` (global KV head index = 2r+j).
  - SV shards `sv.view(160, 4, 2, 128).permute(1,2,3,0)` giving `[4,2,128,160]`.
  - Owner `layer % 4` (8 SVD layers per owner).
- `shadowkv_tp8_attention.py`: views `(B,T,2,128)` for K/V and `(B,T,8,128)` for Q. o_proj input width 1024. Install asserts stay global (32/8).
- Runner, timing and summarizer: the world-size items in §1d. `svd_calls == batch*8`. Allow B16.
- Re-run the three distributed smoke tests at nproc 4: factor reconstruction vs upstream `get_svd`, cache vs full-head reference, and 4K model token parity.
- Effort: **small–medium**. The upstream kernels need no change: `heads` is a runtime parameter, and upstream's own default is 8 heads.

---

## 3. LRQK at TP>1

- [V] No multi-GPU LRQK system benchmark exists for Llama. The LRQK system paths are TP1 only:
  - `benchmarks/system/audit_tp1_request.py:157-196`: upstream `load_model_hack` + `DynamicLRQKCache(num_key_value_groups=4, r=32, num_active_tokens=2048, lite_tokens=64, max_iter=(2,2), tol=1e-8, lwattn_factory=LightAttentionIndicesOffloadPrefill)`, `device_map="cuda:0"`, B1.
  - `bench_tp1_lrqk_local.py` (GPU-local storage variant).
  - Router microbench `lrqk_router.py` / `evaluation/official_lrqk_state.py:19-23`.
- [V] The only TP LRQK code is `evaluation/eval_qwen3_32b_v96_lrqk_tp2.py`. It is a Qwen3-32B quality evaluator (TP2, repo-adapted equations from `basisserve/core/c1_lrqk.py`, V96 values resident, exact K pinned host). It shows the sharding pattern:
  - `prefill_factors_tp2` / `decode_factors_tp2` work on the local heads, and only the convergence scalar is averaged across ranks (`:121-125, 154-215`).
  - The RNG init draws the full 64-head tensor and slices it (`:239-247`).
  - Output uses a local decoder plus `all_reduce` (`:386-393`).
- [V] Upstream LRQK `LRQK_LlamaFlashAttention2.forward` uses `getattr(self,'num_heads', config.num_attention_heads)` (`lrqk_attention.py:2338-2347`). Under HF TP that resolves to the global 32 heads, so the upstream monkeypatched model **cannot run under HF TP as-is**. A TP4 adapter module is needed.

### State layout and sharding (upstream `LightAttentionIndicesFactory`) [V]
| State | Shape per layer | Device | Per-head? | Shards by KV head? |
|---|---|---|---|---|
| `A_K` (key codes) | `[B, Hq, cap, r=32]` BF16, `cap = max(capacity, int(1.5·prefill_len))` (`:881`), so 98,304 at 64K | GPU | per **query** head | yes (8 local Q heads) |
| `B_Q`, `B_K` | `[B, Hq, 128, 32]` | GPU | per Q head | yes |
| `Kgpu`, `Vgpu` (hit cache) | `[B, Hq, 2048+64, 128]` (K/V repeated per Q head) (`:1418-1427`) | GPU | per Q head | yes |
| `KVcpu` (exact K‖V) | `[B, Hkv, cap, 256]` pinned (`:1089-1095`, `:962-964`) | CPU | per KV head | yes |
| `hit_indices`, lite slots | per Q head; `lite_indices` is a token list shared by all heads | GPU/CPU | | yes (deterministic) |
| Convergence test | `delta = max(mse(A_Q),mse(A_K))` (`:401`), `mse(hat_k)` (`:496,604`): **global mean over B·Hq**, host-synchronised `if delta < tol` | — | **no** | needs a scalar AllReduce for TP1 parity, or a tolerance that never triggers (TP1 system runs use `tol=1e-8`, `audit_tp1_request.py:182`, so it always runs 2 iterations) |
| Init | `torch.randn([B,Hq,T,r])` for A_Q and A_K (`:146-153`) | GPU | global RNG stream | For parity, draw the full 32-head tensor and slice (as in the TP2 precedent), or accept a documented per-rank seed |
| B update step size | `_lrqk_decode_gd_hat_lr`: per (b,h) sums `dim=(2,3)` (`:632-650`) | | per head | yes |
| Miss fetch | `take_along_dim_with_mask_python`: indices moved to CPU, then `cpp_kernel.d0_index_select` **CPU gather**, then H2D (`:102-118`) | CPU threads | | yes, but each rank does host-synchronised CPU gathers. Runners set `torch.set_num_threads(2)` |

**Required for TP4 LRQK** [I]:
1. A `LRQKTP4Attention` module modelled on `ShadowKVTP8Attention`:
   - local q/k/v projections;
   - one upstream factory per layer with 8 Q / 2 KV heads (`num_key_value_groups=4`);
   - prefill through the factory's `prefill(q, k, v)` (repeated K inside);
   - prefill attention with the same chunked SDPA;
   - decode via `factory.decode(q,k,v)` then `flash_attn_with_kvcache` over `Kgpu/Vgpu`;
   - o_proj rowwise AllReduce (HF).
2. Parity choices: per-head RNG slicing, plus either a scalar AllReduce of convergence deltas or `tol=1e-8` (deterministic 2 iterations, no extra collectives).
3. Build `external/LRQK/cpp_kernel`.
4. Memory mitigation for B≥8: the per-layer FP32 ALS fit on `[B,8,T,128]` costs about 1.2 GB per request per rank. Fitting per request (loop over batch rows) reduces this. That is exact apart from the convergence and RNG points above.
5. A validation step against TP1 upstream on 4K prompts: per-head selected indices, `A_K`, `B_Q/B_K`, and generated tokens.
- Effort: **medium–large**.

---

## 4. BasisKV at TP4

### 4a. Candidate runtimes [V]
| Runtime | Encoding / reconstruction | Router | K offload | V | Model | TP4 status |
|---|---|---|---|---|---|---|
| **A. TP8 frozen formal path** `basisserve/core/llama31_8b_tp8_combined.py` (JointALSTP8Attention, `arm=basis_joint`) | Source-local: V projection folded with the rank's encoder (`:271-292`), coordinates written to `value_cache` (V96, Base16 = first 16 coordinates after the routing-basis transform), residual R16 (`:433-459`). Output: uniform NCCL AllGather of `[4·96,B]` plus the replicated `global_decoder [3072,4096]` (`:409-423`) | **Full-scan** B16R16 Page32 over all history (`conditional_router_page_lse` through `compile_register` 8 warps, `:490-514`). `select_pack_refresh` picks 62 pages + recent 64 = 2048 (`:39-42, 515-530`) | Exact post-RoPE K in **mapped pinned host** memory (`:313-317`). Persistent 2048 GPU K slots with hit reuse (`slot_*` `:348-359`) | V96, `tp8-combined-v96` = HF `checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6` transformed by `evaluation/prepare_llama31_8b_tp8_combined_factors.py:18-24,63-77` with router `/workspace/runs/l31-cal128/router/ours_b16r16` | Instruct | **TP8-only constants (§1d). Port effort small–medium. Best match to the TP8 ShadowKV grid** |
| B. `basisserve/core/llama_tp4_c1_system.py` (`mode='offload'` / `'sparse_local'`) via `bench_e2e_tp4.py` | Source-local folded encoder, split Base16/tail caches (`:28-39`). AllGather `(8·r,)*4` plus replicated decoder (`:49-67`) | B16R16 page-LSE, `select_fixed_group_max_pages_cuda(62, pinned_prefix_pages=1)` + recent 64 (`:108-120`). Hard 2048 including sink 32 and recent 64 (l40s `SUMMARY.md:7`) | mapped host K read by the fused `mapped_host_paged_attention` **every step, no slots** (`:41,124-129`) | **KL-allocated per-layer ranks, average V96** (`identity['layer_ranks']`, `bench_e2e_tp4.py:46,128`). Factors from `<output>/routing_basis` and `<root>/ours_b16r16` (`:68`; `llama_tp4_c1_system.py:27,44`) | **Llama-3.1-8B base** | Ran at 64K B1/4/8 on a 4x L40S host ("lovelace"). B16 OOM in all modes (l40s `SUMMARY.md` §8) |
| C. `benchmarks/system/tp4_joint_sparse.py` + `smoke_tp4_joint_model.py` | Generic `local_kv=8//world` (`:31`); transform at load (`:39-49`) | **Two-stage**: coarse `Metadata`, 512 candidates, fine B16R16, 62 pages + recent 64 (`:129-160`) | mapped host K + persistent slots (`:70-97,152-156`) | uniform V96 (`llama_tp4_uniform_als.py:35-37`) | Instruct | No recorded results. Prefill is **unchunked** (whole-prompt projections and SDPA, `:173-196`; HF default MLP) [I: heavy at 64K B≥8]. Uses list `dist.all_gather` + `torch.cat` (`llama_tp4_uniform_als.py:95-98`) |
| D. vLLM engine `basisserve/vllm/llama_joint_als.py` + `llama_joint_backend.py` + `kernels/paged_joint_{cache,router,slots}.py`, driven by `benchmarks/system/bench_llama_vllm.py --tp 4 --arm basis_joint` | `JointQKV` folds the encoder (`:22-39`). `group.all_gather(coordinates)` @ replicated decoder (`:97-99`). `world_size in (4,8)` (`:109`) | Two-stage: coarse page min/max bounds, 512 candidates, fine B16R16, 62 pages + recent 64 = 2048 (`paged_joint_router.py:155-181`) | mapped pinned host mirror of the whole block pool, 40 GiB/rank budget (`bench_llama_vllm.py:119-120`; `llama_joint_backend.py:93-97`). 2048 GPU K slots (`paged_joint_slots.py:121`) | V64 or V96 uniform (`:121`). Router from `<factors>/router_b16r16/` with JSON sidecars (`:127-156`) | Instruct | Code is TP4-capable. Only `als_full` (no routing) has run at TP4, and only at 4K (`results/rebuttal/v2q8_palu_g4_tp4/smoke/...`). Block size 32 (`llama_joint_backend.py:76-77`), chunked prefill 8192 (`bench_llama_vllm.py:125`) |

Answers to the specific questions:
- "Which runtime implements source-local encoding + TP AllGather/output reconstruction for B16R16 with K offload/fetch?": all four above do. The **only one validated at 64K with Instruct, uniform V96, full-scan routing and persistent K slots is A, and only at TP8**.
- "Is there a 64K-capable TP4 sparse path?": B (base model, variable ranks, no slots) measured 64K up to B8. A and D are 64K-capable in design but **unmeasured at TP4**.
- V rank: A, C and D use uniform **V96** (D also allows V64). B uses a variable allocation averaging V96. None uses V128 for BasisKV.
- Page and budget: Page32, 62 routed pages + 64 recent tokens = **2048** hard support per (request, KV head). A uses full-scan routing; C and D use a 512-page coarse prefilter.
- [V] Factor availability on this node: only the **retrievalmix** V96 is local, and it has no router. The HF listing already in this session's scratchpad (`hf_audit/hf_files.json`) shows:
  - `checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6/` (the TP8 formal source V96);
  - `checkpoints/k_routing/llama31_8b_instruct_v96_b16r16_32x128k_als40_pcg100/` (the router used by the vLLM TP8 run);
  - `checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix/router_b16r16/` (router for the retrievalmix V96, with vLLM JSON sidecars).
- [I] Whether `/workspace/runs/l31-cal128/router/ours_b16r16` (TP8 bank SHA `3c1523ed…`, `llama31_8b_tp8_combined/SUMMARY.md`) equals the HF `k_routing/…als40_pcg100` bank is not established. Hashes need to be compared.

### 4b. TP4 port of runtime A (recommended BasisKV arm) [V for line refs]
- Constants become `QUERY_HEADS_PER_RANK=8, KV_HEADS_PER_RANK=2`.
- Encoder slice becomes `encoder[2r:2r+2]` with a per-head fold `einsum('hdr,hdm->hrm')` (as `llama_tp4_c1_system.py:34` already does) instead of `:280`.
- Asserts `(256,4096)`, `(1024,4096)`, `(256,4096)`.
- Router narrow widths 2 and 8 (`:97`). `coordinate_base_left` becomes `(2,96,16)`.
- Every `(batch, 1, …)` buffer gets a KV dim of 2. Plan becomes `(8·96,)*4`. `decode_output` view becomes `(B,8,1,96)`.
- Prefill views `(B,T,2,128)`. Runner items as in §1d.
- Kernels need no change.
- Effort **small–medium**, plus validation: a 4K token-parity smoke vs TP8/TP1, and slot hit-rate sanity.

---

## 5. Dense at TP4 [V]
| Path | Prefill | Decode | Collectives | Status |
|---|---|---|---|---|
| `DenseTP8Attention` (TP8 grid dense arm) | SDPA `FLASH_ATTENTION` backend (PyTorch FA2) in 2048-query chunks (`llama31_8b_tp8_combined.py:198-215`) | **custom `gpu_paged_attention`** split-K page32, 16 splits (`:216-229`) | HF rowwise AR ×2/layer + logits AG | TP8 only. **Slow**: 85.6 ms/step at 64K B1 (`tp8_external_baselines/shadowkv/decode_summary.csv`), against 30.7 ms for Flash-SDPA decode at TP8 (`tp8_dense_flash_rerun/SUMMARY.md`). The Flash-SDPA decode code is **not committed** in this repo (`tp8_dense_flash_rerun/README.md` "Working directory: /workspace/BasisServe-CALS-opt") |
| `LlamaTP4KCacheAttention(mode='dense')` | `flash_attn_func` FA2 full prompt per 2048-query chunk (`llama_tp4_k_offload.py:61-67`) | `flash_attn_with_kvcache(num_splits=16)` (`:68-70`) | HF rowwise AR (o_proj `:85`, MLP) | TP4-specific. Measured 64K B1/4/8 on the base model (27.2/28.6/34.9 ms). Needs the `flash_attn` package (absent here) |
| vLLM dense `bench_llama_vllm.py --tp 4 --arm dense` | vLLM FLASH_ATTN backend (`vllm_flash_attn` FA2 on SM89, first in priority, `vllm/platforms/cuda.py` non-MLA branch), chunked prefill 8192 | same, CUDA graphs optional | vLLM NCCL AR (custom AR is not used on PCIe >2 GPUs [I]) | TP4-capable. ShadowKV/LRQK are not available in vLLM |
| HF native SDPA + DynamicCache (`smoke_tp4_joint_model.py --arm dense`, `tp_plan="auto"`) | SDPA | SDPA | HF TP | smoke only |

Recommendation [I]: for a matched four-arm torchrun benchmark, use the HF-TP custom runner family and give dense **FA2 `flash_attn_with_kvcache`** decode. The ShadowKV arm already uses it, and `llama_tp4_k_offload.py` shows how. Building flash_attn 2.8.3.post1 follows `tp8_external_baselines/shadowkv/README.md` §Dependency Preparation; the fallback is `vllm.vllm_flash_attn`. The custom paged-kernel dense arm should not be used as the "optimized dense" baseline.

---

## 6. Prior results at 64K (strict reusability)

| Result | Hardware / TP | Model | Context / batch | Arms | Matched-reusable for L40S TP4 64K? |
|---|---|---|---|---|---|
| `results/system_benchmarks/tp8_external_baselines/shadowkv/` (request + steady, 108/108) | 8x L40S (EPYC 9354), TP8, torch cu130, TF 5.17 | Instruct | 64K & 130048; B1/4/8; 3 cohorts | dense (custom paged), basis_joint V96 full-scan, shadowkv | **No** (TP8, different host). Reference only. 64K B8 request: dense 56.9 s, basis 43.4 s, shadowkv 89.7 s (SVD 29.9 s). Steady: 87.4 / 24.8 / 33.4 ms |
| `llama31_8b_tp8_full_scan/` (decode grid 132/144) | same TP8 | Instruct | 64K B1/4/8/16 (dense B16 prefill OOM) | dense, als_full, basis_joint | No |
| `tp8_joint_batch_sweep/` | same TP8 | Llama & Qwen | 64K B1..16 | dense, basis | No |
| `tp8_dense_flash_rerun/` | same TP8 | Instruct | 64K B1/4/8 (B16 OOM) | dense Flash SDPA | No (useful dense-kernel reference) |
| `quest_tp8/paired/` | same TP8 | Instruct | 64K B1 | Quest vs Basis | No |
| `llama31_8b_tp8_combined/` | same TP8 | Instruct | 64K B1..16 | two-stage basis (historical) | No |
| `results/tp8_joint_l40s_20260923/` | 8x L40S vLLM TP8 | Instruct | 64K B8 | dense, als_full, basis_joint (vLLM two-stage) | No |
| `results/system_benchmarks/l40s/SUMMARY.md` §8 | 4x L40S "lovelace" (bus 28/45/a8/c5, not this node's 0A/0D/98/99), TP4, GPU0 externally occupied | **base** model | 64K B1/4/8 (B16 OOM all modes); 256 generated tokens with 32 discarded | dense (FA2), c1, sparse_local, offload (variable V ranks avg 96) | **No** (different model, different factor allocation, different protocol, different host). Closest TP4 reference: 64K B8 dense prefill peak 33.8 GiB, decode-resident 21.1 GiB, TTFT 50.4 s, offload decode 46.8 ms |
| `results/rebuttal/v2q8_palu_g4_tp4/` | 4x L40S node 1 (bottlenecked), vLLM TP4 | Instruct | **4K** only, B32/64/128 | dense, basis_v96/v64 (no routing), PaLU | No (4K) |
| `tp1_offline/`, `tp1_lrqk_local/` | 1x L40S, TP1, B1 | Instruct | 64K B1 | dense, basis, shadowkv, lrqk (CPU offload 71.0 s request, 401 ms/token) | No. Only references for single-GPU LRQK/ShadowKV behaviour |

No existing result is a matched point for 4x L40S TP4 at 64K with Instruct across these four arms.

---

## 7. Capacity arithmetic (per rank, TP4, prompt 65,536 + 128 generated; L40S = 44.99 GiB physical, about 43.5–44 GiB allocatable after context and NCCL [I])

### 7a. Static components
- **Weights per rank** [V formula; cross-checked against the TP8 measurement]:
  - Sharded layers: 6,979M/4 = 1,745M params. Embedding replicated: 525.3M. lm_head colwise: 131.3M. Total 2,401.9M params, **4.47 GiB** BF16.
  - Cross-check: the same formula at TP8 gives 2.73 GiB, and the measured TP8 dense 64K B1 resident is 3.738 GiB minus 1.002 GiB KV = 2.74 ✓.
  - BasisKV: drop o_proj (−134M), add replicated decoder 32×[3072×4096] (+403M), fold V (−8M). That gives **4.96 GiB**.
- **Dense KV per rank** (your formula): 32 × 8 × 128 × 2 × 2 B = 131,072 B/token, divided by 4 ranks = 32 KiB/token/rank. At 65,664 tokens that is **2.004 GiB per request**: B4 8.02, B8 16.03, B16 32.06 GiB.
- **Prefill activations** [V measured slope]: the TP8 dense 64K prefill peak rises 2.571 GiB per request, of which 1.002 is KV. The rest, about **1.57 GiB per request**, is three replicated `[T,4096]` BF16 tensors: the embedding output kept by the HF loop, the residual, and the layernorm output. This term is TP-independent.

### 7b. Per-request GPU state (decode-resident) and CPU offload, per rank [V from the code's layouts]
| Arm | GPU state per request | CPU (pinned) per request, per rank / all 4 ranks |
|---|---|---|
| dense | K+V 2.004 GiB | 0 |
| basiskv (runtime A layout ×2 heads) | V96 0.751 + R16 0.125 + slot K 0.031 + slot_lookup 0.016 + ids/router ≈0.006 = **0.93 GiB** | exact K 1.002 GiB / **4.0 GiB** |
| shadowkv | U 0.625 (replicated) + landmarks 0.124 + idx 0.004 + K/V buffers (cap 2624) 0.080 + SV 0.002 = **0.84 GiB** | V 1.0 GiB / **4.0 GiB** |
| lrqk (upstream layout, 8 local Q heads, cap = 1.5×T = 98,304) | A_K 1.50 + Kgpu/Vgpu 0.258 + B 0.002 = **1.76 GiB** | K‖V 3.0 GiB / **12.0 GiB** (+0.26 GiB Kgpu/Vgpu temp copies, OffloadPrefill variant) |

Cross-checks:
- ShadowKV TP8 measured decode-ready slope 0.732 GiB per request, predicted 0.735 ✓.
- Basis TP8 resident slope 0.477, predicted 0.464 ✓.
- LRQK single-layer state at TP1 64K: measured 218,956,288 B in `l40s/SUMMARY.md` §4, predicted 218,955,776 B with the 1.5× growth ✓.

### 7c. Estimated peaks [I] (whole-batch one-shot prefill, as the TP8 runner does, `bench_llama31_8b_tp8_request.py:232`)
| Arm | B4 prefill peak / resident | B8 prefill peak / resident | B16 prefill peak / resident | Flag |
|---|---|---|---|---|
| dense | 18.8 / 12.5 | 33.1 / 20.5 | **≈61.7** / 36.6 | B16 **OOM** in prefill. Consistent with measured TP4 base-model peaks of 19.4 and 33.8 GiB at B4/B8 and OOM at B16 |
| basiskv | 15.6 / 8.7 | 25.9 / 12.4 | **≈46** / 19.9 | B16 **OOM likely** in prefill. TP8 needed 36.5 GiB at B16 |
| shadowkv | ≈15–16 / 7.8 | ≈25–26 / 11.2 | **≈44–46** / 17.9 | B16 **OOM likely or borderline**. Prefill temporaries scale with 2 local heads; owner SVD temporaries about 1.3 GiB |
| lrqk | ≈22 / 11.5 | **≈38–41** / 18.6 | **≈75** / 32.6 | B8 **at risk** (whole-batch FP32 ALS ≈1.1 GiB/request/layer transient); B16 **OOM** |

With sequential per-request prefill into batched caches, peak ≈ resident + ~1.6 GiB + ~1 GiB of temporaries. Then B16 fits for every arm [I]: dense ≈38.5, LRQK ≈35.5, basis ≈22.5, shadow ≈21.5 GiB. Dense and LRQK would be tight. None of the TP8 runners support per-request prefill (`assert start == 0 or tokens == 1`, `llama31_8b_tp8_combined.py:178,557`).
- vLLM at `gpu_memory_utilization=0.8` (`bench_llama_vllm.py:127`): the dense B16 64K pool needs about 32 GiB of KV plus 4.5 GiB of weights, more than 36 GiB, so it would fail the runner's `gpu_cache_capacity` check (`:183-192`) [I].
- Host RAM (2 TiB) is not a constraint. LRQK B16 pins about 192 GiB in total.

**Prediction [I]:** with the existing whole-batch-prefill design, the largest batch every arm completes is **B8 if LRQK's prefill fit is done per request, otherwise B4**. B16 needs per-request prefill in all four arms.

### 7d. Communication volume per rank (logical bytes) [V shapes, I totals]
- Decode step, dense, shadowkv and lrqk: 64 AR, each B × 4096 × 2 B = 8 KiB × B (64 KiB at B8, 128 KiB at B16), plus one logits AG with B × 32,064 × 2 B input (501 KiB at B8).
- Decode step, basiskv: 32 AG (input B × 768 × 2 B = 1.5 KiB × B, output 6 KiB × B) + 32 AR (8 KiB × B) + logits AG.
- Prefill, per request: about 1 GiB of AR payload per layer, 32 GiB per request per rank (o_proj per 2048-query chunk plus MLP per 1024 tokens).
- ShadowKV prefill extra, per layer per request: 3 × 32 MiB of K into the owner, a 20 MiB U broadcast, and 3 × 80 KiB SV.
- [I] Prefill is likely link-bound on PCIe with cross-socket SYS paths. Measure first.

---

## 8. Missing adapters and modifications, in dependency order

| # | Item | Depends on | Effort |
|---|---|---|---|
| 1 | **Environment and assets**: build flash_attn 2.8.3.post1 for torch 2.13/cu129/py3.11 (or substitute `vllm.vllm_flash_attn` and document it); clone ShadowKV at `e51904c` + CUTLASS (record its commit) and build `shadowkv*.so`; clone LRQK at `caf1629` and build `cpp_kernel`; fetch the V96 + matching B16R16 router (decide between als6 + TP8 router and retrievalmix + its `router_b16r16`; check hashes against TP8 `a61d334e…`/`3c1523ed…`); materialise the transformed factors (`prepare_llama31_8b_tp8_combined_factors.py`); fetch or rebuild the `p65536_c{0,1,2}` prompt cohorts (`evaluation/prepare_llama31_tp8_benchmark_prompts.py` needs `longbench-v2/tokens.safetensors`; HF raw revision `26b66c9…` reportedly contains them); run `check_interconnect.sh` once the GPUs are idle | — | small–medium |
| 2 | **Runner generalisation** (`bench_llama31_8b_tp8_request.py`, `run_…_grid.py`, `tp8_request_timing.py`, summarizer): world size from env, `DistributedConfig(tp_size=W)`, B16 allowed, rank loops, owner `layer%W`, `svd_calls = B·32/W`, CPU binding (2 GPUs per NUMA node 0/2 here), NCCL env recorded (`NCCL_PROTO`, P2P) | 1 | small |
| 3 | **Dense TP4**: generalise `DenseTP8Attention` to 2 KV / 8 Q heads and switch decode to FA2 `flash_attn_with_kvcache` (as in `llama_tp4_k_offload.py:68-70`) | 2 | small |
| 4 | **BasisKV TP4**: port runtime A (§4b), plus a TP4 smoke (token parity, finite logits, slot hit rate) | 1, 2 | small–medium |
| 5 | **ShadowKV TP4**: generalise `shadowkv_tp8*.py` (§2d) and re-run the 3 distributed smoke tests at nproc 4 | 1, 2 | small–medium |
| 6 | **LRQK TP4 adapter** (new, §3): upstream factory per layer on local heads, per-head RNG slicing, convergence handling (`tol=1e-8` or scalar AR), per-request ALS fit for memory, CPU gather threads, persistent-state accounting (A_K, Kgpu/Vgpu, KVcpu) for the memory report, TP1 parity validation | 1, 2 | medium–large |
| 7 | **Communication instrumentation**: record type, shape, bytes and CUDA-event latency for HF rowwise `dist.all_reduce` (patchable, `tensor_parallel.py:323`), the lm_head DTensor gather (missed by `collective_audit.py`, which patches only `dist.*` and assumes ×4 output), the custom uniform AG (already timed as `output_allgather`, `llama31_8b_tp8_combined.py:417-419`) and the ShadowKV P2P and broadcast (already timed as phases) | 2–6 | small–medium |
| 8 | **Metrics glue**: per-step events in request mode or use steady mode for decode mean/P95, per-rank peak and decode-resident GPU memory, NVML, pinned host bytes for each arm (extend `persistent_state`, `bench_llama31_8b_tp8_request.py:52-65`, for LRQK) | 2–6 | small |
| 9 | **B16 enablement (optional)**: sequential per-request prefill into batched caches, needing row offsets in dense/basis caches, ShadowKV `prefilled_batch` offsets for U/SV writes, and LRQK per-request factories or a merge | 3–6 | medium (dense/basis/shadow) to large (LRQK) |

## Open uncertainties
- Whether HF-TP internals in TF 5.22.2 (`DistributedConfig`, local-tensor inference path) behave as in the 5.17-validated TP8 runs. A TP4 smoke is required.
- Which V96/router pair to use, and whether the TP8 `ours_b16r16` bank equals the HF `k_routing/…als40_pcg100` bank.
- Availability of the frozen 64K prompt cohorts on HF.
- The memory peaks in §7c are estimates from TP8 slopes and code layouts; the allocator, fragmentation and NCCL buffers can shift them by about 1–2 GiB.
- This node's AR latency and bandwidth (cross-socket SYS) are unmeasured.
- The ShadowKV build needs an unpinned CUTLASS.
- LRQK at TP4 changes RNG draws and convergence semantics unless it is explicitly matched to TP1.
