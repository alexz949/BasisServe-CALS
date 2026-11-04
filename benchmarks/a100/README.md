# Paper system benchmarks on 8 x A100-SXM4-80GB (NVSwitch)

The paper's system numbers were measured on L40S (SM89; TP8 over PCIe without NVLink). This directory ports the
archived runtimes to this node: SM80, all eight GPUs on NVSwitch (`NV12` between every pair), conda `basis`
(PyTorch 2.13.0+cu130, vLLM 0.29.0). Archives are read, never modified; every source change is an asserted exact
replacement recorded with before/after sha256.

## Build toolchain

`setup_cuda_home.sh` installs one consistent CUDA 13.0 compiler set (nvcc/crt/nvvm 13.0.88, CCCL 13.0.85, runtime
13.0.96) into `~/cuda-13.0-pip` and exposes it as `CUDA_HOME=~/cuda-13.0`. Needed because the node has no system CUDA
toolkit, and the `basis` env's pip toolchain mixes nvcc 13.4 with 13.0 runtime headers and CCCL 13.3, which CCCL
rejects at compile time. Build environment:

```bash
CUDA_HOME=$HOME/cuda-13.0 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 TORCH_CUDA_ARCH_LIST=8.0 \
  PATH=$HOME/miniforge3/envs/basis/bin:$PATH   # system gcc-12 lacks cc1plus; ninja lives in the env
```

## Source changes

| Change | Where | Reason |
|---|---|---|
| `cub::Max()` -> `::cuda::maximum<>{}` (+ `<cuda/functional>`) | `conditional_router_page32.cu`, `fused_decode_postprocess.cu` (current tree: 3 sites; Table-3 archive: 1 compiled site; Figure-4 bundle: 4 sites) | CUDA 13 CCCL removed `cub::Max`; same `a < b ? b : a` functor |
| Router geometry `w16u1` -> `w8u1` | Table 3 (`audit_tp1_request.py`) | A100 tuning, see below |
| L40S host paths -> local paths | Table 3 model snapshot and router bank | same model revision and archived bank |

## Router geometry on A100

`bench_tp1_router_config.py --phase micro` from the Table-3 archive (Llama-3.1-8B shapes, B16R16, Page32, one layer,
synthetic, warmed CUDA events). All 90 bitwise checks pass (scores and selected pages identical across geometries),
and 52 cases of `validate_register_router.py` match the independent BF16 reference.

| Context | w8u1 (8 warps) | w16u1 (16 warps, L40S choice) | w8u2 (8 warps, unroll 2) |
|---|---:|---:|---:|
| 16K | 0.098 ms | 0.107 ms | 0.087 ms |
| 64K | 0.309 ms | 0.317 ms | 0.308 ms |
| 128K | 0.623 ms | 0.644 ms | 0.621 ms |

On L40S w16u1 lowered full-model decode latency by 1.1-1.5%; on A100 it is 2.5-9% slower per router call, so the
A100 runs use w8u1.

## Table 3: TP1/B1 sparse vs dense decoding (`tp1_paper_scaling.py`)

Llama-3.1-8B-Instruct (revision `0e9e39f2...`), Dense V128, B16R16 Page32 full scan, GPU-local K/V. Same contexts
(16K-128K), cohorts 0/1/2, prompts, 16 warmup + 128 measured decode steps and 32 attention-profile steps as the L40S
formal run; archive inputs pinned to HF revision `63a35906...` with sha256 checks. Outputs:
`results/system_benchmarks/a100/tp1_paper_scaling/{smoke,formal}`.

```bash
/home/sesterce/miniforge3/envs/basis/bin/python benchmarks/a100/tp1_paper_scaling.py --phase smoke
/home/sesterce/miniforge3/envs/basis/bin/python benchmarks/a100/tp1_paper_scaling.py --phase formal
```

Smoke (2026-11-04, 24K, cohort 0, 4 measured steps): both arms complete; source check passed.

## Figure 3: TP8 vLLM serving, Qwen3-32B dense vs C1 V64 (`vllm_32b_tp8.py`)

Runs the archived runtime of the paper's V64 point (`system_benchmarks/vllm_32b_tp8_sm89_4k_budget8k`, 1.307x at
B256 on L40S): a git worktree of base commit `bab891d3` under `~/basiskv-runtimes/` with the archived `source.tar`
overlaid. Same settings: 4096 prompt + 128 output tokens, batch 1-256, 8192-token scheduler budget, FULL_DECODE_ONLY
CUDA Graphs, 1 warmup + 3 repeats, profiles at B1/B32/B256. The current `prepared_c1_output.py` cannot load the
paper's factor banks (factor format label renamed in `ff73e37`), hence the archived runtime.

```bash
python benchmarks/a100/vllm_32b_tp8.py --phase smoke  --variant v64            # then --phase formal
python benchmarks/a100/vllm_32b_tp8.py --phase smoke  --variant v64_nccl_symm
torchrun --nproc-per-node 8 benchmarks/a100/bench_allgather.py --output <json>   # collective microbenchmark
```

A100 issues found and handled:
- FlashInfer JIT-compiles its sampling kernels at engine start and needs cuRAND headers (`setup_cuda_home.sh`).
- The C1 decoder GEMM reads the gathered feature-major arena through a transposed view; A100 cuBLAS (and cuBLASLt)
  picks align-2 kernels for it at 2/4/16 tokens (93-120 us instead of ~35 us per call). Variant `v64` copies the
  arena token-major at <= 16 tokens; the repository `prepared_c1_output.py` does the same on SM80 only.
- vLLM snaps its 3D-attention threshold (128) to the nearest CUDA Graph capture size, so a smoke that captures only
  B1/B2 runs B2 decode attention unsplit (13x slower). Smokes therefore include B128.

Formal, variant `v64` (median of 3; speedup = dense / C1 end-to-end seconds; no preemption in either arm):

| Batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A100 NVSwitch | 0.850 | 0.890 | 0.893 | 0.908 | 0.929 | 0.938 | 0.949 | 0.964 | 1.046 |
| L40S PCIe (paper record) | 0.975 | 1.014 | 1.056 | 1.099 | 1.137 | 1.168 | 1.186 | 1.229 | 1.307 |

Dense is 3.3x faster on A100 than on L40S at B256 (65.9 vs 215.3 s): on NVSwitch vLLM uses its one-shot custom
all-reduce (6.6 us per layer at B1 vs ~37 us PyNccl on PCIe), so compressing the attention-output collective saves
little. The archived summarizer's profile section assumes NCCL all-reduce kernel names and fails on A100 (custom
all-reduce kernels); the timing summary (`summary.csv`) is complete.

Decode collective, `bench_allgather.py` (8 ranks, CUDA Graph replay, us per call, max over ranks):

| Tokens | 1 | 4 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|---:|
| NCCL ring all-gather (archived path) | 17.1 | 22.9 | 20.2 | 21.6 | 34.4 |
| NCCL symmetric-window all-gather | 8.3 | 9.2 | 9.5 | 14.8 | 25.6 |
| vLLM custom all-gather | 15.5 | 15.8 | 14.3 | 16.4 | 25.6 |
| vLLM custom all-reduce [T x 5120] (dense) | 7.8 | 7.3 | 12.1 | 19.2 | 37.0 |

The repository `prepared_c1_output.py` now gathers decode coordinates in place on one NCCL symmetric-window arena when
the TP group is fully NVLink-connected (vLLM custom all-reduce enabled); prefill and PCIe-only groups keep the NCCL
plan. In the model the symmetric kernel (`ncclSymkDevKernel_AllGather_LL`) takes 9.0 us per decode layer at B1 versus
15.7 us for the ring, saving 0.43 ms per step, matching the measured TPOT change (smoke, one run per batch):

| Batch | 1 | 2 | 4 | 16 | 128 |
|---|---:|---:|---:|---:|---:|
| `v64` (NCCL ring) | 0.841 | 0.887 | 0.892 | 0.930 | 0.960 |
| vLLM custom all-gather (exploratory) | 0.888 | 0.919 | 0.929 | 0.946 | 0.964 |
| `v64_nccl_symm` | 0.868 | 0.917 | 0.930 | 0.949 | 0.962 |

Per decode step at B1 (profiled kernel time, ms), V64 with the symmetric gather minus dense: GEMM +1.43, attention
-0.41, all-reduce -0.61, all-gather +0.69, other +0.08. Communication now breaks even on NVLink; the remaining gap is
the replicated [4096 x 5120] decoder GEMM (4x the dense per-rank O projection). A source-partial decoder
([512 x 5120] per rank) followed by the dense-style custom all-reduce would remove it on NVLink; not implemented.

## Figure 4: TP8 joint kernels

The Qwen3-8B (`qwen3_8b_tp8_joint`) and Llama (`tp8_joint_batch_sweep`) source bundles carry identical CUDA sources.
The Qwen3-8B bundle is an overlay on a repository checkout; `evaluation/chunked_prefill_mlp.py` comes from the Llama
bundle (byte-identical to the current tree). With the four `cub::Max` replacements, all five TP8 extensions build
for SM80 (attention+router V96/G4/Page32/B16R16, ALS-full attention, register router w8, decode postprocess, feature
ragged AllGather), and the bundle's 25 focused tests pass on A100
(`tests/test_tp8_joint_batch_sweep.py tests/test_tp_dense_flash_decode.py tests/test_joint_prefix_append.py`).
