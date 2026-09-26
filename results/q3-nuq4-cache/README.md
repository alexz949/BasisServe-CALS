# Qwen3 Packed NUQ4 Backend Preparation

Model: Qwen3-8B-Base. Checkpoints: the frozen adaptive C1 R64/R96 used in
`../q3-kv4-fp8/formal` and `../q3-a8-dec/formal`, not fixed-rank Llama factors.
Environment: `basis`, GPU 2 (L40S) for the short kernel tests. No SHA256.

## Verified

- Actual packed storage: two NUQ4 codes per byte, bitmaps, per-token sparse
  offsets, dynamic V range/offset, and BF16 outlier arenas per physical page.
- Static K quantization before RoPE; dynamic V quantization across the entire
  active V vector, not independently per TP shard. Unsorted LUT tie ordering
  follows the official KVQuant implementation.
- Full-context tiled attention reads the packed pages directly, restores K
  before applying RoPE, and never materializes a full BF16 cache in its forward.
  A separate whole-row gather exists only for test references.
- Loading the archived adaptive factors, codebooks and A8 scales uses structure
  checks only. Selected decoder weights/scales match the previous TP8 boundary
  fixtures exactly for both ranks at layers 0/18/35.
- Final test run: **34 passed in 8.71 s**. This includes packed reconstruction
  versus official KVQuant, incremental append, physical-page reuse, inactive
  slots, exception overflow detection, noncontiguous block tables, chunked
  prefill, decode, 4096-token prefill and CUDA Graph replay.

The first two attempts exposed a column-shaped saved LUT and unsupported Triton
argument unpacking. Both were corrected; their original failures remain in
`tests.log`, followed by the passing runs. Tests use exact BF16 equality for
cache reconstruction, and `atol=0.008, rtol=0.015` versus PyTorch SDPA for
attention. These are kernel checks, not model PPL or TP8 quality evaluation.

## Capacity and Limitations

Each K/V page independently reserves eight BF16 exceptions per token on
average across its 16 tokens. Exceptions are not truncated. Exhaustion sets
a sticky failure flag; the result must be rejected outside graph capture.
Every slot is appended once between page resets, and writing offset zero
resets that physical page. Positions currently start at zero; no prefix cache,
sliding window, or arbitrary position remapping is supported.

Current storage includes all reserved page metadata and exception capacity:

| Local V width | K + V bytes/token/layer/rank | BF16 K + V | Storage ratio |
|---:|---:|---:|---:|
| 64 | 178 | 384 | 2.16x |
| 96 | 198 | 448 | 2.26x |

These are layout calculations, not measured GPU memory. They exclude static
codebooks/bounds, model weights and temporary workspaces. Adaptive checkpoints
use different widths in different layers. Calling this a 4x total cache-memory
reduction would be incorrect. The first layout also reserves unused K scale
fields; it is not a memory-optimal format.

## Still To Integrate

The subsequent vLLM integration is now smoke-tested end to end; see
[serving integration](../q3-nuq4-vllm/README.md). The byte-accurate allocation,
slot lifecycle, global V statistics collective, A8/W8 boundary and overflow
reporting are connected. Split-K decode has been added. The expanded local
suite passes **41 tests**, and the actual TP8 boundary passes 36 changing-input
cases under CUDA Graph replay. These are not formal speed results.

The table above documents the initial 8-exception prototype layout. The first
full model smoke exhausted that arena in some layers and was rejected. Serving
now uses a uniform reserve of **16 exceptions/token**, without changing the
quantile rule or dropping exceptions. Its K+V sizes are 210/230 bytes per token
at local V64/V96. This larger capacity must be included in memory comparisons.

Intended benchmark: TP8, 4096-token prompt, 128-token output, scheduler budget
8192, batches 1/2/4/8/16/32/64/128/256. Encoder BF16; full attention without
sparse routing; frozen NUQ4 + outlier K/V; latent A8 transport and decoder W8A8.
Formal grid requires separate launch confirmation.

## Final Test Command

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q tests/test_nuq4_cache.py tests/test_nuq4_attention.py tests/test_qwen3_nuq4_artifacts.py >> results/q3-nuq4-cache/tests.log 2>&1
```

Nothing from this preparation has been committed or uploaded.
