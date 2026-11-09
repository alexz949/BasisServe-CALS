# Table 3 matched Key-offload benchmark and state-level memory accounting

Rebuttal supplement to the paper's Table 3 (TP1/B1 sparse decoding). It reruns the archived
Table 3 runtime on one L40S and adds a matched arm that keeps the exact historical Keys in
pinned host memory, with phase-level GPU memory measurements and a tensor-level byte inventory.

Results: [arms/formal/SUMMARY.md](arms/formal/SUMMARY.md) (four arms) and
[pages/formal/SUMMARY.md](pages/formal/SUMMARY.md) (Page32 versus Page4). Each summary has
`runs.csv`, `summary.csv` and `inventory.csv` beside it.

## Setup

- Llama-3.1-8B-Instruct BF16, TP1, batch 1, 130,048-token prompts (archived cohorts 0/1/2), 16
  conditioning and 128 measured decode steps after a full prefill; prefill excluded from decode
  timing; greedy token selection outside the timed step. Attention-block components come from a
  separate 32-step profiled pass.
- Archived Table 3 configuration, verified from the frozen runtime: dense V128, B16R16 router,
  Page32 full scan, 1984 routed plus 64 recent tokens, exact K/V on GPU.
- Hardware and software differ from the paper: one NVIDIA L40S on node6 of this cluster, driver
  550.54.15, torch 2.13.0+cu126 (paper: RunPod L40S, torch 2.13.0+cu130). All arms were measured
  on the same GPU in one job; archived-runtime numbers reproduce the paper within 1%.
- Frozen runtime, factors and prompts come from the HF archive revision `86c03092` under
  `system_benchmarks/tp1_offline/`; only the model-cache and factor-directory paths are relocated
  (`runtime_manifest.json`).

## Arms

| Arm | Keys | Base code |
|---|---|---|
| `dense_resident` | GPU | archived dense attention |
| `basis_resident` | GPU | separate Base cache (archived Table 3 runtime) |
| `basis_resident_nobase` | GPU | first 16 coordinates of the rotated Value cache |
| `basis_k_offload_nobase` | mapped pinned host memory, persistent GPU Key slots | first 16 coordinates of the rotated Value cache |

The no-Base arms rotate `v_proj`/`o_proj` offline so the Base code is a view of the Value cache;
this is algebraically lossless but re-rounds the weights to BF16, so it is not bitwise identical
to the archived runtime (16 teacher-forced steps: identical argmax, max logit difference 0.67).
Offload versus its matched resident control is bitwise identical in selected pages, fetched Keys,
cache contents, attention outputs and logits (`arms/check/placement_check.json`,
`pages/check/placement_check.json`).

`_p32`/`_p4` arms use the page-size-general build of the archived register router
(`benchmarks/system/table3_page_router.py`, `table3_page_select.cu`), validated against the
archived kernel and PyTorch in `pages/router_validation.json`.

## Commands (conda env `basis213`, run from the repository root on a GPU node)

```
python benchmarks/system/run_table3_k_offload.py --phase smoke
python benchmarks/system/run_table3_k_offload.py --phase check
python benchmarks/system/run_table3_k_offload.py --phase formal
python benchmarks/system/run_table3_k_offload.py --phase smoke  --page-sizes 32 4
python benchmarks/system/run_table3_k_offload.py --phase check  --page-sizes 32 4
python benchmarks/system/run_table3_k_offload.py --phase formal --page-sizes 32 4
python benchmarks/system/summarize_table3_k_offload.py --phase results/rebuttal/table3_k_offload_128k/arms/formal
python benchmarks/system/summarize_table3_k_offload.py --phase results/rebuttal/table3_k_offload_128k/pages/formal
python benchmarks/system/validate_table3_page_router.py --runtime results/rebuttal/table3_k_offload_128k/runtime --output results/rebuttal/table3_k_offload_128k/pages/router_validation.json
python -m pytest tests/test_table3_key_slots.py
```

## Reading the numbers

- Component times are CUDA-event windows around each stage; when the GPU is idle at a launch they
  include launch latency, so they are diagnostics. The primary metric is the full-step time.
- Offloaded Keys are not eliminated: the same 7.95 GiB sits in pinned host memory, and the slot
  cache and maps add 0.26 GiB on GPU. Decode-only savings do not imply request-level capacity.
- Every step fetches the selected Keys that were not already in a slot (about 25% of 2048 per KV
  head and layer, 32 MiB per step at Page32); this is the whole offload cost.

## Raw artifacts

Per-trial `result.json`, logs, teacher-forced tensors, the relocated runtime copy and factors:
HF repository `alexz949/BasisServe-CALS`, revision `4ab0eb60be2e1ad6a53b1018cd0c7a3918f660da`,
path `system_benchmarks/table3_k_offload_128k/`. The GitHub tree keeps the summaries, CSVs,
manifests and checks only.
