# TP1 Efficiency Figures

Three independent comparisons on Llama-3.1-8B-Instruct BF16, TP1, one L40S:

1. GPU-local sparse computation: attention-block and full-model steady-decode speedups.
2. 64K fixed-active-batch throughput: Dense-local, Dense K-offload, BasisKV K-offload.
3. Complete 128-output-token requests: Dense, BasisKV and ShadowKV.

See the [results summary](RESULTS_SUMMARY.md), [captions](captions.md), and PDFs:
[Figure 1](fig1_sparse_scaling.pdf), [Figure 2](fig2_offload_throughput.pdf),
[Figure 3](fig3_request_latency.pdf).

All BasisKV paths use full-scan routing and full V128. No two-stage results,
MLP changes, quality evaluations, or TP8 experimental kernels are used here.
The longest GPU-local prompt is **130,048**, not 131,072 tokens.
Prompt cohorts are the existing calibration-window prefixes (rows 0/1/2).
This is a timing study, not a held-out or quality-equivalence evaluation.

## Sources

`prepare_data.py` verifies the raw three-cohort/process records before exporting
CSV and point-level `provenance.json`. Historical values are checked at their
published rounding precision. A partial new sweep is never filled with old or
interpolated values, and `fig1_sparse_scaling.py` requires all seven points.
`raw_plotting_values.csv` includes the individual repeat/cohort inputs to every
aggregate, with exact source file and metric field. New/old overlap differences
are saved in `fig1_archive_comparison.csv` after all three cohorts complete.

Figure 1 full-model timing is the CUDA-event latency of the model forward
(including logits), followed by greedy feedback outside the event interval.
Figure 2 throughput instead includes the measured loop's argmax, finite checks,
synchronization and bookkeeping. These distinct archived protocols are preserved,
not silently merged into one latency definition.

The new scaling sweep runs the archived formal full-scan TP1 implementation,
restored byte-for-byte into `results/system_benchmarks/tp1_paper_scaling/runtime/`.
This is the existing formal GPU-local path, not the dirty TP8 optimization tree.
The original source archive, copied source list, launch command, source commit,
timestamps and per-trial logs are recorded. Git commit alone does not identify
the historical dirty runtime; its source archive is authoritative. No SHA256
calculation or validation is performed.

Figure 2 reuses the original **same-source 64K grid**, which has a successful
end-of-run source-byte check. The later chunked-RoPE supplement only ran
128K/B2 and is not included. Figure 3 reuses complete archived requests.
Raw-data HF revisions:

- [GPU-local scaling history, request results, frozen source and factors](https://huggingface.co/alexz949/BasisServe-CALS/tree/86c03092c663dc6654584143130b5e5abf2fedaa/system_benchmarks/tp1_offline).
- [LRQK GPU-local appendix records](https://huggingface.co/alexz949/BasisServe-CALS/tree/86c03092c663dc6654584143130b5e5abf2fedaa/system_benchmarks/tp1_lrqk_local).
- [Capacity/offload results](https://huggingface.co/alexz949/BasisServe-CALS/tree/0d0b7a569eb3adcfbf5d63d78a8beb767980fcad/system_benchmarks/tp1_capacity).

## Publication Files

GitHub includes this figure directory, the scaling runner and figure tests, and
the [new scaling records](../../results/system_benchmarks/tp1_paper_scaling/):
42 formal trials, two smoke trials, JSON, logs, environment and source manifests.
No new HF upload is needed for these small records. The duplicate `runtime/`
source tree, binary `prompts.safetensors`, and generated caches are excluded.
The runner restores runtime sources and prompts from the original offline
archive linked above; `runtime_manifest.json` records the exact source members.

The three plotting scripts work directly from the included CSV files. To
re-export tables from raw data, restore the three HF directories above under
`results/system_benchmarks/`, preserving their names and structure. The factor
byte check in `prepare_data.py` also requires the original factor bank at
`/workspace/runs/l31-router-source/v128-router`, whose files are archived in
`tp1_offline/freeze/factors.tar.gz`. Benchmark reruns additionally require the
local model and factor paths recorded by the frozen drivers. Model weights are
not included in this publication.

## Commands

Repository working directory: `/workspace/BasisServe-CALS`. Environment: `basis`.
GPU commands use `CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9`.
The user explicitly approved direct execution because Slurm has no working
controller configuration on this machine. Every trial saves `run.log`.

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp1_paper_scaling.py --phase smoke --output-root results/system_benchmarks/tp1_paper_scaling
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp1_paper_scaling.py --phase formal --contexts 16384 24576 32768 49152 65536 98304 130048 --cohorts 0 1 2 --output-root results/system_benchmarks/tp1_paper_scaling
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python paper_figures/tp1/prepare_data.py
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python paper_figures/tp1/fig1_sparse_scaling.py
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python paper_figures/tp1/fig2_offload_throughput.py
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python paper_figures/tp1/fig3_request_latency.py
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python -m pytest -q tests/test_tp1_paper_figures.py
```

PDFs are vector, exactly 3.35 by 2.30 inches, with 8 pt axes and 7 pt ticks/legends.
PNGs are 300 dpi. `captions.md` defines all timing boundaries and caveats.
OOM crosses have no measured throughput coordinate; their vertical positions
are for readability. They are never stored as zero throughput in the data.
ShadowKV construction profiles are separate and non-additive. LRQK is confined
to `lrqk_appendix.md` and its CSV, not a main-figure series.
