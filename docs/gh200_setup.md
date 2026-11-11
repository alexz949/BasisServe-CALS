# Delta-GH environment and formal experiment setup

Status on 2026-09-11: environment, GPU check, both model downloads, and the short
Qwen and Nemotron real-model forward checks are complete. Calibration data
preparation and both models' formal dense PPL evaluations are complete. Qwen's
full Fisher/whitening, six PaLU V banks, and their full V-only PPL evaluations
are complete. Qwen C1 fitting is running. Downstream AG fitting and MCQ
evaluation remain outstanding; there is no complete main experiment table yet.

## Formal run progress

Model-specific settings are in `experiments/formal/qwen35_9b.json` and
`experiments/formal/nemotron_h_8b.json`. C1 encoder CG now uses fixed iterations,
with the actual iteration count and numerical breakdowns recorded. The fitting
artifact records this choice, so old iteration-cap-only factors cannot be reused.
O/AG remains the existing exact encoder solve with 12 sweeps.

`evaluation/formal_hybrid_quality.py` reuses the historical Qwen3-8B PPL and
MCQ functions: PPL batch 2, complete 2048-token WT2 chunks, independent 128 C4
validation windows; lm-eval 0.4.11, MCQ batch 8, maximum length 4096, zero-shot,
and no task limit. This corrects the hybrid script's previous inclusion of an
incomplete final WT2 chunk. Nemotron's historical tokenizer path adds its normal
special token, whereas calibration windows intentionally omit special tokens.

Completed job 3133095 (`lowrank`, 1 GPU, 2 CPUs, 10-minute limit) ran:

```bash
source scripts/gh200_env.sh
python -u -m evaluation.run_qwen35_hybrid smoke --model-path "$HF_HUB_CACHE/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" --data "$BASISSERVE_RUN_ROOT/cache/qwen35_9b/data" --output "$BASISSERVE_RUN_ROOT/cache/qwen35_9b/smoke"
python -u -m evaluation.qwen35_hybrid_palu fisher --model-path "$HF_HUB_CACHE/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" --data "$BASISSERVE_RUN_ROOT/cache/qwen35_9b/data" --smoke-only --output "$BASISSERVE_RUN_ROOT/cache/qwen35_9b/fisher_smoke.pt"
```

It passed capture/adapter/cache checks and a full 2048-token Fisher backward
pass (4.3446 seconds, 18.7506 GiB peak). The adapter smoke preceded the fixed-CG
change; 10 CPU solver tests subsequently passed, including the fixed-iteration
regression. Job 3133095 completed in 1:20 with exit code 0 and was scancelled.

Job 3133101 ran `bash scripts/check_formal_quality.sh` in lowrank and completed
in 2:08, exit code 0; scancel was issued. This saved pinned model manifests and
two-window PPL smoke results. Qwen smoke PPL was 8.244832, peak 26.1781 GiB;
Nemotron peak was 48.7131 GiB. These are not full-corpus quality results.

Job 3133116 ran `bash scripts/evaluate_formal_dense_ppl.sh` in lowrank, with
1 GPU, 2 CPUs and a 10-minute limit. A request to extend it to 12 minutes was
denied by Slurm; the limit remains 10 minutes. Completed Qwen dense results:
WT2 PPL **8.645838581848826**, C4 PPL **11.329460112915775**. Nemotron WT2 PPL
is **6.099006062018523**. The job timed out during Nemotron C4 at 10:02 and was
scancelled. Job 3133169 reran only that C4 stage with the same settings/output
path and a six-minute allocation; it completed in 4:21, exit code 0, and was
scancelled. Nemotron C4 PPL is **9.808029401619763**. Results are saved individually under
`/work/hdd/bido/azhang35/BasisServe-CALS/results/<model>/v_ag_ppl_mcq/dense/`.
Completed stages can be reused only after identity and stage checks.

Job 3133114 ran in ghx4 with `bash scripts/collect_formal_qwen_fisher.sh`,
4 GPUs, 8 CPUs, 128 GiB host memory, 10 minutes. Four disjoint shards each cover
64 of the 256 calibration windows. One srun step failed at launch with
`Error configuring interconnect`; shards 0, 1, and 3 completed successfully.
The batch ended FAILED after 5:23, and scancel was issued. Verified these three
shards cover 192 distinct windows and contain finite, nonzero Fisher statistics.
Subsequent launchers use local processes pinned to the GPUs allocated by Slurm,
without separate interconnect-requiring srun steps.

Job 3133172 ran `bash scripts/finish_formal_qwen_palu.sh` on two GPUs, four
CPUs, 96 GiB host memory, with an eight-minute limit. It collects missing Fisher
shard 2 and full 256-window whitening independently, then builds all six Qwen
PaLU banks using canonical Fisher rounding. It completed in 5:57, exit code 0,
and was scancelled. Verified all 256 distinct Fisher windows, finite nonzero
statistics, six banks' hashes/dimensions, and their actual cache budgets.
Banks are under `cache/qwen35_9b/banks/` in the experiment root.

| Method | Target mean | Actual mean | Actual V retention |
| --- | ---: | ---: | ---: |
| PaLU M-LRD | 192 | 188 | 73.4375% |
| PaLU M-LRD | 160 | 164 | 64.0625% |
| PaLU M-LRD | 128 | 132 | 51.5625% |
| PaLU G-LRD2 | 192 | 194 | 75.78125% |
| PaLU G-LRD2 | 160 | 160 | 62.5% |
| PaLU G-LRD2 | 128 | 126 | 49.21875% |

These are the user-allowed canonical rounding deviations. C1 allocation must
still meet its exact budgets. Job 3133196 (`bash scripts/check_formal_palu.sh`,
lowrank) passed six two-window compressed PPL checks in 3:05. Job 3133208
(`bash scripts/evaluate_formal_palu_ppl.sh`, lowrank, three GPUs, six CPUs,
eight-minute limit) completed all twelve full PPL stages in 4:15, exit 0.
Both jobs were scancelled after completion. Results under
`results/qwen35_9b/v_only/` are diagnostics before trajectory-specific AG:

| Method | Target mean | WT2 PPL | C4 PPL |
| --- | ---: | ---: | ---: |
| PaLU M-LRD | 192 | 8.644485 | 11.548475 |
| PaLU M-LRD | 160 | 8.844768 | 11.672281 |
| PaLU M-LRD | 128 | 9.029605 | 11.853149 |
| PaLU G-LRD2 | 192 | 8.650833 | 11.412210 |
| PaLU G-LRD2 | 160 | 8.696246 | 11.522260 |
| PaLU G-LRD2 | 128 | 8.829095 | 11.672703 |

Job 3133194 (`bash scripts/collect_formal_qwen_inputs.sh`, lowrank) captured
all 256 fitting windows in 5:13, exit 0, and was scancelled. The roughly 104 GiB
capture resides on HDD work storage. Job 3133210
(`bash scripts/check_formal_solvers.sh`, lowrank) completed in 4:19, exit 0,
and was scancelled. Full Qwen layer-3/rank-96 benchmark: 207.12 seconds to
load capture tensors, 0.531 seconds per decoder normal application, and
0.507 seconds per encoder normal application. It also passed one full
Nemotron Fisher backward window in 5.374 seconds with 49.319 GiB peak memory.

Job 3133228 fitted Qwen layer 3, rank 96 with
`bash scripts/fit_formal_qwen_first.sh` in lowrank, one GPU, two CPUs,
30-minute limit. It uses all 256 fitting windows, 12 sweeps, fixed encoder
CG16, and a separate decoder cap of 200 iterations. Decoder cap hits are
reported in the log and must not be described as converged solves. The job
completed in 28:17, exit 0, and was scancelled with queue removal confirmed.
Exported relative fit error was 0.0053867141, selected sweep 12. Verified
12 encoder updates of exactly CG16 and 13 decoder updates; 12 decoder
updates hit the 200-iteration cap. Actual fitting time was 1473.69 seconds.

The shared PaLU entry also accepts
`--family nemotron` for its attention-only Fisher and whitening paths; that
extension passed the real-model backward check above. Job 3133232 ran
`bash scripts/collect_formal_nemotron_palu.sh` in lowrank, four GPUs, eight
CPUs, twelve-minute limit: three Fisher shards, one whitening worker, then
six PaLU banks. It completed in 8:45, exit 0; scancel was issued and the queue
was confirmed empty for that job. Verified all 256 distinct Fisher windows,
finite nonzero statistics, and all six banks' hashes, shapes, and budgets.
Both M-LRD and G-LRD4 have actual mean ranks 96, 80, and 64, respectively,
with zero budget deviation. These banks still require formal quality evaluation.
Its preceding native runtime smoke passed: identity relative
error 0.011082 and cached-decode relative error 0.012002, with all 24 native
Mamba layers preserved and value-cache width 64. These smoke factors are
untrained coordinate factors, not formal fitted checkpoints.

`evaluation/nemotron_h_c1_v.py` captures native attention output moments and
uses the existing full-layer routed OV solver for free C1 decoders, 12 sweeps,
fixed CG16, float64 work, and bfloat16 export. Its regression test checks the
exported error against direct reconstruction on raw moments. Solver checks
use the damped quadratic consistently; the factor-selection callback uses
undamped fitting moments, with no held-out windows. The CPU regression passed.
A separate full-rank synthetic check also completed all 12 sweeps: raw fit
relative error zero, bfloat16-export relative error 4.9683e-6. Small FP64
cancellation messages near zero were within the solver's existing tolerance.
Job 3133251 ran `bash scripts/collect_formal_nemotron_c1.sh` in lowrank,
one GPU, two CPUs, 48 GiB host memory, ten-minute limit. It completed in 7:19,
exit 0, and was scancelled with queue removal confirmed. All four layers
(7, 18, 29, 40) have verified finite, symmetric 4096 x 4096 moments from
524,288 fitting rows, 32 query heads, eight KV heads, and width 128.
Stdout/stderr use `logs/nemo-capture-3133251`.

`evaluation/nemotron_h_c1_banks.py` connects the fitted E_V/D_O bank to the
existing shared terminal-KL evaluator and exact Two-Sided allocator. It uses
the requested probes: 32/64/96, 48/80/112, and 64/96/128, alpha 1, 128 profile
windows, and exact mean-rank budgets. All seven rank-bank entries, including
128, use fitted factors. Profiles record and verify model identity and factor
hashes before reuse. The quality entry accepts both Nemotron C1 and PaLU banks.
Seventeen runtime/profile regressions passed; three Nemotron bank tests also
passed, including comparison against exhaustive enumeration of all legal
schedules for known synthetic costs. These are not formal KL/quality runs.

Job 3133278 ran `bash scripts/fit_formal_nemotron_first.sh`
for layer 7/rank 32, full fitting moments, 12 sweeps, and fixed CG16, in
lowrank with one GPU, two CPUs, 48 GiB memory, and ten-minute limit. It
completed in 32 seconds, exit 0, and was scancelled. Fitting itself took
4.815 seconds; exported relative fit error was 0.0934190. All eight encoder
groups used exactly 16 CG iterations in each of 12 sweeps; factor shapes and
finite values were verified. `bash scripts/fit_formal_nemotron_bank.sh`
fits all seven ranks independently on four attention-layer GPU workers.
Its ghx4 job 3133283 remained pending with a roughly 30-minute forecast and
was cancelled before starting. Replacement 3133292 runs in ghx4-interactive
with four GPUs, eight CPUs, 96 GiB memory, and six-minute limit; this partition
charges double. Job 3133292 completed in 56 seconds, exit 0, and was
scancelled. Verified all 28 factors' shapes, finite values, 12 sweeps, and
fixed encoder CG settings. Exported fit errors decrease with increasing rank
at all four layers; full-rank BF16 export errors are 2.83e-6 to 6.51e-6.
Job 3133293 runs `bash scripts/check_formal_nemotron_banks.sh` with four GPUs,
eight CPUs, 128 GiB memory, six-minute limit in ghx4-interactive: nine bank
PPL smokes plus the separate 128-window KL teacher. All nine two-window
smokes completed (PPL 5.5809 to 5.9586). The job completed in 3:53, exit 0,
and was scancelled with queue removal confirmed. Teacher hidden states have
verified finite shape 128 x 2048 x 4096 and matching pinned model identity.
These smokes are limited diagnostics, not full PPL results. Prepared
`bash scripts/evaluate_formal_nemotron_palu_ppl.sh` for the six PaLU banks
with three independent GPU workers, preserving historical PPL batch size
and full datasets. The Nemotron PaLU PPL script has not yet been launched.

Job 3133302 ran in ghx4 for
`bash scripts/evaluate_formal_dense_mcq.sh qwen35 openbookqa`, lowrank,
four GPUs, eight CPUs, 96 GiB memory, six-minute limit. This runs the full
OpenBookQA task with global batch 8, lm-eval 0.4.11, and no sample limit.
The task reached real likelihood evaluation, but repeated DataParallel model
replication made the six-minute allocation insufficient. It was scancelled
at 3:56, with queue removal confirmed; no MCQ result was produced. Inference
DataParallel now retains model replicas across batches. Job 3133343 is pending
in ghx4 with `bash scripts/check_formal_mcq.sh`, four GPUs, eight CPUs,
96 GiB memory, fifteen minutes: validate native/repeated/short-batch outputs
for both models, then rerun the same full OpenBookQA command. Persistent
replication passed GPU validation for both models, including repeated calls
and short tail batches. Qwen relative logit error was 0.0091201 and NLL delta
0.0004506; Nemotron was 0.0109111 and 0.0017998. Repeated 64-token forward
times were 2.153 seconds and 0.540 seconds respectively, so Qwen MCQ remains
substantial. Job 3133343 completed in 9:51, exit 0, and was scancelled with
queue removal confirmed. Full Qwen Dense OpenBookQA acc_norm is **0.43** on
all 500 examples; MCQ stage elapsed time was 477.62 seconds. This is one task,
not the seven-task average. The quality writer normalizes
lm-eval output through its official JSON serialization helper before saving.

Job 3133312 was submitted to ghx4-interactive with
`bash scripts/profile_formal_nemotron_c1.sh`, lowrank, four GPUs, eight CPUs,
128 GiB memory, and a 35-minute limit. It profiles all 27 anchor/intervention
schedules over all 128 windows and then assembles the three exact alpha-1
budgets. Completed profile files are verified before reuse. The prepared
Qwen batch fitter maps IDs 0 through 39 to eight layers times five ranks.
Array 3133326 runs `bash scripts/fit_formal_qwen_batch.sh "$SLURM_ARRAY_TASK_ID"`
for remaining IDs 1-39, at most eight concurrently, each one GPU, two CPUs,
48 GiB memory, 45-minute limit in ghx4. The first three tasks have started.
Job 3133329 is pending for `bash scripts/evaluate_formal_nemotron_palu_ppl.sh`,
three GPUs, six CPUs, 128 GiB memory, initially twenty-minute limit in ghx4.
While pending, its limit was successfully increased to forty minutes after
the user clarified acceptable allocation lengths. Pending Qwen array tasks
8-39 were similarly increased to ninety minutes; already-started tasks 1-7
retain their original 45-minute limit. Slurm rejected the change for task 7
because it had started, while applying it to the remaining pending tasks.

Slurm limits were queried directly: account/user job limits were empty;
partition QoS `qos_ghx4int` limits interactive jobs to one running and two
submitted per user. Ordinary `qos_ghx4` does not have these limits. Hence
ordinary arrays and evaluation jobs can be queued alongside the KL job.
The user clarified that allocations below four hours are acceptable; roughly
twelve hours is too long. Extending running KL job 3133312 to two hours was
denied by Slurm, so its current limit remains 35 minutes. If needed, reuse
verified complete profiles and resume missing ones with a two-hour allocation.

Job 3133409 was submitted to ghx4 for
`bash scripts/collect_formal_nemotron_output.sh palu_mlrd_v96`, lowrank,
one GPU, two CPUs, 64 GiB memory, thirty minutes. This collects all 24 Mamba
output moments for this specific V trajectory, using the full 256 windows.
It completed in 8:12, exit 0, and was scancelled. All 24 moment files were
checked for 8192 x 8192 covariance / 4096 x 8192 weight shapes, 524,288 rows,
and matching upstream V identity. Dependent job 3133439 ran
`bash scripts/fit_formal_nemotron_output.sh palu_mlrd_v96 0` after successful
capture, one GPU, two CPUs, 64 GiB memory, thirty minutes, fitting the first
Mamba output with exact 12-sweep ALS and source rank 768. It completed in
1:02, exit 0, and was scancelled. Fitting took 37.659 seconds, with BF16
export relative error 0.0072196124; finite factors and 12 sweeps were verified.

Array 3133469 runs
`bash scripts/fit_formal_nemotron_output_shard.sh palu_mlrd_v96 "$SLURM_ARRAY_TASK_ID"`
for four shards, each one GPU, two CPUs, 64 GiB memory, thirty minutes.
Dependent job 3133470 runs
`bash scripts/evaluate_formal_nemotron_output.sh palu_mlrd_v96`, one GPU,
two CPUs, 64 GiB memory, forty minutes: assemble all 24 output factors,
then smoke and full WT2/C4 PPL on the V+AG model.

Dense MCQ array 3133468 runs
`bash scripts/evaluate_formal_dense_mcq_array.sh "$SLURM_ARRAY_TASK_ID"`
for IDs 0-5 and 7-13, excluding already-complete Qwen OpenBookQA. Each task
has four GPUs, eight CPUs, 96 GiB memory, four-hour limit; at most two run
concurrently. IDs 0-6 are Qwen and 7-13 Nemotron; within each, tasks are
ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, BoolQ, OpenBookQA.
These jobs use the identity-bound response cache.

Job 3133430 runs `bash scripts/collect_formal_qwen_teacher.sh` in lowrank,
one GPU, two CPUs, 48 GiB memory, fifteen minutes. It collects the shared
128-window Qwen teacher and records/validates full model identity. It
completed in 1:36, exit 0, and was scancelled; finite hidden states of shape
128 x 2048 x 4096 and model identity were verified. The Qwen
allocator's obsolete rank-256 error entry was removed: candidate curves must
contain exactly 96/128/160/192/224. Eight assembly/profile tests passed.

Future MCQ invocations enable lm-eval's built-in response cache, bound to
model/bank identities, runtime hashes, environment, GPU count, task, and
batch/tokenizer settings. This avoids repeating completed requests after
an interrupted evaluation. Current OpenBookQA started before that addition.

`evaluation/formal_hybrid_results.py` validates matching model/V/AG identities
across PPL and per-task MCQ files, checks the historical evaluation protocol,
and only computes the seven-task average when all seven scores exist. The
final writer requires all expected stages before producing JSON/CSV. Two
regression tests passed: incomplete averages remain absent, and mixed model
identities are rejected. No final results table has been generated.

The Nemotron output stage now has a trajectory-specific 256-window moment
collector for all 24 Mamba output projections, the existing exact source ALS
with 12 sweeps, and bank assembly. TP8 source ranks are 768/640/512 for V
targets 96/80/64. Its runtime reuses the source-local encode/concatenate/decode
operator, preserving Mamba modules and state. The quality entry composes V
and output factors. Nine runtime/output-solver tests passed; no formal Mamba
output fitting or quality run has yet occurred. Qwen GDN output default rank
is now 768, matching the confirmed 75% source-width retention.

All four jobs' stdout/stderr logs are under the experiment root's `logs/`, using
`q35-check-3133095`, `quality-check-3133101`, `dense-ppl-3133116`, and
`q35-fisher-3133114` prefixes. No complete compressed-model result table exists
yet; fitting, allocation, downstream AG, and seven-task evaluation remain required.

`evaluation/formal_batch_parallel.py` wraps PyTorch DataParallel so a global
lm-eval batch of eight can be distributed across GPUs while retaining native
model forwards. It gathers only logits, since MCQ evaluation does not consume
the returned recurrent/KV cache. Job 3133185 checks native versus four-GPU
logits/NLL on the same eight 64-token calibration sequences for each model.
Its command is `python -u -m evaluation.smoke_formal_batch_parallel` with
`--family`, `--model-path`, `--data`, and `--output` recorded in
`logs/batch-check-3133185.err`. It has a five-minute limit, four GPUs, eight CPUs,
64 GiB host memory. It completed in 1:52, exit code 0, and was scancelled.
Qwen relative logit error was 0.00912010 and absolute NLL delta 0.00045061;
Nemotron relative logit error was 0.01091105 and absolute NLL delta 0.00179982.
Both passed the short BF16 smoke thresholds. This does not establish worst-case
4096-token memory use or compressed-AG adapter replication correctness.
The quality entry exposes `--batch-parallel` for MCQ with global batch size 8;
no formal MCQ task has run yet. The user queue was empty after cleanup.

## Confirmed calibration data

Each model uses 256 windows of 2048 tokens for fitting and Fisher, plus 128
separate windows of 2048 tokens for KL profiling. No extra held-out selection
or confirmation windows are used. Factor selection uses the fitting data.
All methods for a model reuse its frozen windows. Independent C4 validation
and WikiText test data remain available for final PPL evaluation.

Command: `bash scripts/prepare_gh200_data.sh` (activates `lowrank`). Slurm job
3133083 completed successfully in 1 minute 57 seconds, exit code 0. Both model
outputs contain fit `[256, 2048]`, profile `[128, 2048]`, and C4 evaluation
`[128, 2048]`. WikiText token counts are 297193 for Qwen and 301616 for Nemotron.
Outputs are under `/work/hdd/bido/azhang35/BasisServe-CALS/cache/`, in
`qwen35_9b/data` and `nemotron_h_8b/data`. Logs are under the experiment root's
`logs/data-3133083.out` and `.err`. Tokenizer length warnings occurred while
tokenizing complete documents; model inputs must use the prepared windows.

## Environment

- Conda: Miniforge 26.7.2 at `/u/azhang35/miniforge3`.
- Environment: `lowrank`, Python 3.11.15, Linux aarch64.
- PyTorch: 2.6.0+cu126. The Qwen3-8B reference used 2.6.0+cu124 on x86_64;
  this environment uses the official ARM64 CUDA 12.6 wheel.
- Transformers: 5.15.1; datasets: 5.0.0; lm-eval: 0.4.11;
  kernels: 0.16.1.
- `pip check` passed. Full package freeze and installer logs are under
  `/u/azhang35/.cache/basis_setup/`.

Installation commands:

```bash
bash /u/azhang35/.cache/basis_setup/Miniforge3-Linux-aarch64.sh -b -p /u/azhang35/miniforge3
/u/azhang35/miniforge3/bin/conda create -y -n lowrank python=3.11.15 pip
/u/azhang35/miniforge3/envs/lowrank/bin/python -m pip install --no-cache-dir -r requirements-gh200.txt
```

The Miniforge installer was checked against its published SHA256 before execution.

## GPU check

```bash
source /u/azhang35/miniforge3/etc/profile.d/conda.sh
conda activate lowrank
export OMP_NUM_THREADS=2
python -u scripts/check_gh200_environment.py
```

Job 3129798 was cancelled while pending in `ghx4`, with no allocation.
Replacement job 3129801 ran in `ghx4-interactive` on gh152: one GPU, two CPUs,
8 GiB host memory, five-minute limit. It completed in 24 seconds with exit code 0.
`scancel` was issued afterwards and `squeue` confirmed no remaining user jobs.
The estimated charge is 24/3600*2 = 0.01333 GPU-equivalent billing hours.

- Device name: NVIDIA GH200 120GB; compute capability: 9.0.
- **PyTorch-visible device memory: 95.07635 GiB**; use measured memory for planning.
- BF16 512x512 matmul: finite; relative error to FP32: 0.00165831.
- HFLM and Qwen3_5ForCausalLM imports passed.
- Logs: `results/environment/gh200/smoke-3129801.out` and `.err`.

## Approved experiment clarifications

- PaLU retains its canonical Fisher allocation and block rounding. Exact budget
  matching is not required for PaLU; record nominal rank, actual mean rank,
  actual V retention, and budget deviation. C1 remains exact-budget allocation.
- O/AG retains the existing exact encoder solver with 12 sweeps. CG16 applies
  to C1 V fitting, not O/AG.
- MCQ uses lm-eval 0.4.11, batch size 8, max length 4096, zero-shot and all
  examples of the seven existing tasks. Select acc_norm when available,
  otherwise acc, and report their unweighted average.

## Code changes so far

- `requirements-gh200.txt`: core version pins and ARM64 torch wheel.
- `scripts/check_gh200_environment.py`: short GPU and import check.
- `evaluation/qwen35_hybrid_palu.py`: configurable target ranks (192, 160, 128)
  and groups (M-LRD, G-LRD2); derives dimensions from model config, records
  rounded budget accounting, and passes verified model identity to AG fitting.
- `evaluation/qwen35_hybrid_wo.py`: configurable sweeps defaulting to 12,
  fixed minimum sweeps, explicit exact-solver metadata, and rejection of
  cached records with different sweep settings.
- `evaluation/qwen35_hybrid_banks.py`: formal candidate bank
  96/128/160/192/224, targets 128/160/192, linear two-sided KL alpha=1,
  and ALS12/CG16 provenance checks.
- `evaluation/run_qwen35_hybrid.py` and `basisserve/core/qwen35_gated_v_als.py`:
  formal fit defaults and separately recorded encoder/decoder iteration limits.
  Encoder CG is capped at 16; the existing decoder cap stays at 200.
- `basisserve/core/qwen35_gated_v_runtime.py`: uses Transformers 5.15
  DynamicCache layers for compressed attention and cache fork/reset.
- Existing runtime, profile-sharding, and output-solver tests updated for
  the new interface and formal settings.

Syntax, CLI imports, and `git diff --check` passed. The following command
passed 32 small CPU tests in the lowrank environment:

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests/test_qwen35_gated_v_als.py tests/test_qwen35_gated_v_runtime.py tests/test_qwen35_hybrid_wo_solver.py tests/test_qwen35_hybrid_profile_shards.py tests/test_qwen35_hybrid_model_identity.py
```

Log: `/u/azhang35/.cache/basis_setup/qwen-adapter-tests.log`.
Full model integration is not yet tested; these checks do not establish
formal experiment correctness or quality metrics.

## Outstanding setup

Storage has recovered. Verified quotas: projects 266.5/500 GiB used before the
downloads; work/nvme 872.1/1000 GiB used; work/hdd about 53 MiB/1000 GiB used.
`scripts/gh200_env.sh` selects:

- Models: `/projects/bido/azhang35/hf/.cache/huggingface/hub`.
- Datasets: `/work/hdd/bido/azhang35/hf/.cache/huggingface/datasets`.
- Experiment root: `/work/hdd/bido/azhang35/BasisServe-CALS`.
- Authentication remains in the user's private home token file.

Both model snapshots have four safetensors shards and their index/config files.
The Nemotron Base-8K selection was explicitly confirmed by the user.
Model sizes: Qwen 19,306,310,880 bytes, Nemotron 16,201,739,992 bytes.
Downloads used `huggingface_hub.snapshot_download` in lowrank with pinned
revisions and JSON/safetensors/tokenizer patterns. Logs are under the experiment
root's `logs/qwen35-download.log` and `logs/nemotron-download.log`.

Qwen smoke job 3133073 completed with exit code 0 in 63 seconds and was cleaned
up. Its 18-token forward was finite, peak allocated memory was 16.7957 GiB,
and the BF16 full-rank V adapter's hidden-state relative error was 0.0097363.
This passed the short smoke threshold of 0.02; it is not a formal PPL result.
Nemotron smoke job 3133076 completed successfully in 58 seconds with the same
1 GPU, 2 CPU, 48 GiB host memory and five-minute limit. Its forward was finite,
with peak allocated memory 16.1843 GiB. Target discovery found 24 Mamba2 layers
and four full-attention layers.

Commands:

```bash
source scripts/gh200_env.sh
python -u -m evaluation.smoke_formal_qwen35 --model /projects/bido/azhang35/hf/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
python -u -m evaluation.smoke_formal_nemotron --model /projects/bido/azhang35/hf/.cache/huggingface/hub/models--nvidia--Nemotron-H-8B-Base-8K/snapshots/94ea861e008c2dfced3e8e1302094024077aa04e
```

### Earlier storage outage (resolved)

Home has a 100 GiB soft quota. Model weights and large experiment artifacts
must go to project storage, using an HF cache such as
`/projects/bido/azhang35/hf/.cache/huggingface` after access is verified.
Reads of `/projects/bido` and work paths initially hung on gh-login01, so model
downloads were deferred until storage recovered.

Additional storage diagnosis: direct SSH to gh-login02 requires Kerberos/Duo
authentication unavailable to this session. DeltaAI rejected a CPU-only Slurm
probe (`Jobs requesting no GPU resources cannot be submitted on DeltaAI`).
Job 3129822 therefore requested one GPU, one CPU, 512 MiB host memory, and a
one-minute limit in ghx4-interactive to run:

```bash
hostname
timeout -k 1 5 stat -c '%A %U %G %n' /projects/bido /work/hdd/bido /work/nvme/bido
```

It reached gh152 but the directory query also hung there and the timeout wrapper
was killed. `scancel 3129822` was issued after 29 seconds. Accounting records
the job as cancelled. After a transient COMPLETING state, `squeue` confirmed
the job had left the queue and cleanup completed. Logs are
`results/environment/gh200/storage-3129822.out` and `.err`.
Avoid more probes until storage health is confirmed. This reproduces the
access problem on both a login node and a compute node; the underlying cause
requires administrator diagnosis, rather than assuming a permission error.

Qwen uses `Qwen/Qwen3.5-9B` revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Nemotron uses the user-confirmed
`nvidia/Nemotron-H-8B-Base-8K` revision
`94ea861e008c2dfced3e8e1302094024077aa04e`.

No Git commit or push has been performed.

### 2026-09-11: ordinary partition and verified intermediate cleanup

The user permits allocations up to four hours and prohibits further interactive
partition use. Interactive job 3133329 was cancelled and left the queue;
its identical Nemotron PaLU V-only PPL command was resubmitted in ordinary
`ghx4` as 3133509 (three GPUs, six CPUs, 40 minutes).

The user explicitly authorized deleting regenerable captures/moments only after
all corresponding factors are verified. All 24 `palu_mlrd_v96` output factors
passed layer/provenance, 12-sweep, 524288-row, BF16, shape and finite-value checks
in `lowrank`. Their 24 raw moment files (14,495,552,064 bytes) were removed;
`moments/manifest.json`, all factors, banks, models, tokens and results remain.
Qwen raw captures remain until its entire factor bank is verified.

`/work/hdd` and `/work/nvme` share project quota 21965: 1000 GiB soft,
1100 GiB hard. After cleanup, `lfs quota -p 21965 /work/hdd` reported
1,037,662,040 KiB, below the soft limit. Other group members' data is untouched.

The four AG fitting tasks 3133469 completed successfully and were scancelled.
Job 3133470 passed the V+AG smoke and is evaluating full PPL. Two additional
trajectory-specific captures were submitted: 3133524 (`c1_twosided_v96`) and
3133525 (`c1_twosided_v80`), each ordinary ghx4, one GPU, two CPUs, 30 minutes,
using `bash scripts/collect_formal_nemotron_output.sh <name>` in `lowrank`.

Job 3133532 checks real compressed Nemotron V+AG single-versus-four-GPU logits,
NLL, persistent replicas and tail batches using
`python -u -m evaluation.smoke_formal_batch_parallel` with the `palu_mlrd_v96`
V and output banks. It requests four GPUs/eight CPUs for ten minutes in ghx4.
The smoke script compiles and `git diff --check` passes; GPU results are pending.
All job logs are under `$BASISSERVE_RUN_ROOT/logs`. No sbatch files were created.

C1 output fit arrays 3133533 (V96, afterok:3133524) and 3133534
(V80, afterok:3133525) each run four one-GPU/two-CPU shards, 30-minute limits,
with `bash scripts/fit_formal_nemotron_output_shard.sh <name> "$SLURM_ARRAY_TASK_ID"`.
Both retain the approved 12-sweep exact encoder solver. Qwen factor tasks 7 and
8 completed successfully and were scancelled; neither remains in the queue.

The first Nemotron `palu_mlrd_v96` V+AG full WikiText-2 result is
6.606468919001816 PPL; C4 was still running at this update. This is not a
completed main-table row until all seven MCQ tasks also finish.

Added `scripts/evaluate_formal_compressed_mcq.sh <family> <name> <mode> <task>`
for stage-scoped compressed evaluations with the shared batch-8 protocol.
`formal_hybrid_results` now exports all requested budget/allocation/output
setting columns and separate main/diagnostic CSV and Markdown tables. Its
existing two validation tests pass; final export still requires complete data.
No final result table has been generated and no Git write was performed.

Job 3133470 completed with exit code 0. The first Nemotron PaLU M-LRD
V96 + Mamba output/AG rank768 configuration has full WT2 PPL
6.606468919001816 and C4 PPL 10.251971943849224 (128 documents, batch2,
2048-token sequences, FP32 loss). `scancel 3133470` was issued and its queue
entry is absent. Command: `bash scripts/evaluate_formal_nemotron_output.sh
palu_mlrd_v96`; environment: lowrank.

Its seven-task MCQ array 3133550 is submitted with afterok dependencies on
3133470 and compressed multi-GPU smoke 3133532. It runs
`bash scripts/evaluate_formal_compressed_mcq.sh nemotron palu_mlrd_v96
v_ag_ppl_mcq "$task"` for ARC-Easy, ARC-Challenge, HellaSwag, PIQA,
WinoGrande, BoolQ and OpenBookQA. Each task requests four GPUs/eight CPUs,
96 GiB host memory, four hours, ordinary ghx4; concurrency is two.
Global MCQ batch size remains eight. Results are pending, not inferred from PPL.

Queued dependent PPL jobs: 3133556 runs
`bash scripts/evaluate_formal_nemotron_output.sh c1_twosided_v96` after fit
array 3133533, and 3133557 runs the V80 command after array 3133534.
Each uses lowrank, ordinary ghx4, one GPU/two CPUs, 64 GiB memory, 40 minutes.

Qwen C1 V-only PPL array 3133562 waits for bank assembly 3133500. Its three
one-GPU/two-CPU tasks run `bash scripts/evaluate_formal_qwen_c1_vonly.sh <rank>`
for 192, 160 and 128, each with 48 GiB host memory and a 40-minute ghx4 limit.
The new script passes bash syntax validation and runs smoke before full WT2/C4.
Qwen output/AG capture is still gated on all three C1 V-only PPL and MCQ
results plus dense evaluation; no Qwen AG capture has been submitted.

Qwen compressed multi-GPU smoke job 3133570 waits for assembly 3133500,
then checks all three C1 V-only banks with `evaluation.smoke_formal_batch_parallel`.
It uses four GPUs/eight CPUs, 96 GiB memory, 20 minutes in ordinary ghx4.
The smoke script now supports Qwen V and optional AG as well as Nemotron.
Python compilation passes. Its GPU results remain pending.

Qwen V-only MCQ array 3133571 (0-20, concurrency2) depends on both smoke
3133570 and PPL array3133562. It runs the compressed MCQ launcher for each
of ranks192/160/128 and all seven tasks, four GPUs/eight CPUs per task,
96 GiB host memory, four-hour limit, ghx4, lowrank, global batch8.
No Qwen AG task is launched before the complete V-only diagnostic gate.

Nemotron C1 capture jobs 3133524 and3133525 completed with exit0 and were
scancelled; neither remains in the queue. Their raw moments are retained
pending complete output-factor verification.

Compressed Nemotron four-GPU smoke 3133532 completed with exit0 and passed:
batch8, length64, relative logit error0.016057351604104042,
absolute NLL delta0.0009188652038574219; persistent replicas and short batches
verified. Repeated forward took0.2186428252607584 seconds. The result is
`results/nemotron_h_8b/parallel_ag_smoke.json`, tied to both V and AG bank hashes.
This is a short-sequence numerical check, not a full MCQ result. The job was
scancelled and left the queue. All prerequisites for MCQ array3133550 succeeded.

All four C1 V96 output-fit shards3133533 completed with exit0 and were
scancelled after completion. All24 factors passed layer/provenance, BF16,
finite-value, TP8rank768 geometry,524288-row and12-sweep checks. Approved
cleanup removed14,495,552,064 bytes of raw moments; the manifest and all
factors/banks/results remain. Subsequent assembly needs no raw moments.

Slurm unexpectedly marked PPL job3133556's array dependency
`afterok:3133533_*(failed)` despite all four sacct records showing COMPLETED0:0.
After the complete factor audit, its stale dependency was cleared with
`scontrol update JobId=3133556 Dependency=`. No duplicate fit/evaluation was
submitted. Watch other completed arrays for the same scheduler behavior;
clear a stale dependency only after verifying every prerequisite artifact.

Nemotron C1 V64 pipeline submitted: capture3133609,
output-fit array3133610 (0-3%4, afterok capture), and PPL3133611
(afterok fit). Commands are the existing collect/fit-shard/evaluate output
scripts with `c1_twosided_v64`; rank512,12 exact encoder sweeps,lowrank.
Each task uses one GPU/two CPUs,64 GiB host memory,ordinary ghx4;
capture/fit limits30 minutes and PPL40 minutes. V80 raw moments remain
until all its factors pass verification. V96 PPL3133556 is now running.
V80 shard3133534_0 completed0:0, was scancelled and left the queue.

PaLU V-only job3133509 is running. Its complete M-LRD WikiText-2 results
so far are V96=6.409010543631558,V80=6.510937853452374,
V64=6.554416229328524. Other stages remain pending/running; these are
V-only diagnostics, not the matched V+AG main rows.

C1 V96+AG passed its small PPL smoke (5.843964875869705; not a full score).
Seven-task MCQ array3133614 is queued afterok:3133556, using
`bash scripts/evaluate_formal_compressed_mcq.sh nemotron c1_twosided_v96
v_ag_ppl_mcq "$task"`,lowrank,ghx4,four GPUs/eight CPUs,96 GiB,4h,
concurrency2,global batch8. Full results remain pending.
Qwen factor task3133326_11 and V80 fit tasks3133534_1,_3 completed0:0,
were scancelled and left the queue. V80's remaining shard is still running.

V80 C1 all24 output factors passed full finite/BF16/shape/provenance,
12-sweep and524288-row checks;14,495,552,064 bytes of raw moments were removed,
retaining the manifest and all factors. Its PPL3133557 started normally.
PaLU M-LRD V-only C4 results completed: V96=10.036212353099957,
V80=10.134509260839295,V64=10.166841216667805.

PaLU M-LRD V80 output pipeline: capture3133626,fit array3133629,
followed by dependent PPL. Existing collect/fit-shard/evaluate scripts take
`palu_mlrd_v80`;source rank640,12 exact encoder sweeps,lowrank,ordinary ghx4,
one GPU/two CPUs/64 GiB per task,30-minute capture/fit and40-minute PPL.
Qwen factor tasks9,10,12,13,14 completed0:0 and were scancelled/removed.

C1 V96+AG full WikiText-2 PPL completed:6.276884293363173;
C4 remains running. C1 V80+AG smoke completed:6.135544217154545
(short subset only). V80 seven-task MCQ array3133631 depends on PPL3133557:
`bash scripts/evaluate_formal_compressed_mcq.sh nemotron c1_twosided_v80
v_ag_ppl_mcq "$task"`,lowrank,ghx4,four GPUs/eight CPUs,96 GiB,4h,
concurrency2,batch8. M-LRD V80 dependent PPL job ID is3133630.

C1 V96+AG full PPL job3133556 completed0:0: WT2=6.276884293363173,
C4=10.033984565315963. Command was
`bash scripts/evaluate_formal_nemotron_output.sh c1_twosided_v96` in lowrank.
The job was scancelled and left the queue; MCQ is still pending.
C1 V64 capture3133609 also completed0:0 after all256 windows and was
scancelled/removed. Qwen factor task3133326_16 completed0:0 and was released.

C1 V64 all24 output factors verified: TP8rank512,BF16,finite,correct V/moment
hashes,524288 rows,12 sweeps. Raw moments14,495,552,064 bytes were removed;
manifest/factors preserved. All3133610 shards completed0:0 and were scancelled.
C1 V80 PPL3133557 completed0:0: WT2=6.483710764779422,
C4=10.262509139492622. PaLU V-only PPL3133509 also completed0:0;
both jobs were scancelled and left the queue. G-LRD4 V-only WT2/C4:
V96=6.220256030727058/9.937553101237162;
V80=6.268907153895327/9.98105828883038;
V64=6.352849685643104/10.050481892613014.
These use the existing evaluation scripts in lowrank; MCQ remains pending.

M-LRD V64 trajectory-specific AG pipeline submitted: capture3133646,
fit array3133647,dependent PPL. Existing collect/fit-shard/evaluate scripts
with `palu_mlrd_v64`;rank512,12 exact encoder sweeps;ordinary ghx4,
one GPU/two CPUs,64 GiB per task,30-minute capture/fit,40-minute PPL.

M-LRD V80 capture3133626 completed0:0, was scancelled and left the queue;
all four dependent AG fit shards3133629 started. M-LRD V64 PPL job is3133651.
C1 V64+AG smoke passed (6.543643894360883 on the small subset).
Its seven-task MCQ array is submitted afterok PPL3133611, using the existing
compressed MCQ script with `nemotron c1_twosided_v64 v_ag_ppl_mcq "$task"`,
lowrank,ghx4,four GPUs/eight CPUs,96 GiB,4h,concurrency2,global batch8.
Full C1 V64 PPL and all its MCQ results remain pending/running.

C1 V64+AG WikiText-2 full PPL=6.857825893092207; C4 still running.
All M-LRD V80 output factors passed the24-layer provenance/geometry/BF16/
finite/12-sweep/524288-row audit. Removed14,495,552,064 raw moment bytes,
retaining manifest/factors; all3133629 shards completed0:0 and were scancelled.
Its PPL3133630 started normally. C1 V64 MCQ job ID is3133657.

G-LRD4 V96 pipeline submitted: capture3133677,fit array3133678,dependent PPL.
Existing collect/fit-shard/evaluate scripts with `palu_glrd4_v96`,rank768,
12 exact encoder sweeps,lowrank,ordinary ghx4,one GPU/two CPUs/64 GiB,
30-minute capture/fit and40-minute PPL. Model,calibration and factors are preserved.

C1 V64+AG PPL3133611 completed0:0:WT2=6.857825893092207,
C4=10.662730334302427. All three C1 Nemotron main configurations now have
full PPL, but MCQ remains pending. PPL3133611 and M-LRD V64 capture3133646
both completed0:0, were scancelled and left the queue.

M-LRD V80+AG passed its small smoke (6.48171322251892).
Its seven-task MCQ array3133681 depends on full PPL3133630 and runs
`bash scripts/evaluate_formal_compressed_mcq.sh nemotron palu_mlrd_v80
v_ag_ppl_mcq "$task"`,lowrank,ghx4,four GPUs/eight CPUs,96 GiB,4h,
concurrency2,batch8. G-LRD4 V96 dependent PPL job is3133679.

M-LRD V64 all24 output factors verified (hashes,rank512 geometry,BF16,finite,
12 sweeps,524288 rows);14,495,552,064 raw moment bytes removed,manifest and
factors retained. All3133647 shards completed0:0, were scancelled/removed;
PPL3133651 started. Its small smoke PPL=6.864123898224326.
The seven-task MCQ array is submitted afterok3133651 with the standard
compressed MCQ script,lowrank,ghx4,four GPUs/eight CPUs,96 GiB,4h,
concurrency2,batch8.

G-LRD4 V80 pipeline:capture3133699,fit array3133701,PPL3133704.
Commands: existing collect/fit-shard/evaluate output scripts with
`palu_glrd4_v80`,rank640,12 exact sweeps;lowrank,ghx4,one GPU/two CPUs,
64 GiB,30-minute capture/fit and40-minute PPL. No sbatch files created.

M-LRD V80+AG full PPL3133630 completed0:0:WT2=6.942110162076282,
C4=10.55308233959299. Command:
`bash scripts/evaluate_formal_nemotron_output.sh palu_mlrd_v80`,lowrank.
The job was scancelled and left the queue. Its MCQ remains queued.
M-LRD V64 MCQ job ID is3133705; its full PPL is still running.

G-LRD4 V96 all24 output factors verified (provenance,BF16,finite,rank768,
12 sweeps,524288 rows);raw moments14,495,552,064 bytes removed,manifest and
factors retained. All3133678 shards completed0:0 and were scancelled/removed;
PPL3133679 started. Its capture3133677 also completed0:0 and was released.
Last Nemotron AG pipeline G-LRD4 V64:capture3133740,fit3133741,PPL3133743.
Existing output scripts with `palu_glrd4_v64`,rank512,12 exact encoder sweeps,
lowrank,ghx4,one GPU/two CPUs/64 GiB,30-minute capture/fit,40-minute PPL.

M-LRD V64 full PPL3133651 completed0:0:WT2=7.4541466239560545,
C4=10.95027289917603. It was scancelled and left the queue.
Command:`bash scripts/evaluate_formal_nemotron_output.sh palu_mlrd_v64`,lowrank.
All three M-LRD main configurations now have full PPL, but MCQ is pending.

G-LRD4 V96+AG smoke passed (5.946727265342599 on the small subset).
Seven-task MCQ array3133749 depends on PPL3133679 and uses
`bash scripts/evaluate_formal_compressed_mcq.sh nemotron palu_glrd4_v96
v_ag_ppl_mcq "$task"`,lowrank,ghx4,four GPUs/eight CPUs/96 GiB,4h,
concurrency2,batch8. Qwen factor tasks17,18,19,21 completed0:0 and were
scancelled/removed. Qwen complete bank and MCQ remain outstanding.

G-LRD4 V96+AG full WikiText-2 PPL=6.373161723540018; C4 remains running.
All G-LRD4 V80 output-fit shards3133701 completed0:0 and were scancelled.
All24 factors passed provenance,rank640 geometry,BF16,finite,12-sweep and
524288-row checks. Removed14,495,552,064 raw moment bytes;manifest/factors
remain. PPL3133704 is eligible to run. Capture3133699 completed0:0 and was
scancelled earlier. Qwen factor task3133326_23 completed0:0 and was released.

G-LRD4 V96 full PPL3133679 completed0:0:WT2=6.373161723540018,
C4=10.127236732058849. Command:existing evaluate output script with
`palu_glrd4_v96`,lowrank. Job scancelled/removed. Qwen tasks22 and24 also
completed0:0 and were released. G-LRD4 V80 smoke passed (6.209912643763375).
Its seven-task MCQ array is submitted afterok3133704 with standard compressed
MCQ script,lowrank,ghx4,four GPUs/eight CPUs/96 GiB,4h,concurrency2,batch8.

G-LRD4 V64 capture3133740 completed0:0 and was scancelled/removed;all24 raw
moment files passed source/geometry/524288-row checks. All four fit shards
3133741 subsequently completed0:0 and were scancelled/removed. All24 output
factors passed V/moment provenance,rank512 geometry,BF16,finite,12-sweep and
524288-row checks. Removed14,495,552,064 raw moment bytes;manifest and all
factors remain. All nine Nemotron trajectory-specific output factor sets are
now fitted and verified. Some assembled banks/PPL and all remaining MCQ
are still outstanding, so the experiment is not complete.
G-LRD4 V80 MCQ job ID is3133788. V64 PPL3133743 is queued by priority.

G-LRD4 V80+AG full WikiText-2 PPL=6.602187284118187;C4 remains running.
G-LRD4 V64+AG smoke passed (6.718387225349096 on the small subset).
Its seven-task MCQ array is submitted afterok3133743 with the standard
compressed MCQ script and `nemotron palu_glrd4_v64 v_ag_ppl_mcq "$task"`,
lowrank,ordinary ghx4,four GPUs/eight CPUs/96 GiB,4h,concurrency2,batch8.
All nine Nemotron main compressed configurations now have MCQ submitted,
but their results are not yet available. Do not treat queued jobs as results.

Prepared `scripts/collect_formal_qwen_output.sh <bank-name>` for the later
Qwen AG stage. Before loading/capturing the model, it uses the existing
quality-row verifier to require complete dense and all three C1 V-only
PPL/seven-task MCQ records and asserts that those rows have no output bank.
It then calls the existing Qwen output capture implementation under the
selected frozen V bank,TP4. Bash syntax and git diff whitespace checks pass.
This script has NOT been submitted: the Qwen V-only diagnostic gate is not
complete. No Qwen AG capture has started.

G-LRD4 V80 full PPL3133704 completed0:0:WT2=6.602187284118187,
C4=10.34357900511813. Command:existing evaluate output script with
`palu_glrd4_v80`,lowrank. The job was scancelled and left the queue.
G-LRD4 V64 MCQ job ID is3133810;its PPL remains running.

Prepared Qwen output fit-shard and assemble/PPL scripts, reusing existing
implementations: `scripts/fit_formal_qwen_output_shard.sh` and
`scripts/evaluate_formal_qwen_output.sh`. They explicitly set TP4,Full512,
DeltaNet768,12 sweeps,float64 work (existing exact encoder solver).
Bash syntax and git whitespace checks pass. Neither has been submitted;
Qwen's required complete V-only PPL/MCQ gate remains pending.

Final Nemotron main PPL job3133743 (G-LRD4 V64+AG) completed0:0:
WT2=7.0722993474332565,C4=10.750388161361808. Command:
`bash scripts/evaluate_formal_nemotron_output.sh palu_glrd4_v64`,lowrank.
It was scancelled and left the queue. All ten Nemotron main rows now have
full PPL, but the seven-task MCQ results remain pending. The full goal is
not complete; Qwen factor bank,KL,V-only diagnostics and AG results remain.

Qwen factor array3133326 concurrency increased from8 to14 so its final six
pending independent rank fits can use fragmented ordinary ghx4 capacity.
Command remains `bash scripts/fit_formal_qwen_batch.sh "$SLURM_ARRAY_TASK_ID"`;
each task still one GPU/two CPUs/48 GiB,90-minute limit,lowrank,unchanged
calibration/12 sweeps/CG16. No duplicate tasks or altered factor settings.
MCQ remains queued on ghx4; scheduler start estimates are forecasts,not guarantees.

Two-GPU compressed Nemotron numerical check3133867 submitted in ordinary
ghx4 (two GPUs,four CPUs,96 GiB,10 minutes,lowrank) using the existing
smoke_formal_batch_parallel command with M-LRD V96+AG banks and output
`results/nemotron_h_8b/parallel_ag_2gpu_smoke.json`. Global batch remains8.
This explores better scheduling on partially occupied nodes; formal MCQ
allocations have not been changed. The check is still pending by priority.
Qwen task3133326_26 completed0:0, was scancelled and left the queue.
After increasing array concurrency, tasks34 and35 have started.

Two-GPU compressed Nemotron smoke3133867 completed0:0 and passed:
relative logit error0.012921490706503391,NLL delta0.000039577484130859375,
batch8,length64,persistent replicas and short batches verified.
It was scancelled/removed. To validate real MCQ with this packing, the pending
M-LRD V96 OpenBookQA task3133550_6 was cancelled (confirmed pending first),
and the same full500-example command was submitted with two GPUs/four CPUs,
96 GiB,one hour,ordinary ghx4,lowrank. Global batch8 and the evaluation protocol
are unchanged. Remaining MCQ allocations are unchanged pending this result.

Qwen fit now loads raw capture lazily only for a missing factor. Existing
factor reuse retains capture-manifest hash and solver-setting checks plus
model-config identity validation, and no longer needs regenerable raw files.
Verified in lowrank with the real l03_r096 factor and a temporary capture
directory containing only the identical manifest (no raw files);reuse passed.
Python compilation and git whitespace checks pass. This changes no solver
or fit settings and does not affect already-running Python processes.
Qwen tasks27,28,29,30,31 completed0:0 and were scancelled/removed earlier.
Two-GPU full OpenBookQA job ID is3133874 and remains pending by priority.

OpenBookQA3133874 failed before model loading: an unconstrained `--gpus=2`
allocation spanned gh048 and gh149, exposing only one GPU on the batch host.
It was scancelled and verified absent from the queue. Resubmitted as3133902
with explicit `--nodes=1 --ntasks=1`, otherwise identical two-GPU resources,
one-hour limit,lowrank and full evaluation command:
`bash scripts/evaluate_formal_compressed_mcq.sh nemotron palu_mlrd_v96 v_ag_ppl_mcq openbookqa`.
No score was produced by the failed attempt. Existing four-GPU dense MCQ
allocations already explicitly constrain a single node.

Applied the same explicit single-node constraint to pending compressed MCQ
jobs3133550,3133614,3133631,3133657,3133681,3133705,3133749,3133788,3133810
and Qwen parallel check/MCQ3133570/3133571 using `scontrol update`.
Verified each has NumNodes=1-1,NumTasks=1,NumCPUs=8,CPUs/Task=8 and four GPUs;
commands,limits,dependencies and evaluation settings are unchanged.
On this Slurm, updating NumTasks reset NumCPUs to1 even when NumCPUs=8 was
supplied in the same call; a subsequent NumCPUs=8 update restored and verified
the intended CPU allocation before execution.
The collector successfully read existing actual bank budgets and all nine
Nemotron output settings. It correctly reports incomplete:20 main rows and
15 diagnostic rows,with184 stage files still absent at this check.

Two-GPU Nemotron M-LRD V96+AG OpenBookQA3133902 completed0:0 in4:34
(evaluation243.575s),all500examples,acc_norm=0.472,recorded peak31.9005GiB.
Protocol verified:lm-eval0.4.11,lowrank,globalbatch8,max4096,zero-shot,
no BOS,no limit,two GPUs. Job scancelled and absent from queue.
Replaced pending BoolQ3133550_5 with single-node two-GPU3133916 (four CPUs,
96GiB,four hours) to validate longer inputs using the same M-LRD V96+AG bank.
Replaced pending Qwen dense WinoGrande3133468_4 with3133918 (same resources),
which first runs the existing two-GPU numerical smoke then the full task.
Smoke output:results/qwen35_9b/parallel_dense_2gpu_smoke.json.
Other eight Nemotron OpenBookQA tasks were confirmed pending and cancelled,
then replaced by two-GPU array3133920 (0-7%2,four CPUs,96GiB,one hour):
C1 V96,V80,V64;M-LRD V80,V64;G-LRD4 V96,V80,V64,in that order.
Commands use existing evaluate_formal_compressed_mcq.sh with openbookqa;
no model,factor,calibration or evaluation protocol changes.
Qwen factor3133326_32 completed0:0 and was scancelled/removed;33/40 factors
are now present,with seven remaining fits running at this check.

Qwen factor tasks33,34,35,36 subsequently completed0:0 and were scancelled,
verified absent from queue. All37 completed factors now audited,including
new l27 ranks160/192/224 and l31 ranks96/128:finite BF16,expected dimensions,
matching capture hash,12 sweeps,fixed encoderCG16,decoder cap200.
Only l31 ranks160/192/224 remain running;raw capture is retained.
Nemotron OpenBookQA array3133920_0/_1 completed0:0 and were scancelled/removed:
C1 V96+AG acc_norm0.478 (500examples,244.886s,peak31.9343GiB),
C1 V80+AG acc_norm0.464 (500examples,228.156s,peak31.6454GiB).
Both retain lowrank,lm-eval0.4.11,globalbatch8,two GPUs and full task protocol.

Two-GPU probes3133916(BoolQ) and3133918(Qwen dense WinoGrande) had their
pending time limits shortened to one hour,with commands unchanged.
BoolQ3133916 then failed before scoring:datasets refused preparation because
shared /work usage1056901432KiB exceeded the1048576000KiB soft quota.
It was scancelled/removed. Resubmit the same BoolQ command after verified
Qwen raw-capture cleanup restores space;no result exists yet.
Qwen WinoGrande3133918 has started on gh119 and passed its two-GPU smoke.
A hold attempted while it was starting was immediately released;verified
RUNNING,Reason=None. It is now executing the full WinoGrande command.
Nemotron OpenBookQA3133920_2/_3 completed0:0 and were scancelled/removed:
C1 V64+AG acc_norm0.470,234.615s;M-LRD V80+AG acc_norm0.458,191.350s;
both full500examples. Array tasks4/5 are running;6/7 remain pending.

Qwen WinoGrande3133918 failed before scoring with the same /work soft-quota
dataset preparation error;it was scancelled and left the queue. Its numerical
two-GPU smoke passed and is retained (relative logit0.00900813,NLL delta0.00220275).
Nemotron OpenBookQA3133920_4/_5 completed0:0 and were scancelled/removed:
M-LRD V64+AG acc_norm0.462,183.460s;G-LRD4 V96+AG acc_norm0.464,186.278s.

All40 Qwen factors are now complete and fully verified. Last fit tasks37/38/39
completed0:0 and were scancelled/removed. Audit saved to
cache/qwen35_9b/factors/audit.json:all expected layer/rank pairs,finite BF16,
exact E/R shapes,12 actual fixed-CG16 encoder updates,decoder cap200,
selected decoder-closed sweep12,capture provenance and individual file hashes.
After verifying these hashes again,deleted only32 manifest-listed fit_*.pt
raw capture files,total111669427904bytes (104GiB),under prior user permission.
Capture manifest/weights,factors,tokens,teacher,models and results retained.
Quota subsequently dropped to947855120KiB,below1048576000KiB soft limit.
The completed factor array's afterok dependency nevertheless became
DependencyNeverSatisfied. After full artifact audit,cleared only3133499's
stale dependency;KL tasks0/1/2 are now RUNNING,remaining tasks pending.
Resubmitted unchanged two-GPU one-hour full tasks after space recovered:
3133970:`bash scripts/evaluate_formal_compressed_mcq.sh nemotron palu_mlrd_v96 v_ag_ppl_mcq boolq`;
3133971:`bash scripts/evaluate_formal_dense_mcq.sh qwen35 winogrande`.
No repeated Qwen numerical smoke or overwritten result;existing smoke retained.

All nine compressed Nemotron OpenBookQA tasks are now complete,each verified
500examples,globalbatch8,two GPUs. Acc_norm by budget96/80/64:
C1=0.478/0.464/0.470;M-LRD=0.472/0.458/0.462;G-LRD4=0.464/0.460/0.472.
Final array tasks3133920_6/_7 completed0:0 and were scancelled/removed;
the entire3133920 array is absent from the queue.
Pending dense Nemotron OpenBookQA3133468_13 was cancelled and replaced by
single-node two-GPU3133974,four CPUs,96GiB,one hour,lowrank:
`bash scripts/evaluate_formal_dense_mcq.sh nemotron openbookqa`.
Qwen KL tasks3133499_0/_1/_2/_3 completed0:0 and were scancelled/removed;
each full128-window profile is retained. Tasks4-8 are running at this check.

Qwen KL tasks0-14 have now completed0:0 and all have been scancelled/removed.
The first seven profile artifacts were independently verified against their
exact schedules,factor hashes,model identity,128-window values and mean KL.
Final JSON collection now also embeds both formal experiment configs and
calibration manifests,with source paths/hashes. A real-artifact collection
successfully serialized with allow_nan=False,20 main rows and both manifests
verified fit256x2048/profile128x2048. It remains explicitly incomplete;
no final result export was written while required stages are absent.

Qwen KL tasks15-20 completed0:0 and were scancelled/removed (21/51 done).
Since ordinary two-GPU start forecasts remain hours away and no prior
single-GPU batch8 OOM was established,the pending BoolQ3133970 was cancelled
and replaced by3134017:ordinary ghx4,one GPU,two CPUs,48GiB,one hour,lowrank.
It invokes the existing formal_hybrid_quality mcq entry directly for Nemotron
M-LRD V96+AG,boolq,without --batch-parallel. Global batch remains8,all examples,
same max4096/tokenizer/zero-shot protocol and output path. This is a real
single-GPU long-input feasibility run;other MCQ allocations remain unchanged.
No result is yet available for this BoolQ configuration.

All51 Qwen KL tasks3133499_0-50 completed0:0 and were scancelled/removed.
Full artifact audit saved to cache/qwen35_9b/kl/audit.json:each exact schedule,
factor hash,model identity,128 finite window KL values and their mean verified.
Uniform anchor KL means:128=0.02927958049216795,160=0.015618978486190706,
192=0.007529557934644606. These are calibration KL,not evaluation metrics.
Assembly3133500 showed DependencyNeverSatisfied despite successful profiles;
after the full51-profile audit,cleared its stale dependency. The unchanged
assemble_formal_qwen_c1.sh is pending ordinary Slurm resources.
Single-GPU BoolQ3134017 has started and passed dataset preparation,which had
failed before raw-capture cleanup. It is preparing/scoring all6540 requests
(3270 examples),globalbatch8;no final BoolQ result yet.

Single-GPU BoolQ3134017 failed2:04 at the first scoring batch:the native
Nemotron mamba2_chunk_scan C/B broadcast requested an88GiB intermediate
with74.12GiB free on the95.08GiB GPU. No scoring requests completed.
Job scancelled/removed. This establishes that this native long-input path
cannot run batch8 on one GH200;restored the same full BoolQ command to two
GPUs,four CPUs,96GiB,one hour (globalbatch8 unchanged). Do not infer that
short OpenBookQA memory predicts BoolQ memory.

Restored two-GPU BoolQ job ID3134063 (pending). Qwen assembly3133500
completed0:0 in11s and was scancelled/removed. Verified all three output
banks' factor hashes and exact mean budgets. Layer order3,7,11,15,19,23,27,31:
R192=[128,192,192,192,224,224,224,160],retention0.75;
R160=[96,160,128,128,224,192,224,128],retention0.625;
R128=[96,96,96,96,192,160,192,96],retention0.5.
All three V-only PPL tasks3133562_0/_1/_2 have started. Parallel V-only
check3133570 is pending by priority;V-only MCQ3133571 waits for PPL and check.
Qwen AG capture has not been launched:the required full V-only PPL/MCQ
diagnostic gate still applies.

All three Qwen V-only PPL jobs3133562_0/_1/_2 completed0:0,were scancelled
and verified absent. Matched full WT2/C4 PPL by budget:
192=8.239483907036066/11.365440651620789;
160=8.201419160755293/11.425212745508341;
128=8.2950887238354/11.533454545099163.
Quality collector verified model/bank identities and dense output projections.
Qwen dense WinoGrande3133971 completed0:0,all1267examples,acc0.7292817679558011,
361.752s,peak18.5723GiB. Nemo dense OpenBookQA3133974 completed0:0,
all500examples,acc_norm0.472,220.941s,peak19.9749GiB. Both scancelled/removed.

Pending four-GPU Qwen V-only check3133570 was replaced by two-GPU3134067,
20minutes. V192/V160 checks passed and their immutable results are retained.
V128 failed the short-tail batch3-vs-native-batch8 logit threshold;job cleaned.
Diagnostic rerun3134069 quantified this:parallel-vs-batch8=0.0214923,
native-batch3-vs-batch8=0.0356694,parallel-vs-native-batch3=0.0244341,
parallel-vs-identically-split-native(2+1)=0.0. It was scancelled/removed.
The smoke now strictly asserts_close short parallel logits against native
logits with identical chunk shapes,to isolate replication from BF16 batch-shape
effects. All diagnostics are retained;full batch8 logit/NLL tolerances unchanged.
Only V128 rerun submitted as3134072,two GPUs,10minutes,same output path.
V-only MCQ3133571 now depends on3134072;all prior PPL prerequisites verified.

Two-GPU Nemo BoolQ3134063 also OOMed on its first batch:44GiB native Mamba
intermediate request,30.94GiB free,44.35GiB reserved but unallocated.
No scores produced;scancelled/removed. Restored four-GPU eight-CPU96GiB
one-hour BoolQ as3134070,unchanged script/globalbatch8. Other long-task
four-GPU requests remain unchanged.

V128 check3134072 completed0:0 in23s and was scancelled/removed.
Full batch8 relative logit0.00959659647,NLL delta0.00040435791;
short parallel versus matching native chunks exactly0. All three V-only
checks and PPL prerequisites now pass. MCQ3133571 became eligible.
Confirmed pending and replaced its six OpenBookQA/WinoGrande tasks
(indices4,6,11,13,18,20) with two-GPU one-hour array3134088 (0-5%2):
rank192 OpenBookQA/WinoGrande,rank160 OpenBookQA/WinoGrande,rank128 same.
Command is existing evaluate_formal_compressed_mcq.sh,qwen35,C1 bank,v_only,
task;globalbatch8 unchanged. Remaining15 longer tasks stay on four GPUs.

Qwen short-task array3134088_0/_1 started (V192 OpenBookQA/WinoGrande).
To reduce reservation length,52 still-pending four-GPU ARC-Easy,
ARC-Challenge,PIQA and WinoGrande tasks had TimeLimit changed4h->1h via
scontrol,after confirming each was pending. GPU/CPU/memory,commands,batch8,
datasets and cache identities unchanged. HellaSwag/BoolQ limits unchanged.
Queue verification:57 pending one-hour tasks,27 pending four-hour tasks,
and2 running one-hour short tasks at this check.

Qwen V192-only OpenBookQA3134088_0 completed0:0,all500examples,acc_norm0.430,
201.780s,was scancelled and removed. WinoGrande3134088_1 remains running.
Pending Nemotron M-LRD V96 ARC-Easy/PIQA3133550_0/_3 were replaced by
two-GPU array3134147 (0=arc_easy,1=piqa,concurrency2),one hour,four CPUs,
96GiB,ordinary ghx4,lowrank. Existing compressed MCQ script and globalbatch8
unchanged. This tests these tasks' real two-GPU memory/runtime before
changing remaining four-GPU ARC/PIQA allocations.

Qwen V192-only WinoGrande3134088_1 completed0:0,all1267examples,
acc0.7324388318863457,275.109s,was scancelled and removed.
Still-pending four-GPU BoolQ3134070 time limit shortened1h->15minutes
to seek a short backfill slot and establish four-GPU long-input feasibility.
Same full task/globalbatch8/command/cache identity;no sample limit introduced.
If it reaches walltime,the existing official response cache supports resuming
the remaining requests. This is not yet a completed BoolQ evaluation.

Qwen V160-only OpenBookQA3134088_2 completed0:0,all500examples,acc_norm0.434,
210.705s;WinoGrande3134088_3 completed0:0,all1267examples,acc0.7198105761641673,
249.457s. Both scancelled/removed.
Four-GPU BoolQ3134070 started and successfully scored the initial longest
batches (528/6540 requests by2:05 job elapsed),unlike one/two-GPU attempts.
Based on measured throughput,remaining pending four-GPU short tasks and
Nemotron BoolQ were shortened to20minutes (59 tasks);Qwen BoolQ to30minutes
(4 tasks). HellaSwag unchanged. Only pending tasks were updated;globalbatch8,
full samples,commands and response cache identities unchanged. Walltime
interruptions,if any,must resume cached remaining requests before final scoring.

Pending Qwen V128 short tasks3134088_4/_5 were shortened1h->20minutes,
then started and completed0:0. OpenBookQA acc_norm0.432,all500examples,
210.024s;WinoGrande acc0.7190213101815311,all1267examples,238.635s.
Both scancelled/removed;all six Qwen V-only OpenBookQA/WinoGrande tasks
are complete,while their other five tasks per bank remain pending.
Four-GPU Nemotron M-LRD V96+AG BoolQ3134070 completed0:0 in9:08,
all3270examples,acc0.8547400611620795,evaluation533.044s,peak39.9793GiB.
It was scancelled and removed. The one/two-GPU native long-input OOM attempts
produced no scores;this four-GPU result uses unchanged globalbatch8/full protocol.

Qwen Dense ARC-Easy3133468_0 has started but is materially slower than the
short-task timings used for initial reservations (~1119/9501 requests after
5:13 scoring). Attempt to extend its running20minute limit to1h was denied
by Slurm. Let it continue;on timeout,scancel and resume the same four-GPU
command/cache with1h. Restored all seven pending Qwen ARC-Easy/Challenge
tasks to1h (dense challenge and six C1 tasks),based on this runtime evidence.
Nemotron two-GPU ARC-Easy/PIQA3134147_0/_1 have started;full results pending.

Pending Qwen Dense ARC-Challenge3133468_1 was replaced by two-GPU3134197,
four CPUs,96GiB,one hour,unchanged evaluate_formal_dense_mcq.sh qwen35
arc_challenge command/globalbatch8. It is running.
Nemotron M-LRD V96+AG PIQA3134147_1 completed0:0,all1838examples,
acc_norm0.8193688792165397,386.606s,peak38.8250GiB. Scancelled/removed.
User-requested remaining-work census at this point:20/161 individual MCQ
evaluations complete (Qwen main2/70,C1 V-only6/21,Nemotron main12/70).
141 remain:Qwen dense5,C1 V-only15,Qwen AG63,Nemotron58.
All35 expected WT2/C4 pairs include26 complete;remaining9 Qwen AG pairs
require18 PPL stages plus nine trajectory-specific AG fits after the full
Qwen dense/C1 V-only gate. No Qwen AG run has started.

Following the user's request to reduce GPU reservations, 47 pending MCQ tasks
were updated in place to two GPUs/four CPUs (both GPU TRES and CPU task TRES
verified). Commands, full datasets and global batch8 remain unchanged.
Nemotron BoolQ stays at four GPUs following real one/two-GPU OOMs;
compressed Nemotron HellaSwag stays at four pending dense two-GPU feasibility.
Qwen pending MCQ now uses two GPUs. Running four-GPU tasks were not resized.
The BoolQ OOM is native FP32 Mamba broadcast temporaries (88GiB single GPU,
44GiB two GPUs), not an inherent four-GPU requirement of the 8B weights.

Remaining nine Nemotron PIQA evaluations were moved to array3134213_0-8%3,
two GPUs/four CPUs/96GiB/20minutes,ordinary ghx4,lowrank. Mapping: dense,
C1v96,C1v80,C1v64,MLRDv80,MLRDv64,GLRD4v96,GLRD4v80,GLRD4v64.
Commands: bash scripts/evaluate_formal_dense_mcq.sh nemotron piqa for dense;
bash scripts/evaluate_formal_compressed_mcq.sh nemotron <variant> v_ag_ppl_mcq piqa
for compressed models. Original pending PIQA tasks were cancelled first.

New complete full-task results (globalbatch8,0shot,max4096,noBOS,no limit,
lm-eval0.4.11,lowrank; effective sample counts verified):
- Qwen dense ARC-Challenge3134197: acc_norm0.5597269624573379,1172 samples,
  401.018s,peak18.572GiB,two GPUs.
- Nemotron MLRDv96+AG ARC-Easy3134147_0: acc_norm0.8417508417508418,
  2376 samples,752.436s,peak58.899GiB,two GPUs.
- Nemotron dense PIQA3134213_0: acc_norm0.824265505984766,1838 samples,
  393.359s,peak24.868GiB,two GPUs.
- Nemotron C1v96+AG PIQA3134213_1: acc_norm0.8150163220892275,1838 samples,
  372.80s,peak40.75GiB,two GPUs.
- Nemotron C1v80+AG PIQA3134213_2: acc_norm0.8182807399347116,1838 samples,
  389.587s,peak38.178GiB,two GPUs.
All five terminal tasks were scancelled and verified absent from squeue.

Qwen dense ARC-Easy3133468_0 timed out at20minutes and was scancelled.
Superseding the prior four-GPU resume plan in accordance with the user's
resource preference, replacement3134251 requests two GPUs/four CPUs/96GiB/1h
in ordinary ghx4. Command: bash scripts/evaluate_formal_dense_mcq.sh qwen35 arc_easy.
Globalbatch8 unchanged; changing GPU count selects a separate response cache,
so the old four-GPU partial cache is retained but not reused by this job.

User approved fixing the native Mamba broadcast memory overhead. Added isolated
basisserve/core/nemotron_h_scan.py with FP32 einsum contractions replacing
broadcast-then-reduce operations for G, diagonal output, intra-chunk states,
inter-chunk recurrence and off-diagonal output. Native scan/cache interface,
model parameters, core dependency versions and globalbatch8 remain unchanged.
This implementation is not yet enabled in formal evaluations.
Validation3134264 submitted in ordinary ghx4,one GPU/two CPUs/64GiB/15minutes,
lowrank. Command: python -u -m evaluation.smoke_nemotron_scan --model
<cached Nemotron Base-8K snapshot> --data <runroot>/cache/nemotron_h_8b/data
--output <runroot>/results/nemotron_h_8b/scan_smoke.json.
Checks cover padded/multiple chunks, initial/final states, full-model short
logits/NLL and one-GPU batch8x2048 peak memory. Full BoolQ validation follows
only after these pass. Logs: logs/nemo-scan-check-3134264.out/.err.

Scan check3134264 completed0:0 in35seconds and was scancelled/removed.
All six scan/state checks passed with maximum absolute error1.90735e-6.
Full BF16 model batch8x64 relative logit error0.0101721240 and NLL delta
0.0002005100 passed existing numerical tolerances. Crucially, single-GPU
batch8x2048 completed in1.338seconds with peak37.14384GiB (dense model).
This is a memory feasibility check, not a complete BoolQ result.
Formal Nemotron evaluation now enters the efficient scan context; its source
hash is included in response-cache identity and result metadata. Existing
completed artifacts remain intact. Core versions and experimental settings
are unchanged; floating-point reduction order differs and is documented.

Full single-GPU MLRDv96+AG BoolQ validation3134272 submitted,ordinary ghx4,
one GPU/two CPUs/64GiB/30minutes,lowrank. Command: python -u -m
 evaluation.formal_hybrid_quality mcq --family nemotron --model <cached snapshot>
 --manifest <cache>/model.json --data <cache>/data
 --bank <cache>/banks/palu_mlrd_v96.pt
 --wo-bank <cache>/wo/palu_mlrd_v96/factors/wo_bank.pt
 --tasks boolq --output <runroot>/results/nemotron_h_8b/scan_boolq.json.
Uses all3270samples,globalbatch8; separate validation artifact preserves the
original four-GPU score for comparison. Pending at submission.

Nemotron C1v64+AG PIQA3134213_3 completed0:0 in6:23,all1838samples,
and was scancelled. Qwen dense PIQA3133468_3 reached all3676 loglikelihood
requests but timed out during finalization without a result JSON;scancelled.
Replacement3134276 uses two GPUs/four CPUs/96GiB/30minutes,ordinary ghx4,
lowrank,command: bash scripts/evaluate_formal_dense_mcq.sh qwen35 piqa.
Old partial cache retained;new GPU/runtime identity uses a separate cache.

Full single-GPU MLRDv96+AG BoolQ3134272 completed0:0 in7:55,scancelled and
removed. All3270samples,globalbatch8,model/V/AG hashes match old four-GPU run.
Optimized scan accuracy0.8550458715596331 vs old0.8547400611620795 (net one
correct answer difference); numerical reduction/batch shape changes are not
bitwise identical. Peak42.44459GiB on one GPU,eval457.819s versus old four-GPU
533.044s. Scoring itself185seconds; shared-filesystem SQLite flushing caused
additional wait,confirmed live I/O and open journal,not quota exhaustion.
Artifacts scan_smoke.json and scan_boolq.json preserve validation separately.

Updated nine still-pending Nemotron BoolQ tasks in place to one GPU/two CPUs:
3133468_12,3133614_5,3133631_5,3133657_5,3133681_5,3133705_5,3133749_5,
3133788_5,3133810_5. Both GPU and CPU task TRES verified. Existing20minute
limits/commands/data/globalbatch8 unchanged. Dense and compressed MCQ scripts
now only pass --batch-parallel when SLURM_GPUS_ON_NODE exceeds one,so a
single-GPU allocation uses the native HFLM interface with the efficient scan.
Other tasks retain their current allocations pending their own validation.

Nemotron optimized two-GPU PIQA results are complete for MLRDv80,
MLRDv64 and GLRD4v96 (3134213_4/_5/_6): all1838samples,acc_norm respectively
0.8204570184983678,0.8155603917301415,0.8204570184983678; elapsed240.182,
240.162,234.568seconds;peak36.689,36.447,36.859GiB.
Their Python result stages completed successfully, but outer bash exited2
because the script was edited while those shells were executing its earlier
contents. Result JSON/protocol/full sample counts verified;do not rerun valid
results. All three terminal tasks scancelled. Current scripts pass bash -n;
avoid further edits to shell scripts while processes are executing them.

Still-pending dense Nemotron HellaSwag3133468_9 changed to one GPU/two CPUs,
1h for full-task single-GPU feasibility with optimized scan. Command unchanged:
bash scripts/evaluate_formal_dense_mcq.sh nemotron hellaswag (lowrank,batch8).
Other compressed HellaSwag tasks keep existing resources until pilot passes.

Collector provenance extended per source to retain environment, protocol/GPU
count, explicit optimized Mamba metadata and response runtime hashes. This
makes mixed native/optimized historical results visible rather than implying
bitwise-identical execution. Two collector tests passed (14.29seconds).
Read-only full collector audit on current artifacts succeeded, including bank
and output identities and JSON serialization: main23 MCQ tasks/11 PPL pairs,
diagnostics6 MCQ tasks/15 PPL pairs. Thus29/161 MCQ tasks and26/35 PPL pairs
complete;zero main rows have all seven MCQs yet. This excludes the separate
single-GPU BoolQ validation duplicate. Qwen AG gate still lacks19 MCQ tasks
at this check (Dense4,C1 V-only5 per rank);Dense ARC-Easy/PIQA are running.

Qwen Dense PIQA3134276 completed0:0 in5:51,two GPUs,all1838samples,
acc_norm0.8003264417845484,eval335.399seconds,peak18.95550GiB.
Command: bash scripts/evaluate_formal_dense_mcq.sh qwen35 piqa;lowrank,
globalbatch8. Result verified,scancelled and confirmed absent from squeue.
Formal MCQ count now30/161;Qwen AG gate18 tasks remaining. Dense ARC-Easy
3134251 remains live with more than half of9501 requests scored at this check.

Nemotron GLRD4v80/v64+AG PIQA3134213_7/_8 completed0:0 in4:13/4:41,
all1838samples,acc_norm0.8188248095756256/0.8117519042437432;
eval238.330/252.582seconds,peak36.924/36.634GiB,two GPUs,lowrank.
Command: bash scripts/evaluate_formal_compressed_mcq.sh nemotron
palu_glrd4_v80 (or palu_glrd4_v64) v_ag_ppl_mcq piqa.
Both scancelled. All ten Nemotron main PIQA artifacts verified complete with
full sample counts and globalbatch8. Formal total now32/161 MCQ tasks.

Resource audit found Slurm MinCPUsNode remained8 after CPU task/TRES reductions.
Updated dense HellaSwag3133468_9 minimum to2,then55 other still-pending tasks'
MinCPUsNode to their actual CPUs/Task (2 for single-GPU BoolQ,4 for two-GPU
MCQ),verifying every update. No command,algorithm,GPU count or time limit changed.
Immediately subsequent queue observation showed seven additional jobs running:
3133550_1/_4 and3133614_5,3133631_5,3133657_5,3133749_5,3133788_5.
These starts coincide with the resource correction; no guaranteed queue-time
claim is made. Qwen dense ARC-Easy3134251 remains running (~74% requests).

Nemotron MLRDv96+AG WinoGrande3133550_4 completed0:0 in2:59,
all1267samples,acc0.760852407261247,eval164.026seconds,peak30.083GiB,
two GPUs,optimized scan,lowrank/globalbatch8. Command:
bash scripts/evaluate_formal_compressed_mcq.sh nemotron palu_mlrd_v96 v_ag_ppl_mcq winogrande.
Verified complete and scancelled. Formal MCQ count33/161 at this check.

Qwen dense ARC-Easy3134251 completed0:0 in11:55,all2376samples,
acc_norm0.7424242424242424,eval702.753seconds,peak18.572GiB,two GPUs.
Command: bash scripts/evaluate_formal_dense_mcq.sh qwen35 arc_easy (lowrank).
Nemotron MLRDv96+AG ARC-Challenge3133550_1 completed0:0 in3:59,
all1172samples,acc_norm0.6032423208191127,eval226.291seconds,peak45.076GiB,
two GPUs,optimized scan. Command: bash scripts/evaluate_formal_compressed_mcq.sh
nemotron palu_mlrd_v96 v_ag_ppl_mcq arc_challenge (lowrank).
Both full-task results verified,globalbatch8 unchanged,scancelled and removed.
Formal MCQ35/161;Qwen AG gate17 remaining (Dense BoolQ/HellaSwag and15C1).

Five formal single-GPU BoolQ tasks completed0:0,all3270samples/globalbatch8,
lowrank,optimized FP32 scan. Commands use evaluate_formal_compressed_mcq.sh
nemotron <variant> v_ag_ppl_mcq boolq. Results (acc,eval seconds,peakGiB):
- C1v96,3133614_5:0.8513761467889909,303.956,42.446.
- C1v80,3133631_5:0.8428134556574923,298.158,42.162.
- C1v64,3133657_5:0.8293577981651377,298.114,41.884.
- GLRD4v96,3133749_5:0.8532110091743119,284.006,42.728.
- GLRD4v80,3133788_5:0.8422018348623853,294.964,42.397.
All verified,scancelled and absent from queue. Formal MCQ count40/161.
MLRDv80,MLRDv64 and GLRD4v64 BoolQ are now running single-GPU;
dense BoolQ and HellaSwag remain pending.

Qwen pending C1 ARC jobs have no unresolved dependencies and corrected CPU
minimums. Based on actual two-GPU Dense ARC-Easy11:55/ARC-Challenge6:57,
shortened six pending C1 ARC task limits1h->30minutes (3133571_0/_1/_7/_8/_14/_15).
This supersedes the earlier one-hour settings based on slow four-GPU execution.
Commands/globalbatch8/two GPUs/four CPUs/96GiB/lowrank unchanged; no resubmit.

Pending Qwen Dense BoolQ3133468_5 changed to one GPU/two CPUs (including
MinCPUsNode2),retaining30minutes/96GiB/fulltask/globalbatch8/lowrank.
Command: bash scripts/evaluate_formal_dense_mcq.sh qwen35 boolq.
This is Qwen's full-task single-GPU feasibility run;other Qwen MCQs retain
two GPUs until real long-input memory/result validation. No claim of
single-GPU Qwen BoolQ feasibility is made before completion.

Four further full-task Nemotron results verified (lowrank,globalbatch8):
- MLRDv80+AG BoolQ3133681_5:3270samples,acc0.8431192660550458,
  eval326.724s,peak42.167GiB,one GPU,job5:40.
- MLRDv64+AG BoolQ3133705_5:3270samples,acc0.8235474006116208,
  eval265.005s,peak41.879GiB,one GPU,job4:38.
- GLRD4v64+AG BoolQ3133810_5:3270samples,acc0.8345565749235474,
  eval263.346s,peak42.067GiB,one GPU,job4:36.
- C1v96+AG ARC-Challenge3133614_1:1172samples,acc_norm0.6083617747440273,
  eval238.928s,peak44.661GiB,two GPUs,job4:15.
All completed0:0,scancelled and verified absent from queue. Commands:
bash scripts/evaluate_formal_compressed_mcq.sh nemotron <variant> v_ag_ppl_mcq <task>.
Formal MCQ count44/161. All nine compressed Nemotron BoolQ results complete;
dense BoolQ still pending. Qwen AG gate still17 MCQ tasks pending.

Three Nemotron C1 ARC tasks completed0:0,full samples,lowrank/globalbatch8,
two GPUs,optimized scan;verified/scancelled/absent from queue:
- C1v96 ARC-Easy3133614_0:2376samples,acc_norm0.8476430976430976,
  eval370.399s,peak56.618GiB,job6:27.
- C1v80 ARC-Easy3133631_0:2376samples,acc_norm0.8396464646464646,
  eval368.827s,peak56.349GiB,job6:22.
- C1v80 ARC-Challenge3133631_1:1172samples,acc_norm0.5938566552901023,
  eval185.999s,peak44.485GiB,job3:21.
Commands: bash scripts/evaluate_formal_compressed_mcq.sh nemotron
<c1_twosided_v96 or c1_twosided_v80> v_ag_ppl_mcq <arc_easy or arc_challenge>.
Formal MCQ47/161. ARC remains two-GPU:measured ARC-Easy GPU0 peak~56.6GiB
means BoolQ's single-GPU success alone does not prove ARC single-GPU feasibility.

C1v96+AG WinoGrande3133614_4 completed0:0 in2:49,all1267samples,
acc0.7529597474348856,eval153.717seconds,peak30.847GiB,two GPUs,lowrank.
Command: bash scripts/evaluate_formal_compressed_mcq.sh nemotron
c1_twosided_v96 v_ag_ppl_mcq winogrande. Verified/scancelled/absent.
Formal MCQ48/161.

Qwen C1v192-only ARC-Challenge3133571_1 pending allocation reduced to one
GPU/two CPUs/MinCPUsNode2,retaining30minutes/96GiB/globalbatch8. Command:
bash scripts/evaluate_formal_compressed_mcq.sh qwen35 c1_twosided_v192 v_only arc_challenge.
This is a full-task single-GPU C1 feasibility run motivated by dense two-GPU
ARC-Challenge18.57GiB peak; no single-GPU success claim until validated.
Other Qwen C1 tasks retain two GPUs.

Three more Nemotron C1 results completed0:0 (lowrank,globalbatch8,two GPUs,
optimized scan),verified full sample counts and scancelled/removed:
- C1v80 WinoGrande3133631_4:1267samples,acc0.7348066298342542,
  eval199.915s,peak29.843GiB,job3:36.
- C1v64 ARC-Easy3133657_0:2376samples,acc_norm0.8253367003367004,
  eval454.354s,peak56.875GiB,job7:48.
- C1v64 ARC-Challenge3133657_1:1172samples,acc_norm0.5793515358361775,
  eval298.932s,peak44.605GiB,job5:14.
Commands: bash scripts/evaluate_formal_compressed_mcq.sh nemotron
<variant> v_ag_ppl_mcq <task>. All three Nemotron C1 ranks now have both ARC
tasks complete. Formal MCQ51/161;Qwen AG gate remains17 tasks incomplete.

Five further full MCQ results verified,all completed0:0 and scancelled/removed:
- Nemotron MLRDv80 ARC-Challenge3133681_1:1172samples,
  acc_norm0.5938566552901023,eval295.990s,peak44.039GiB,two GPUs,job5:10.
- Nemotron MLRDv80 ARC-Easy3133681_0:2376samples,
  acc_norm0.8341750841750841,eval546.835s,peak58.276GiB,two GPUs,job9:24.
- Nemotron MLRDv64 ARC-Easy3133705_0:2376samples,
  acc_norm0.8202861952861953,eval487.982s,peak55.315GiB,two GPUs,job8:22.
- Nemotron MLRDv64 ARC-Challenge3133705_1:1172samples,
  acc_norm0.5699658703071673,eval239.117s,peak44.529GiB,two GPUs,job4:16.
- Qwen C1v192-only ARC-Challenge3133571_1:1172samples,
  acc_norm0.575938566552901,eval261.655s,peak28.738GiB,one GPU,job4:38.
Commands use evaluate_formal_compressed_mcq.sh with documented family,
variant,mode/task;lowrank/globalbatch8/full protocol unchanged.
Formal MCQ56/161;Qwen AG gate16 remaining.

Following the successful full-task Qwen single-GPU C1 ARC validation,
remaining five pending Qwen C1 ARC tasks3133571_0/_7/_8/_14/_15 changed to
one GPU/two CPUs/MinCPUsNode2,retaining30minutes/96GiB and original commands.
Resource fields verified after updates;other task types retain their allocations.

Three further Nemotron results completed0:0,full samples,lowrank/globalbatch8,
two GPUs,optimized scan;verified/scancelled/absent:
- GLRD4v96 ARC-Easy3133749_0:2376samples,acc_norm0.8425925925925926,
  eval380.773s,peak56.131GiB,job6:35.
- GLRD4v96 ARC-Challenge3133749_1:1172samples,acc_norm0.5938566552901023,
  eval183.299s,peak45.358GiB,job3:17.
- MLRDv64 WinoGrande3133705_4:1267samples,acc0.7269139700078927,
  eval126.603s,peak29.554GiB,job2:21.
Commands: bash scripts/evaluate_formal_compressed_mcq.sh nemotron
<variant> v_ag_ppl_mcq <task>. Formal MCQ59/161.
Qwen C1v192-only ARC-Easy3133571_0 is running with one GPU.

Qwen C1v192 single-GPU ARC-Easy3133571_0 passed the longest initial batches
and exceeded50% requests without OOM (~45requests/s at this observation).
Based on completed single-GPU C1 ARC-Challenge and short-task memory evidence,
three pending Qwen C1 PIQA tasks3133571_3/_10/_17 reduced to one GPU/two CPUs,
MinCPUsNode2,retaining20minutes/96GiB/lowrank/globalbatch8/original commands.
Command: bash scripts/evaluate_formal_compressed_mcq.sh qwen35
c1_twosided_v<192|160|128> v_only piqa. Resource fields verified.

Three Nemotron WinoGrande tasks completed0:0,all1267samples,lowrank/globalbatch8,
two GPUs,optimized scan;verified/scancelled/absent:
- C1v64,3133657_4:acc0.7308602999210734,eval124.522s,peak29.969GiB,job2:18.
- MLRDv80,3133681_4:acc0.739542225730071,eval123.508s,peak30.226GiB,job2:17.
- GLRD4v96,3133749_4:acc0.7529597474348856,eval118.705s,peak30.396GiB,job2:12.
Commands: bash scripts/evaluate_formal_compressed_mcq.sh nemotron <variant>
v_ag_ppl_mcq winogrande.

Qwen C1v192-only single-GPU ARC-Easy3133571_0 completed0:0 in5:49,
all2376samples,eval321.429s,peak34.776GiB,lowrank/globalbatch8.
Command: bash scripts/evaluate_formal_compressed_mcq.sh qwen35
c1_twosided_v192 v_only arc_easy. Full sample/GPU/batch protocol verified;
scancelled. Formal MCQ63/161;Qwen AG gate15 MCQ tasks remain.

Based on completed single-GPU Qwen C1v192 ARC timings4:38/5:49,shortened
four still-pending C1v160/v128 ARC tasks3133571_7/_8/_14/_15 from30 to15minutes.
One GPU/two CPUs/96GiB/globalbatch8/lowrank and existing commands unchanged.
The limits retain >2x measured walltime;on real timeout use existing cache and
resubmit before treating a task as complete. No running job was shortened.

Three further GLRD4 ARC results completed0:0,full samples,lowrank/globalbatch8,
two GPUs,optimized scan;verified/scancelled/absent:
- V80 ARC-Easy3133788_0:2376samples,acc_norm0.8337542087542088,
  eval347.655s,peak56.201GiB,job6:02.
- V64 ARC-Challenge3133810_1:1172samples,acc_norm0.5810580204778157,
  eval180.242s,peak44.795GiB,job3:13.
- V80 ARC-Challenge3133788_1:1172samples,full result verified,job3:17;
  exact metrics retained in results/nemotron_h_8b/v_ag_ppl_mcq/palu_glrd4_v80/mcq_arc_challenge.json.
Commands: bash scripts/evaluate_formal_compressed_mcq.sh nemotron <variant>
v_ag_ppl_mcq <task>. Formal MCQ66/161.
Qwen C1v160 ARC-Easy/Challenge3133571_7/_8 started with one GPU/15minutes.

Three Nemotron GLRD4 tasks completed0:0,full samples,lowrank/globalbatch8,
two GPUs,optimized scan;verified/scancelled/absent:
- V80 WinoGrande3133788_4:1267samples,acc0.7426992896606156,
  eval182.199s,peak30.061GiB,job3:15.
- V64 WinoGrande3133810_4:1267samples,acc0.7348066298342542,
  eval186.596s,peak29.757GiB,job3:21.
- V64 ARC-Easy3133810_0:2376samples,acc_norm0.8207070707070707,
  eval456.546s,peak56.285GiB,job7:50.
Commands: bash scripts/evaluate_formal_compressed_mcq.sh nemotron <variant>
v_ag_ppl_mcq <task>. All nine compressed Nemotron configurations now lack
only HellaSwag among their seven MCQs;dense still has pending tasks.

Qwen C1v160-only ARC-Challenge3133571_8 completed0:0 in5:23,one GPU,
all1172samples;full protocol verified,scancelled. Command:
bash scripts/evaluate_formal_compressed_mcq.sh qwen35 c1_twosided_v160 v_only arc_challenge.
Formal MCQ70/161;Qwen AG gate14 tasks remain (V160 ARC-Easy still running).

Revisited nine compressed Nemotron HellaSwag reservations after optimized
scan and multiple full two-GPU ARC runs succeeded with peaks below59GiB.
Superseding the earlier four-GPU hold plan,all nine pending HellaSwag task2
allocations (3133550,3133614,3133631,3133657,3133681,3133705,3133749,3133788,
3133810) reduced4->2 GPUs,8->4 CPUs including MinCPUsNode,4h->2h.
Commands unchanged: bash scripts/evaluate_formal_compressed_mcq.sh nemotron
<variant> v_ag_ppl_mcq hellaswag,lowrank/globalbatch8/full samples.
All resource fields verified. This is justified resource reduction,not a claim
that HellaSwag has already passed;monitor its own actual peaks/completion.
Dense single-GPU HellaSwag3133468_9 retains1h as separate feasibility run.

Qwen C1v160-only ARC-Easy3133571_7 completed0:0 in8:27,all2376samples,
one GPU,eval492.465s,peak34.751GiB,lowrank/globalbatch8. Full sample/GPU
protocol verified,scancelled. Command: bash scripts/evaluate_formal_compressed_mcq.sh
qwen35 c1_twosided_v160 v_only arc_easy. Formal MCQ71/161;Qwen AG gate13 remain.

Queue check found no running jobs;all remaining work pending. Qwen C1 MCQ
array3133571 still had throttle2 from its original multi-GPU configuration.
Increased ArrayTaskThrottle2->4 to allow more independent now-single-GPU
short tasks to use available slots. Per-task GPU/CPU/memory/commands/time
limits/protocol unchanged;Slurm remains responsible for actual allocation.
No pending task was cancelled or resubmitted merely due to waiting.

Queue coverage audit:98 currently eligible formal MCQ stages (Nemotron70,
Qwen dense7,C1 V-only21) have71 complete files and27 missing files. Every
missing stage maps to a live pending Slurm array task;27 queue tasks total,
zero uncovered stages. The additional63 Qwen V+AG MCQs remain intentionally
unsubmitted until the prescribed full dense/C1 V-only gate and nine AG fits.
Ordinary ghx4 has no fully idle nodes at this observation;remaining jobs report
Priority with no estimated start. Existing pending jobs retained;no duplicate
resubmission or partition change.

Host-memory audit:completed Qwen ARC jobs report Slurm MaxRSS~1.7-4.2GiB,
validated single-GPU Nemotron BoolQ~5.6GiB and succeeded with64GiB request.
These sampled RSS values do not capture every possible peak/cache usage;
retain substantial margin. Reduced all27 still-pending MCQ host-memory
requests96->64GiB using MinMemoryNode=65536 (scontrol requires integer MiB),
verified ReqTRES mem=64G for each. Per-task GPU/CPU/time/commands/batch protocol
unchanged. No running job was modified. AG scripts retain full prerequisite
gate;no AG capture launched before dense/C1 V-only MCQs finish.
