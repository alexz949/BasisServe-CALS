# K routing: 32K fitting to 64K RULER

## Live update: September 12, 2026, 01:57 EDT

Latest: compact ShadowKV 8310220_3 passed both dual-GPU inputs (5m18s,
~141s each). Compact ours 8310220_4 OOMed in prefill on GPU 0, requesting
768 MiB with 504 MiB free. New sources_residual snapshot stores only prompt
Residual16 and shared per-device prompt RoPE tensors, reconstructing prompt
Base K from resident V per layer during decode; generated-token full sidecars
remain cached. This removes ~8 GiB long-prompt Base-K storage. Prompt GEMM
shape and token-major layout are preserved. The first all-token-reconstruction
attempt failed strict equality only on newly generated FP32 tokens (~5e-10
global relative RMSE); its failed diagnostics are 8310232/8310233. Do not use
that abandoned reconstruction scheme.

Revised CPU strict parity 8310235 passed (8s): FP32/BF16 each exactly match
the materialized reference at 65,536/65,537/65,538 tokens across cache append.
GPU strict parity 8310236 is queued. Ours dual-GPU smoke 8310238 depends on
both tests and writes eval64k_residual. The sources_residual protocol is new;
successful sources_compact smokes cannot be silently mixed into its gate.
Nemotron 8310179_6 is running; LRQK GPU-count decision remains unanswered.

Compact-source Qwen Dense 8310220_0 and Exact Sparse 8310220_1 both passed
two dual-GPU 64K smoke inputs, exit 0 (5m30s and 5m29s). Saved protocols,
samples, and first-token agreement were verified for both inputs. ShadowKV
8310220_3 and ours 8310220_4 are now running concurrently on two GPUs each.
Obsolete Qwen gate 8310121, eval array 8310122 and summary 8310067 were
canceled; they referenced the previous failed/protocol-incompatible chain.

GPU old/new Page32 parity 8310219 passed (5s): FP32 and BF16 both had zero
relative output RMSE, identical selected pages, and identical statistics at
the full 65,537-token Qwen geometry. Array 8310220_0 and _1 are now running
the compact-source Dense and Exact Sparse dual-GPU smokes concurrently.
Nemotron 8310179_5 completed all 22 predictions (25m53s, exit 0), bringing
Exact Sparse to 44/88; its remaining arrays are queued behind these smokes.

Full-geometry CPU old/new Page32 parity 8310222 passed (8s, basis): for
65,537 tokens, 64 Query heads, 8 KV heads, V96 and routing rank 144, both
FP32 and BF16 had zero relative output RMSE, identical selected pages, and
identical statistics. GPU parity 8310219 remains pending; do not treat the
CPU comparison as evidence of GPU parity or full-model memory sufficiency.

Newest Qwen state: 8310210 passed all three 64K full-decode tests and both
Dense dual-GPU smoke inputs (5m57s total, ~131s per input including repeat).
Results remain in smoke_2gpu_chunk with sources_2gpu. 8310211 exact_sparse
failed expanding complete FP32 GQA V; shadowkv and ours failed allocating
prefill output on GPU 1. A separate sources_compact snapshot now avoids
complete K/V/sidecar head expansion: grouped proxy GEMM and direct selected
token gathers retain the same Page32 policy and statistics. Its two-GPU
map uses 33/31 layers, and ShadowKV frees pre-RoPE K after state construction.
CPU tests 8310218 passed 4/4. GPU 64K old/new parity diagnostic 8310219 is
queued; four-arm dual-GPU smoke array 8310220 depends on that diagnostic,
using eval64k_2gpu.
LRQK is excluded from that array pending the user's resource decision.

Later verified state: Nemotron 8310179_[0-3] all completed, yielding all 88
Dense predictions with matching protocols and unique sample IDs. Eleven-task
mean is 66.28787879%; task percentages in order single1/2/3, multikey1/2,
multiquery, multivalue, vt, fwe, qa1/2 are
[100,100,100,62.5,0,90.625,71.875,0,91.66666667,50,62.5]. This baseline
includes both C1 V and folded compressed Mamba Wo. Do not attribute its
failures to K routing. Samples 32/33 show repeated digits or continuation of
unrelated question-answer text; vt 56/57 emit wrong variable names and then
continue question text. Finish the other arms before comparative conclusions.

Qwen GPU diagnostic 8310184 confirmed GC frees 5.625 GiB total: allocation
changed from [31.1101,34.1605] to [28.3855,31.2601] GiB. Prefill completed,
but Dense decode then OOMed in PyTorch SDPA GQA while allocating 2 GiB.
Frozen sources_2gpu now use the existing compressed-V Triton decode kernel
for full attention; prior source is archived in sources/updates/decode_before.
New 8310210 runs 64K GQA parity tests then Dense dual-GPU smoke; 8310211
runs exact_sparse/shadowkv/ours dual-GPU smokes, at most two concurrently.
These jobs have not yet passed. Qwen LRQK resource choice remains unanswered.

CPU module-lifetime diagnostic 8310186 passed (12s, basis): with actual
Accelerate dispatch hooks, displaced attention modules and their old V/O
weights remained alive before explicit gc.collect() and all weak references
cleared afterward. The frozen Qwen sources_2gpu evaluator now collects once
after C1 installation; the previous evaluator is preserved under
sources/updates/gc_before/evaluation/. GPU diagnostic 8310184 remains queued
and will measure the actual before/after allocation inside its install wrapper.
Nemotron formal sample 0 completed with score 1.0 in 76.015s; one sample does
not establish aggregate quality.

Subsequent verified progress: 8310176 completed all four remaining arms;
8310000 passed all five-arm smoke checks (exit 0, 27s). Formal Nemotron
8310179_0 is now running. Array nice was set to 100 so short queued Qwen
diagnostics can obtain GPUs between formal shards without interrupting a shard.
Qwen diagnostic 8310178 failed at layer 56: GPU 0 had 37.721 GiB allocated
while GPU 1 ran out allocating a 1,022 MiB attention output. The automatic
device map placed layers 0-30 on GPU 0 and 31-63 on GPU 1; actual C1 parameter
sizes were 28.373 and 31.245 GiB. Pre-layer-0 allocations were 31.735 and
34.192 GiB. Uncollected displaced attention modules are a hypothesis for
part of the excess allocation, not yet proven. Diagnostic 8310184 repeats
the same Dense smoke with explicit post-install gc.collect() and prints
before/after allocated bytes. It is pending. Its diagnostic wrapper is not
yet a production evaluator change; any successful fix must be incorporated
and the resulting final evaluator protocol validated before Qwen formal runs.

Nemotron native 64K Dense smoke 8310165 completed successfully (exit 0,
5m40s): both 65,389-token and 65,477-token inputs completed. Four-token
smoke scores are not quality measurements. The remaining four arms run in
8310176_[1-4%1]; gate 8310000 now depends on this array. Replacement formal
array 8310179_[0-19%1] depends on the gate and enables expandable CUDA
segments for every arm. Summary 8310002 depends on 8310179. Obsolete held
formal array 8310001 was canceled. Do not edit native evaluator sources
while this smoke/evaluation chain is active.

Qwen two-GPU smoke 8310169 failed in all five arms. Static BF16 storage
accounting from actual checkpoint headers and the V-rank manifest gives
59.618 GiB C1 model weights, 14 GiB K/V at 65,536 tokens, and 16 GiB LRQK
AK state. Thus LRQK needs at least 89.618 GiB before temporaries, exceeding
two L40S cards' roughly 89.04 GiB available capacity. Dense K's corresponding
lower bound is 73.618 GiB; our materialized base/residual sidecar adds 9 GiB.
These are storage lower bounds, not measured allocator peaks. A user question
is pending about allowing four GPUs for LRQK versus CPU offload on two GPUs.
Do not silently change that resource choice. Dense two-GPU memory tracing
smoke 8310178 uses evaluation/diagnose_qwen_memory.py and the unchanged frozen
sources_2gpu evaluator. Native BF16 LRQK state / FP32 solves remain in effect.

The current user request supersedes the earlier fitting-only stop condition.
Start with Llama-3.1-8B, frozen formal Two-Sided-KL C1-V96. Compare full exact
K, exact sparse Page32, LRQK, ShadowKV, and our Base16/Residual16 router on the
same 11 RULER tasks, eight samples per task. Include all 88 samples. Do not
exclude historical sample 86. This is an 11-task pilot, not the full 13-task suite.

## Decision and remaining model scope

If our 88-sample score is within four percentage points of full K with the same
V96, do not refit at 64K; proceed to Qwen3-32B and
`nvidia/Nemotron-H-47B-Reasoning-128K`. Otherwise redo calibration and residual
fitting at 64K and repeat the five-arm comparison. Always report per-task scores
and anomalies.

### Latest verified state

Memory/runtime update: Qwen sources_2gpu now uses BF16 LRQK state with
head-blocked FP32 prefill (8 heads, shared stopping decision), and tokenwise
MLP plus RMSNorm chunks1024. Helpers live in evaluation/lrqk_chunked_prefill.py
and evaluation/chunked_prefill_mlp.py. CPU8310134 passed factor/selection/
stopping and MLP comparisons;8310146 passed MLP and exact RMSNorm comparisons.
GPU8310137 with MLP-only chunking failed in Q RMSNorm (2GiB allocation).
Before-norm sources are preserved under Qwen sources/updates/norm_before.
Replacement five-arm two-GPU smoke8310148 uses sources_2gpu and output
smoke_2gpu_chunk. Formal/summary dependencies still need rebuilding after
successful new smoke; historical three Qwen full predictions remain untouched.

Native smoke8309999 arms0/1/2 failed illegal CUDA access; pending3/4 canceled.
Minimal reproduction8310141: contiguous dt passes32768 and65389, strided dt
passes32768 but fails65389 (maximum offset2427202815, sequence stride37120).
This isolates scan offset overflow; evaluation/nemotron_h_scan_layout.py now
packs dt before calling the native scan. Existing device guard remains unchanged.
Unstarted sync-only diagnostic8310140 canceled; full-model fixed smoke8310149
uses basis, native isolated dependencies, CUDA_LAUNCH_BLOCKING=1,4L40S/96GiB.
Its complete success is still unverified. Native remaining five-arm gate and
formal dependencies must be rebuilt after the fixed smoke passes.

LRQK precision correction requested by user: upstream pinned lrqk_attention.py
lines784-801 and818-840 compute factors in FP32 then cast outputs to input
dtype. The FP32RoutingState wrapper was our FP16/V100 overflow workaround,
not an upstream requirement for BF16 inference. Working native evaluator and
new Qwen sources_bf16 now use the original LRQKState: BF16 stored factors/codes,
FP32 solves. Historical FP32 records are preserved and must be labeled separately.
GPU job8310131 runs existing LRQK tests and two-L40S LRQK smoke to smoke_bf16,
afterany8310125; native smoke8309999 now follows afterany8310131.
Original two-GPU smoke8310118 failed all five arms. Retry8310125 enables
expandable_segments for every arm, but Dense/Exact still OOM at the unchunked
64K MLP; LRQK FP32 also OOMs in factor fitting. Two-GPU formal8310122 remains
gated and cannot run while the previous smoke fails. Its precision/protocol
and summary dependency must be replaced after the new BF16/memory tests.

September12 scheduling update: native capture8309996 completed all four
shards in16:02. Native K fit8309997 completed all five attention layers
(longest shard9:19); audit8309998 passed in1:32, retaining PCG tolerance
warnings. Qwen formal8310066 produced full-arm samples0,4,8, each score1,
before cancellation at the user's request to use two GPUs per question and
two concurrent workers. The existing results remain in eval64k_json.
Two-GPU five-arm smoke8310118 uses the unchanged sources_json evaluator,
two L40S and64GiB host RAM per worker, array concurrency2, output smoke_2gpu.
CPU gate8310121 follows success; formal8310122 (20 shards, concurrency2,
two L40S/64GiB each) follows gate success and resumes eval64k_json.
Summary8310067 now depends on8310122. Native smoke8309999 waits afterany
8310118 to permit this resource test; both native Wo and K audits have passed.
Two-GPU feasibility is still being tested, not yet established.
User also authorized using available A100/H200 GPUs for suitable pending work;
the last check found all healthy A100/H200 GPUs allocated.

September 12 update: FP64 Wo smoke8310070 passed, and formal8309994
completed all45 Mamba Wo layers with12 sweeps. Audit8309995 failed because
install_nemotron_h_wo.py still required float32 work precision. Its check now
requires float64; BF16 factors are unchanged. Before/after sources are retained
under sources/updates/wo_audit_fp64_before and wo_audit_fp64. Replacement
CPU audit8310110 uses basis,2 CPUs,8GiB,30 minutes; smoke8309999 now depends
on K audit8309998 and Wo audit8310110. Native capture8309996 has all four
shards running. Corrected Qwen five-arm smoke gate8310065 passed; formal
8310066 is waiting for resources. Earlier status entries below are historical.

### Next experiment requested by the user

After the current block, prioritize Qwen3.5 V192 K-router fitting, without
initial RULER evaluation. Primary Base16+Residual16 and diagnostic capacity
control Base32+Residual32 share64x32K fitting and16x32K diagnostic windows,
Q64, Page32,40 BCD sweeps and PCG100. Finish calibration before fitting.
Plot diagnostic relative MSE against actual zero-indexed full-attention layers
3,7,11,15,19,23,27,31 alongside Qwen3-8B with matching metric definitions.
Keep fit, held-out diagnostic, and any independently fresh measurements
distinct. Qwen3.5 head dimension256 gives router rank fractions12.5% and25%;
Qwen3 head dimension128 gives25% for B16R16. Compare late-layer trends and
capacity sensitivity; these are observations, not a causal architecture proof.
Confirm the local V192 artifact and comparison protocol before launch.

The user additionally requested attention recall mass. On the same16x32K
diagnostic windows, report dense-teacher attention probability mass captured
by selected pages for B16R16, B32R32, and Exact-K page selection. Match Page32,
token budget, mandatory prefix treatment, causal visibility, query sampling,
and GQA page aggregation across arms and the Qwen3-8B comparison. Plot the
layer curve alongside relative MSE. Recall mass uses the original dense
probabilities, without renormalizing over selected tokens. Exact-K is a
matched routing reference, not necessarily an upper bound for each query
when GQA heads share selected pages. No initial RULER run is required.

Wo audit8310110 subsequently completed successfully in21 seconds, exit0,
verifying all45 layers. The pending native smoke dependency is now healthy.

Native KL8309988 completed in5:03, exit0, selecting ranks32/96/112/112/128
for attention layers17/38/49/60/86 (mean96). Confirmation terminal KL is
0.003452043887 for selected versus0.004834729568 for uniform96; these are
8-window calibration-confirmation values, not RULER quality.
JSON protocol CPU check8310068 passed all ten previous smoke records, requiring
only the intentional evaluator source hash change. Corrected Qwen smoke8310063
has started. Native Wo smoke8309993 failed in15s: the FP32 exact two-sided
solver rejected left minimum eigenvalue6.165805e-11 versus tolerance1.407673e-7.
Working Wo fitter now uses FP64 work precision, retaining the same solver,
objective, covariance damping,12 sweeps, totalrank6144 and BF16 output.
Original and changed sources are archived with source_snapshot_wo_fp64.json.
Replacement same-output smoke8310070 is queued,1 L40S/16GiB/2 CPUs/1h,basis;
formal Wo8309994 now requires afterok8310070. Full-size FP64 memory/runtime
and numerical success remain to be verified before formal fitting starts.

Qwen audit8309226 completed all64 layers in19:30, exit0. Ours64K smoke8309234
completed both inputs in6:09, exit0 (132.17/133.17 seconds); these are4-token
smokes, not quality results. Gate8309235 failed because runtime config
id2label has integer keys in memory and string keys after JSON persistence.
All five arms' saved protocols match each other. The evaluator now canonicalizes
the complete protocol with strict JSON roundtrip before returning it. Original
Qwen sources and eval64k records are preserved; corrected sources are in
sources_json, with manifest source_snapshot_json.json. New smoke8310063_0–4
waits afterany native KL8309988; gate8310065, formal8310066_0–19 and
summary8310067 follow success dependencies, output eval64k_json. Original
unstarted formal8309420 and summary8309237 were canceled. CPU check8310068
checks canonical roundtrip and all ten prior smoke protocol/sample/bank records,
requiring the evaluator source hash to be the only intentional difference.
Working native evaluator received the same fix before native smokes begin;
previous/current versions are archived and source_snapshot_json_protocol.json
records the change. No active native KL source was modified.
Native V bank8309983 and all seven merges8309984 completed successfully;
native KL8309988 is running with finite measured probe KL values so far.

Qwen formal K fitting8309981 completed all64 layers: all four shards exited0,
with elapsed58:17,59:18,57:48,59:36. Full-layer audit8309226 is running.
Full80-window reader validation8310017 completed in7:28, exit0, peak Slurm
MaxRSS33556044KiB (approximately32GiB). Both layers0 and4 reported
`BITWISE FIT REFERENCE MATCH`; the comparison covers every saved factor tensor.
This establishes two-layer full-size equivalence, not every-layer memory bounds.
Native V bank8309983 has started. Pending native K fit8309997 now requests
36GiB per worker and four concurrent workers (144GiB total), retaining2 CPUs,
one L40S each and its original fitting command. Its capture-success dependency
remains intact. The extra4GiB per worker provides headroom over this validation;
monitor actual native memory and preserve the node parent151GiB limit.

After discussing calibration memory with the user, the working K capture
reader was changed to load one V/K window at a time using safetensors slices.
Candidate Q remains materialized for the unchanged fit-only query selection;
Base accumulation, Fisher construction and40-sweep residual solver are unchanged.
`evaluation/k_routing_capture_windows.py` owns the reader and clones each
returned window so it cannot modify or retain the whole source mapping.
The working fitter records the loading mode and new helper hash. Active Qwen
jobs still execute their old frozen sources; no running fit was modified.

CPU test8310015 passed5 tests in11.67s. L40S smoke8310016 COMPLETED in1:33
with `BITWISE FIT REFERENCE MATCH 63` against the existing Qwen smoke bank;
saved losses (all40 sweeps), reconstruction metrics and numerical smoke also
match exactly. Its output is `qwen3_32b/smoke/window_fit`. This is a1-fit/1-diag
smoke and does not establish full80-window memory usage.
Full-size validation8310017 is queued afterok8309981 and8310016:1 L40S,
32GiB RAM,2 CPUs,basis, testing formal layers0 and4 across a layer boundary
against the frozen formal factors. The command is
`python -u evaluation/fit_k_routing_captures.py --identity
results/k_routing_fit/qwen3_32b/manifests/v96.json --windows
results/k_routing_fit/qwen3_32b/calibration/windows.safetensors --captures
results/k_routing_fit/qwen3_32b/captures --output
results/k_routing_fit/qwen3_32b/smoke/window_full --layers 0,4 --num-shards 1
--reference-bank results/k_routing_fit/qwen3_32b/ours_b16r16`.
Native V bank8309983 now waits afterany8310017 as well as8309981 for resources.
Native K fit8309997 additionally requires afterok8310016 and8310017; its
64GiB/two-worker allocation has not yet been reduced or increased in concurrency.
The changed sources/tests are archived under native `sources/updates/window_reader`,
with `manifests/source_snapshot_window_reader.json`; the previous archive is
preserved. Verify full-size peak memory and exact factor agreement before
raising native fitting concurrency.

Qwen fit8309981_0 and_1 COMPLETED exit0 in58:17 and59:18 respectively.
Their32 layer factors cover IDs0mod4 and1mod4, including60/61. A basis
read-only metadata check verified the32 factor SHA256 values, shared protocol,
40 saved sweeps and finite saved residual metrics; all32 final PCG residuals
remain above1e-5. This does not replace the final all-layer tensor/capture audit.
Slurm automatically started8309981_2 and_3 after the first two completed.
The overall fit is32/64 layers complete; the second pair covers IDs2mod4 and
3mod4. No new OOM occurred in either completed64GiB worker. All native and
formal Qwen evaluation jobs remain queued on the previously documented chain.

Qwen fit8309981_0/_1 remain RUNNING at22:06 elapsed, with12 completed
layers0/1/4/5/8/9/12/13/16/17/20/21. A basis standard-library read-only check
verified their factor SHA256 values, shared protocol/frozen V96 binding,
per-layer ranks,40 saved sweeps, and finite nonnegative residual metrics.
Diagnostic Page-Fisher NMSE spans0.0787618–0.5021799. All12 final query PCG
residuals exceed1e-5; retain the warning and fixed40-sweep endpoint. This
partial check does not replace the queued all64-layer tensor/capture audit.
Slurm current RSS dropped to55555520/57980044KiB after layers20/21 completed,
despite earlier peaks near64GiB; no terminal failure or restart is indicated.

Native RULER data8309992 COMPLETED in1:57: all11 tasks/eight samples,
65536 total-token cap, seed42, prompt margin8, pinned RULER revision
c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a. Its manifest is under
`results/k_routing_fit/nemotron_h_47b/ruler64k/manifest.json`.
The remaining native pipeline is now queued (not yet executed):

| Job | Stage | Resources / dependencies |
|---|---|---|
| 8309993 | Real-size Mamba Wo smoke, layer0 | 1 L40S,16GiB; afterok V96 KL8309988 |
| 8309994_0–3 | All45 Wo layers | 1 L40S/16GiB each,4 concurrent; afterok8309993 |
| 8309995 | All45 Wo artifact audit | CPU8GiB; afterok8309994 |
| 8309996_0–3 | Dense80×32K K captures | 1 L40S/32GiB each,4 concurrent; afterok8309988,afterany8309994 for resources |
| 8309997_0–3 | Base16/Residual16 K fitting | 1 L40S/64GiB each,2 concurrent; afterok8309996 |
| 8309998 | Native5-attention-layer K audit | CPU8GiB; afterok8309997 |
| 8309999_0–4 | Five64K arm smokes | 4 L40S/96GiB,1 replica; afterok8309998 and8309995 |
| 8310000 | Five-arm smoke gate | CPU8GiB; afterok8309999 |
| 8310001_0–19 | Five arms × four22-prompt shards | 4 L40S/96GiB,1 replica; afterok8310000 |
| 8310002 | Verify all440 predictions and summarize | CPU8GiB; afterok8310001 |

All jobs use basis and2 CPUs. Native model capture/evaluation additionally use
the isolated `results/tools/nemotron_deps` PYTHONPATH. LRQK GPU jobs export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Each job logs to its
`logs/nh-*` name with job/array ID. The Wo fitter now accepts an explicit
`--layers 0` subset for smoke; formal shards still cover all45 layers. Smoke
outputs are separate under `smoke/wo`, with the same12-sweep,TP4,totalrank6144
protocol as formal `wo`. The new `evaluation/audit_nemotron_h_wo.py` verifies
all45 factor hashes/shapes/BF16 finiteness, exact source covariance hashes,
256/64×2048 split,12 completed sweeps, selected-sweep bounds, finite stored
metrics, and disjoint complete shard coverage. Native RULER now requires its
successful `manifests/wo_audit.json` and binds that audit into every arm.
This is an artifact audit, not a new full-size independent numerical solve.

545 current Python sources were copied and hash-verified into native `sources/`,
with `manifests/source_snapshot.json`. Queued native jobs still execute the
working-tree paths recorded in their batch commands; the archive preserves
the current bytes. If a required source changes before execution, preserve
and document that new version rather than claiming the archive is unchanged.
Qwen's existing frozen source tree is untouched. Its two fit workers remain
RUNNING at19:46 elapsed; layer exports continue and no terminal OOM is observed.

Native evaluator integration tests8309989 passed7 tests; follow-up8309991
passed15 tests in11.97s under basis on CPU. The latter includes an actual small
Nemotron mixed Mamba/attention/MLP model: identity V payload installation and
the evaluator's greedy multi-step generation agree with the unmodified native
model, exercising the hybrid decode mask and recurrent cache. It also checks
folded Wo installation against explicit source-factor execution. These tests
do not validate the full47B model with compressed Wo or64K inputs.
The working evaluator now requires native audit/full-model smoke/Wo bank,
binds their hashes into the common five-arm protocol, verifies fast Mamba
bindings, installs device guards and the same folded Wo factors for every arm,
and records infinite native time_step_limit as a JSON string. Qwen's active
frozen sources remain unchanged.

Native64K RULER data preparation8309992 is submitted on CPU, basis2 CPUs/16GiB.
Command: `python -u evaluation/prepare_qwen3_ruler_v1.py --ruler-root
results/tools/RULER --tokenizer-path
/deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--nvidia--Nemotron-H-47B-Reasoning-128K/snapshots/18c2a0e52e2d028dd96c3b4252af2a4f8fa54a43
--max-seq-length 65536 --num-samples 8 --random-seed 42 --workers 2 --prompt-margin 8
--tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2
--output-dir results/k_routing_fit/nemotron_h_47b/ruler64k`.
Qwen fit8309981_0/_1 remain running past13 minutes, with layers0/1/4/5/8/9
exported. Peak RSS has approached64GiB per worker, so memory headroom needs
monitoring; no terminal OOM has been observed for these live jobs.

Qwen formal layer0 fit completed40 sweeps and exported factors. Its residual
Page-Fisher NMSE is0.3188561 fit/0.5021799 diagnostic, with final maximum PCG
relative residual0.00231257 (>1e-5). Retain this warning under the fixed40-sweep
protocol; do not silently change fitting settings or interpret it as RULER
quality. Worker0 has moved to layer4, while worker1 is completing its first
layer. Full64-layer fitting and audit remain incomplete.

Formal covariance audit8309982 COMPLETED exit0 in3:25, verifying all50 formal
artifacts. Qwen fit8309981_0/_1 are in Page-Fisher sweeps with decreasing losses;
Slurm reports peak RSS53352340/53354544KiB (~50.9GiB each), explaining the
previous32GiB worker OOM. Current64GiB/two-worker allocation remains live.
Native KL allocation8309988 is queued afterok V bank merge8309984 and
afterany Qwen ours smoke8309234,4 L40S/96GiB/2 CPUs. It runs
`python -u evaluation/allocate_nemotron_h_v96.py --audit
results/k_routing_fit/nemotron_h_47b/manifests/native_audit.json --full-smoke
results/k_routing_fit/nemotron_h_47b/manifests/full_smoke.json --windows
results/k_routing_fit/nemotron_h_47b/vcal/windows.safetensors --snapshots
results/k_routing_fit/nemotron_h_47b/snapshots/full_attention --bank
results/k_routing_fit/nemotron_h_47b/vbank --output
results/k_routing_fit/nemotron_h_47b/v96 --identity-output
results/k_routing_fit/nemotron_h_47b/manifests/v96.json` in basis+isolated deps.
The allocator now explicitly imports native fast dependencies inside main and
checks their function bindings/native implementation hash against the audit.
Qwen formal array8309420 additionally waits afterany8309988 for GPU ordering;
its afterok five-arm smoke gate remains intact. No native KL measurements yet.

Native32K capture audit8309987 passed all5 attention artifacts17/38/49/60/86:
shared protocol/window IDs[0,64], tensor SHA256, expected shapes/BF16 dtype,
finite tensors, and saved post-K equal pre-RoPE K bitwise (no native RoPE).
The inline basis Python validation is preserved in the submitted Slurm batch
script and its `logs/nh-k-capture-audit-8309987.out` output. This is artifact
and invariant validation, not an independent full-teacher activation reference.
Qwen fit8309981_0/_1 remain running beyond2 minutes, with no new OOM reported.
Formal covariance audit8309982 has reached Mamba85 and is still running.

Native32K capture retry8309980 completed through attention86 and wrote
`smoke/capture/complete_0.json`; Qwen fit8309981_0 and_1 have now started on
L40S, with_2/_3 held by the two-worker concurrency limit. Native capture
tensor contents/hashes still require post-run audit before claiming the
complete capture smoke verified.

Formal covariance audit8309982 is running on CPU (basis,2 CPUs/12GiB):
`python -u evaluation/audit_nemotron_h_covariances.py --audit
results/k_routing_fit/nemotron_h_47b/manifests/native_audit.json --snapshots
results/k_routing_fit/nemotron_h_47b/snapshots --output
results/k_routing_fit/nemotron_h_47b/manifests/covariance_audit.json`.
It verifies all50 hashes, shapes, dtypes, finite tensors and320-window splits.
Native V bank8309983_[0-27%4] is queued afterok8309982 and afterany8309981.
Each worker uses1 L40S/16GiB/2 CPUs. Task rank is
`[32,48,64,80,96,112,128][id//4]`, layer shard is`id%4`, four shards/rank.
It uses the passed native fitter/ALS12/CG16 settings,256 fit/64 heldout,
FP32 work/BF16 factors, full-layer decoder objective, and outputs`vbank/r{rank}`.
Merge8309984_[0-6%2] waits afterok8309983 and runs the native fitter's CPU
`merge --audit .../manifests/native_audit.json --output-dir .../vbank/r{rank}`.
Qwen ours smoke8309234 now also waits afterany8309983 for GPU ordering,
retaining afterok Qwen fit audit8309226. Full native bank fitting has not run.

Latest terminal jobs: ShadowKV8309233 PASSED both repeated64K smoke samples
in5:24. Formal Nemotron covariance8309409 COMPLETED exit0 in19:35, all50 targets
saved; a full independent artifact hash audit remains to be run. Native V96
layer17 fit smoke8309415 COMPLETED exit0 in45s, heldout relative MSE0.048299172.
Qwen fit8309225_0–3 were all OOM-killed after2:00 (host memory). Retry8309981
keeps identical frozen fit command/data/output and four layer shards, but runs
at most2 workers,64GiB RAM/1 L40S/2 CPUs each. It waits afterany8309980.
Fit audit8309226 now waits afterok8309981; ours smoke8309234 waits afterok8309226
(the Native V-fit smoke resource wait is already satisfied and removed).
Native32K capture smoke8309414 failed while writing the first attention-layer
JSON: native config time_step_limit contains positive infinity, rejected by
strict JSON. All preceding block finite checks passed. The metadata now records
that unbounded limit as string `inf`, leaving actual config/math unchanged.
Retry8309980 uses the identical capture command and output with this fix,
1 L40S/16GiB/2 CPUs. Its uncommitted first-layer tensor artifact is retained;
the retry will regenerate the same capture and finish its manifest.

LRQK retry8309417 COMPLETED exit0 in5:36. Both64K samples passed repeated
generation/first-logit equality in148.795s/150.571s using expandable segments.
ShadowKV8309233 is now RUNNING. Formal Qwen array8309420 replaces never-started
array8309236 (cancelled after replacement was submitted). The same20 tasks,
four-GPU allocations, arm/sample mapping and afterok8309235 gate remain;
only LRQK tasks additionally export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, matching their passed smoke.
All array tasks export `CONDA_DEFAULT_ENV=basis` and `OMP_NUM_THREADS=2`.
The frozen Python evaluate command is unchanged. Summary8309237 now waits
afterok8309420. The submitted batch script was read back to verify the mapping
and allocator setting. No formal Qwen evaluation has started yet.

LRQK retry8309417 completed its first64K smoke sample with repeated equality
in148.795s; second sample is running. Expandable segments avoided the first
sample's prior OOM, but the whole smoke has not finished yet.
`evaluation/install_nemotron_h_wo.py` is prepared to validate all45 Mamba
factor records, common V96 identity/audit, factor hashes and the matched6144
AllGather width, then fold BF16 E/D factors through FP32 products into BF16
dense Wo for paired quality evaluation. This reuses the existing core folding
routine; it does not benchmark an actual collective or compress recurrent
state. Syntax checked; no45-layer Wo bank exists yet and installation has not
been exercised on the47B model. All five native evaluation arms must use this
same Wo approximation once the bank is ready.

Working-tree LRQK/ShadowKV installers now use `c1_attention_layers` to select
native Nemotron full-attention mixers by actual index, preserving recurrent
and MLP blocks. CPU job8309418 passed9 tests in11.50s in basis:
`python -m pytest -q tests/test_c1_hybrid_routing_install.py
tests/test_nemotron_h_c1_attention.py tests/test_c1_lrqk.py`.
The new native meta-model checks verify only attention forwards change,
identity-position hooks survive, and cache layer types match native DynamicCache.
Existing native attention/cache and LRQK numeric tests also passed. One warning
is a pinned upstream invalid escape sequence. This is installation validation,
not native47B routed decoding. Frozen Qwen sources are unchanged.
LRQK retry8309417 remains running on its first64K smoke sample without a new
reported OOM at the latest poll.

Qwen LRQK smoke8309232 FAILED exit1 in1:01: MLP allocation3.12GiB OOM on
GPU0,31.87GiB allocated and10.94GiB reserved-but-unallocated (1.21GiB free).
Retry8309417 keeps4 L40S/64GiB/2 CPUs and the exact frozen LRQK smoke command,
adding `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to address possible
fragmentation. FP32 routing state, rank/budgets, inputs and output location
are unchanged. ShadowKV8309233 now waits afterok8309417. If this allocator
setting is needed, propagate it to the queued formal LRQK evaluation launch
before that array starts; current formal array8309236 does not yet include it.
No successful LRQK smoke result exists yet.

Qwen exact-sparse smoke8309231 COMPLETED exit0 in6:01. Both64K samples passed
repeated generated-token and first-logit equality (136.243s/136.546s).
Qwen LRQK smoke8309232 is now RUNNING; ShadowKV8309233 follows it.
Smoke scores retain the4-token cap and do not substitute for the formal88
sample results per arm.

Native V fitting smoke8309415 is queued afterok formal covariance8309409 and
afterany Qwen fit8309225,1 L40S/16GiB/2 CPUs/1h. Basis command:
`python -u evaluation/fit_nemotron_h_c1_joint.py fit-shard --audit
results/k_routing_fit/nemotron_h_47b/manifests/native_audit.json --snapshot-dir
results/k_routing_fit/nemotron_h_47b/snapshots/full_attention --output-dir
results/k_routing_fit/nemotron_h_47b/smoke/vfit --layers 17 --layer-shard-index 0
--layer-shard-count 1 --cache-rank 96 --fit-windows 256 --validation-windows 64
--work-dtype float32 --factor-dtype bfloat16 --decoder-objective full_layer
--encoder-cg-mode fixed --encoder-cg-fixed-iterations 16 --device cuda:0
--torch-num-threads 2 --resume`. Native profile fixes ALS at12 sweeps.
It tests the real covariance fitting path before the full seven-rank bank;
it has not run yet. Qwen ours smoke8309234 now also waits afterany8309415 for
resource ordering, retaining afterok fit audit8309226.

Working-tree `capture_k_routing.py` now supports native Nemotron: audited
checkpoint key mapping, native embeddings/mixed blocks, no Q/K normalization
or RoPE, attention-only captures, and per-block weight-device scopes. It checks
the passed full-model smoke and actual native fast-kernel bindings. Frozen
Qwen sources remain unchanged. Native two-window32K capture smoke8309414 is
queued afterany8309233 on1 L40S/16GiB/2 CPUs. It uses
`python -u evaluation/capture_k_routing.py --identity
results/k_routing_fit/nemotron_h_47b/smoke/capture_identity.json --native-audit
results/k_routing_fit/nemotron_h_47b/manifests/native_audit.json --full-smoke
results/k_routing_fit/nemotron_h_47b/manifests/full_smoke.json --windows
results/k_routing_fit/nemotron_h_47b/calibration/windows.safetensors --output
results/k_routing_fit/nemotron_h_47b/smoke/capture --rope native --num-shards 1
--fit-count 1 --diagnostic-count 1`, basis with isolated Nemotron PYTHONPATH.
The identity is explicitly geometry-only for dense capture smoke; it is not a
V96 bank. Capture syntax checked; native long-window GPU execution pending.
Formal cov8309409 now waits afterok8309233 and afterany8309414 for resources;
its covariance-smoke prerequisite8309159 has already passed and been audited.

Nemotron covariance smoke8309159 COMPLETED exit0 in2:03, capturing fit window0
and heldout256 at128 tokens for full-attention17 and Mamba0. Post-run basis
verification checked both artifact hashes, window counts and covariance/weight
shapes: attention8192×8192 covariance, Mamba16384×16384 covariance, weights
8192×input-width. Files are under `nemotron_h_47b/snapshots_smoke`; capture
itself asserted finite hidden states and saved tensors. This validates the
small capture path, not formal320-window/all50-target memory usage.
Qwen exact-sparse smoke8309231 is now RUNNING. Formal covariance8309409 still
waits for the remaining Qwen baseline smoke chain8309233.

Qwen full smoke8309408 COMPLETED exit0 in5:49. Both64K samples passed repeated
generation/first-logit equality: sample0 took130.847s, sample56 took131.520s.
Thus corrected prefill device handling is now validated by this full-model
smoke, as well as the15 kernel tests. The4-token smoke caps are not formal
RULER quality measurements. Nemotron covariance smoke8309159 is now RUNNING;
Qwen exact-sparse8309231 waits for it to finish. Formal Qwen fit/evaluation and
Nemotron covariance/fitting remain incomplete.

Qwen corrected full smoke8309408 completed sample0 with repeated output/first
logit equality in130.847s, peak33.3444GiB/device; sample56 is now running.
The earlier cudaFree stack was a transient wait, not proof of permanent hang.
Sample0's smoke score0 uses a4-token cap and is not a formal quality result.
Formal Nemotron dense covariance8309409 is queued afterok8309159 and8309233,
four L40S/128GiB/2 CPUs/8h. Command in basis+isolated Nemotron deps:
`python -u evaluation/capture_nemotron_h_covariances.py
--audit results/k_routing_fit/nemotron_h_47b/manifests/native_audit.json
--full-smoke results/k_routing_fit/nemotron_h_47b/manifests/full_smoke.json
--windows results/k_routing_fit/nemotron_h_47b/vcal/windows.safetensors
--output results/k_routing_fit/nemotron_h_47b/snapshots
--fit-windows 256 --heldout-windows 64 --sequence-length 2048`.
It captures all5 full-attention and45 Mamba Wo targets. Qwen fit8309225 now
also waits afterany8309409 for resource ordering; numerical settings and the
afterok baseline gate are unchanged. The completed capture8309215 gate was
removed from the update because Slurm had purged that finished job from its
active dependency records (initial update rejected). Capture completion was
already verified, and the fitter independently checks all capture manifests.
No formal covariances exist yet.

Qwen corrected full smoke8309408 is still running beyond3 minutes on its
first sample. A fresh GDB sample (detached successfully) now shows
`CUDACachingAllocator::malloc -> release_cached_blocks -> cudaFree` inside the
CUDA driver, rather than the previous Triton loadBinary stack. Evidence:
`logs/q3-full-smoke-8309408-stack.txt`. The prefill guard passed unit tests but
has not yet established successful Qwen full-model execution. This observation
does not prove an OOM or terminal failure; continue tracking the existing job.

Nemotron full smoke8309407 PASSED: all577 checkpoint tensors verified, all
block outputs finite,129/2049-token full and cached logits finite. Full versus
cached relative RMSE is0.015951233/0.013202986 (<0.05); argmax matches at both
lengths. Peak memory is at most23.3941GiB/device. The same BF16 model with the
Mamba weight-device guard resolves the previous nonfinite failure in this
smoke; this does not establish64K quality. Artifact: `manifests/full_smoke.json`.
The Nemotron KL allocator now installs the same guard and validates its hash
against this smoke, with helper source included in its protocol hashes.
No KL allocation has run yet. Qwen full smoke8309408 and covariance smoke
8309159 remain the next GPU jobs in the recorded dependency order.

Qwen full smoke8309223 ended TIMEOUT at30:05 with no completed sample.
Device tests8309370 completed15 passed in4.44s, including non-default tensor
device/stream cases. Frozen Qwen prefill now includes the tested device guard;
its old source is retained in `sources/history/`, and the previous manifest in
`manifests/source_snapshot_before_device_fix.json`; current manifest570files.
No successful Qwen eval artifacts existed when this change was made.
Nemotron diagnostic8309362 completed: GPU1 tensors/currentGPU0 are finite but
relative RMSE0.042357169 versusGPU0; GPU1 tensors/currentGPU1 match GPU0 with
zero RMSE. This demonstrates a device-context correctness defect, without yet
establishing full-model recovery. `evaluation/nemotron_h_runtime.py` now scopes
native Mamba forwards to their weight device, preserving dtype and arithmetic.
Full Nemotron smoke8309407 is running, same4GPU basis+isolated-deps command as
earlier with the new helper. Qwen full smoke8309408 follows afterany8309407,
same command/settings as8309223 with corrected frozen prefill. Covariance
smoke8309159 now waits afterok8309407 and afterany8309408 and applies/verifies
the same Mamba guard. Qwen exact-sparse8309231 waits afterok8309408 and
afterany8309159; remaining baseline/fit/eval chain is unchanged.

Local Triton3.2 inspection confirms JIT run uses the current CUDA device and
its current stream. The working-tree compressed-V prefill entry lacked the
device guard already present in decode; it now launches inside
`torch.cuda.device(query.device)`. The running frozen Qwen kernel is unchanged.
Two-L40S test8309370 waits afterany8309223 and can share the node with two-GPU
Nemotron diagnostic8309362. Basis command:
`python -m pytest -q tests/test_compressed_v_decode_attention.py -k
'prefill_uses_tensor_device or flex_prefill_matches'`.
The new tests place tensors on device1 and the caller on device0, use a
non-default device1 stream, check device restoration and compare ranks32/96/128
against causal reference attention; existing small prefill cases also run.
This correction has not yet passed GPU tests or resolved the full-model delay.
Qwen exact-sparse smoke8309231 additionally waits afterany8309370 for resources.

Qwen full smoke8309223 remains running after15 minutes on its first sample;
the completed Llama counterpart took32.66 seconds per repeated smoke sample.
Read-only Slurm process inspection found the Python process3352608 consuming
one CPU core, without compiler children. A short GDB attachment was detached
successfully; the main stack is inside CUDA driver calls beneath Triton's
`cuda_utils.so::loadBinary`, not Python generation work at the sampled instant.
Evidence: `logs/q3-full-smoke-8309223-stack.txt`. This is an abnormal delay and
does not establish a root cause or terminal job failure. Keep tracking8309223;
do not restart solely on observation timeout. The node forbids `nvidia-smi`.

Nemotron K-router C4 windows are now prepared by CPU job8309368, completed
exit0 in2:02, using basis and `python -u evaluation/prepare_v96kl_data.py windows
--model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--nvidia--Nemotron-H-47B-Reasoning-128K/snapshots/18c2a0e52e2d028dd96c3b4252af2a4f8fa54a43
--calibration results/k_routing_fit/nemotron_h_47b/calibration`.
Post-run verification checked the tensor file SHA256,80×32768 shape,64 fit/16
diagnostic IDs and640 unique document hashes. Tensor SHA256 is
`d23649f05e0e33da9856cb2f83d63c83ea858abbd6fc136c0658f2d7a1e781df`.
This uses the shared pinned C4 long-window protocol and native tokenizer;
it is separate from the2048-token V/Wo calibration set. No activations have
been captured for these long windows yet. Logs: `logs/nh-k-windows-8309368.*`.

The working-tree fit-only entry now uses checked routing geometry and actual
attention-layer payload IDs. `routing_position_embeddings` supplies identity
phases for native Nemotron (no RoPE), preserving existing CPU-initialized
Llama/Qwen phases. These helpers are included in fit provenance hashes.
CPU job8309366 passed4 tests in11.29s in basis with
`python -m pytest -q tests/test_k_routing_config.py tests/test_k_routing_hybrid_audit.py`:
identity rotation preserves keys bitwise, Qwen phases match, YaRN calibration
and evaluation phases agree, and hybrid payload IDs are validated.
Two Transformers warnings concern the deprecated rotary constructor `device`
argument. This is preparation only: no Nemotron router fitting has run, and
its dense capture and runtime evaluation still need native-model integration.
Frozen Qwen sources are unchanged.

Update after capture completion: Qwen capture8309215_0–3 completed with exit0
in44:44 each. Full runtime smoke8309223 is running on4 L40S. Nemotron
single-device diagnostic8309218 completed: native BF16, FP32 chunk and the
independent selective scan all produced finite outputs for the saved layer23
input on GPU0. Native block relative RMSE versus selective scan is0.000290405;
FP32 chunk is0.000220408. This does not reproduce or fix the full-model failure.
Two-device replay8309362 is queued afterany8309223, comparing tensors/current
CUDA device pairs(0,0),(1,0),(1,1), without intermediate module hooks/syncs.
Its basis command is `python -u evaluation/diagnose_nemotron_h_device.py
--audit results/k_routing_fit/nemotron_h_47b/manifests/native_audit.json
--failure results/k_routing_fit/nemotron_h_47b/manifests/nonfinite_full_129_layer_023.json
--output results/k_routing_fit/nemotron_h_47b/manifests/layer23_device_diagnostic.json`,
with isolated `PYTHONPATH=results/tools/nemotron_deps`; logs are
`logs/nh-device-diag-8309362.{out,err}`. Pending Qwen exact-sparse smoke8309231
now waits for afterok8309223 and afterany8309362 for resource ordering.
The user reconfirmed the3–4-point tolerance; the decision remains a maximum
four-percentage-point mean drop, retaining per-task reporting.

Llama formal array8309065 and summary8309066 completed. The summary independently
verified all440 predictions, shared inputs/protocols, first tokens, EOS/caps and
recomputed scores. Scores are full79.6969697%, exact-sparse78.4090909%,
LRQK81.1363636%, ShadowKV75.5492424%, ours76.7234848%. The full-minus-ours
drop is2.97348485 percentage points, so the user-authorized branch is **skip64K
refitting and proceed to Qwen/Nemotron**. Per-task results are in
`results/k_routing_fit/llama31_8b/eval64k/summary.md` and its paired JSON.
Ours versus full has the largest task drops on vt(-15 points),
niah_single_3(-12.5), and fwe(-8.3333); niah_multikey_2 improves12.5 points.
Each task has only8 prompts. Keep the final PCG tolerance warnings below.

Fit-only smoke8309140 passed: all five exported layer0 factor tensors matched
the original Llama2-fit/1-diagnostic smoke bitwise. The separate capture and
fit-only path is therefore numerically checked against that original smoke.
Qwen two-window/all64-layer32K capture smoke8309206 is queued after Nemotron
covariance smoke8309159 originally; its dependency was subsequently released
as noted below. It uses `capture_k_routing.py --rope yarn2 --num-shards 1
--fit-count 1 --diagnostic-count 1` and output
`results/k_routing_fit/qwen3_32b/smoke/capture`.

After Llama summary completed, `eval_k_routing_ruler.py` was updated to require
explicit `--rope native|yarn2`, load the bounded runtime config, record it in
the shared evaluation protocol, and verify Qwen router-bank RoPE agreement.
Llama's original executed evaluator remains in its existing source snapshot.
CPU workflow/config regression job8309207 passed5 tests in11.12s.
Qwen source files are frozen under `results/k_routing_fit/qwen3_32b/sources`
with568 file hashes and package versions in `manifests/source_snapshot.json`.
Qwen evaluations execute that snapshot. Its first full smoke8309210 failed
before loading because the snapshot omitted an imported `scripts` dependency;
that directory was added, all recorded files retained, and the frozen CLI
import was verified. Two-L40S full smoke8309213 then loaded the model but OOMed
in the64K MLP prefill (3.12GiB requested, GPU0 free286MiB). Identical smoke
8309223 now requests4 L40S, after diagnostic8309218. No prompt, algorithm or
generation settings changed. Qwen capture smoke8309206 completed all64 layers.
First/last-layer fit smoke8309214 completed: layer63 Page-Fisher NMSE is
0.0533245 fit/0.615209 diagnostic on its one-fit/one-diagnostic tiny subset;
this is not a formal quality result. Its final PCG residual0.00226235 exceeds
1e-5, retained under the fixed40-sweep policy.

Formal capture8309215_0–3 is running on all4 L40S,20GiB RAM and2 CPUs per
worker, with16 fit/4 diagnostic windows per shard. It started immediately when
8309214 passed, before a dependency update could add8309223; Slurm rejected
that update because the array was already running. Keep the live captures:
their streamed32K teacher path passed smoke and is separate from the64K
evaluation memory failure. Formal fit8309225_0–3 waits for both8309215 and
baseline smoke chain8309233, using one GPU/32GiB/2 CPUs per worker and16 layers per shard.
CPU audit8309226 waits for all fit shards. Capture/fit/audit commands execute
the frozen Qwen source tree, keeping later Nemotron changes isolated.

The queued Qwen runtime chain is: full smoke8309223 → exact-sparse8309231 →
LRQK8309232 → ShadowKV8309233 → formal fit8309225 → CPU fit audit8309226 →
ours smoke8309234 → CPU five-arm gate8309235 → formal evaluation8309236 →
summary8309237. Every GPU runtime job requests4 L40S/64GiB/2 CPUs. Evaluation
has20 array elements, mapping `arm = [full,exact_sparse,lrqk,shadowkv,ours][id//4]`
and `sample_shard = id%4`; at most one four-GPU replica runs at a time. All
commands include `--rope yarn2`, the pinned Qwen identity,64K data and the
same frozen V96 bank. A failed gate prevents dependent jobs from running.
The five-arm summary checks440 complete paired predictions before reporting.

Before any successful Qwen evaluation and while only its capture entry was
active, the evaluator's hardcoded Llama Markdown heading was corrected to a
model-neutral heading. Numerical code is unchanged. The previous evaluator
is retained in `sources/history/` and its previous manifest in
`manifests/source_snapshot_before_title_fix.json`; the current source manifest
records the corrected evaluator and569 source files. Active capture/fitter
source files were not changed by this editorial correction.

Nemotron full-model smoke8309124 failed after verifying all577 weights against
the checkpoint: at least one of full/cached logits was nonfinite. This is a
terminal numerical failure, not an observation timeout. It does not establish
usable full-model inference. The smoke now checks every block output and saves
the first nonfinite block's input for reproduction. Retry8309211 uses the same
model, data, dependencies and4-L40S/96GiB allocation. It also failed, now
locating the first nonfinite output at Mamba layer23 during129-token full
forward, with finite input max_abs424. The input and preceding layer checks
are saved as `manifests/nonfinite_full_129_layer_023.{json,safetensors}`.
The learned A_log values at layers0 and23 have finite exponentials, ruling
out simple A_log exponential overflow at the first failing layer.
Single-GPU diagnostic8309218 is queued to replay that exact input and compare
native chunk scan, FP32 chunk scan and the library's separate selective scan.
It records module/scan tensor ranges and nonfinite counts; it does not change
formal numerical settings or establish a validated fix. Covariance smoke
8309159 remains gated on a successful full-model smoke; its current failed
dependency8309211 must be replaced after diagnosis. Qwen no longer waits for
Nemotron. Do not restart a live capture merely to free diagnostic resources.

The working-tree router audit now resolves checkpoint entries by actual layer
ID and verifies that attention IDs and Q/KV/head/hidden geometry match the
native model config. This removes the previous contiguous-layer indexing
assumption for Nemotron. CPU job8309238 passed its native98-block config test
in10.04s (`python -m pytest -q tests/test_k_routing_hybrid_audit.py`, basis),
including reordered payloads, invalid Mamba IDs and duplicate entries. The
running Qwen jobs continue using their unchanged frozen audit source; their
attention layers are contiguous. No Nemotron fitted bank has been audited yet.

The native full-forward Nemotron allocator is prepared in
`evaluation/allocate_nemotron_h_v96.py`: five attention layers only,
anchor64/compression32/expansion96, alpha1, exact mean96,8 profile and8
independent confirmation windows,2047 prediction positions per2048-token
window with full-vocabulary KL. It reuses canonical-gauge decoder closure and
the established two-sided costs/exact allocation. Attention profiling keeps
Mamba dense; separate Wo fitting remains required afterward. It exports local
factors and a V96 identity; no Git/HF upload is involved. No47B KL profiling
job has run. Unit job8309205 passed3 tests in8.40s, checking encoder subspace
preservation/held-fixed solve, deployed reconstruction improvement, full-rank
dense equivalence, and exact-budget allocation against exhaustive enumeration.
The initial unit8309204 incorrectly expected the canonicalized encoder bytes
to equal the pre-canonicalization bytes; the test now checks the actual fixed
subspace contract and the solver verifies unchanged bytes during its solve.

The chronological notes below retain earlier intermediate states; this section
and the completed artifacts above supersede their pending-status statements.

Nemotron revision inspected: `18c2a0e52e2d028dd96c3b4252af2a4f8fa54a43`.
Its attention has no RoPE or Q/K RMSNorm. Use the installed Transformers native
implementation, whose block names are `full_attention`, `linear_attention`,
and `mlp`. The checkpoint's old remote code uses `attention`/`mamba` instead;
an early adapter edit based only on that remote code was reverted. The original
adapter already matches the native implementation. The earlier five mock tests
(job8309067) did not establish native-model compatibility; a native meta-model
test has now been added.
Job8309087 passed all6 adapter tests in13.33s in the basis environment,
including discovery on an actual native Transformers meta model.
Nemotron C1 attention adapter `basisserve/checkpoint/gqa_vo_nemotron_h.py`
reuses the existing compressed-V kernels with identity Q/K normalization and
identity positional rotation, preserving the native absence of RoPE. Its
pre-hook remains active when a routing implementation replaces forward.
CPU test job8309152 passed all8 tests in11.38s:
`python -m pytest -q tests/test_nemotron_h_c1_attention.py tests/test_nemotron_h_c1.py`.
The two new small native-attention tests cover full and half V rank, prefill,
two cached decode steps, unchanged K and actual compressed V cache width.
They compare against native attention with the same V projection folded into
dense weights. This does not validate47B compressed-model quality or GPU kernels.

Native geometry audit8309086 passed: all577 checkpoint tensors match the native
meta model after the library's built-in key renaming, with no tensor transforms,
missing/extra keys, or shape mismatches. There are45 Mamba,5 full-attention,
and48 MLP blocks. The initial audit8309085 failed because it compared raw names
before applying the library's `backbone`→`model` renaming. Audit records are in
`results/k_routing_fit/nemotron_h_47b/manifests/native_audit.json`.
Full model execution and numerical equivalence remain untested.
Native fast Mamba dependencies still need setup. The basis environment has
Torch2.6.0+cu124 and lacks mamba-ssm/causal-conv1d/kernels. Lowrank has
Torch2.11.0+cu130 and kernels0.16.1, but lovelace's loaded NVIDIA driver is
550.54.15. Do not change the active basis environment's dependencies while
Llama fitting/evaluation is running. The native Transformers kernel decorator
can use the original mamba-ssm and causal-conv1d packages directly; check
matching Torch/CUDA/C++ ABI wheels before installing isolated Nemotron deps.
Isolated dependencies now live in `results/tools/nemotron_deps`, activated
only through Nemotron commands' PYTHONPATH. Job8309091 installed official
Torch2.6/cu12/ABI-False/CPython3.11 wheels for mamba-ssm2.2.5 and
causal-conv1d1.5.0.post8. Smoke8309092 rejected a Torch-only scan binding;
explicit imports in8309093 exposed an undefined PyTorch C++ symbol in the
2.2.5 wheel. Job8309095 cleanly uninstalled that package and installed the
official matching2.2.4 wheel. The basis environment itself was not modified.
Smoke8309098 got past the compiled extension but exposed removed Transformers
generation aliases used by2.2.4. The latest official2.3.2.post1 source uses
GenerateDecoderOnlyOutput and provides a matching Torch2.6/cu12/ABI-False
wheel. Job8309100 cleanly replaced2.2.4 with2.3.2.post1 in the same isolated
directory; the GPU smoke is being repeated unchanged.
Smoke8309101 also found an undefined C++ symbol in the latest prebuilt wheel.
Job8309103 is building the pinned2.3.2.post1 release from source with
the existing basis Torch2.6/cu124/ABI-False, using cluster module
`nvidia/cuda12/cuda/12.4.1`, MAMBA_FORCE_BUILD=TRUE, MAX_JOBS=2, and pip wheel
with --no-deps/--no-build-isolation. Built wheels go to
`results/tools/nemotron_wheels`; install only after successful compilation.
Job8309110 waits for successful compilation, verifies a single matching wheel,
uninstalls the incompatible prebuilt package from the isolated directory, and
installs the locally compiled wheel. GPU smoke8309112 depends on that install.
The smoke records compiled-extension hashes as well as package versions.
Compilation8309103 completed in15m07s. Wheel SHA-256:
`e9afe7788d305bf781b683cb55925c37edf34578cb7a289e366fb30d57240baf`.
The login shell placed it under `/deac/csc/yangGrp/zhangal/results/tools/`
instead of the repository-relative path, so installer8309110 failed its file
existence check before uninstalling anything. The wheel was copied to the
intended directory without overwriting files, and its hash matched the build
log. Installer8309115 repeats the same command; smoke8309112 now depends on it.
Use absolute output paths for any future build launched through a login shell.
Installer8309115 succeeded. Smoke8309112 got past the compiled extension but
Mamba3's package import requires triton.set_allocator, absent in Triton3.2.
The upstream3.3.1 source provides this API. Job8309119 installs Triton3.3.1
only into the isolated Nemotron directory. Its compatibility with this
Torch2.6 workload still requires the actual GPU smoke; Llama/Qwen continue
using basis Triton3.2 without this PYTHONPATH.
GPU smoke8309120 passed in57s with the locally compiled Mamba2.3.2.post1,
causal-conv1d1.5.0.post8 and isolated Triton3.3.1. Original-package accelerated
bindings were confirmed. At129/2049 tokens, cached-vs-full last-token relative
RMSE was0.003061/0.002586 for Mamba layer0 and0.001031/0.000483 for attention
layer17. Peak GPU allocation was1.608GiB. These are real checkpoint block
tests, not a whole-model or long-context quality result.
The smoke source hash was checked against the executed script, and the
validated package versions were saved in the Nemotron manifests/environment.json.
Whole-model smoke8309124 is queued after successful Llama summary8309066,
using4 L40S,2 CPUs and96GiB host memory. The basis command is
`PYTHONPATH=$PWD/results/tools/nemotron_deps python -u evaluation/smoke_nemotron_h_full.py --audit results/k_routing_fit/nemotron_h_47b/manifests/native_audit.json --block-smoke results/k_routing_fit/nemotron_h_47b/manifests/native_smoke.json --output results/k_routing_fit/nemotron_h_47b/manifests/full_smoke.json`.
It checks every loaded tensor against the checkpoint after dtype conversion,
requires GPU-only placement across4 devices, and compares129/2049-token full
prefill against cached last-token logits. This is not a RULER quality test.
`evaluation/smoke_nemotron_h_native.py` verifies original-package function
bindings and tests real first-Mamba/first-attention weights at129 and2049
tokens, comparing full prefill with cached last-token decode. It does not
claim whole-model quality or equivalence to the old remote implementation.
V fitting is restricted to
attention layers. New V fitting follows 256 C4 windows × 2048, 12 encoder
sweeps, fixed 16 CG iterations, candidate ranks 32/48/64/80/96/112/128, and
Two-Sided-KL alpha=1 with exactly mean rank96 over attention layers.
CPU data job8309155 completed `evaluation/prepare_nemotron_c1_windows.py`
with the pinned native audit and output `results/k_routing_fit/nemotron_h_47b/vcal`.
The bank contains336 document-disjoint2048-token excerpts:256 fit,64 heldout,
8 KL-profile and8 confirmation windows. This follows the existing C1 fit/
heldout and fresh-KL split sizes; it also excludes duplicate document text.
Seed20260821, C4 revision1588ec454efa1a09f29cd18ddd04fe05fc8653a2,
shuffle buffer10000, uniform within-document starts and no special tokens.
Manifest/payload/script hashes, split IDs and document/text uniqueness were
independently verified. No V or Wo factors have been fitted from this bank yet.
`evaluation/capture_nemotron_h_covariances.py` prepares a single dense teacher
pass that collects full-attention o_proj and Mamba out_proj input covariances
using the existing streaming accumulator. It separates fit/heldout statistics,
checks that every selected hook fires, validates finite final hidden states and
saved matrices, and records the actual window IDs. Completed per-kind artifacts
are verified on resume. Full attention retains the existing covariance schema;
Mamba Wo has an explicit separate format. No compression is applied during capture.
Smoke8309159 waits for whole-model smoke8309124 and uses4 L40S with1 fit/
1 heldout window,128 tokens, layers0 and17, writing `snapshots_smoke`.
The full256/64-window capture has not been submitted. For that full capture,
allow approximately92.5GiB for both offloaded covariance splits plus process
overhead; request128GiB host memory and keep the151GiB parent limit in mind.
`evaluation/fit_nemotron_h_c1_joint.py` specializes the existing full-layer C1
solver using the attention indices from the native audit. The shared fitter
now supports an explicit attention subset in a hybrid decoder stack; ordinary
Llama/Qwen profiles continue to cover every decoder layer. Nemotron uses12
fixed encoder sweeps and fixed16 CG iterations, FP32 work and BF16 artifacts,
and accepts only the declared seven candidate ranks. Full rank128 retains the
existing analytic exact endpoint. Mamba indices are rejected for V fitting.
CPU test8309165 passed23 tests in3.61s, including hybrid layer filtering and
the existing routed-OV solver suite. No Nemotron V fitting has been submitted.

The user additionally requests Mamba2 Wo compression matching full-attention
Wo. Current working interpretation follows the existing Nemotron Wo entry
point's `_source_rank`: communication reduction is measured against Dense
AllReduce, whose width depends on output8192 for both layer types. Thus
attention V96 means average total AllGather rank6144 and62.5% reduction;
Mamba Wo uses the same total6144, or source rank1536 at TP4. Mamba's retained
input fraction is37.5% because its native input is16384, versus75% for
attention input8192. The earlier tentative12288 interpretation used equal
input fractions and was explicitly corrected before any Wo fitting. The user
has not separately confirmed numerical ranks; the current interpretation is
based on the repository's actual communication-reduction definition.
`evaluation/fit_nemotron_h_wo.py` prepares45-layer sharded fitting from the
full Mamba covariance bank and a verified final attention V96 identity. It
derives total rank from64 query heads × mean V rank96 and checks identical
wire reduction against the attention reference before fitting each layer.
It uses the existing exact per-source two-sided Cholesky encoder solver,
12 sweeps and the existing decoder-closed heldout selector, with FP32 work
and BF16 stored factors. Attention V's fixed-CG16 requirement is separate.
CPU test8309177 passed all6 TP-source Wo tests in2.30s. Added coverage checks
TP4 wire accounting (9216 bytes/token/rank,62.5% AllReduce reduction for both
layer types) and stored-factor heldout MSE against direct folded execution.
The driver is prepared but has not run on the45 real Mamba layers.
No Nemotron factors or weights have been modified or fitted.
Read-only download of the pinned Nemotron model repository was queued as
job8309080 (small partition, basis environment, two download workers).
It ended OUT_OF_MEMORY with4GiB allocated; the identical Python download
command was resubmitted as8309081 with16GiB and completed in2m13s.
Downloading repository files does not execute its remote model code.

Qwen3-32B cached model revision is
`9216db5781bf21249d130ec9da846c4624c16137`: 64 layers, 64 query heads,
8 KV heads, head dimension128, hidden5120. Its unchanged config has
max_position_embeddings40960 and no RoPE scaling. The
[official model card](https://huggingface.co/Qwen/Qwen3-32B) recommends
YaRN factor2 for typical 65536-token contexts, with original context32768.
The stated working runtime choice is YaRN factor2/original32768, used
identically for calibration, router fitting, and all five eval arms.
Do not silently run beyond the native context or modify cached model files.
The new streamed-capture entry point supports this explicit option; the
current evaluation script still needs the matching runtime override.
`evaluation/k_routing_config.py` now provides an explicit bounded native/
YaRN2 runtime configuration with identity geometry and config-hash checks.
CPU test8309185 passed in11.18s: static YaRN gives bitwise-identical32K
position encodings when computed as the prefix of64K, preserves the original
checkpoint config file, and rejects native Qwen context overflow. This helper
has not yet been wired into the running evaluator or streamed capture entry.
Job8309107 downloaded and verified the frozen allocated
`ICLR-results/qwen3-32b/checkpoints/Q3-32B-C1-R96` from the same pinned
BasisServe-CALS revision. All64 selected factors passed hashes, shapes,
finite values, and rank checks; the allocation result is factorized KL alpha1
with mean96. Its original encoder fitting used6 sweeps; these existing frozen
factors are reused per the request, not replaced by a new uniform bank.
Identity: `results/k_routing_fit/qwen3_32b/manifests/v96.json`.
The preparation entry point now accepts explicit model/prefix/output arguments.
Qwen C4 preparation8309126 completed with64 fit and16 diagnostic32K windows
using the existing `prepare_v96kl_data.py windows` command and Qwen tokenizer.
No Qwen router fitting or RULER evaluation has started.

`evaluation/capture_k_routing.py` streams checkpoint weights one layer at a
time and shards dense captures by window. Its Llama layer0 smoke uses the
same2 fit/1 diagnostic windows as8309044 and requires bitwise capture equality.
Smokes8309127/8309128 found equal pre-RoPE K but approximately0.027% relative
RMSE after RoPE when initializing frequency buffers on GPU. The script now
initializes those buffers on CPU before transfer, matching the original
teacher path. The failed attempt's `smoke/streamed_capture/protocol.json` is
retained as historical provenance; successful captures carry their own
complete protocol, and that initial file is not consumed on resume.
Corrected streamed-capture smoke8309133 passed in16s: raw-V/post-RoPE-K
rows, pre-RoPE keys and candidate queries all matched the original capture
bitwise at32K. This validates layer0 capture only; the separate sharded-fit
consumer and Qwen runtime still need validation before formal use.

Llama replacement fitting8309072 completed both remaining shards in32m29s
and34m08s. All32 layers passed audit8309059. Mean Base16 key MSE is0.184307
fit/0.188126 diagnostic; mean Page-Fisher residual NMSE is0.120470 fit/
0.154908 diagnostic. All32 layers' final PCG relative residual exceeds1e-5;
the fixed40-sweep endpoint is retained, and RULER remains the decision test.
All five smoke arms8309055 completed and smoke gate8309064 passed.
Formal440-prediction array8309065 is running with four sample shards at a time.
Its full-K baseline completed all88 prompts with mean79.6969697%. Per-task
percent scores, in the declared task order, are100,100,100,100,62.5,87.5,
87.5,97.5,66.6666667,37.5,37.5. These are baseline results only; the
five-arm summary remains pending. The four-point rule corresponds to ours
at least75.6969697% before rounding. Array tasks4–7 (exact_sparse) are now
running. Spot checks of QA failures showed wrong/incomplete answers with
normal native EOS or the official32-token cap, not a crashed generation.
Exact Sparse completed all88 prompts with mean78.4090909%, a1.2878788-point
drop from full K with the same V96. Per-task percent scores are100,100,100,
100,87.5,81.25,81.25,87.5,50,37.5,37.5 in the same task order.
LRQK array tasks8–11 are starting as those GPU slots are released. The
refit decision still requires the completed ours arm and five-arm summary.
An intermediate check of the176 completed full/exact predictions verified
identical per-prompt inputs, shared protocol and first generated tokens,
generation caps and recomputed scores, with11 tasks ×8 prompts in each arm.
LRQK has now completed all88 prompts with mean81.1363636%, or1.4393939
points above full K. Its per-task scores are100,100,100,100,87.5,87.5,87.5,
92.5,62.5,37.5,37.5. ShadowKV tasks12–15 are running; ours remains queued.
ShadowKV subsequently completed all88 prompts with mean75.5492424%, a
4.1477273-point drop from full K. Per-task scores are100,100,100,100,87.5,
71.875,50,80,66.6666667,37.5,37.5. Ours tasks16–19 are now starting as
the final four sample shards. The user's four-point decision applies to ours,
not the ShadowKV score; no refit decision has been made yet.
Qwen64K RULER data job8309138 completed in1m44s with the same11 tasks,
eight samples per task, seed42, completion template and8-token margin.

After Llama fitting and its audit had finished, capture persistence was moved
out of `fit_k_routing.fit_layer` without changing its numerical fitting steps.
`evaluation/fit_k_routing_captures.py` now assembles verified window shards in
memory and calls that fitter without a loaded teacher. Capture manifests
reference original shard files; no second full copy of raw captures is saved.
The audit supports those explicit shard references. Llama's executed original
sources remain in its existing source snapshot; its active evaluation source
has not changed. Fit-only smoke8309140 waits for formal evaluation8309065 and
compares all five layer0 factor tensors bitwise against the original2-fit/
1-diagnostic smoke bank. This path is not yet cleared for formal Qwen fitting.
Workflow test job8309143 passed all4 tests in11.55s in basis:
`python -m pytest -q tests/test_k_routing_workflow.py`. The additional test
places diagnostic window64 before training window0 within a shard and verifies
that every assembled tensor restores global order0,1,64. It checks the
fit/diagnostic boundary independently of physical shard order.

Before fitting Qwen, address the151GiB parent-memory limit: the current Llama
worker retains all teacher hidden states and model weights while fitting.
Larger-model worker replicas must not be launched assuming their summed Slurm
allocations are usable. A separate calibration phase and fit-only layer workers
would allow fit workers to release the teacher and hidden-state buffers; any
such change needs a numerical smoke and must preserve the fitting protocol.

Evaluation now supports multiple visible GPUs with Accelerate balanced model
placement, reserving12GiB per GPU beyond model weights for cache/workspace.
CPU/disk model placement is rejected. Inputs use the embedding device, and
results record placement and each GPU's peak allocation. This path still needs
a real multi-GPU smoke. Llama uses the existing single-GPU loading path.
For larger models, choose a feasible GPU group per replica before choosing
the number of simultaneous sample shards; four model replicas cannot be
assumed to fit on four L40S cards.
Observed Llama peak allocation is28.937GiB for full/exact,36.979GiB for
LRQK, and31.110GiB over the completed ShadowKV prompts. LRQK records show
per-query-head FP32 key codes with shape[1,32,N,32]. For Qwen64 layers ×64
query heads ×65536 tokens ×rank32 ×4 bytes, those codes alone require32GiB.
Thus a two-L40S Qwen LRQK replica is not a safe allocation; test a larger
GPU group while preserving the specified per-query LRQK algorithm. Other
arms can still be assessed for two-GPU replicas by actual memory smoke.

## Frozen Llama identity

- Model: `meta-llama/Llama-3.1-8B`, revision
  `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- C1 repository: `alexz949/BasisServe-CALS`, revision
  `f1a6253b5d5c747a2475cbf9e704a67d97930b31`.
- Artifact: `ICLR-results/llama31-8b/checkpoints/L31-8B-C1-R96`.
- The downloaded model/result/factor manifests and all 32 selected factor
  hashes, shapes and finite values were checked. The allocation result agrees
  with the manifest and has factorized exponent1.
- Identity and actual rank schedule:
  `results/k_routing_fit/llama31_8b/manifests/v96.json`.

## Fit protocol

- C4 `allenai/c4`, revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`.
- Seed20260828, shuffle buffer10000; 640 unique documents, each contributes a
  4096-token excerpt. Eight excerpts form each deterministic 32K window.
  This reuses the existing repository's packing procedure, not native
  contiguous 32K documents. IDs0–63 fit, IDs64–79 diagnostic.
- Dense native BF16 teacher, actual model RoPE, separately captured raw V,
  pre-RoPE K, post-RoPE K and candidate post-RoPE queries.
- Base16: FP32 token products with FP64 accumulated moments, closed-form affine
  RRR from frozen V latent to directly captured pre-RoPE K. Unlike historical
  scripts, the pre-RoPE target is not reconstructed by inverting rounded RoPE.
- Fit Query-Gram Q64: four strata, 16 pivots each. Diagnostic Q32: eight per
  stratum. Both position selections use fit windows only. No diagnostic-driven
  endpoint selection. Query positions are shared across windows within a layer.
- Residual16: causal non-sink Page-Fisher, Page32, excluded prefix pages1,
  top-eigenvector initialization, 40 BCD sweeps, PCG damping/tolerance1e-5,
  maximum100 iterations. No Adam. TF32 disabled via `configure()`.
- Reuse existing `fit_affine_reduced_rank_map`,
  `build_multi_query_statistics` and `_fit_residual_grid` numerical routines.
- Four independent L40S workers, layer index modulo4. Each replays the dense
  teacher and fits only its assigned layers. Two CPUs and96GiB host allocation
  per worker. Captures, selected queries and factors are retained separately.

## RULER protocol

- Generator: NVIDIA/RULER revision
  `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.
- Source JSONs reused from `external/ShadowKV/data/ruler/synthetic/json/`.
- Base completion template, seed42, maximum65536 tokens including reserved
  generation, prompt margin8; all lengths checked by the preparation script.
- Tasks: niah_single_1/2/3, niah_multikey_1/2, niah_multiquery,
  niah_multivalue, vt, fwe, qa_1, qa_2.
- Shared full causal compressed-C1 Triton prefill; greedy generation, native
  model EOS and official task caps. Full K is not original dense V.
- Exact sparse: FP32 exact-QK page mass with normalized per-query-head mass,
  GQA max aggregation, Page32/B2048, sink page included in budget. Selected
  attention is also FP32, then cast back to payload dtype; this precision
  difference is recorded rather than claimed to be a perfectly matched kernel.
- Ours: native BF16 Base16/Residual16 sidecar and Page32/B2048, same sink and
  GQA aggregation. Captures/factors remain FP32 on disk.
- LRQK: rank32, topk2048 per query head, recent64, two prefill/decode iterations,
  tolerance1e-8, seed0, FP32 internal factors.
- ShadowKV: SVD160, chunk8, routed2048, outlier chunks48; native local tail and
  generated tokens. Do not cap its total physical support to2048.
- Resident quality experiment; no offload-throughput or cache-memory-speedup claim.

## Execution record

All Python commands use `/home/zhangal/.conda/envs/basis/bin/python`.
Jobs were submitted with `sbatch --wrap`; no temporary sbatch files were created.

| Job | Action | Status at submission record |
|---|---|---|
| 8309042 | Frozen V96 download/audit and C4 windows | Completed, exit0 |
| 8309044 | Layer0 2-fit/1-diagnostic 32K numerical smoke | Completed, exit0 |
| 8309045_0–3 | Formal 64-fit/16-diagnostic four-way fitting | Running |
| 8309049 | Generate/audit 64K RULER data | Completed, exit0 |
| 8309053 | CPU numerical tests | Failed: pytest missing; tests did not run |
| 8309054 | Same tests after installing pytest9.1.1 | Completed; 3 passed |
| 8309055_0–4 | Five-arm 64K smoke | Depends on 8309045 |
| 8309059 | Independent all-layer artifact audit | Depends on 8309045 |
| 8309064 | Verify all five smoke arms and fit audit | Depends on 8309055 and 8309059 |
| 8309065_0–19 | Five arms × four sample shards; at most four GPUs | Depends on 8309064 |
| 8309066 | Verify and summarize all 440 predictions | Depends on 8309065 |
| 8309070 | CPU routing regression after multi-GPU entry-point change | Completed; 3 passed in10.17s |
| 8309072_0,3 | Same fit commands for failed/stopped shards | Depends on 8309045_1 and 8309045_2; at most2 concurrent |

At07:12:57 EDT, original shard8309045_0 ended OUT_OF_MEMORY. Kernel evidence
identifies `oom_memcg=/system.slice`, whose memory.max is162157244416 bytes
(about151GiB), despite Slurm advertising750000MB for the node and96GiB per
fit task. Shard0 RSS was about40GiB; the aggregate parent cgroup was full.
Thus free physical node memory and per-job limits did not diagnose the real
constraint. Shard3 was intentionally cancelled to protect live shards1 and2.
The same unchanged fitter commands for shards0 and3 are queued as8309072,
after shards1 and2. Existing completed capture artifacts remain intact.
The smoke and all-layer audit dependencies were updated to require these four
successful shards, replacing the failed original array dependency. Fitting
temporarily uses two GPUs concurrently; evaluation remains planned for four.
Original shards1 and2 completed successfully in1h02m28s and1h02m47s,
including the earlier shared-memory stall. Their16 layers are saved.
Replacement shards8309072_0 and3 are now running, reusing completed captures.
Do not alter the node's global cgroup limits or other users' processes.

After recovery, parent usage was about76GiB while shards1 and2 continued.
Baseline smoke tasks8309055_0–3 were released early in a serial chain
(full, exact sparse, LRQK, ShadowKV); they do not require router factors.
The array throttle is now1 to limit host-memory loading peaks. Ours smoke
task4 still waits for all four fit shards. The final smoke gate still requires
all five arms and the independent all-layer audit before formal evaluation.
The full, exact-sparse, LRQK and ShadowKV smoke jobs completed successfully.
Each tests two64K prompts and repeats generation with a four-token cap; their
truncated scores are not formal quality measurements. Pairwise checks confirmed
identical inputs/protocols, first-token agreement, and decode exercised across
all four baselines on both prompts. Ours still waits for the full router bank.
The13 sources referenced by the fit and eval protocols were verified against
their recorded hashes and archived under `sources/`; package versions are in
`manifests/source_snapshot.json`.

Small fit smoke: Page-Fisher fit NMSE0.0218577, diagnostic NMSE0.0844336.
Final query solve hit100 iterations, maximum relative residual0.00530061,
above tolerance. Finite factors do not imply convergence. Its raw reconstruction
MSE is not the Page-Fisher objective. Smoke outputs are not formal factors.

The user already authorized experiment execution and retries. Preserve completed
artifacts, investigate failures before resubmitting, and never restart a job
merely because an observation timed out. Current source commands and job logs
are in `logs/kr-*`; layer records also retain commands and device information.
Direct SSH and a standalone nvidia-smi monitoring step were denied by node
permissions; Slurm job status, sstat and the permitted ps step are used instead.

## Entry points

```bash
python -u evaluation/prepare_k_routing_llama.py
python -u evaluation/fit_k_routing.py --identity results/k_routing_fit/llama31_8b/manifests/v96.json --windows results/k_routing_fit/llama31_8b/calibration/windows.safetensors --output results/k_routing_fit/llama31_8b --shard-index SHARD --num-shards 4
python -u evaluation/audit_k_routing_fit.py --root results/k_routing_fit/llama31_8b
python -m pytest -q tests/test_k_routing_workflow.py
python -u evaluation/eval_k_routing_ruler.py STAGE --arm ARM --identity results/k_routing_fit/llama31_8b/manifests/v96.json --data results/k_routing_fit/llama31_8b/ruler64k --bank results/k_routing_fit/llama31_8b/ours_b16r16 --output results/k_routing_fit/llama31_8b/eval64k --shard-index SHARD --num-shards 4
```

STAGE is smoke/audit-smoke/evaluate/summarize. ARM is
full/exact_sparse/lrqk/shadowkv/ours. Formal evaluation additionally requires
the saved successful smoke audit. Final five-arm summary verifies all440
predictions, EOS/caps, identical input tokens, first tokens, and recomputed scores.
All-layer audit checks hashes and shapes against actual nonuniform ranks.

No Git commit, push, or Hugging Face upload has been performed.
