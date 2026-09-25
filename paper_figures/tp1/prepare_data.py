"""Verify raw TP1 records and export publication tables and point provenance."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics as stats
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/system_benchmarks"
OFFLINE = RESULTS / "tp1_offline"
CAPACITY = RESULTS / "tp1_capacity/formal"
SCALING = RESULTS / "tp1_paper_scaling"
CONTEXTS = [16384, 24576, 32768, 49152, 65536, 98304, 130048]
OFFLINE_REV = "86c03092c663dc6654584143130b5e5abf2fedaa"
CAPACITY_REV = "0d0b7a569eb3adcfbf5d63d78a8beb767980fcad"


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def export(name, rows):
    assert rows
    with (HERE / name).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def check(actual, expected, decimals=3):
    assert math.isfinite(actual) and abs(actual - expected) <= 0.51 * 10 ** -decimals, (actual, expected)


def source(path, field, value):
    return dict(file=relative(path), metric_field=field, value=value,
                command_file=relative(path.parent / "command.json"),
                log_file=relative(path.parent / "run.log"))


def point(figure, context, method, batch=1, *, placement, routing, warmup,
          steps, repeats, aggregation, prefill, sources, rerun=False):
    return dict(figure=figure, model="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16",
        hardware="NVIDIA L40S", tp_degree=1, batch=batch, context_tokens=context,
        method=method, kv_placement=placement, routing=routing, warmup_count=warmup,
        measured_steps=steps, repeat_count=repeats, requested_repeats=3,
        aggregation=aggregation, prefill_included=prefill, newly_rerun=rerun,
        sources=sources, conda_environment="basis", cuda_graph=False)


def steady_values(folder, context, method):
    paths = [folder / f"steady_{method}_t{context}_c{c}_n128/result.json" for c in range(3)]
    full, attention = [], []
    for cohort, path in enumerate(paths):
        data = read(path)
        assert (data["status"], data["method"], data["cohort"], data["context_tokens"], data["mode"]) == (
            "complete", method, cohort, context, "steady")
        assert data["gpu"] == "NVIDIA L40S" and data["environment"] == "basis"
        assert data["configuration"]["storage"] == "GPU-local K/V"
        assert data["configuration"]["routing"] == ("full_scan" if method == "basis" else "dense")
        measured = data["measurement"]
        assert len(measured["cuda_ms"]) == 128 and measured["all_logits_finite"]
        assert len(measured["generated_tokens"][0]) == 144
        assert abs(stats.median(measured["cuda_ms"]) - measured["cuda_median_ms"]) < 1e-9
        profile = data["attention_profile"]
        assert len(profile["components"]) == 32 and profile["all_logits_finite"]
        full.append(measured["cuda_median_ms"])
        attention.append(stats.fmean(r["attention_block"] for r in profile["components"]))
    return dict(full=stats.median(full), attention=stats.median(attention),
                full_sources=[source(p, "measurement.cuda_median_ms", v) for p, v in zip(paths, full)],
                attention_sources=[source(p, "mean(attention_profile.components[*].attention_block)", v)
                                   for p, v in zip(paths, attention)])


def scaling_row(context, values, origin):
    dense, basis = values["dense"], values["basis"]
    return dict(context_tokens=context, context_label="128K" if context == 130048 else f"{context // 1024}K",
        dense_attention_ms=dense["attention"], basis_attention_ms=basis["attention"],
        attention_speedup=dense["attention"] / basis["attention"],
        dense_full_model_ms=dense["full"], basis_full_model_ms=basis["full"],
        full_model_speedup=dense["full"] / basis["full"], repeats=3, origin=origin)


def fig1(provenance):
    expected = {16384: (3.814, 4.401, 27.124, 27.597), 32768: (6.705, 5.840, 30.050, 28.988),
                65536: (12.583, 8.694, 35.994, 31.853), 130048: (23.857, 14.649, 47.370, 37.817)}
    historical = []
    for context, targets in expected.items():
        values = {m: steady_values(OFFLINE / "formal", context, m) for m in ("dense", "basis")}
        row = scaling_row(context, values, "archive")
        for field, target in zip(("dense_attention_ms", "basis_attention_ms", "dense_full_model_ms",
                                  "basis_full_model_ms"), targets):
            check(row[field], target)
        historical.append(row)
        provenance["historical_verification"].append(dict(figure=1, values=row, sources=values))
    export("fig1_historical_verified.csv", historical)
    rows = []
    for context in CONTEXTS:
        paths = [SCALING / "formal" / f"steady_{m}_t{context}_c{c}_n128/result.json"
                 for c in range(3) for m in ("dense", "basis")]
        if not all(p.exists() and read(p)["status"] == "complete" for p in paths):
            continue
        values = {m: steady_values(SCALING / "formal", context, m) for m in ("dense", "basis")}
        rows.append(scaling_row(context, values, "new_same_version_sweep"))
        for metric in ("attention", "full"):
            item = point(1, context, "attention_speedup" if metric == "attention" else "full_model_speedup",
                placement="Both methods: exact K128 and full V128 GPU-local",
                routing="Basis: full-scan B16R16/Page32, 16 warps, 1984 routed + 64 recent; Dense: full attention",
                warmup=16, steps=32 if metric == "attention" else 128, repeats=3,
                aggregation="ratio of three-cohort medians; per-cohort " + (
                    "mean attention-block profile" if metric == "attention" else "CUDA-event step median"),
                prefill=False, sources={m: values[m][metric + "_sources"] for m in values}, rerun=True)
            item.update(runtime_source=relative(SCALING / "runtime_manifest.json"),
                run_manifest=relative(SCALING / "formal/manifest.json"),
                benchmark_dates=[read(p.parent / "outcome.json")["started_utc"] for p in paths],
                source_git_commit=provenance["offline_archive"]["repository_head"],
                same_shape_warmup_requests=1, measured_in_separate_pass=metric == "attention",
                context_definition="Initial prompt prefix; cache grows during conditioning and measured decode",
                cache_capacity_tokens=context + 145, key_reuse=False,
                compilation_policy="Extension/JIT setup and same-shape runtime warmup precede primary measurement")
            provenance["points"].append(item)
    provenance["fig1_status"] = dict(complete_contexts=[r["context_tokens"] for r in rows],
                                     pending_contexts=[n for n in CONTEXTS if n not in {r["context_tokens"] for r in rows}])
    if len(rows) == len(CONTEXTS):
        manifest = read(SCALING / "formal/manifest.json")
        settings = manifest["settings"]
        assert settings == dict(phase="formal", contexts=CONTEXTS, cohorts=[0, 1, 2],
            methods=["dense", "basis"], warmup=16, measured_steps=128, separate_attention_steps=32)
        outcomes = read(SCALING / "formal/outcomes.json")
        assert len(outcomes) == 42 and all(r["status"] == "complete" for r in outcomes)
        assert len({(r["context"], r["method"], r["cohort"]) for r in outcomes}) == 42
        source_check = read(SCALING / "formal/source_check.json")
        assert source_check["status"] == "passed"
        provenance["new_formal_verification"] = dict(status="passed", trials=42,
            source_check=relative(SCALING / "formal/source_check.json"),
            manifest=relative(SCALING / "formal/manifest.json"))
    if rows:
        export("fig1_sparse_scaling_data.csv", rows)
    return rows


def fig2(provenance):
    manifest = read(CAPACITY / "manifest.json")
    assert read(CAPACITY / "source_check.json")["changed_files"] == []
    assert manifest["settings"]["warmup"] == 8 and manifest["settings"]["steps"] == 128
    assert manifest["settings"]["repeats"] == 3
    outcomes = [r for r in read(CAPACITY / "outcomes.json") if r["context"] == 65536]
    expected = {"dense_local": [27.66, 41.12], "dense_k_offload": [5.08, 5.42, 5.62],
                "basis_k_offload": [29.36, 46.10, 68.18]}
    rows = []
    for method in expected:
        for index, batch in enumerate(sorted({r["batch"] for r in outcomes if r["method"] == method})):
            trials = [r for r in outcomes if r["method"] == method and r["batch"] == batch]
            successful = [r for r in trials if r["status"] == "success"]
            sources, tps, latency, peak, resident = [], [], [], [], []
            row = dict(context_tokens=65536, method=method, active_batch=batch, status="complete",
                throughput_tok_s="", mean_decode_ms="", throughput_interval_mean_ms="", gpu_peak_through_prefill_gib="", gpu_decode_resident_gib="",
                repeats=len(successful), attempted_repeats=len(trials), oom_phase="", origin="archive_same_version")
            for trial in trials:
                directory = CAPACITY / Path(trial["directory"]).name
                if trial["status"] == "success":
                    path = directory / "result.json"
                    data = read(path)
                    assert data == trial["result"]
                    assert (data["context"], data["batch"], data["warmup_steps"], data["decode_steps"]) == (
                        65536, batch, 8, 128)
                    assert data["active_batch"] == [batch] * 128 and data["all_logits_finite"]
                    assert abs(data["aggregate_tokens_per_second"] - batch * 128 / data["decode_seconds"]) < 1e-8
                    assert abs(data["mean_step_ms"] - stats.fmean(data["step_ms"])) < 1e-8
                    assert data["decode_seconds"] * 1000 >= sum(data["step_ms"])
                    tps.append(data["aggregate_tokens_per_second"])
                    latency.append(data["mean_step_ms"])
                    peak.append(data["decode_resident"]["gpu_peak_allocated_bytes"] / 2**30)
                    resident.append(data["decode_resident"]["gpu_allocated_bytes"] / 2**30)
                    sources.append(source(path, "aggregate_tokens_per_second", tps[-1]))
                else:
                    assert trial["status"] == "gpu_oom" and not successful
                    assert "out of memory" in (directory / "run.log").read_text().lower()
                    assert read(directory / "progress.json")["phase"] == trial["phase"]
                    row.update(status="gpu_oom", oom_phase=trial["phase"])
                    sources.append(dict(file=relative(CAPACITY / "outcomes.json"), metric_field="status, phase",
                                        selector=dict(method=method, batch=batch, context=65536),
                                        log_file=relative(directory / "run.log"), value="gpu_oom"))
            if successful:
                assert len(successful) == 3 and {r["repeat"] for r in successful} == {0, 1, 2}
                row.update(throughput_tok_s=stats.median(tps), mean_decode_ms=stats.median(latency),
                           throughput_interval_mean_ms=1000 * batch / stats.median(tps),
                           gpu_peak_through_prefill_gib=stats.median(peak), gpu_decode_resident_gib=stats.median(resident))
                check(row["throughput_tok_s"], expected[method][index], decimals=2)
            rows.append(row)
            item = point(2, 65536, method, batch,
                placement="exact K128 and full V128 GPU" if method == "dense_local" else
                          "historical exact K128 pinned/mapped CPU; full V128 GPU",
                routing="full-scan B16R16/Page32, 1984 routed + 64 recent, persistent GPU K slots"
                        if method == "basis_k_offload" else "dense full-context attention",
                warmup=8, steps=128, repeats=len(successful), aggregation="median of three process throughputs",
                prefill=False, sources=sources)
            item.update(status=row["status"], oom_phase=row["oom_phase"],
                attempted_repeats=len(trials), source_git_commit=manifest["git_commit"],
                benchmark_dates=[datetime.fromtimestamp(read(CAPACITY / Path(t["directory"]).name /
                    "progress.json")["time"], timezone.utc).isoformat() for t in trials],
                date_note="Last recorded phase timestamp in progress.json, not inferred from file mtime",
                source_archive=relative(CAPACITY / "source.tar.gz"), source_check=relative(CAPACITY / "source_check.json"),
                hf_revision=CAPACITY_REV)
            provenance["points"].append(item)
    export("fig2_offload_throughput_data.csv", rows)
    provenance["fig2_reuse_reason"] = "All 64K points share one source archive and passed end-of-run byte comparison; no chunked-RoPE supplement mixed in."
    return rows


def request_values(context, method):
    paths = [OFFLINE / "formal" / f"request_{method}_t{context}_c{c}_n128/result.json" for c in range(3)]
    request, tail, build = [], [], []
    for cohort, path in enumerate(paths):
        data = read(path)
        assert data["method"] == method and data["gpu"] == "NVIDIA L40S" and data["environment"] == "basis"
        assert (data["status"], data["cohort"], data["output_tokens"], data["context_tokens"]) == (
            "complete", cohort, 128, context)
        m = data["measurement"]
        assert m["actual_decode_calls"] == 127 and m["final_logits_finite"]
        assert len(m["generated_tokens"][0]) == 128
        assert abs(sum(m[k] for k in ("prefill_inclusive_seconds", "post_prefill_phase_seconds",
                   "decode_phase_seconds")) - m["request_seconds"]) < 1e-8
        assert m["steady_tail_decode_steps"] == 111
        request.append(m["request_seconds"])
        tail.append(m["steady_tail_wall_mean_ms"])
        profile = data["build_profile"]
        build.append(profile.get("construction_including_preparation_seconds", profile.get("prompt_specific_fit_seconds")))
    return dict(request=stats.median(request), tail=stats.median(tail), build=stats.median(build), paths=paths,
                request_sources=[source(p, "measurement.request_seconds", v) for p, v in zip(paths, request)],
                tail_sources=[source(p, "measurement.steady_tail_wall_mean_ms", v) for p, v in zip(paths, tail)],
                build_sources=[source(p, "build_profile.construction_including_preparation_seconds" if method in
                                     ("shadowkv", "lrqk") else "build_profile.prompt_specific_fit_seconds", v)
                               for p, v in zip(paths, build)])


def fig3(provenance):
    expected = {32768: [8.433, 8.306, 19.131], 65536: [16.709, 16.399, 29.508],
                130048: [43.418, 42.898, 58.805]}
    rows = []
    for context, targets in expected.items():
        values = {m: request_values(context, m) for m in ("dense", "basis", "shadowkv")}
        ratio = values["shadowkv"]["request"] / values["basis"]["request"]
        for (method, data), target in zip(values.items(), targets):
            check(data["request"], target)
            rows.append(dict(context_tokens=context, context_label="128K" if context == 130048 else f"{context // 1024}K",
                method=method, request_seconds=data["request"], request_tail_ms=data["tail"],
                separate_nonadditive_build_profile_s=data["build"], shadow_over_basis=ratio,
                output_tokens=128, actual_decode_calls=127, cohorts=3, origin="archive"))
            item = point(3, context, method,
                placement="CPU V / low-rank reconstructed K" if method == "shadowkv" else "GPU-local exact K128/full V128",
                routing="ShadowKV rank160/chunk8/support2048" if method == "shadowkv" else
                        "full-scan B16R16/Page32,16 warps,1984 routed+64 recent" if method == "basis" else "dense full context",
                warmup="one complete same-shape request, then reset", steps=127, repeats=3,
                aggregation="median of three fixed prompt cohorts", prefill=True,
                sources=dict(request=data["request_sources"], tail=data["tail_sources"], build=data["build_sources"]))
            item.update(output_tokens=128, request_tail_conditioning_calls=16,
                source_git_commit=provenance["offline_archive"]["repository_head"],
                benchmark_date=None, archive_created_utc=provenance["offline_archive"]["created_utc"],
                source_archive=relative(OFFLINE / "freeze/dependencies.tar.gz"), hf_revision=OFFLINE_REV,
                excluded=["model/factor loading", "cache preallocation", "tokenization", "input/network transfer"],
                build_profile_is_nonadditive=True)
            provenance["points"].append(item)
        check(values["shadowkv"]["build"], {32768: 11.671, 65536: 14.699, 130048: 20.345}[context])
        if context == 130048:
            check(values["basis"]["tail"], 37.522)
            check(values["shadowkv"]["tail"], 32.594)
    export("fig3_request_latency_data.csv", rows)
    return rows


def appendix(provenance):
    root = RESULTS / "tp1_lrqk_local"
    rows = []
    for context in (32768, 65536, 130048):
        trials = [r for r in read(root / "outcomes.json") if r["context"] == context and r["output_tokens"] == 128]
        assert len(trials) == 3
        ok = [r for r in trials if r["status"] == "complete"]
        row = dict(context_tokens=context, completed=len(ok), requested_cohorts=3, request_seconds="",
                   request_tail_ms="", nonadditive_build_s="", failure_phase="",
                   cpu_request_seconds="", cpu_request_tail_ms="")
        cpu = request_values(context, "lrqk") if context != 130048 else None
        if cpu:
            row.update(cpu_request_seconds=cpu["request"], cpu_request_tail_ms=cpu["tail"])
        if context == 32768:
            check(cpu["request"], 55.603)
            check(cpu["tail"], 379.791)
        paths = [root / f"t{context}_n128_c{c}/result.json" for c in range(3)]
        if len(ok) == 3:
            data = [read(p) for p in paths]
            assert all((d["status"], d["context_tokens"], d["cohort"], d["output_tokens"]) == (
                "complete", context, c, 128) for c, d in enumerate(data))
            row.update(request_seconds=stats.median(d["measurement"]["request_seconds"] for d in data),
                request_tail_ms=stats.median(d["measurement"]["steady_tail_wall_mean_ms"] for d in data),
                nonadditive_build_s=stats.median(d["build_profile"]["construction_including_preparation_seconds"] for d in data))
            check(row["request_seconds"], 28.659)
            check(row["request_tail_ms"], 187.501)
            check(row["nonadditive_build_s"], 0.910)
        else:
            assert not ok and all(r["status"] == "gpu_oom" for r in trials)
            row["failure_phase"] = "warmup prefill MLP allocation" if context == 65536 else "warmup online fitting"
            for path in paths:
                assert "out of memory" in (path.parent / "run.log").read_text().lower()
        rows.append(row)
        provenance["lrqk_appendix"].append(dict(values=row, source_results=[relative(p) for p in paths if p.exists()],
            source_outcomes=relative(root / "outcomes.json"), source_logs=[relative(p.parent / "run.log") for p in paths],
            source_interpretation=relative(root / "COMPARISON.md"), hf_revision=OFFLINE_REV,
            cpu_sources=dict(request=cpu["request_sources"], tail=cpu["tail_sources"]) if cpu else
                dict(status="Only 2/3 original CPU-offload cohorts completed; no median reported")))
    export("lrqk_appendix_data.csv", rows)
    (HERE / "lrqk_appendix.md").write_text(
        "# LRQK Runtime Supplement\n\nTP1/B1, Llama-3.1-8B-Instruct BF16, one L40S, basis environment.\n\n"
        "| Prompt | GPU-local request (s) | Request-tail decode (ms/token) | Complete |\n"
        "|---:|---:|---:|---:|\n" + "".join(
            f"| {r['context_tokens']} | {r['request_seconds'] or '-'} | {r['request_tail_ms'] or '-'} | {r['completed']}/3 |\n"
            for r in rows) + f"\nAt 32K, CPU-offload request/tail values were {rows[0]['cpu_request_seconds']:.3f} s / "
        f"{rows[0]['cpu_request_tail_ms']:.3f} ms; GPU-local build profiling measured "
        f"{rows[0]['nonadditive_build_s']:.3f} s in a separate, non-additive pass. "
        "64K failed during warmup prefill MLP allocation; 130,048 failed during online fitting. "
        "These are properties of the evaluated implementation/runtime, not proven algorithmic capacity limits. "
        "Graph-break/recompilation warnings remain in the original logs. LRQK is not a main-figure series.\n")


def captions(scaling, capacity, requests):
    ready = len(scaling) == len(CONTEXTS)
    endpoint = scaling[-1] if ready else None
    offload = {r["method"]: r["throughput_tok_s"] for r in capacity if r["active_batch"] == 4 and r["status"] == "complete"}
    ratios = [r["shadow_over_basis"] for r in requests if r["method"] == "basis"]
    text = "# TP1 Figure Captions\n\n## Figure 1: Sparse Decode Scaling With Context Length\n\n"
    text += ("Llama-3.1-8B-Instruct sparse decode scaling (BF16, TP1/B1) on one L40S with all exact K/V state resident on GPU. "
        "The full-scan routing overhead is visible at short context, while sparse-attention benefit grows with context. "
        f"At 130,048 tokens (labelled 128K), attention-block speedup is {endpoint['attention_speedup']:.2f}x "
        f"and full-model steady-decode speedup is {endpoint['full_model_speedup']:.2f}x. "
        "The attention block includes cache update, routing, selection and selected attention, after QKV/RoPE and before output projection. "
        "Full-model timing includes the unchanged MLP and other components. Three fixed-cohort medians; "
        "16 conditioning calls, 128 measured calls, and a separate 32-step attention profile. "
        "The context axis uses logarithmic spacing. This is not complete-request latency.\n\n" if ready else
        "PENDING: the new seven-point, same-version sweep has not completed. Historical data are verified in "
        "`fig1_historical_verified.csv`; no missing point is interpolated and no final Figure 1 is emitted.\n\n")
    text += ("## Figure 2: Long-Context Decode Throughput Under K Offload\n\n"
        "Llama-3.1-8B-Instruct decode throughput (BF16, TP1) at 65,536 tokens on one L40S. "
        "All requests remain active; prefill is excluded. "
        "Dense-local reaches the tested cache-allocation limit at B=4. Offloading historical Keys allows dense execution "
        "to continue but requires full-history K movement each step. BasisKV retrieves selected/missing exact Keys into "
        "persistent GPU slots; all methods retain full V128 on GPU. "
        f"At B=4, BasisKV K-offload reaches {offload['basis_k_offload']:.2f} tok/s versus "
        f"{offload['dense_k_offload']:.2f} tok/s for Dense K-offload "
        f"({offload['basis_k_offload'] / offload['dense_k_offload']:.2f}x). "
        "This ratio compares the two K-offload methods, not BasisKV against Dense-local. "
        "Both offload methods fail cache allocation at B=8. Cross markers indicate failed configurations, not zero throughput; "
        "their vertical positions are for readability and have no measured throughput meaning. "
        "Largest successful tested batches are 2 / 4 / 4, not exact maximum capacities. "
        "Eight warmup steps, 128 measured decode forwards, medians of three independent processes. "
        "Throughput uses active_batch * 128 / total measured loop seconds, including loop bookkeeping; "
        "the raw per-step mean excludes the small between-step bookkeeping interval. "
        "All points reuse the original same-source 64K grid; the later 128K RoPE supplement is not mixed in.\n\n"
        "## Figure 3: Complete Request Latency With Offline Routing\n\n"
        "Llama-3.1-8B-Instruct complete 128-output-token request latency (BF16, TP1/B1) on one L40S. "
        "Exact prompts are 32,768, 65,536 and 130,048 tokens "
        "(labelled 32K, 64K and 128K). BasisKV uses offline-learned routing factors with no prompt-specific fitting; "
        "the evaluated ShadowKV path constructs a prompt-specific representation. ShadowKV/BasisKV request-latency ratios are "
        f"{ratios[0]:.2f}x, {ratios[1]:.2f}x and {ratios[2]:.2f}x. "
        "At the longest context, ShadowKV's request-tail steady decode is faster (32.594 vs. 37.522 ms/token), "
        "so the request difference is not a pure steady-state kernel speedup. Cache placement and runtime implementation "
        "also differ; the entire gap cannot be attributed to construction alone. ShadowKV construction/preparation profiles "
        "are 11.671 / 14.699 / 20.345 s in separate, non-additive instrumented passes. They are neither stacked with nor "
        "subtracted from primary request latency. BasisKV's ordinary projection, encoding and cache writes remain in prefill. "
        "These are warmed device-side requests, excluding model/factor loading, cache preallocation, tokenization and "
        "network transfer; not TTFT or network-service latency. Each value requires all three fixed cohorts to complete.\n")
    (HERE / "captions.md").write_text(text)


def verify_factors():
    archive_path = OFFLINE / "freeze/factors.tar.gz"
    directory = Path("/workspace/runs/l31-router-source/v128-router")
    files = []
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.isfile():
                path = directory / member.name
                assert path.resolve().is_relative_to(directory)
                assert path.read_bytes() == archive.extractfile(member).read(), str(path)
                files.append(str(path))
    assert files
    return dict(status="passed", archive=relative(archive_path), file_count=len(files),
                files=files, comparison="direct bytes; no SHA256")


def report(scaling, capacity, requests, provenance):
    historical = {r["values"]["context_tokens"]: r["values"] for r in provenance["historical_verification"]}
    differences = []
    for row in scaling:
        if row["context_tokens"] in historical:
            old = historical[row["context_tokens"]]
            for key in ("dense_attention_ms", "basis_attention_ms", "dense_full_model_ms", "basis_full_model_ms"):
                differences.append(dict(context_tokens=row["context_tokens"], metric=key,
                    archive_ms=old[key], new_ms=row[key], percent_change=100 * (row[key] / old[key] - 1)))
    if differences:
        export("fig1_archive_comparison.csv", differences)
    text = (f"# TP1 Efficiency Figure Results\n\nGenerated: {provenance['created_utc']}.\n\n"
            "Environment: `basis`; Llama-3.1-8B-Instruct BF16; "
            "TP1/B1 for Figures 1/3, fixed active batch for Figure 2; one NVIDIA L40S.\n\n"
            "## Verification and Run Status\n\n"
            "All supplied historical main-figure values match archived raw results at published precision. "
            "The 24K Dense/Basis smoke passed. Figure 1 is a new same-version seven-context sweep "
            "with three fixed cohorts in independent processes per method/context. "
            "Figure 2 reuses the original same-source 64K grid; Figure 3 reuses archived complete requests. "
            "No quality benchmarks or LRQK reruns were launched.\n\n")
    outcomes_path = SCALING / "formal/outcomes.json"
    outcomes = read(outcomes_path) if outcomes_path.exists() else []
    text += f"New formal trials complete: **{sum(r['status'] == 'complete' for r in outcomes)}/42**. "
    text += f"Failures: {sum(r['status'] != 'complete' for r in outcomes)}. "
    text += f"Fully aggregated context pairs: {len(scaling)}/7.\n\n"
    text += ("## Figure 1: GPU-Local Sparse Decode\n\n"
             "| Prompt | Dense attention ms | Basis attention ms | Attention speedup | Dense model ms | Basis model ms | Model speedup |\n"
             "|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in scaling:
        text += (f"| {r['context_tokens']} | {r['dense_attention_ms']:.3f} | {r['basis_attention_ms']:.3f} | "
                 f"{r['attention_speedup']:.3f}x | {r['dense_full_model_ms']:.3f} | "
                 f"{r['basis_full_model_ms']:.3f} | {r['full_model_speedup']:.3f}x |\n")
    if len(scaling) < 7:
        text += "\nPending points are not replaced with archive values or interpolated.\n"
    if differences:
        text += "\nExact new/archived latencies and percentage changes are in `fig1_archive_comparison.csv`. "
        text += f"The largest absolute change among completed historical overlaps is {max(abs(r['percent_change']) for r in differences):.2f}%.\n"
    text += ("\n## Figure 2: 64K K-Offload Throughput\n\n"
             "| Method | Batch | tok/s | Status / failure phase |\n|---|---:|---:|---|\n")
    for r in capacity:
        value = f"{r['throughput_tok_s']:.3f}" if r["status"] == "complete" else "-"
        text += f"| {r['method']} | {r['active_batch']} | {value} | {r['oom_phase'] or r['status']} |\n"
    text += "\nLargest successful tested batches: Dense-local 2, Dense K-offload 4, BasisKV K-offload 4. "
    text += "All plotted capacity failures are cache-allocation OOMs, not decode-kernel OOMs.\n\n"
    text += ("## Figure 3: Complete 128-Output-Token Requests\n\n"
             "| Prompt | Dense s | BasisKV s | ShadowKV s | Shadow/Basis |\n|---:|---:|---:|---:|---:|\n")
    for context in (32768, 65536, 130048):
        group = {r["method"]: r for r in requests if r["context_tokens"] == context}
        text += (f"| {context} | {group['dense']['request_seconds']:.3f} | {group['basis']['request_seconds']:.3f} | "
                 f"{group['shadowkv']['request_seconds']:.3f} | {group['basis']['shadow_over_basis']:.3f}x |\n")
    text += ("\nAll three cohorts completed for every main request value. ShadowKV's longest-context tail "
             "decode is faster than BasisKV's; complete requests are slower. Separate build profiles are non-additive. "
             "LRQK runtime failures and compiler warnings remain appendix-only.\n\n"
             "## Artifacts and Provenance\n\n"
             "Figures and raw plotting CSV: `paper_figures/tp1/fig{1,2,3}_*.{pdf,png,csv}`. "
             "Captions: `captions.md`; point-level source fields: `provenance.json`; "
             "LRQK: `lrqk_appendix.md` / `lrqk_appendix_data.csv`. "
             "New raw trial JSON/logs: `results/system_benchmarks/tp1_paper_scaling/formal/`. "
             "Exact launch/reproduction commands: `README.md`, per-trial `command.json` and run manifests.\n\n"
             f"Historical source HEAD: `{provenance['offline_archive']['repository_head']}`. "
             "The archived source bytes, not a dirty checkout or commit alone, define the runtime. "
             f"Offline HF revision: `{OFFLINE_REV}`; capacity HF revision: `{CAPACITY_REV}`. "
             "No SHA256 checks. Historical raw-data archive links and the publication file scope are in `README.md`.\n")
    (HERE / "RESULTS_SUMMARY.md").write_text(text)


def raw_plotting_values(points):
    rows = []

    def visit(item, prefix, metadata):
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, prefix, dict(metadata, repeat_index=index))
        elif "metric_field" in item:
            rows.append(dict(metadata, input_metric=prefix, source_file=item["file"],
                             metric_field=item["metric_field"], value=item["value"]))
        else:
            for key, child in item.items():
                visit(child, key, metadata)

    for p in points:
        visit(p["sources"], "primary", dict(figure=p["figure"], context_tokens=p["context_tokens"],
            batch=p["batch"], plotted_method=p["method"], newly_rerun=p["newly_rerun"]))
    export("raw_plotting_values.csv", rows)


def main():
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_task="paper_figures/tp1/inputs/Pasted text.txt",
        command="python paper_figures/tp1/prepare_data.py", conda_environment="basis",
        offline_archive=read(OFFLINE / "freeze/provenance.json"), historical_verification=[],
        points=[], lrqk_appendix=[], validation="Raw JSON arithmetic and archived source metadata; no SHA256")
    provenance["factor_verification"] = verify_factors()
    provenance["new_environment_packages"] = relative(SCALING / "environment.json")
    provenance["plot_settings"] = dict(width_inches=3.35, height_inches=2.30,
        axis_label_pt=8, tick_legend_annotation_pt=7, pdf_fonttype=42, png_dpi=300,
        x_scale={"figure1": "log2", "figure2": "log2", "figure3": "categorical"},
        oom_markers="No measured y value; vertical placement is typographic only")
    provenance["archived_verification"] = read(OFFLINE / "verification.json")
    assert provenance["archived_verification"]["status"] == "passed"
    scaling = fig1(provenance)
    capacity = fig2(provenance)
    requests = fig3(provenance)
    appendix(provenance)
    captions(scaling, capacity, requests)
    report(scaling, capacity, requests, provenance)
    raw_plotting_values(provenance["points"])
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(archive_checks="passed", fig1=provenance["fig1_status"],
                         fig2_points=len(capacity), fig3_points=len(requests))))


if __name__ == "__main__":
    main()
