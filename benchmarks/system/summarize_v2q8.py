"""Tabulate V2Q8 TP1/TP4 run directories (result.json or, for failed runs, progress.json + log) under a root.

  python benchmarks/system/summarize_v2q8.py --root results/rebuttal/v2q8_l40s_sparse/tp1_128k/smoke

Writes runs.csv and RUNS.md into the root. A failed run is classified from its log (OOM, other error) and its last
recorded phase. Selection checks (selection_check.json) and RULER samples (sample_*.json) are tabulated too.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

GIB = 2 ** 30


def failure(directory, log):
    progress = directory / "progress.json"
    phase = json.loads(progress.read_text()).get("phase") if progress.exists() else None
    text = log.read_text(errors="replace")
    if phase is None:  # TP runs print progress lines instead of writing progress.json
        phase = next((json.loads(line)["phase"] for line in reversed(text.splitlines())
                      if line.startswith("{") and '"phase"' in line), None)
    kind = ("cuda_oom" if "OutOfMemoryError" in text or "CUDA out of memory" in text else
            "host_oom" if "Killed" in text or "Cannot allocate memory" in text else
            "error" if "Traceback" in text or "Error" in text else "no_result")
    last = next((line for line in reversed(text.splitlines()) if "Error" in line), "")
    return dict(status=kind, last_phase=phase, error=last[:200])


def tp1_row(result):
    m, row = result["measurement"], {}
    if result["mode"] == "request":
        row.update(request_s=m["request_seconds"], ttft_s=m["prefill_inclusive_seconds"],
                   decode_s=m["decode_phase_seconds"], tail_ms=m.get("steady_tail_wall_mean_ms"),
                   peak_gib=m["memory_after_request"]["peak_allocated"] / GIB)
    elif result["mode"] == "steady":
        row.update(cuda_median_ms=m["cuda_median_ms"], wall_mean_ms=m["wall_mean_ms"], wall_p95_ms=m["wall_p95_ms"],
                   tokens_s=m["decode_tokens_per_second"],
                   prefill_peak_gib=m["memory"]["after_prefill_and_placement"]["peak_allocated"] / GIB,
                   decode_ready_gib=m["memory"]["decode_ready"]["allocated"] / GIB,
                   decode_peak_gib=m["memory"]["decode_end"]["peak_allocated"] / GIB,
                   host_gib=result["inventory"]["host_backing_bytes"] / GIB)
    else:
        row.update(attention_ms=m["attention_module_mean_ms"], construction_s=m["construction_seconds"])
    return row


def tp_row(result):
    ranks, row = result["ranks"], {}
    if result["mode"] == "request":
        ms = [r["measurement"] for r in ranks]
        row.update(request_s=max(m["request_seconds"] for m in ms), ttft_mean_s=statistics.fmean(ms[0]["ttft_seconds"]),
                   batch_ready_s=max(m["batch_ready_seconds"] for m in ms),
                   decode_ms_step=max(m["decode_mean_ms_per_step"] for m in ms),
                   tokens_s=min(m["decode_tokens_per_second"] for m in ms),
                   prefill_peak_gib_max_rank=max(m["memory_after_prefill"]["peak_allocated"] for m in ms) / GIB,
                   decode_peak_gib_max_rank=max(m["memory_after_decode"]["peak_allocated"] for m in ms) / GIB)
    else:
        steps = list(zip(*[r["measurement"]["wall_ms"] for r in ranks]))
        wall = [max(step) for step in steps]
        ordered = sorted(wall)
        row.update(wall_mean_ms=statistics.fmean(wall), wall_p95_ms=ordered[int(0.95 * (len(ordered) - 1))],
                   tokens_s=result["batch"] * 1000 / statistics.fmean(wall),
                   decode_ready_gib_max_rank=max(r["measurement"]["memory"]["decode_ready"]["allocated"]
                                                 for r in ranks) / GIB,
                   host_gib_all_ranks=sum(r["inventory"]["host_backing_bytes"] for r in ranks) / GIB)
    row["support_tokens"] = ranks[0]["support"]["mean_tokens_per_request_kv_head"]
    row["tokens_agree"] = result["cross_rank_tokens_agree"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for log in sorted((args.root / "logs").glob("*.log")):
        # One log per run: TP4 runs live at root/<name>, TP1 runs at root/<label>/<rest of name>.
        label, _, rest = log.stem.partition("_")
        candidates = [args.root / log.stem, args.root / label / rest]
        directory = next((c for c in candidates if c.is_dir()), candidates[0])
        result = directory / "result.json"
        base = dict(run=log.stem)
        if (directory / "selection_check.json").exists():
            check = json.loads((directory / "selection_check.json").read_text())
            rows.append(dict(base, status="check", identical_fraction=check["identical_fraction"],
                             mean_jaccard=check["mean_jaccard"], argmax_agreement=check["argmax_agreement"]))
        elif list(directory.glob("sample_*.json")):
            samples = [json.loads(p.read_text()) for p in sorted(directory.glob("sample_*.json"))]
            rows.append(dict(base, status="ruler", samples=len(samples),
                             score=statistics.fmean(s["score"] for s in samples),
                             archived_score=statistics.fmean(s["archived"]["score"] for s in samples),
                             same_ids=sum(s["same_ids_as_archive"] for s in samples)))
        elif result.exists():
            data = json.loads(result.read_text())
            rows.append(dict(base, status=data["status"], **(tp_row(data) if "ranks" in data else tp1_row(data))))
        else:
            rows.append(dict(base, **failure(directory, log)))
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with (args.root / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[k]:.3f}" if isinstance(row.get(k), float) else str(row.get(k, ""))
                                       for k in fields) + " |")
    (args.root / "RUNS.md").write_text(f"# Runs under {args.root}\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
