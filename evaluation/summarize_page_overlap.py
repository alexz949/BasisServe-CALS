"""Validate saved page sets and report unweighted all-layer overlap."""

import argparse
import json
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.export_page_rankings import page_row, write_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, help="Verify q32 and exact saved ranks/sets against the old audit")
    p.add_argument("--ranking-arm", help="Export the 20 highest head-mean teacher-mass missed pages per layer")
    args = p.parse_args()
    torch.set_num_threads(2)
    assert not (args.root / "summary.json").exists()
    assert not (args.root / "summary.md").exists()
    first = json.loads((args.root / "evaluate/l0_g0/result.json").read_text())
    arms = tuple(first["aggregate"])
    assert args.ranking_arm is None or args.ranking_arm in arms
    layers, input_hashes = {}, {}
    ranking_rows = []
    scalar_metrics = {arm: [] for arm in arms}
    total = 0
    common_positions = list(range(24831, 32768, 256))
    for layer in range(36):
        samples = {arm: [] for arm in arms}
        layer_rankings = []
        for group in range(8):
            directory = args.root / "evaluate" / f"l{layer}_g{group}"
            record = json.loads((directory / "result.json").read_text())
            assert record["status"] == "complete" and record["stage"] == "evaluate"
            assert record["layer"] == layer and record["group"] == group
            assert record["rows"] == 512 and record["documents"] == list(range(64, 80))
            assert record["query_positions"] == common_positions
            assert set(record["aggregate"]) == set(arms)
            path = directory / "pages.safetensors"
            assert record["pages_sha256"] == sha256(path)
            input_hashes[f"l{layer}_g{group}"] = sha256(directory / "result.json")
            tensors = load_file(str(path))
            if args.reference_root:
                old_dir = args.reference_root / "evaluate" / f"l{layer}_g{group}"
                old = json.loads((old_dir / "result.json").read_text())
                old_path = old_dir / "pages.safetensors"
                assert sha256(old_path) == old["pages_sha256"]
                assert all(record["inputs"][key] == value for key, value in old["inputs"].items()
                           if key in record["inputs"])
                previous = load_file(str(old_path))
                for key, value in previous.items():
                    if key.startswith(("exact.", "q32.")) or key in ("document", "query_position", "page_count"):
                        assert torch.equal(tensors[key], value), (layer, group, key)
                del previous
            assert tensors["document"].tolist() == [d for d in range(64, 80) for _ in common_positions]
            assert tensors["query_position"].tolist() == common_positions * 16
            exact = tensors["exact.selected"]
            assert exact.shape == (512, 1024) and exact[:, 0].all()
            assert (exact.sum(-1) == 64).all()
            page_ids = torch.arange(1024)[None, :]
            valid = page_ids < tensors["page_count"][:, None]
            assert not (exact & ~valid).any()
            for arm in arms:
                selected = tensors[f"{arm}.selected"]
                assert selected[:, 0].all() and (selected.sum(-1) == 64).all()
                assert not (selected & ~valid).any()
                intersection = (exact & selected).sum(-1)
                metrics = record["per_query_metrics"][arm]
                assert len(metrics) == 512
                for row, count in zip(metrics, intersection.tolist()):
                    assert row["page_recall"] == count / 64
                    assert abs(row["non_sink_page_recall"] - (count - 1) / 63) < 1e-7
                    assert row["missed_count"] == row["extra_count"] == 64 - count
                samples[arm].extend(intersection.tolist())
                scalar_metrics[arm].extend(metrics)
            if args.ranking_arm:
                missed = exact & ~tensors[f"{args.ranking_arm}.selected"]
                weights = tensors["exact.teacher_mass"].masked_fill(~missed, -1)
                k = min(20, int(missed.sum()))
                indices = torch.argsort(weights.flatten(), descending=True, stable=True)[:k]
                for index in indices.tolist():
                    row, page = divmod(index, 1024)
                    layer_rankings.append(page_row(tensors, row, page, arms, layer, group))
            total += 512
        layer_rankings.sort(key=lambda r: (-r["teacher_mass_mean"], r["group"], r["document"], r["query_position"], r["page"]))
        ranking_rows.extend(layer_rankings[:20])
        layers[str(layer)] = {}
        for arm, counts in samples.items():
            assert len(counts) == 4096
            mean = sum(counts) / len(counts)
            layers[str(layer)][arm] = {
                "conditions": len(counts), "mean_intersection_pages": mean,
                "page_overlap": mean / 64,
                "non_pinned_page_overlap": (mean - 1) / 63,
            }
        print(f"validated layer {layer}: " + json.dumps(layers[str(layer)]), flush=True)
    assert total == 147456
    overall = {arm: {key: sum(layers[str(l)][arm][key] for l in range(36)) / 36
                     for key in ("mean_intersection_pages", "page_overlap", "non_pinned_page_overlap")}
               for arm in arms}
    report = {
        "status": "complete", "conditions_per_arm": total, "arm_comparisons": total * len(arms),
        "definition": "mean |S_exact intersect S_proxy| / 64; unweighted page overlap, not mass coverage or IoU",
        "protocol": "36 layers, 8 GQA groups, 16 C4 diagnostic windows, 32 common terminal8k Q; Page32/B2048, page0 pinned",
        "reference": "FP32 exact QK from BF16 captures; same per-head non-pinned normalized page mass then GQA-max selector",
        "proxy": "native BF16 closed-form Base16 + Fisher R8; named banks recorded per group; frozen C1-V80",
        "scope": "all layers use the same sparse rule in this diagnostic, including layers 0/1; no model rollout, fitting or RULER",
        "layers": layers, "overall": overall, "input_result_sha256": input_hashes,
        "scalar_metric_means": {arm: {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}
                                for arm, rows in scalar_metrics.items()},
        "old_reference_verified": str(args.reference_root) if args.reference_root else None,
        "command": shlex.join(sys.argv), "python": sys.executable,
    }
    lines = ["# All-layer exact/proxy page overlap", "", report["protocol"], "",
             "Overlap is the intersection size divided by 64, not attention-mass coverage and not IoU. "
             "The non-pinned metric subtracts the shared page0 and divides by 63. "
             "All means equally weight queries, windows, GQA groups and layers.", "",
             "The exact and proxy selectors use the same GQA-max policy and budget. "
             "The exact reference computes FP32 QK from captured BF16 Q/K; proxy arithmetic is native BF16. "
             "This is an offline diagnostic using dense-teacher captures, not an end-to-end accuracy test. "
             "Layers 0/1 are also subjected to the sparse rule for this comparison.", "",
             f"{total:,} shared conditions per arm; {total * len(arms):,} arm comparisons. All 288 output hashes, "
             "budgets and per-query overlap metrics were checked against saved selected-page masks.", "",
             "## Overall", "",
             "| Router | Mean intersection / 64 | Overlap | Excluding pinned page0 |",
             "| --- | ---: | ---: | ---: |"]
    for arm, r in overall.items():
        lines.append(f"| {arm.upper()} | {r['mean_intersection_pages']:.4f} | {100*r['page_overlap']:.4f}% | {100*r['non_pinned_page_overlap']:.4f}% |")
    lines += ["", "## Per-layer overlap (including pinned page0)", "",
              "| Layer | " + " | ".join(arms) + " |", "| --- | " + " | ".join("---:" for _ in arms) + " |"]
    for layer, values in layers.items():
        lines.append("| " + layer + " | " + " | ".join(f"{100*values[a]['page_overlap']:.4f}%" for a in arms) + " |")
    csv_rows = [{"layer": int(layer), **{f"{arm}.{key}": value for arm, metrics in values.items()
                 for key, value in metrics.items()}} for layer, values in layers.items()]
    write_rows(args.root / "layer_overlap.csv", csv_rows)
    if ranking_rows:
        write_rows(args.root / "missed_page_rankings.csv", ranking_rows)
        lines += ["", "## Detailed rankings", "",
                  "All valid pages, not just misses, are saved in evaluate/l{layer}_g{group}/pages.safetensors. Rows are keyed by document and query_position; pages >= page_count are padding.",
                  "Each arm stores group_score, rank_min/rank_max (inclusive tie intervals), selected IDs/masks, cutoff, and owning GQA head. Page0 is pinned and has rank0; routed cutoff rank is63, not64. Actual selected masks resolve boundary ties.",
                  "Exact teacher mass is saved both per head and head-averaged. Scores use non-sink normalized per-head mass then GQA max, so they are not head-mean teacher mass.",
                  "missed_page_rankings.csv contains the20 highest head-mean teacher-mass misses per layer for " + args.ranking_arm + "; this is a diagnostic subset, not the full distribution. It includes all arms, rank intervals, selection categories, head owners and score-minus-cutoff margins.",
                  "evaluation/export_page_rankings.py can export every valid page for any saved layer/group/document/query to CSV without model inference.",
                  "layer_overlap.csv contains all36 layer averages. Historical exact and q32 tables were verified bitwise against --reference-root." if args.reference_root else "No historical table verification was requested."]
    lines += ["", "## Environment and commands", "", "Conda environment: `basis`.", "",
              "Evaluation command for layer0/group0; layer and group vary over all 36 × 8 combinations:", "", "```bash",
              json.loads((args.root / "evaluate/l0_g0/result.json").read_text())["command"], "```", "",
              "Aggregation command:", "", "```bash", report["command"], "```", ""]
    write_json(args.root / "summary.json", report)
    (args.root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
