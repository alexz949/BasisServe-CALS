"""Compare a release eval.ruler output with the paper's per-sample RULER records on the same prompts.

usage: python compare_ruler.py OURS_DIR PAPER_PREDICTIONS_JSONL
Per task: our mean, the paper's mean on the same (task, ordinal) prompts, the paper's mean over its 100 prompts,
and how many predictions are identical. Means are unweighted over the tasks present in OURS_DIR.
"""
import json
import sys
from pathlib import Path

ours_dir, paper_file = Path(sys.argv[1]), Path(sys.argv[2])
ours = {}
for path in sorted((ours_dir / "samples").glob("*.json")):
    record = json.load(open(path))
    if record.get("status") == "complete" and not record.get("excluded"):
        ours[(record["task"], record["source_index"])] = record
paper = {}
for line in open(paper_file):
    record = json.loads(line)
    paper[(record["sample"]["task"], record["sample"]["ordinal"])] = record
tasks = sorted({task for task, _ in ours}, key=lambda t: ["niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2",
                "niah_multiquery", "niah_multivalue", "qa_1", "qa_2", "vt", "fwe", "niah_single_3"].index(t))
print(f"{'task':18s} {'n':>3s} {'ours':>7s} {'paper@n':>8s} {'paper@100':>9s} {'same pred':>9s}")
means = {"ours": [], "paper@n": [], "paper@100": []}
for task in tasks:
    keys = sorted(k for k in ours if k[0] == task)
    o = [ours[k]["score"] for k in keys]
    p = [paper[k]["result"]["score"] for k in keys if k in paper]
    p100 = [r["result"]["score"] for (t, _), r in paper.items() if t == task]
    same = sum(ours[k]["prediction"] == paper[k]["result"]["prediction"] for k in keys if k in paper)
    row = (100 * sum(o) / len(o), 100 * sum(p) / len(p), 100 * sum(p100) / len(p100))
    for name, value in zip(means, row):
        means[name].append(value)
    print(f"{task:18s} {len(o):3d} {row[0]:7.2f} {row[1]:8.2f} {row[2]:9.2f} {same:5d}/{len(keys)}")
print(f"{'mean':18s} {'':3s} " + " ".join(f"{sum(v) / len(v):{w}.2f}" for v, w in zip(means.values(), (7, 8, 9))))
