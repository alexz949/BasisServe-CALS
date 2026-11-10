"""B128 routing diagnosis from per-rank traces and actual slot counters."""

from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np


def main():
    root = Path("results/system_benchmarks/tp4_router_opt/profile_b128")
    source = json.loads((root / "kernel_profile_summary.json").read_text())
    stages = defaultdict(lambda: defaultdict(float))
    for row in source["kernels"]:
        path = Path(row["trace"])
        arm = path.parent.parent.name
        rank = int(re.search(r"rank(\d+)\.", path.name)[1])
        steps = json.loads((path.parent.parent / "config.json").read_text())["measured_decode_steps"]
        stages[(arm, rank)][row["stage"]] += row["gpu_duration_sum_us"] / steps / 1000
    rows = [dict(arm=arm, rank=rank, kernel_ms_per_step=dict(values))
            for (arm, rank), values in sorted(stages.items())]
    support = []
    for line in (root / "basis_joint" / "support_stats.jsonl").read_text().splitlines():
        record = json.loads(line)
        counts = np.asarray(record["counts"], dtype=np.int64)
        assert counts.ndim == 5 and counts.shape[-1] == 2
        selected = counts.sum(-1)
        misses = counts[..., 1]
        support.append(dict(
            rank=record["rank"], hit_rate=float(counts[..., 0].sum() / selected.sum()),
            selected_mean=float(selected.mean()), selected_max=int(selected.max()),
            misses_mean_per_layer_request_kv_group=float(misses.mean()),
            misses_p95_per_layer_request_kv_group=float(np.percentile(misses, 95)),
            host_k_miss_payload_mean_mib_per_step=float(misses.sum((1, 2, 3)).mean() * 128 * 2 / 2**20),
        ))
    result = dict(
        scope="Per-rank kernel duration sums divided by 32 profile steps; not additive wall latency. "
              "Host payload is logical unique miss K128 BF16 bytes, not a PCIe hardware counter.",
        stage_breakdown=rows, support=support,
    )
    (root / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    for arm in sorted({row["arm"] for row in rows}):
        ranks = [row for row in rows if row["arm"] == arm]
        names = {stage for row in ranks for stage in row["kernel_ms_per_step"]}
        totals = {stage: float(np.mean([row["kernel_ms_per_step"].get(stage, 0) for row in ranks])) for stage in names}
        print(arm, json.dumps(dict(sorted(totals.items(), key=lambda p: -p[1]))))
    print("support", json.dumps(support))


if __name__ == "__main__":
    main()
