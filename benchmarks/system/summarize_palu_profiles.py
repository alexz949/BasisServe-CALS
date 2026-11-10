"""Per-step GPU time by category for the PaLU G-LRD4 TP4 decode profiles.

Reads the per-rank key-average tables (profiles/profiler_out_<rank>.txt) written by
bench_llama_vllm.py --experiment profile and splits each rank's self CUDA time into NCCL,
attention, GEMM and other kernels. NCCL kernel time includes waiting for the peer ranks, so
where that wait lands differs per rank; the per-rank NCCL total is the robust quantity.
Operator rows are skipped because they repeat their kernels' CUDA time; kernels missing from the
printed top rows are counted as other.
"""

import argparse
import json
from pathlib import Path
import re

UNITS = {"us": 1e-3, "ms": 1.0, "s": 1e3}
ATTENTION = ("flash_fwd", "unified_attention", "reduce_segments", "_decode_scores", "_apply_weights", "SoftMax")


def milliseconds(text):
    value, unit = re.fullmatch(r"([0-9.]+)(us|ms|s)", text).groups()
    return float(value) * UNITS[unit]


def category(name):
    if "nccl" in name.lower():
        return "nccl_allgather" if "AllGather" in name else "nccl_allreduce" if "AllReduce" in name else "nccl_other"
    if any(key in name for key in ATTENTION):
        return "attention"
    if re.search(r"gemm|cutlass|splitKreduce", name):
        return "gemm"
    return "other"


def read_rank(path):
    totals, listed = dict.fromkeys(("nccl_allreduce", "nccl_allgather", "nccl_other", "attention", "gemm", "other"), 0.0), 0.0
    total = None
    for line in path.read_text().splitlines():
        if line.startswith("Self CUDA time total:"):
            total = milliseconds(line.split(":")[1].strip())
            continue
        columns = re.split(r"\s{2,}", line.strip())
        if len(columns) != 11 or columns[0] == "Name" or not re.fullmatch(r"[0-9.]+(us|ms|s)", columns[6]):
            continue
        # Operator rows repeat the CUDA time of the kernels they launch; GPU kernels have no CPU time.
        # "nccl:<collective>" rows are GPU-side annotations spanning the NCCL kernel.
        if milliseconds(columns[4]) != 0 or columns[0].startswith("nccl:"):
            continue
        self_cuda = milliseconds(columns[6])
        totals[category(columns[0])] += self_cuda
        listed += self_cuda
    totals["other"] += total - listed
    return total, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="profile_eager directory")
    args = parser.parse_args()
    rows = []
    for cell in sorted(args.root.glob("*/p4096_b*")):
        if not (cell / "summary.json").exists():
            continue
        config = json.loads((cell / "config.json").read_text())
        steps = config["measured_decode_steps"]
        wall = json.loads((cell / "summary.json").read_text())["medians"]["decode_step_p50_ms"]
        ranks = [read_rank(path) for path in sorted((cell / "profiles").glob("profiler_out_*.txt"))]
        mean = {key: sum(r[1][key] for r in ranks) / len(ranks) / steps for key in ranks[0][1]}
        nccl = [sum(v for k, v in r[1].items() if k.startswith("nccl")) / steps for r in ranks]
        rows.append(dict(arm=cell.parent.name, batch=config["batch"], ranks=len(ranks), steps=steps,
                         profiled_step_p50_ms=wall, gpu_ms_per_step=sum(r[0] for r in ranks) / len(ranks) / steps,
                         nccl_ms_per_step_each_rank=nccl, **{f"{k}_ms_per_step": v for k, v in mean.items()}))
    (args.root / "profile_breakdown.json").write_text(json.dumps(rows, indent=2) + "\n")
    header = ["Arm", "Batch", "Profiled step p50 ms", "GPU busy ms/step", "NCCL ms/step (mean)", "NCCL per rank",
              "Attention ms/step", "GEMM ms/step", "Other ms/step"]
    lines = ["# Decode-step GPU time by category (mean over TP ranks)", "",
             "Profiled run: one trial, 32 decode steps after 16 conditioning steps, eager. The torch profiler "
             "adds host overhead, so the profiled step is slower than the unprofiled grid cell. NCCL time "
             "includes waiting for peer ranks.", "",
             "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        nccl = r["nccl_allreduce_ms_per_step"] + r["nccl_allgather_ms_per_step"] + r["nccl_other_ms_per_step"]
        lines.append(f"| `{r['arm']}` | {r['batch']} | {r['profiled_step_p50_ms']:.1f} | {r['gpu_ms_per_step']:.1f} | "
                     f"{nccl:.1f} | {' / '.join(f'{x:.1f}' for x in r['nccl_ms_per_step_each_rank'])} | "
                     f"{r['attention_ms_per_step']:.1f} | {r['gemm_ms_per_step']:.1f} | {r['other_ms_per_step']:.1f} |")
    (args.root / "PROFILE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
