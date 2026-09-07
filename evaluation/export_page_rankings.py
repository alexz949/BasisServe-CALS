"""Export all valid page ranks for one saved layer/group/window/query to CSV."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.fit_qwen3_8b_residual_kl_bank import sha256


def page_row(tensors, row, page, arms, layer, group):
    assert 0 <= page < int(tensors["page_count"][row])
    result = dict(layer=layer, group=group, document=int(tensors["document"][row]),
                  query_position=int(tensors["query_position"][row]), page=page,
                  token_start=32 * page, token_end_inclusive=32 * page + 31,
                  pinned=page == 0,
                  teacher_mass_mean=float(tensors["exact.teacher_mass"][row, page]),
                  teacher_non_sink_mass_mean=float(tensors["exact.teacher_non_sink_mass"][row, page]))
    for head in range(4):
        result[f"teacher_mass_h{head}"] = float(tensors["exact.teacher_mass_by_head"][row, head, page])
    exact = bool(tensors["exact.selected"][row, page])
    for arm in ("exact", *arms):
        for key in ("group_score", "rank_min", "rank_max", "owner", "selected"):
            result[f"{arm}.{key}"] = tensors[f"{arm}.{key}"][row, page].item()
        cutoff = float(tensors[f"{arm}.cutoff"][row])
        result[f"{arm}.cutoff"] = cutoff
        result[f"{arm}.score_minus_cutoff"] = result[f"{arm}.group_score"] - cutoff
        result[f"{arm}.global_owner_head"] = group * 4 + result[f"{arm}.owner"]
        result[f"{arm}.rank_min_minus_63"] = result[f"{arm}.rank_min"] - 63 if page else None
        if arm != "exact":
            selected = result[f"{arm}.selected"]
            result[f"{arm}.status"] = ("intersection" if selected else "missed") if exact else ("extra" if selected else "neither")
    return result


def write_rows(path, rows):
    assert rows and not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--document", type=int, required=True)
    p.add_argument("--query-position", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    directory = args.root / "evaluate" / f"l{args.layer}_g{args.group}"
    record = json.loads((directory / "result.json").read_text())
    assert record["status"] == "complete"
    path = directory / "pages.safetensors"
    assert sha256(path) == record["pages_sha256"]
    tensors = load_file(str(path))
    indices = torch.where((tensors["document"] == args.document) &
                          (tensors["query_position"] == args.query_position))[0]
    assert len(indices) == 1
    row = int(indices[0])
    rows = [page_row(tensors, row, page, tuple(record["aggregate"]), args.layer, args.group)
            for page in range(int(tensors["page_count"][row]))]
    write_rows(args.output, rows)
    print(f"Exported {len(rows)} valid pages to {args.output}; ranks exclude pinned page0 and preserve ties")


if __name__ == "__main__":
    main()
