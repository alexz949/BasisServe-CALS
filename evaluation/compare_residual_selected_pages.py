"""Saved-page audit on shared C4 diagnostic queries; no fitting/model forward."""

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar, conditional_routing_query_projector,
)
from basisserve.core.page_selection_comparison import selection_details, compare_selection
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _discover_capture, _load_direct, _rotary_embeddings,
)
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from scripts.capture_qwen3_8b_q16 import verify_document


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--c1-checkpoint", type=Path, required=True)
    p.add_argument("--query-capture", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, choices=range(36), required=True)
    p.add_argument("--group", type=int, choices=range(8), required=True)
    p.add_argument("--repair-root", type=Path, help="Explicit optional root containing l{layer}_g{group} repaired factors")
    p.add_argument("--comparison-bank", action="append", metavar="NAME=PATH",
                   help="Named factor banks; q32 must be included as the frozen Base reference")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    layer, group = args.layer, args.group
    output = args.output_dir / args.stage / f"l{layer}_g{group}"
    assert not (output / "result.json").exists() and not (output / "pages.safetensors").exists()
    started = time.monotonic()
    banks, inputs = {}, {}
    bank_paths = {f"q{n}": ROOT / f"results/checkpoints/mse_base_q{n}_r8" for n in (32, 64, 128)}
    if args.comparison_bank:
        pairs = [entry.split("=", 1) for entry in args.comparison_bank]
        assert all(len(pair) == 2 and pair[0].isidentifier() and pair[0] != "exact" for pair in pairs)
        bank_paths = {name: Path(path) for name, path in pairs}
        assert len(bank_paths) == len(pairs) and "q32" in bank_paths
    for arm, bank_path in bank_paths.items():
        file = bank_path / f"layer_{layer:03d}.safetensors"
        record = json.loads(file.with_suffix(".json").read_text())
        assert record["status"] == "complete" and record["sha256"] == sha256(file)
        assert record["protocol"]["base_kind"] == "closed_form_rrr"
        assert record["protocol"]["page_size"] == 32 and record["protocol"]["excluded_prefix_pages"] == 1
        assert record["protocol"]["model_config_sha256"] == sha256(args.model / "config.json")
        assert record["protocol"]["c1_manifest_sha256"] == sha256(args.c1_checkpoint / "results.json")
        banks[arm] = load_file(str(file))
        assert all(torch.isfinite(t).all() for t in banks[arm].values())
        inputs[arm] = record["sha256"]
    source = banks["q32"]
    for bank in banks.values():
        assert all(torch.equal(source[key], bank[key]) for key in source if key.startswith("base_"))
    factors = {arm: (bank["residual_encoder_b16_r8"][group].to(device).bfloat16(),
                     bank["residual_query_b16_r8"][group * 4:group * 4 + 4].to(device).bfloat16())
               for arm, bank in banks.items()}
    if args.repair_root is not None:
        repair_root = args.repair_root / f"l{layer}_g{group}"
        repair_record = json.loads((repair_root / "result.json").read_text())
        assert repair_record["layer"] == layer and repair_record["group"] == group
        assert repair_record["inputs"]["bank"] == inputs["q32"]
        assert repair_record["artifact_sha256"] == sha256(repair_root / "group_factors.safetensors")
        inputs["q32_repaired"] = repair_record["artifact_sha256"]
        repaired = load_file(str(repair_root / "group_factors.safetensors"))
        factors["q32_repaired"] = (repaired["residual_encoder"].to(device).bfloat16(),
                                    repaired["residual_query"].to(device).bfloat16())
    capture = json.loads((args.query_capture / "manifest.json").read_text())
    assert capture["status"] == "complete" and capture["overlap_bitwise_equal"]
    assert capture["protocol"]["model_config_sha256"] == sha256(args.model / "config.json")
    positions = capture["protocol"]["query_positions"]
    assert positions == list(range(24831, 32768, 256))
    dr, dm = _discover_capture(ROOT / "results/calibration", split="validation", layer=layer)
    assert sha256(dr / "manifest.json") == capture["protocol"]["inputs"]["validation"][str(layer)]["direct_manifest_sha256"]
    inputs["direct_manifest"] = sha256(dr / "manifest.json")
    inputs["query_manifest"] = sha256(args.query_capture / "manifest.json")
    _, raw = _load_direct(dr, dm, layer)
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    file = args.c1_checkpoint / c1["artifacts"][str(layer)]["file"]
    inputs["c1"] = sha256(file)
    encoder = load_file(str(file))["value_coordinate_encoders"][group].to(device).bfloat16()
    cos, sin = _rotary_embeddings(args.model, sequence=32768, device=device)
    indices = [64] if args.stage == "smoke" else list(range(64, 80))
    query_indices = [31] if args.stage == "smoke" else list(range(32))
    count = len(indices) * len(query_indices)
    tables = {"document": torch.empty(count, dtype=torch.int32),
              "query_position": torch.empty(count, dtype=torch.int32),
              "page_count": torch.empty(count, dtype=torch.int32)}
    tables["exact.teacher_mass_by_head"] = torch.zeros(count, 4, 1024)
    tables["exact.teacher_non_sink_mass_by_head"] = torch.zeros(count, 4, 1024)
    arms = ["exact", *factors]
    for arm in arms:
        for name in ("group_score", "teacher_mass", "teacher_non_sink_mass"):
            if arm != "exact" and name.startswith("teacher_"):
                continue
            tables[f"{arm}.{name}"] = torch.zeros(count, 1024)
        for name in ("rank_min", "rank_max", "owner"):
            tables[f"{arm}.{name}"] = torch.zeros(count, 1024, dtype=torch.int16)
        tables[f"{arm}.selected"] = torch.zeros(count, 1024, dtype=torch.bool)
        tables[f"{arm}.cutoff"] = torch.empty(count)
        tables[f"{arm}.selected_ids"] = torch.full((count, 64), -1, dtype=torch.int16)
        if arm != "exact":
            for name in ("intersection_ids", "missed_ids", "extra_ids"):
                tables[f"{arm}.{name}"] = torch.full((count, 64), -1, dtype=torch.int16)
    metrics = {arm: [] for arm in factors}
    row = 0
    for index in indices:
        rec, queries = verify_document(args.query_capture, index, capture["protocol"])
        assert rec["sha256"] == capture["artifacts"][str(index)]["sha256"]
        inputs[f"query_{index}"] = rec["sha256"]
        current = raw[index - 64, :, group].to(device).bfloat16()
        key = current[:, 128:]
        codes = (current[:, :128] @ encoder)[None, None]
        sidecar = build_conditional_routing_sidecar(codes, key[None, None],
            base_left=source["base_left_b16"][group:group + 1],
            base_right=source["base_right_b16"][group:group + 1],
            base_bias=source["base_bias_b16"][group:group + 1],
            residual_encoder=factors["q32"][0][None], cos=cos, sin=sin)[0, 0]
        base = sidecar[:, :128]
        residual = key - base
        sidecars = {arm: torch.cat((base, residual @ e), -1) for arm, (e, _) in factors.items()}
        torch.testing.assert_close(sidecars["q32"], sidecar, rtol=0, atol=0)
        for qi in query_indices:
            stop = positions[qi] + 1
            q = queries[layer, qi, group * 4:group * 4 + 4].to(device)
            exact = selection_details((q.float() @ key[:stop].float().T) * (128 ** -.5))
            details = {"exact": exact}
            for arm, (_, u) in factors.items():
                projected = torch.einsum("hd,hdr->hr", q, conditional_routing_query_projector(u))
                scores = (projected @ sidecars[arm][:stop].T) * (128 ** -.5)
                details[arm] = selection_details(scores)
                comparison = compare_selection(exact, details[arm])
                metrics[arm].append({k: v for k, v in comparison.items() if not isinstance(v, torch.Tensor)})
                for name in ("intersection", "missed", "extra"):
                    ids = torch.where(comparison[name])[0].cpu().short()
                    tables[f"{arm}.{name}_ids"][row, :len(ids)] = ids
            page_count = stop // 32
            tables["document"][row] = index
            tables["query_position"][row] = stop - 1
            tables["page_count"][row] = page_count
            tables["exact.teacher_mass"][row, :page_count] = exact["full_mass"].mean(0).cpu()
            tables["exact.teacher_non_sink_mass"][row, :page_count] = exact["non_sink_mass"].mean(0).cpu()
            tables["exact.teacher_mass_by_head"][row, :, :page_count] = exact["full_mass"].cpu()
            tables["exact.teacher_non_sink_mass_by_head"][row, :, :page_count] = exact["non_sink_mass"].cpu()
            for arm, detail in details.items():
                for name in ("group_score", "rank_min", "rank_max", "owner", "selected"):
                    tables[f"{arm}.{name}"][row, :page_count] = detail[name].cpu()
                tables[f"{arm}.selected_ids"][row] = detail["ids"].cpu().short()
                tables[f"{arm}.cutoff"][row] = detail["cutoff"].cpu()
            row += 1
        print(f"[pages] layer={layer} group={group} window={index} rows={row}/{count}", flush=True)
    aggregate = {arm: {name: sum(r[name] for r in rows) / len(rows) for name in rows[0]}
                 for arm, rows in metrics.items()}
    output.mkdir(parents=True, exist_ok=True)
    file = output / "pages.safetensors"
    save_file(tables, str(file))
    write_json(output / "result.json", {
        "status": "complete", "stage": args.stage, "layer": layer, "group": group,
        "documents": indices, "query_positions": [positions[i] for i in query_indices],
        "rows": count, "aggregate": aggregate, "per_query_metrics": metrics,
        "inputs": inputs, "pages_sha256": sha256(file),
        "bank_paths": {arm: str(path.resolve()) for arm, path in bank_paths.items()},
        "reference": "FP32 exact QK of cached BF16 Q/K; same non-sink head-normalized GQA-max selector",
        "proxy": "native BF16 concatenated Base128+R8 sidecar; frozen C1-V80/Base16",
        "scope": "offline dense-teacher C4 diagnostic captures; no fitting, model rollout or RULER",
        "schema": "row keyed by document/query_position; only pages < page_count valid; page ID = token_start//32; -1 pads ID lists; pinned page0 rank=0; other ranks are inclusive tie intervals among routed pages; owner is local query head 0..3; cutoff is 63rd routed group score",
        "mean_mass_warning": "exact GQA-max selection need not maximize head-mean full teacher mass; missed minus extra mass can be negative",
        "command": shlex.join(sys.argv), "python": sys.executable, "gpu": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "basisserve/core/page_selection_comparison.py", "evaluation/compare_residual_selected_pages.py",
            "basisserve/core/c1_conditional_page_attention.py")},
    })
    print(json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
