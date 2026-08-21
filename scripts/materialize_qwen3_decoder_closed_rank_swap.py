#!/usr/bin/env python3
"""Materialize an accepted DC-GKL one-swap endpoint as a full rank bank."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_qwen3_decoder_closed_rank_swap_candidates import (
    CANDIDATE_FORMAT,
    METHOD,
)
from scripts.build_qwen3_gqa_routed_ov_solver_ablation import (
    LAYER_FORMAT,
    _SafetensorWeightReader,
    _write_json,
)


FORMAT = "basisserve.dc_gkl_swap.materialized_endpoint.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-rank-bank", required=True)
    parser.add_argument("--base-rank-bank", required=True)
    parser.add_argument("--base-schedule", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--audit-decision", required=True)
    parser.add_argument("--output-rank-bank", required=True)
    parser.add_argument("--output-schedule", required=True)
    parser.add_argument(
        "--refinement-iteration",
        type=int,
        default=None,
        help="1-based sequential swap index; defaults to parent history length plus one",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_schedule_sha256(selected_ranks: Any) -> str:
    normalized = [list(map(int, layer)) for layer in selected_ranks]
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _refinement_history(schedule_payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = schedule_payload.get("dc_gkl_refinements")
    if history is None:
        legacy = schedule_payload.get("dc_gkl_refinement")
        history = [] if legacy is None else [legacy]
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("invalid DC-GKL refinement history")
    return [dict(item) for item in history]


def _materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = parse_args()
    model_root = Path(args.model).expanduser().resolve()
    source_root = Path(args.source_rank_bank).expanduser().resolve()
    base_root = Path(args.base_rank_bank).expanduser().resolve()
    base_schedule_path = Path(args.base_schedule).expanduser().resolve()
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    decision_path = Path(args.audit_decision).expanduser().resolve()
    output_bank = Path(args.output_rank_bank).expanduser().resolve()
    output_schedule = Path(args.output_schedule).expanduser().resolve()
    if output_bank.exists() or output_schedule.exists():
        raise FileExistsError("refusing to overwrite a materialized DC-GKL output")
    for required in (
        model_root / "config.json",
        source_root / "config.json",
        base_root / "config.json",
        base_schedule_path,
        candidate_dir / "candidate_index.json",
        decision_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
    decision = decision_payload.get("decision")
    if not decision or not decision.get("accepted"):
        raise RuntimeError("the final audit did not accept a rank swap")
    selected = decision_payload["candidates"]
    if len(selected) != 1 or selected[0]["candidate_id"] != decision["candidate_id"]:
        raise ValueError("audit decision/candidate mismatch")
    candidate = selected[0]
    factor_path = candidate_dir / candidate["factor_path"]
    if _sha256(factor_path) != candidate["factor_sha256"]:
        raise RuntimeError("accepted candidate factor checksum mismatch")
    factors = torch.load(factor_path, map_location="cpu", weights_only=True)
    if factors.get("format") != CANDIDATE_FORMAT:
        raise ValueError("unsupported accepted candidate factor format")
    affected_layer = int(factors["layer_index"])
    before = tuple(map(int, factors["ranks_before"]))
    after = tuple(map(int, factors["ranks_after"]))

    model_config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    num_layers = int(model_config["num_hidden_layers"])
    hidden_size = int(model_config["hidden_size"])
    num_heads = int(model_config["num_attention_heads"])
    num_groups = int(model_config["num_key_value_heads"])
    head_dim = int(model_config.get("head_dim", hidden_size // num_heads))
    heads_per_group = num_heads // num_groups
    schedule_payload = json.loads(base_schedule_path.read_text(encoding="utf-8"))
    schedule = [list(map(int, layer)) for layer in schedule_payload["selected_ranks"]]
    history = _refinement_history(schedule_payload)
    inferred_iteration = len(history) + 1
    refinement_iteration = (
        inferred_iteration
        if args.refinement_iteration is None
        else int(args.refinement_iteration)
    )
    if refinement_iteration != inferred_iteration:
        raise ValueError(
            f"refinement iteration {refinement_iteration} does not follow "
            f"parent history length {len(history)}"
        )
    if tuple(schedule[affected_layer]) != before:
        raise ValueError("accepted candidate does not start from the base schedule")
    schedule[affected_layer] = list(after)
    changed = [
        (layer, group)
        for layer in range(num_layers)
        for group in range(num_groups)
        if int(schedule_payload["selected_ranks"][layer][group]) != schedule[layer][group]
    ]
    if len(changed) != 2 or {group for layer, group in changed if layer == affected_layer} != {
        int(candidate["receiver_group"]),
        int(candidate["donor_group"]),
    }:
        raise ValueError("materialized schedule must change exactly two coordinates")
    old_rank_sum = sum(sum(map(int, layer)) for layer in schedule_payload["selected_ranks"])
    new_rank_sum = sum(sum(layer) for layer in schedule)
    if old_rank_sum != 18432 or new_rank_sum != old_rank_sum:
        raise ValueError("accepted rank swap changed the global rank budget")
    if sum(before) != sum(after):
        raise ValueError("accepted rank swap changed the layer rank budget")

    current_rank_schedule_sha = _rank_schedule_sha256(
        schedule_payload["selected_ranks"]
    )
    new_rank_schedule_sha = _rank_schedule_sha256(schedule)
    lineage = [str(value) for value in schedule_payload.get("rank_schedule_lineage", [])]
    if not lineage:
        base_config_for_lineage = json.loads(
            (base_root / "config.json").read_text(encoding="utf-8")
        )
        parent_schedule = base_config_for_lineage.get("base_schedule")
        if parent_schedule is not None:
            parent_path = Path(parent_schedule).expanduser().resolve()
            if parent_path.is_file():
                parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
                parent_sha = _rank_schedule_sha256(parent_payload["selected_ranks"])
                if parent_sha != current_rank_schedule_sha:
                    lineage.append(parent_sha)
        lineage.append(current_rank_schedule_sha)
    elif lineage[-1] != current_rank_schedule_sha:
        raise ValueError("parent rank-schedule lineage does not end at the base schedule")
    if new_rank_schedule_sha in lineage:
        raise ValueError("accepted rank swap would revisit an earlier rank schedule")
    lineage.append(new_rank_schedule_sha)

    base_config = json.loads((base_root / "config.json").read_text(encoding="utf-8"))
    base_profile = json.loads(
        (base_root / base_config["profile"]).read_text(encoding="utf-8")
    )
    source_config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    source_profile = json.loads(
        (source_root / source_config["profile"]).read_text(encoding="utf-8")
    )
    reader = _SafetensorWeightReader(model_root)
    materialization = Counter()
    output_bank.mkdir(parents=True)
    profile_layers: dict[str, Any] = {}
    for layer in range(num_layers):
        if layer != affected_layer:
            ranks_payload = {}
            for rank_text, entry in base_profile["layers"][str(layer)]["ranks"].items():
                source = base_root / entry["factor_path"]
                destination = output_bank / entry["factor_path"]
                materialization[_materialize(source, destination)] += 1
                ranks_payload[rank_text] = dict(entry)
            profile_layers[str(layer)] = {
                "module_name": base_profile["layers"][str(layer)]["module_name"],
                "ranks": ranks_payload,
            }
            continue

        module_name = base_profile["layers"][str(layer)]["module_name"]
        prefix = f"{module_name}."
        dense_v = reader.tensor(prefix + "v_proj.weight")
        dense_o = reader.tensor(prefix + "o_proj.weight")
        dense_v_bias = reader.tensor(prefix + "v_proj.bias", required=False)
        o_bias = reader.tensor(prefix + "o_proj.bias", required=False)
        if dense_v is None or dense_o is None or dense_v_bias is not None:
            raise ValueError("unsupported dense V/O geometry for materialization")
        ranks_payload = {}
        for rank in sorted(set(after)):
            if rank == head_dim:
                base_v = dense_v.detach().cpu().to(torch.bfloat16).clone()
                base_o = dense_o.detach().cpu().to(torch.bfloat16).clone()
            else:
                entry = source_profile["layers"][str(layer)]["ranks"].get(str(rank))
                if entry is None:
                    raise KeyError(f"source bank lacks layer {layer} rank {rank}")
                payload = torch.load(
                    source_root / entry["factor_path"],
                    map_location="cpu",
                    weights_only=True,
                )
                base_v = payload["v_proj_compressed_weight"].clone().to(torch.bfloat16)
                base_o = payload["o_decoder_weight"].clone().to(torch.bfloat16)
            for group, selected_rank in enumerate(after):
                if selected_rank != rank:
                    continue
                base_v[group * rank : (group + 1) * rank] = factors[
                    "v_group_weights"
                ][group]
                first_head = group * heads_per_group
                base_o[
                    :,
                    first_head * rank : (first_head + heads_per_group) * rank,
                ] = factors["o_group_weights"][group]
            relative = Path(f"layer_{layer:04d}") / f"rank_{rank:04d}.pt"
            destination = output_bank / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(
                {
                    "format": LAYER_FORMAT,
                    "method": METHOD,
                    "endpoint": f"dc_gkl_swap_iter{refinement_iteration}",
                    "module_name": module_name,
                    "layer_index": layer,
                    "rank_per_kv_head": rank,
                    "v_proj_compressed_weight": base_v,
                    "v_proj_compressed_bias": None,
                    "o_decoder_weight": base_o,
                    "o_decoder_bias": (
                        None if o_bias is None else o_bias.detach().cpu().to(torch.bfloat16)
                    ),
                    "diagnostics": factors["diagnostics"],
                },
                temporary,
            )
            os.replace(temporary, destination)
            ranks_payload[str(rank)] = {
                "factor_path": str(relative),
                "rank": rank,
                "physical_v_width": num_groups * rank,
                "schedule_specific": True,
            }
        profile_layers[str(layer)] = {"module_name": module_name, "ranks": ranks_payload}

    candidate_ranks = sorted({rank for layer in schedule for rank in layer})
    profile = {
        "format": "basisserve.a3_gqa_vo.rank_profile.v1",
        "schema_version": 1,
        "method": METHOD,
        "model": str(model_root),
        "candidate_ranks": candidate_ranks,
        "model_config": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_query_heads": num_heads,
            "num_kv_heads": num_groups,
            "head_dim": head_dim,
        },
        "layers": profile_layers,
    }
    bank_config = {
        "format": "basisserve.a3_gqa_vo.rank_bank.v1",
        "model": str(model_root),
        "model_type": "qwen3",
        "profile": "profile.json",
        "candidate_ranks": candidate_ranks,
        "method": METHOD,
        "endpoint": f"dc_gkl_swap_iter{refinement_iteration}",
        "refinement_iteration": refinement_iteration,
        "source_rank_bank": str(source_root),
        "base_rank_bank": str(base_root),
        "base_schedule": str(base_schedule_path),
        "audit_decision": str(decision_path),
        "candidate_id": candidate["candidate_id"],
        "full_rank_factor_overrides": True,
        "alpha": 0.75,
        "head_coupling_mode": "full_layer",
        "rank_sum": new_rank_sum,
        "materialization": dict(materialization),
    }
    _write_json(output_bank / "profile.json", profile)

    histogram = Counter(rank for layer in schedule for rank in layer)
    new_schedule = dict(schedule_payload)
    # The source DP contributions and scalar proxy describe the pre-swap
    # schedule.  Keeping them in the refined artifact would silently attach
    # stale per-coordinate costs to the new ranks.
    new_schedule.pop("contributions", None)
    new_schedule.pop("predicted_proxy_error", None)
    refinement_record = {
        "iteration": refinement_iteration,
        "method": METHOD,
        "candidate_id": candidate["candidate_id"],
        "affected_layer": affected_layer,
        "receiver_group": int(candidate["receiver_group"]),
        "donor_group": int(candidate["donor_group"]),
        "ranks_before": list(before),
        "ranks_after": list(after),
        "rank_schedule_sha256_before": current_rank_schedule_sha,
        "rank_schedule_sha256_after": new_rank_schedule_sha,
        "audit_decision": str(decision_path),
    }
    new_schedule.update(
        {
            "allocator": "decoder_closed_conditional_global_kl_sequential_swap",
            "schedule_name": f"dc_gkl_swap_iter{refinement_iteration}",
            "selected_ranks": schedule,
            "rank_histogram": {
                str(rank): histogram.get(rank, 0)
                for rank in sorted(map(int, source_config["candidate_ranks"]))
            },
            "dc_gkl_refinement": refinement_record,
            "dc_gkl_refinements": [*history, refinement_record],
            "refinement_iteration": refinement_iteration,
            "rank_schedule_lineage": lineage,
        }
    )
    new_schedule["budget"] = dict(new_schedule["budget"])
    new_schedule["budget"]["rank_sum"] = new_rank_sum
    _write_json(output_schedule, new_schedule)
    bank_config.update(
        {
            "schedule": str(output_schedule),
            "schedule_sha256": _sha256(output_schedule),
            "rank_schedule_sha256": new_rank_schedule_sha,
            "rank_schedule_lineage": lineage,
            "refinement_history": [*history, refinement_record],
        }
    )
    _write_json(output_bank / "config.json", bank_config)
    (output_bank / "BUILD_COMPLETE").touch()
    report = {
        "format": FORMAT,
        "status": "complete",
        "refinement_iteration": refinement_iteration,
        "refinement_history": [*history, refinement_record],
        "candidate_id": candidate["candidate_id"],
        "affected_layer": affected_layer,
        "ranks_before": list(before),
        "ranks_after": list(after),
        "changed_coordinates": changed,
        "rank_sum_before": old_rank_sum,
        "rank_sum_after": new_rank_sum,
        "layer_rank_sum_before": sum(before),
        "layer_rank_sum_after": sum(after),
        "v_cache_width_unchanged": sum(before) == sum(after),
        "compressed_o_width_unchanged": (
            heads_per_group * sum(before) == heads_per_group * sum(after)
        ),
        "compressed_parameter_count_before": (
            (1 + heads_per_group) * hidden_size * old_rank_sum
        ),
        "compressed_parameter_count_after": (
            (1 + heads_per_group) * hidden_size * new_rank_sum
        ),
        "output_rank_bank": str(output_bank),
        "output_schedule": str(output_schedule),
        "output_schedule_sha256": _sha256(output_schedule),
    }
    _write_json(output_bank / "materialization_report.json", report)
    print(
        f"[Done] candidate={candidate['candidate_id']} layer={affected_layer} "
        f"rank_sum={new_rank_sum} bank={output_bank} schedule={output_schedule}",
        flush=True,
    )


if __name__ == "__main__":
    main()
