"""Single-layer, single-GQA-group real-cache boundary-repair smoke."""

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
from basisserve.core.residual_page_ranking import (
    PageRoutingExample, mass_coverage, repair_page_boundary, proxy_scores,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _discover_capture, _load_direct, _rotary_embeddings,
)
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from scripts.capture_qwen3_8b_q16 import verify_document


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--c1-checkpoint", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--query-capture", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, choices=(15, 33), default=33)
    p.add_argument("--group", type=int, choices=range(8), default=0)
    p.add_argument("--calibration-root", type=Path, default=ROOT / "results/calibration")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    started = time.monotonic()
    layer, group = args.layer, args.group
    assert not (args.output_dir / "result.json").exists()
    assert not (args.output_dir / "group_factors.safetensors").exists()
    path = args.bank / f"layer_{layer:03d}.safetensors"
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["status"] == "complete" and record["sha256"] == sha256(path)
    assert record["protocol"]["base_kind"] == "closed_form_rrr"
    source = load_file(str(path))
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    c1_path = args.c1_checkpoint / c1["artifacts"][str(layer)]["file"]
    value_encoder = load_file(str(c1_path))["value_coordinate_encoders"][group].to(device).bfloat16()
    capture = json.loads((args.query_capture / "manifest.json").read_text())
    assert capture["status"] == "complete" and capture["overlap_bitwise_equal"]
    assert capture["protocol"]["model_config_sha256"] == sha256(args.model / "config.json")
    positions = capture["protocol"]["query_positions"]
    assert positions == list(range(24831, 32768, 256))
    query_indices = [7, 15, 23, 31]
    cos, sin = _rotary_embeddings(args.model, sequence=32768, device=device)
    e = source["residual_encoder_b16_r8"][group].to(device)
    u = source["residual_query_b16_r8"][4 * group:4 * group + 4].to(device)
    groups = {}
    inputs = {"bank": sha256(path), "c1": sha256(c1_path),
              "query_manifest": sha256(args.query_capture / "manifest.json")}
    for split, indices in (("fit", [0, 1]), ("validation", [64, 65])):
        dr, dm = _discover_capture(args.calibration_root, split=split, layer=layer)
        direct_hash = sha256(dr / "manifest.json")
        assert direct_hash == capture["protocol"]["inputs"][split][str(layer)]["direct_manifest_sha256"]
        inputs[f"direct_{split}"] = direct_hash
        _, all_rows = _load_direct(dr, dm, layer)
        examples = []
        for index in indices:
            query_record, query = verify_document(args.query_capture, index, capture["protocol"])
            assert query_record["sha256"] == capture["artifacts"][str(index)]["sha256"]
            inputs[f"query_{index}"] = query_record["sha256"]
            slot = index if split == "fit" else index - 64
            rows = all_rows[slot, :, group].to(device).bfloat16()
            key = rows[:, 128:]
            codes = (rows[:, :128] @ value_encoder)[None, None]
            sidecar = build_conditional_routing_sidecar(
                codes, key[None, None],
                base_left=source["base_left_b16"][group:group + 1],
                base_right=source["base_right_b16"][group:group + 1],
                base_bias=source["base_bias_b16"][group:group + 1],
                residual_encoder=e[None], cos=cos, sin=sin)[0, 0]
            base = sidecar[:, :128].contiguous()
            residual = (key - base).contiguous()
            for qi in query_indices:
                stop = positions[qi] + 1
                q = query[layer, qi, group * 4:group * 4 + 4].to(device)
                exact = (q.float() @ key[:stop].float().T) * (128 ** -.5)
                mass = exact.softmax(-1).reshape(4, -1, 32).sum(-1)
                example = PageRoutingExample(q, base[:stop], residual[:stop], mass)
                # Verify the native score assembly against the production sidecar.
                projected = torch.einsum("hd,hdr->hr", q, conditional_routing_query_projector(u.bfloat16()))
                reference = ((projected @ sidecar[:stop].T) * (128 ** -.5)).float()
                torch.testing.assert_close(proxy_scores(example, e, u, native=True), reference, rtol=0, atol=0)
                examples.append(example)
        groups[split] = examples
    before = {split: mass_coverage(examples, e, u) for split, examples in groups.items()}
    fitted_e, fitted_u, history = repair_page_boundary(groups["fit"], e, u, sweeps=2)
    after = {split: mass_coverage(examples, fitted_e, fitted_u) for split, examples in groups.items()}
    assert after["fit"]["mass"] >= before["fit"]["mass"]
    assert torch.equal(e.cpu(), source["residual_encoder_b16_r8"][group])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "group_factors.safetensors"
    save_file({"residual_encoder": fitted_e.cpu().contiguous(),
               "residual_query": fitted_u.cpu().contiguous()}, str(artifact))
    write_json(args.output_dir / "result.json", {
        "status": "complete", "layer": layer, "group": group,
        "fit_indices": [0, 1], "diagnostic_indices": [64, 65],
        "query_positions": [positions[i] for i in query_indices],
        "base": "frozen closed-form Base16", "initial_residual": "terminal-Q32 Fisher R8",
        "objective": "head-mean full teacher mass covered by native physical page set",
        "selector": "BF16 concatenated sidecar; non-sink per-head normalization; GQA max; Page32/B2048 pinned page0",
        "optimization": "two E/U alternating local squared-hinge least-squares sweeps; no autograd/Adam",
        "margin": .05, "relative_damping": .01, "maximum_pairs_per_query": 8,
        "line_search": [1, .5, .25, .125], "acceptance": "fit mass non-decrease AND fixed-pair native loss decrease",
        "diagnostic_used_for_acceptance": False,
        "scope": "offline dense-teacher raw V/K and frozen C1 encoding; not an end-to-end C1 rollout or RULER",
        "before": before, "after": after, "history": history, "inputs": inputs,
        "artifact_sha256": sha256(artifact), "command": shlex.join(sys.argv),
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "basisserve/core/residual_page_ranking.py", "evaluation/smoke_residual_page_ranking.py",
            "basisserve/core/c1_conditional_page_attention.py")},
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "python": sys.executable, "gpu": torch.cuda.get_device_name(0),
    })
    print(json.dumps({"before": before, "after": after, "history": history}, indent=2), flush=True)


if __name__ == "__main__":
    main()
