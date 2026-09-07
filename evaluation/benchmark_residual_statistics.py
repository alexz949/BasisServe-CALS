"""Small real-capture oracle/timing check for window-major Fisher statistics."""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _build_residual_statistics, _discover_capture, _load_direct, _rotary_embeddings,
)
from evaluation.fit_qwen3_8b_q8_fisher_residual import base_maps, build_multi_query_statistics
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from basisserve.core.gqa_joint_routing_payload_s80_fisher import compact_softmax_fisher_loss


@torch.inference_mode()
def reference_statistics(queries, rows, *, query_positions, cos, sin, **kwargs):
    """Single-query oracle reproducing the previous query-major execution."""
    parts, metrics = [], {}
    for index, position in enumerate(query_positions):
        result, reconstruction = _build_residual_statistics(
            queries[:, index], rows[:, :position + 1],
            cos=cos[:, :position + 1], sin=sin[:, :position + 1], **kwargs)
        parts.append(result[16])
        metrics[str(position)] = reconstruction
    return {16: replace(parts[0],
                        queries_by_head=torch.cat([s.queries_by_head for s in parts], dim=1),
                        fisher_grams_by_head=torch.cat([s.fisher_grams_by_head for s in parts], dim=1),
                        teacher_fisher_energy=sum(s.teacher_fisher_energy for s in parts))}, metrics


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--c1-checkpoint", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--query-capture", type=Path, required=True)
    p.add_argument("--calibration-root", type=Path, default=ROOT / "results/calibration")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", default="0,33")
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    assert 1 <= args.windows <= 4 and 1 <= args.repeats <= 5
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True  # Match fitting arithmetic.
    device = torch.device("cuda:0")
    cos, sin = _rotary_embeddings(args.model, sequence=32768, device=device)
    capture = json.loads((args.query_capture / "manifest.json").read_text())
    assert capture["status"] == "complete"
    positions = capture["protocol"]["query_positions"]
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    records = []
    for layer in map(int, args.layers.split(",")):
        direct, manifest = _discover_capture(args.calibration_root, split="fit", layer=layer)
        _, all_rows = _load_direct(direct, manifest, layer)
        rows = all_rows[:args.windows].clone()  # Materialize input before timing.
        queries = torch.stack([
            load_file(str(args.query_capture / f"window_{i:03d}.safetensors"))["queries"][layer]
            for i in range(args.windows)])
        source = load_file(str(args.bank / f"layer_{layer:03d}.safetensors"))
        encoder = load_file(str(args.c1_checkpoint / c1["artifacts"][str(layer)]["file"]))["value_coordinate_encoders"]
        settings = dict(value_encoder=encoder, base_maps={16: base_maps(source)},
                        page_size=32, excluded_prefix_pages=1, device=device, cos=cos, sin=sin)
        timings = {"query_major": [], "window_major": []}
        with redirect_stdout(io.StringIO()):
            build_multi_query_statistics(queries[:, -1:], rows, query_positions=positions[-1:], **settings)
            for repeat in range(args.repeats):
                outputs = {}
                order = [("query_major", reference_statistics), ("window_major", build_multi_query_statistics)]
                if repeat % 2:
                    order.reverse()
                for name, builder in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs[name] = builder(queries, rows, query_positions=positions, **settings)
                    torch.cuda.synchronize()
                    timings[name].append(time.perf_counter() - start)
                old, old_metrics = outputs["query_major"]
                new, new_metrics = outputs["window_major"]
                torch.testing.assert_close(old[16].queries_by_head, new[16].queries_by_head, rtol=0, atol=0)
                torch.testing.assert_close(old[16].fisher_grams_by_head, new[16].fisher_grams_by_head,
                                           rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(torch.tensor(old[16].teacher_fisher_energy),
                                           torch.tensor(new[16].teacher_fisher_energy), rtol=2e-5, atol=1e-7)
                for position in old_metrics:
                    for key, value in old_metrics[position][16].items():
                        torch.testing.assert_close(torch.tensor(value), torch.tensor(new_metrics[position][16][key]),
                                                   rtol=2e-5, atol=1e-7)
                factors = dict(routing_payload_encoders=source["residual_encoder_b16_r8"],
                               routing_query_factors=source["residual_query_b16_r8"])
                old_loss = compact_softmax_fisher_loss(old[16], **factors)
                new_loss = compact_softmax_fisher_loss(new[16], **factors)
                torch.testing.assert_close(torch.tensor(old_loss), torch.tensor(new_loss), rtol=2e-5, atol=1e-7)
                difference = new[16].fisher_grams_by_head - old[16].fisher_grams_by_head
                relative = float(difference.norm() / old[16].fisher_grams_by_head.norm().clamp_min(1e-30))
                maximum = float(difference.abs().max())
        medians = {k: float(torch.tensor(v, dtype=torch.float64).median()) for k, v in timings.items()}
        record = {"layer": layer, "seconds": timings, "median_seconds": medians,
                  "speedup_statistics_only": medians["query_major"] / medians["window_major"],
                  "gram_relative_frobenius_error": relative, "gram_max_abs_error": maximum,
                  "oracle_fisher_loss": float(old_loss), "window_major_fisher_loss": float(new_loss)}
        print(json.dumps(record), flush=True)
        records.append(record)
        del outputs, old, new, rows, all_rows, difference
    write_json(args.output_dir / "result.json", {
        "status": "complete", "records": records, "windows": args.windows,
        "query_positions": positions, "repeats": args.repeats,
        "scope": "statistics builder only; no BCD fit or RULER; input materialization excluded",
        "gpu": torch.cuda.get_device_name(0), "python": sys.executable,
        "torch": torch.__version__, "allow_tf32": True, "command": shlex.join(sys.argv),
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "evaluation/benchmark_residual_statistics.py", "evaluation/fit_qwen3_8b_q8_fisher_residual.py",
            "evaluation/eval_qwen3_8b_v80_conditional_residual_router.py",
            "basisserve/core/c1_v_conditional_k_router.py")},
    })


if __name__ == "__main__":
    main()
