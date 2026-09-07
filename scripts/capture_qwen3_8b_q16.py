#!/usr/bin/env python3
"""Capture nested Q16–Q128 on existing C4 windows with exact overlap checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import rotate_half

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _discover_capture
from evaluation.eval_qwen3_8b_v80_fisher_base import (
    _check_query_alignment, _discover_query_statistics, _load_query_observations,
)
from evaluation.fit_qwen3_8b_qaware_base_fisher_bank import QUERY_POSITIONS as Q8_POSITIONS
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json

TERMINAL_Q16_POSITIONS = list(range(25087, 32768, 512))
UNIFORM_Q16_POSITIONS = list(range(2047, 32768, 2048))
Q16_POSITIONS = TERMINAL_Q16_POSITIONS
FORMAT = "basisserve.qwen3.q16_queries.v1"
Q32_POSITIONS = list(range(24831, 32768, 256))
UNIFORM_Q32_POSITIONS = list(range(1023, 32768, 1024))
Q32_FORMAT = "basisserve.qwen3.q32_queries.v1"


def query_positions(layout, count=16):
    assert layout in ("terminal8k", "uniform32k")
    assert count in (16, 32, 64, 128)
    if count > 32:
        assert layout == "terminal8k"
        stride = 8192 // count
        return list(range(24576 + stride - 1, 32768, stride))
    if count == 32:
        return Q32_POSITIONS if layout == "terminal8k" else UNIFORM_Q32_POSITIONS
    return TERMINAL_Q16_POSITIONS if layout == "terminal8k" else UNIFORM_Q16_POSITIONS


def tensor_sha256(tensor):
    # Q8 is saved as FP32 values originating from BF16 teacher activations.
    return hashlib.sha256(tensor.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def selected_rotated_queries(module, hidden_states, position_embeddings, positions):
    """Project/normalize full rows like the original capture, rotate selected rows."""
    batch, sequence, _ = hidden_states.shape
    dim = module.head_dim
    query = module.q_norm(module.q_proj(hidden_states).view(batch, sequence, -1, dim))
    index = torch.as_tensor(positions, dtype=torch.long, device=query.device)
    query = query.index_select(1, index).transpose(1, 2)
    cos, sin = (item.index_select(1, index).unsqueeze(1) for item in position_embeddings)
    return (query * cos + rotate_half(query) * sin).transpose(1, 2)


def assert_q8_overlap(queries, reference, positions=Q16_POSITIONS):
    overlap = [p for p in Q8_POSITIONS if p in positions]
    assert overlap
    indices = [positions.index(p) for p in overlap]
    reference_indices = [Q8_POSITIONS.index(p) for p in overlap]
    assert torch.isfinite(queries).all()
    assert torch.equal(queries[indices].float(), reference[reference_indices].float()), "Q16/Q8 teacher trajectory mismatch"


def assert_q16_overlap(queries, reference):
    assert queries.shape[-3] in (32, 64, 128) and reference.shape[-3] == 16
    stride = queries.shape[-3] // 16
    assert torch.isfinite(queries).all()
    assert torch.equal(queries[..., stride - 1::stride, :, :].float(), reference.float()), "Q16 teacher trajectory mismatch"


def load_sources(model, calibration_root, *, layout="terminal8k", count=16):
    windows = calibration_root / "qwen3_8b_c4_64f16h_s32768/windows.safetensors"
    positions = query_positions(layout, count)
    overlap = [p for p in Q8_POSITIONS if p in positions]
    specification = {
        "format": f"basisserve.qwen3.q{count}_queries.v1", "model_config_sha256": sha256(model / "config.json"),
        "windows_sha256": sha256(windows), "sequence_length": 32768,
        "fit_windows": 64, "validation_windows": 16,
        "query_layout": layout, "query_positions": positions,
        "query_span": 8192 if layout == "terminal8k" else 32768,
        "overlap_positions": overlap, "overlap_requirement": "bitwise FP32 values at every shared old Q8 position",
        "capture": "dense BF16 teacher SDPA, batch1, full Q projection and Q norm; no C1 installation",
        "stored_dtype": "bfloat16", "stored_shape": [36, count, 32, 128],
        "source_sha256": sha256(Path(__file__)), "inputs": {},
    }
    references = {}
    for split, start, count in (("fit", 0, 64), ("validation", 64, 16)):
        references[split] = []
        specification["inputs"][split] = {}
        for layer in range(36):
            qr, qm = _discover_query_statistics(calibration_root, split=split, layer=layer)
            dr, dm = _discover_capture(calibration_root, split=split, layer=layer)
            _check_query_alignment(dm, qm)
            assert qm["source"]["model_config_sha256"] == specification["model_config_sha256"]
            assert qm["source"]["windows_sha256"] == specification["windows_sha256"]
            assert dm["calibration"]["fit_start"] == qm["calibration"]["start"] == start
            assert dm["calibration"]["routing_fit_windows"] == qm["calibration"]["documents"] == count
            queries, positions = _load_query_observations(qr, qm, layer=layer)
            assert positions.tolist() == Q8_POSITIONS and queries.shape == (count, 8, 32, 128)
            assert torch.isfinite(queries).all() and torch.equal(queries, queries.bfloat16().float())
            references[split].append(queries)
            specification["inputs"][split][str(layer)] = {
                "query_manifest_sha256": sha256(qr / "manifest.json"),
                "direct_manifest_sha256": sha256(dr / "manifest.json"),
                "query_values_sha256": tensor_sha256(queries),
            }
    return specification, references


def verify_document(root, index, specification):
    path = root / f"window_{index:03d}.safetensors"
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["status"] == "complete" and record["document"] == index
    assert record["protocol"] == specification and record["overlap_bitwise_equal"]
    assert sha256(path) == record["sha256"]
    tensors = load_file(str(path))
    assert set(tensors) == {"queries"}
    queries = tensors["queries"]
    assert list(queries.shape) == specification["stored_shape"]
    assert queries.dtype == torch.bfloat16 and torch.isfinite(queries).all()
    return record, queries


def reference_q16_queries(root, index, specification):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["overlap_bitwise_equal"]
    assert sha256(root / "manifest.json") == specification["q16_reference_manifest_sha256"]
    recorded = manifest["protocol"]
    assert recorded["format"] == FORMAT
    stride = len(specification["query_positions"]) // 16
    assert recorded["query_positions"] == specification["query_positions"][stride - 1::stride]
    for field in ("model_config_sha256", "windows_sha256", "sequence_length", "inputs",
                  "fit_windows", "validation_windows"):
        assert recorded[field] == specification[field]
    record, queries = verify_document(root, index, recorded)
    assert manifest["artifacts"][str(index)]["sha256"] == record["sha256"]
    assert manifest["artifacts"][str(index)]["file"] == f"window_{index:03d}.safetensors"
    return queries


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--calibration-root", type=Path, default=ROOT / "results/calibration")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/calibration/q8_q16_queries")
    p.add_argument("--stage", choices=("smoke", "capture", "summarize", "candidates", "selected"), required=True)
    p.add_argument("--candidate-window-count", type=int, default=64)
    p.add_argument("--candidate-stride", type=int, default=64)
    p.add_argument("--candidate-layers", type=int, nargs="+", default=list(range(36)))
    p.add_argument("--windows", type=Path)
    p.add_argument("--query-position-manifest", type=Path)
    p.add_argument("--query-layout", choices=("terminal8k", "uniform32k"), default="terminal8k")
    p.add_argument("--query-count", type=int, choices=(16, 32, 64, 128), default=16)
    p.add_argument("--q16-reference", type=Path, help="Required immutable Q16 reference for denser capture")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--torch-num-threads", type=int, default=2)
    return p


@torch.inference_mode()
def main():
    args = parser().parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage in ("candidates", "selected"):
        from scripts.query_position_capture import capture_position_queries
        capture_position_queries(args)
        return
    specification, references = load_sources(args.model, args.calibration_root,
                                            layout=args.query_layout, count=args.query_count)
    if args.query_count > 16:
        assert args.q16_reference is not None
        assert args.output_dir.resolve() != args.q16_reference.resolve()
        specification["q16_reference"] = str(args.q16_reference.resolve())
        specification["q16_reference_manifest_sha256"] = sha256(args.q16_reference / "manifest.json")
    positions = specification["query_positions"]
    if args.stage == "summarize":
        artifacts = {}
        for index in range(80):
            record, queries = verify_document(args.output_dir, index, specification)
            if args.query_count > 16:
                assert_q16_overlap(queries, reference_q16_queries(args.q16_reference, index, specification))
            split, slot = ("fit", index) if index < 64 else ("validation", index - 64)
            for layer in range(36):
                assert_q8_overlap(queries[layer], references[split][layer][slot], positions)
            artifacts[str(index)] = {"file": f"window_{index:03d}.safetensors", "sha256": record["sha256"]}
        write_json(args.output_dir / "manifest.json", {
            "status": "complete", "protocol": specification, "artifacts": artifacts,
            "overlap_bitwise_equal": True, "command": shlex.join(sys.argv),
        })
        print(f"Q{args.query_count} AUDIT PASSED: 80 windows, all 36 layers, required overlaps bitwise equal", flush=True)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA L40S"
    root = args.output_dir / "smoke" if args.stage == "smoke" else args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="sdpa", device_map={"": 0},
    ).eval()
    model.config.use_cache = False
    assert len(model.model.layers) == 36
    windows = load_file(str(args.calibration_root / "qwen3_8b_c4_64f16h_s32768/windows.safetensors"))["input_ids"]
    assert windows.shape == (80, 32768)
    captured = {}
    active = {}

    def hook(layer):
        def collect(module, positional, kwargs):
            hidden = kwargs.get("hidden_states", positional[0] if positional else None)
            query = selected_rotated_queries(module, hidden, kwargs["position_embeddings"], positions)[0].cpu()
            assert_q8_overlap(query, references[active["split"]][layer][active["slot"]], positions)
            if args.query_count > 16:
                assert_q16_overlap(query, active["q16"][layer])
            assert torch.equal(query.float(), query.bfloat16().float())
            captured[layer] = query.bfloat16().contiguous()
        return collect

    handles = [layer.self_attn.register_forward_pre_hook(hook(i), with_kwargs=True)
               for i, layer in enumerate(model.model.layers)]
    indices = [0] if args.stage == "smoke" else list(range(args.shard_index, 80, args.num_shards))
    for index in indices:
        active["split"], active["slot"] = ("fit", index) if index < 64 else ("validation", index - 64)
        if args.query_count > 16:
            active["q16"] = reference_q16_queries(args.q16_reference, index, specification)
        path = root / f"window_{index:03d}.safetensors"
        if path.with_suffix(".json").exists():
            _, queries = verify_document(root, index, specification)
            if args.query_count > 16:
                assert_q16_overlap(queries, active["q16"])
            for layer in range(36):
                assert_q8_overlap(queries[layer], references[active["split"]][layer][active["slot"]], positions)
            print(f"[Q{args.query_count} resume] window={index}", flush=True)
            continue
        started = time.monotonic()
        captured.clear()
        torch.cuda.reset_peak_memory_stats()
        output = model.model(input_ids=windows[index:index + 1].to("cuda:0"), use_cache=False)
        del output
        assert set(captured) == set(range(36))
        temporary = path.with_suffix(".tmp")
        save_file({"queries": torch.stack([captured[layer] for layer in range(36)])}, str(temporary))
        temporary.replace(path)
        record = {"status": "complete", "document": index, "protocol": specification,
                  "sha256": sha256(path), "overlap_bitwise_equal": True,
                  "wall_seconds": time.monotonic() - started,
                  "maximum_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "command": shlex.join(sys.argv), "python": sys.executable,
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
        write_json(path.with_suffix(".json"), record)
        print(f"[Q{args.query_count} captured] window={index} seconds={record['wall_seconds']:.2f}", flush=True)
    for handle in handles:
        handle.remove()
    write_json(root / f"shard_{args.shard_index}.json", {
        "status": "complete", "protocol": specification, "documents": indices,
    })


if __name__ == "__main__":
    main()
