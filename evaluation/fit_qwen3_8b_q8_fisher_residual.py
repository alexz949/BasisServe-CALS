#!/usr/bin/env python3
"""Freeze Base16 and refit uniform R8 with captured causal Q positions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.fit_qwen3_8b_qaware_base_fisher_bank import (
    QUERY_POSITIONS, base_maps, parser as base_parser, protocol as source_protocol,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _apply_base, _discover_capture, _fit_residual_grid, _load_direct,
    _post_rope_rows, _rotary_embeddings, _stack_base_map, _value_codes,
)
from basisserve.core.c1_v_conditional_k_router import residual_page_fisher_gram
from basisserve.core.gqa_joint_routing_payload_s80_fisher import S80CompactSoftmaxFisherRouting
from evaluation.eval_qwen3_8b_v80_fisher_base import (
    _discover_query_statistics, _load_query_observations,
)
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from scripts.capture_qwen3_8b_q16 import (
    Q16_POSITIONS, Q32_POSITIONS, UNIFORM_Q32_POSITIONS,
    assert_q8_overlap, assert_q16_overlap, load_sources, reference_q16_queries, verify_document, query_positions,
)


def parser():
    p = base_parser()
    p.description = __doc__
    p.set_defaults(initial_bank=ROOT / "results/checkpoints/q8_qbase_fisher_bank",
                   output_dir=ROOT / "results/checkpoints/q8_qbase_fisher8_r8")
    p.add_argument("--query-capture", type=Path,
                   help="Audited terminal Q16–Q128 or uniform Q32 capture; omit for existing Q8")
    p.add_argument("--base-kind", choices=("qaware", "closed_form_rrr"), default="qaware")
    p.add_argument("--query-position-manifest", type=Path)
    p.add_argument("--sampler-smoke", action="store_true", help="Build one group of statistics only; never fit factors")
    p.add_argument("--smoke-layer", type=int, default=15)
    p.add_argument("--smoke-group", type=int, default=0)
    return p


def frozen_base_specification(frozen, base_kind):
    assert frozen["base_rank"] == 16
    assert frozen["page_size"] == 32 and frozen["excluded_prefix_pages"] == 1
    if base_kind == "qaware":
        assert frozen["format"] == "basisserve.qaware_base_fisher_bank.v1"
        assert frozen["base_query_positions"] == QUERY_POSITIONS
        assert frozen["residual_query_positions"] == [32767]
        return {"base_kind": base_kind,
                "base_selection": "frozen completed Q-aware Base16; no Base update in this run"}
    assert base_kind == "closed_form_rrr"
    assert frozen["format"] == "basisserve.residual_kl_bank.v1"
    assert frozen["fit_windows"] == 64 and frozen["sequence_length"] == 32768
    assert frozen["fit_query"] == "last token of each window"
    return {
        "format": "basisserve.closed_form_base_fisher_bank.v1",
        "base_kind": base_kind,
        "base_objective": "unweighted affine rank-16 pre-RoPE Key reconstruction MSE",
        "base_query_positions": [],
        "base_original_solver": "closed-form affine RRR by whitening and truncated SVD",
        "base_selection": "frozen MSE-RRR Base16; no Base update or validation selection in this run",
    }


def expanded_queries(root, split, layer, positions=Q16_POSITIONS):
    assert split in ("fit", "validation") and 0 <= layer < 36
    indices = range(64) if split == "fit" else range(64, 80)
    queries = []
    for index in indices:
        with safe_open(root / f"window_{index:03d}.safetensors", framework="pt", device="cpu") as handle:
            queries.append(handle.get_slice("queries")[layer])
    return torch.stack(queries), torch.tensor(positions, dtype=torch.long)


@torch.inference_mode()
def build_multi_query_statistics(queries, rows, *, query_positions, cos, sin,
                                 value_encoder, base_maps, page_size,
                                 excluded_prefix_pages, device,
                                 excluded_recent_tokens=0):
    """Compute token features once/window; retain exact per-Q causal Grams.

    Output order stays query-position-major, document-minor. The teacher
    softmax, page grouping and Fisher kernel are unchanged. Only Q-independent
    C1/Base/RoPE/residual work is moved outside the query loop.
    """
    assert queries.ndim == 4 and queries.shape[0] == rows.shape[0]
    positions = torch.as_tensor(query_positions, dtype=torch.long).tolist()
    assert positions and len(positions) == queries.shape[1] and len(set(positions)) == len(positions)
    assert all(page_size * excluded_prefix_pages <= p < rows.shape[1] for p in positions)
    # Tokens the deployed selector always retains never compete for a routed page,
    # so they are dropped from the causal prefix the Fisher statistics are built on.
    assert excluded_recent_tokens >= 0
    assert all(p + 1 - excluded_recent_tokens > page_size * excluded_prefix_pages for p in positions)
    documents, query_count, heads, dim = map(int, queries.shape)
    groups = int(rows.shape[2])
    assert rows.ndim == 4 and rows.shape[-1] == 2 * dim and heads % groups == 0
    heads_per_group = heads // groups
    stop = max(positions) + 1
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    factors = {rank: _stack_base_map(maps, device=device) for rank, maps in base_maps.items()}
    grams = {rank: torch.empty(heads, query_count * documents, dim, dim, dtype=torch.float32)
             for rank in factors}
    # Keep the original per-query summation order for the scalar diagnostics.
    energy = {rank: [0.0] * query_count for rank in factors}
    errors = {rank: [0.0] * query_count for rank in factors}
    key_energy = [0.0] * query_count
    scaling = dim**-0.5
    for document in range(documents):
        print(f"  residual features/Fisher window={document + 1}/{documents} Q={query_count}", flush=True)
        current = rows[document, :stop].to(device=device, dtype=torch.float32)
        current_queries = queries[document].to(device=device, dtype=torch.float32)
        exact_key = current[..., dim:]
        codes = _value_codes(current[..., :dim], encoder)
        for index, position in enumerate(positions):
            key_energy[index] += float(exact_key[:position + 1].square().sum())
        for rank, factor in factors.items():
            base_pre = _apply_base(codes, factor)
            base_post = _post_rope_rows(base_pre, cos[:, :stop], sin[:, :stop])
            residual = exact_key - base_post
            for index, position in enumerate(positions):
                prefix = position + 1 - excluded_recent_tokens
                errors[rank][index] += float(residual[:prefix].square().sum())
                slot = index * documents + document
                for group in range(groups):
                    first = group * heads_per_group
                    last = first + heads_per_group
                    gram, teacher_energy = residual_page_fisher_gram(
                        current_queries[index, first:last], exact_key[:prefix, group],
                        residual[:prefix, group], scaling=scaling, page_size=page_size,
                        excluded_prefix_pages=excluded_prefix_pages,
                    )
                    grams[rank][first:last, slot].copy_(gram.float().cpu())
                    energy[rank][index] += teacher_energy
            del base_pre, base_post, residual
        del current, current_queries, exact_key, codes
    query_rows = queries.permute(2, 1, 0, 3).reshape(heads, query_count * documents, dim).contiguous().float()
    combined = {
        rank: S80CompactSoftmaxFisherRouting(
            queries_by_head=query_rows, fisher_grams_by_head=grams[rank],
            head_to_kv_group=torch.arange(heads, dtype=torch.long) // heads_per_group,
            value_dim=0, key_dim=dim, scaling=scaling,
            teacher_fisher_energy=sum(energy[rank]),
        ) for rank in factors
    }
    reconstruction = {
        str(position): {
            rank: {"post_rope_relative_mse": errors[rank][index] / key_energy[index],
                   "residual_squared_error": errors[rank][index],
                   "exact_key_squared_energy": key_energy[index]}
            for rank in factors
        } for index, position in enumerate(positions)
    }
    return combined, reconstruction


def protocol(args):
    specification = source_protocol(args)
    frozen = json.loads((args.initial_bank / "layer_000.json").read_text())["protocol"]
    base_specification = frozen_base_specification(frozen, args.base_kind)
    for layer in range(36):
        record = json.loads((args.initial_bank / f"layer_{layer:03d}.json").read_text())
        assert record["protocol"] == frozen
        if args.base_kind == "closed_form_rrr":
            paths = sorted(Path(frozen["base_root"]).glob(f"shard_*/layer_{layer:03d}.safetensors"))
            assert len(paths) == 1
            original = load_file(str(paths[0]))
            current = load_file(str(args.initial_bank / f"layer_{layer:03d}.safetensors"))
            for name in ("base_left_b16", "base_right_b16", "base_bias_b16"):
                assert original[name].dtype == current[name].dtype == torch.float32
                assert torch.equal(original[name], current[name])
            specification["inputs"][str(layer)]["closed_form_base_sha256"] = sha256(paths[0])
    specification.update({
        "ranks": [8],
        "base_optimizer": None,
        "base_selection": "frozen completed Q-aware Base16; no Base update in this run",
        "frozen_base_bank": str(args.initial_bank.resolve()),
        "frozen_base_protocol": frozen,
        "residual_query_positions": QUERY_POSITIONS,
        "residual_objective": "sum of separate causal exact-teacher non-sink Page-Fisher losses",
        "residual_example_order": "query position major, document minor; equal example weights",
        "residual_statistics_execution": "window-major; C1/Base/RoPE/residual once per window; unchanged per-Q causal Fisher kernel",
        "residual_fit_examples": 64 * 8,
        "residual_validation_examples": 16 * 8,
        "rank_allocation": "none; uniform R8",
    })
    specification.update(base_specification)
    specification["code_sha256"]["evaluation/fit_qwen3_8b_q8_fisher_residual.py"] = sha256(Path(__file__))
    if args.query_position_manifest is not None:
        from evaluation.select_query_positions import load_position_manifest
        selected = load_position_manifest(args.query_position_manifest)
        assert args.base_kind == "closed_form_rrr" and args.query_capture is not None
        assert selected['num_fit_windows'] == 64 and set(selected['layers']) == set(map(str,range(36)))
        assert selected['model_config_sha256'] == specification['model_config_sha256']
        assert selected['context_length'] == specification['sequence_length'] == 32768
        path = args.query_capture/'manifest.json'
        capture = json.loads(path.read_text())
        assert capture['status'] == 'complete' and capture['protocol']['format'] == 'basisserve.selected_queries.v1'
        assert capture['protocol']['position_manifest_sha256'] == sha256(args.query_position_manifest)
        assert capture['protocol']['documents'] == list(range(80))
        assert capture['protocol']['fit_token_sha256'] == selected['fit_token_sha256']
        input_tokens = load_file(str(args.calibration_root/'qwen3_8b_c4_64f16h_s32768/windows.safetensors'))['input_ids']
        assert hashlib.sha256(input_tokens[:64].contiguous().numpy().tobytes()).hexdigest() == selected['fit_token_sha256']
        assert capture['protocol']['document_token_sha256'] == {
            str(d):hashlib.sha256(input_tokens[d].contiguous().numpy().tobytes()).hexdigest() for d in range(80)}
        positions = {l:r['selected_positions'] for l,r in selected['layers'].items()}
        assert capture['protocol']['positions_by_layer'] == positions
        query_count = selected['num_bins'] * selected['queries_per_bin']
        assert query_count in (8,32) and all(len(p)==query_count for p in positions.values())
        specification.update({'residual_query_positions':positions, 'residual_query_count':query_count,
            'residual_fit_examples':64*query_count,'residual_validation_examples':16*query_count,
            'query_position_manifest':str(args.query_position_manifest.resolve()),
            'query_position_manifest_sha256':sha256(args.query_position_manifest),
            'query_capture':str(args.query_capture.resolve()),'query_capture_manifest_sha256':sha256(path),
            'query_capture_protocol':capture['protocol'],
            'query_capture_validation':'layer-specific sorted positions, immutable artifact hashes checked while loading; fit-only sampling'})
    elif args.query_capture is not None:
        path = args.query_capture / "manifest.json"
        capture = json.loads(path.read_text())
        assert capture["status"] == "complete" and capture["overlap_bitwise_equal"]
        captured = capture["protocol"]
        positions = captured["query_positions"]
        query_count = len(positions)
        assert positions in (Q16_POSITIONS, Q32_POSITIONS, UNIFORM_Q32_POSITIONS,
                             query_positions("terminal8k", 64), query_positions("terminal8k", 128))
        layout = "uniform32k" if positions == UNIFORM_Q32_POSITIONS else "terminal8k"
        expected, references = load_sources(args.model, args.calibration_root, count=query_count, layout=layout)
        assert captured["format"] == expected["format"]
        # Reuse immutable observations, not the current generator's source identity.
        # Original source provenance stays in the captured protocol; numerical
        # agreement with every current Q8 reference is checked again below.
        for field in ("model_config_sha256", "windows_sha256", "sequence_length",
                      "fit_windows", "validation_windows", "query_positions", "query_span",
                      "overlap_positions", "capture", "stored_dtype", "stored_shape", "inputs"):
            assert captured[field] == expected[field]
        assert set(capture["artifacts"]) == {str(i) for i in range(80)}
        for index in range(80):
            record, queries = verify_document(args.query_capture, index, captured)
            if query_count > 16:
                assert_q16_overlap(queries, reference_q16_queries(Path(captured["q16_reference"]), index, captured))
            assert capture["artifacts"][str(index)]["file"] == f"window_{index:03d}.safetensors"
            assert capture["artifacts"][str(index)]["sha256"] == record["sha256"]
            split, slot = ("fit", index) if index < 64 else ("validation", index - 64)
            for layer in range(36):
                assert_q8_overlap(queries[layer], references[split][layer][slot], positions)
        specification.update({
            "residual_query_positions": positions,
            "residual_fit_examples": 64 * query_count, "residual_validation_examples": 16 * query_count,
            "query_capture": str(args.query_capture.resolve()),
            "query_capture_manifest_sha256": sha256(path), "query_capture_protocol": captured,
            "query_capture_validation": "immutable inputs and all 80 x 36 Q8 overlaps reverified bitwise; denser captures also check every Q16 overlap",
        })
        specification["code_sha256"]["scripts/capture_qwen3_8b_q16.py"] = expected["source_sha256"]
    return specification


@torch.inference_mode()
def fit_layer(args, layer, cos, sin, device):
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    encoder = load_file(str(args.c1_checkpoint / c1["artifacts"][str(layer)]["file"]))[
        "value_coordinate_encoders"]
    source = load_file(str(args.initial_bank / f"layer_{layer:03d}.safetensors"))
    maps = base_maps(source)
    statistics, reconstruction = {}, {}
    for split, count in (("fit", 64), ("validation", 16)):
        dr, dm = _discover_capture(args.calibration_root, split=split, layer=layer)
        _, rows = _load_direct(dr, dm, layer)
        if args.query_position_manifest is not None:
            from evaluation.select_query_positions import manifest_queries
            queries, positions = manifest_queries(args.query_position_manifest,args.query_capture,split,layer)
            expected_positions = positions.tolist()
        elif args.query_capture is None:
            qr, qm = _discover_query_statistics(args.calibration_root, split=split, layer=layer)
            queries, positions = _load_query_observations(qr, qm, layer=layer)
            expected_positions = QUERY_POSITIONS
        else:
            manifest = json.loads((args.query_capture / "manifest.json").read_text())
            expected_positions = manifest["protocol"]["query_positions"]
            assert expected_positions in (Q16_POSITIONS, Q32_POSITIONS, UNIFORM_Q32_POSITIONS,
                                          query_positions("terminal8k", 64), query_positions("terminal8k", 128))
            queries, positions = expanded_queries(args.query_capture, split, layer, positions=expected_positions)
        assert positions.tolist() == expected_positions
        assert queries.shape == (count, len(expected_positions), 32, 128) and torch.isfinite(queries).all()
        assert rows.shape == (count, 32768, 8, 256)
        statistics[split], reconstruction[split] = build_multi_query_statistics(
            queries, rows, query_positions=positions, value_encoder=encoder,
            base_maps={16: maps}, cos=cos, sin=sin, page_size=32,
            excluded_prefix_pages=1, device=device,
        )
    residuals, diagnostics = _fit_residual_grid(
        statistics["fit"], statistics["validation"], residual_ranks=(8,),
        sweeps=40, relative_damping=1e-5, iterative_tolerance=1e-5,
        iterative_max_iterations=100, device=device,
    )
    # Copy Base tensors exactly, without balancing, truncation or dtype conversion.
    output = {name: source[name].clone() for name in
              ("base_left_b16", "base_right_b16", "base_bias_b16")}
    output["residual_encoder_b16_r8"], output["residual_query_b16_r8"] = residuals[(16, 8)]
    assert all(torch.isfinite(value).all() for value in output.values())
    assert all(torch.equal(output[name], source[name]) for name in output if name.startswith("base_"))
    return output, {"residual": diagnostics, "reconstruction_by_query_position": reconstruction,
                    "base_bitwise_equal_to_source": True}


@torch.inference_mode()
def sampler_smoke(args):
    from evaluation.select_query_positions import load_position_manifest, manifest_queries
    selected = load_position_manifest(args.query_position_manifest)
    assert args.base_kind == 'closed_form_rrr' and selected['num_fit_windows'] <= 4
    assert args.query_capture is not None and 0 <= args.smoke_group < 8
    assert selected['model_config_sha256'] == sha256(args.model/'config.json')
    assert selected['context_length'] == 32768
    input_tokens = load_file(str(args.calibration_root/'qwen3_8b_c4_64f16h_s32768/windows.safetensors'))['input_ids']
    assert hashlib.sha256(input_tokens[selected['fit_document_ids']].contiguous().numpy().tobytes()).hexdigest() == selected['fit_token_sha256']
    layer,group = args.smoke_layer,args.smoke_group
    queries,positions = manifest_queries(args.query_position_manifest,args.query_capture,'fit',layer)
    count = len(queries)
    dr,dm = _discover_capture(args.calibration_root,split='fit',layer=layer)
    _,rows = _load_direct(dr,dm,layer)
    rows = rows[:count,:,group:group+1]
    queries = queries[:,:,group*4:(group+1)*4]
    source_path = args.initial_bank/f'layer_{layer:03d}.safetensors'
    source_record = json.loads(source_path.with_suffix('.json').read_text())
    assert source_record['status'] == 'complete' and source_record['sha256'] == sha256(source_path)
    frozen_base_specification(source_record['protocol'],args.base_kind)
    assert source_record['protocol']['c1_manifest_sha256'] == sha256(args.c1_checkpoint/'results.json')
    source = load_file(str(source_path))
    c1 = json.loads((args.c1_checkpoint/'results.json').read_text())
    encoder = load_file(str(args.c1_checkpoint/c1['artifacts'][str(layer)]['file']))['value_coordinate_encoders'][group:group+1]
    cos,sin = _rotary_embeddings(args.model,sequence=32768,device=torch.device('cuda:0'))
    statistics,reconstruction = build_multi_query_statistics(queries,rows,query_positions=positions,
        cos=cos,sin=sin,value_encoder=encoder,base_maps={16:base_maps(source)[group:group+1]},
        page_size=32,excluded_prefix_pages=1,device=torch.device('cuda:0'))
    routing = statistics[16]
    assert routing.fisher_grams_by_head.shape == (4,count*len(positions),128,128)
    assert torch.isfinite(routing.fisher_grams_by_head).all()
    expected = queries.permute(2,1,0,3).reshape(4,count*len(positions),128).float()
    assert torch.equal(routing.queries_by_head,expected)
    # Independently rebuild the first position/window Gram to verify query-major order.
    current = rows[0].to('cuda:0').float()
    codes = _value_codes(current[...,:128],encoder.to('cuda:0').float())
    maps = _stack_base_map(base_maps(source)[group:group+1],device=torch.device('cuda:0'))
    residual = current[...,128:]-_post_rope_rows(_apply_base(codes,maps),cos,sin)
    stop = int(positions[0])+1
    gram,_ = residual_page_fisher_gram(queries[0,0].to('cuda:0').float(),current[:stop,0,128:],
        residual[:stop,0],scaling=128**-.5,page_size=32,excluded_prefix_pages=1)
    torch.testing.assert_close(gram.cpu(),routing.fisher_grams_by_head[:,0],rtol=1e-5,atol=1e-6)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    assert not (args.output_dir/'result.json').exists()
    write_json(args.output_dir/'result.json',{'status':'complete','scope':'sampler integration smoke only; no fitting/RULER',
        'layer':layer,'group':group,'fit_windows':count,'query_positions':positions.tolist(),
        'statistics_shape':list(routing.fisher_grams_by_head.shape),
        'teacher_fisher_energy':routing.teacher_fisher_energy,'query_major_order_verified':True,
        'independent_first_gram_verified':True,'source_base_sha256':sha256(source_path),
        'position_manifest_sha256':sha256(args.query_position_manifest),
        'command':' '.join(sys.argv),'python':sys.executable,'gpu':torch.cuda.get_device_name(0)})
    print('SAMPLER STATISTICS SMOKE PASSED',flush=True)


def main():
    p = parser()
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    assert args.initial_bank.resolve() != args.output_dir.resolve()
    torch.set_num_threads(args.torch_num_threads)
    if args.sampler_smoke:
        sampler_smoke(args)
        return
    specification = protocol(args)
    label = f"Q{specification.get('residual_query_count',len(specification['residual_query_positions']))} residual"
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "protocol": specification}, indent=2))
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    cos, sin = _rotary_embeddings(args.model, sequence=32768, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = []
    for layer in range(args.shard_index, 36, args.num_shards):
        output = args.output_dir / f"layer_{layer:03d}.safetensors"
        record_path = output.with_suffix(".json")
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record["status"] == "complete" and record["protocol"] == specification
            assert record["sha256"] == sha256(output)
        else:
            started = time.monotonic()
            print(f"[{label}] layer={layer} start", flush=True)
            tensors, diagnostics = fit_layer(args, layer, cos, sin, device)
            temporary = output.with_suffix(".tmp")
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(temporary))
            temporary.replace(output)
            record = {"status": "complete", "layer": layer, "protocol": specification,
                      "sha256": sha256(output), "diagnostic": diagnostics,
                      "wall_seconds": time.monotonic() - started, "command": shlex.join(sys.argv),
                      "python": sys.executable, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(device)}
            write_json(record_path, record)
            print(f"[{label}] layer={layer} done seconds={record['wall_seconds']:.1f}", flush=True)
        complete.append(layer)
    write_json(args.output_dir / f"shard_{args.shard_index}.json",
               {"status": "complete", "layers": complete, "protocol": specification})


if __name__ == "__main__":
    main()
