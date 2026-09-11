"""Recapture fit-only dense-teacher Q; independent window shards, no fitting."""

import argparse
from functools import partial
from pathlib import Path
import shlex
import sys
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basisserve.core.query_position_sampling import candidate_positions
from evaluation.v96kl_common import (
    MODEL, CALIBRATION, configure, read_json, write_json, save_tensors, sha256, tensor_hash,
)


def shard_documents(window_count, num_shards, shard_index):
    assert 0 < window_count <= 64 and 0 < num_shards <= window_count
    assert 0 <= shard_index < num_shards
    return list(range(shard_index, window_count, num_shards))


def collect_query(module, positional, kwargs, *, layer, grid, captured):
    x = kwargs['hidden_states']
    query = module.q_norm(module.q_proj(x).view(*x.shape[:-1], -1, module.head_dim)).transpose(1, 2)
    cos, sin = kwargs['position_embeddings']
    # Use the installed model's RoPE arithmetic. The second result is discarded.
    query, _ = apply_rotary_pos_emb(query, query, cos, sin)
    selected = query[0, :, grid].transpose(0, 1).contiguous().cpu()
    assert selected.dtype == torch.bfloat16 and selected.shape == (512, 32, 128)
    assert torch.isfinite(selected).all() and layer not in captured
    captured[layer] = selected


@torch.inference_mode()
def capture(args):
    documents = shard_documents(args.window_count, args.num_shards, args.shard_index)
    layers = list(map(int, args.layers.split(',')))
    assert layers == sorted(set(layers)) and all(0 <= l < 36 for l in layers)
    manifest = read_json(args.calibration / 'manifest.json')
    window_path = args.calibration / 'windows.safetensors'
    assert manifest['status'] == 'complete' and manifest['sha256'] == sha256(window_path)
    assert manifest['model_config_sha256'] == sha256(args.model / 'config.json')
    assert manifest['fit_ids'] == list(range(64)) and manifest['validation_ids'] == list(range(64, 80))
    windows = load_file(str(window_path))['input_ids']
    assert windows.shape == (80, 32768)
    grid = candidate_positions(32768)
    protocol = dict(format='basisserve.candidates_queries.v1',
                    model_config_sha256=manifest['model_config_sha256'],
                    model_index_sha256=sha256(args.model / 'model.safetensors.index.json'),
                    windows_sha256=manifest['sha256'], fit_token_sha256=tensor_hash(windows[:64]),
                    calibration_manifest_sha256=sha256(args.calibration / 'manifest.json'),
                    context_length=32768, candidate_stride=64, layers=layers,
                    positions_by_layer={str(l): grid for l in layers},
                    fit_document_ids=list(range(args.window_count)),
                    documents=list(range(args.window_count)), num_shards=args.num_shards,
                    teacher='dense BF16 SDPA; full-window model backbone, use_cache=False',
                    capture='post q_norm, post RoPE; full-size q_proj; no C1 or residual fitting',
                    provenance='Local fixed v96kl windows; not historical remote captures',
                    verify_repeat=args.verify_repeat,
                    torch_version=torch.__version__, transformers_version=transformers.__version__,
                    source_sha256=sha256(Path(__file__)))
    directory = args.output_dir / f'shard_{args.shard_index}'
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'protocol.json', protocol)
    completed = {}
    for document in documents:
        record_path = directory / f'window_{document:03d}.json'
        if record_path.exists():
            record = read_json(record_path)
            assert record['document'] == document and record['input_ids_sha256'] == tensor_hash(windows[document])
            assert record['sha256'] == sha256(directory / record['file'])
            completed[str(document)] = record
    if len(completed) != len(documents):
        assert torch.cuda.is_available()
        # Match the old fitter's FP32 matmul setting; inference operands remain BF16.
        torch.backends.cuda.matmul.allow_tf32 = True
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, local_files_only=True,
            attn_implementation='sdpa').eval().to('cuda')
        assert model.config.num_hidden_layers == 36
        captured = {}
        handles = [model.model.layers[l].self_attn.register_forward_pre_hook(
            partial(collect_query, layer=l, grid=grid, captured=captured), with_kwargs=True) for l in layers]
        torch.cuda.reset_peak_memory_stats()
        for document in documents:
            if str(document) in completed:
                continue
            started = time.monotonic()
            captured.clear()
            inputs = windows[document:document + 1].long().cuda()
            model.model(input_ids=inputs, use_cache=False, return_dict=True)
            assert set(captured) == set(layers)
            queries = torch.stack([captured[l] for l in layers])
            if args.verify_repeat:
                captured.clear()
                model.model(input_ids=inputs, use_cache=False, return_dict=True)
                assert torch.equal(queries, torch.stack([captured[l] for l in layers]))
            name = f'window_{document:03d}.safetensors'
            save_tensors(directory / name, {'queries': queries})
            record = dict(document=document, file=name, sha256=sha256(directory / name),
                          input_ids_sha256=tensor_hash(windows[document]), shape=list(queries.shape),
                          repeat_verified=args.verify_repeat, seconds=time.monotonic() - started,
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                          gpu=torch.cuda.get_device_name(0), python=sys.executable,
                          command=shlex.join(sys.argv))
            write_json(directory / f'window_{document:03d}.json', record)
            completed[str(document)] = record
            print(f'document={document} complete seconds={record["seconds"]:.2f} '
                  f'peak_GiB={record["peak_gpu_allocated_bytes"] / 2**30:.3f}', flush=True)
        for handle in handles:
            handle.remove()
    write_json(directory / 'manifest.json', dict(status='complete', protocol=protocol, artifacts=completed))
    print(f'shard={args.shard_index} complete windows={len(completed)}', flush=True)


def merge(args):
    records = [read_json(args.output_dir / f'shard_{s}' / 'manifest.json') for s in range(args.num_shards)]
    protocol = records[0]['protocol']
    assert protocol['num_shards'] == args.num_shards
    artifacts = {}
    for shard, record in enumerate(records):
        assert record['status'] == 'complete' and record['protocol'] == protocol
        expected = shard_documents(len(protocol['fit_document_ids']), args.num_shards, shard)
        assert set(record['artifacts']) == set(map(str, expected))
        for doc, item in record['artifacts'].items():
            assert doc not in artifacts
            item = dict(item, file=f'shard_{shard}/{item["file"]}')
            assert item['sha256'] == sha256(args.output_dir / item['file'])
            artifacts[doc] = item
    assert set(artifacts) == set(map(str, protocol['fit_document_ids']))
    write_json(args.output_dir / 'manifest.json', dict(status='complete', protocol=protocol, artifacts=artifacts))
    print(f'merged {len(artifacts)} windows', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('capture', 'merge'))
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--calibration', type=Path, default=CALIBRATION)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layers', default='0,15,35')
    parser.add_argument('--window-count', type=int, default=64)
    parser.add_argument('--num-shards', type=int, default=2)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--verify-repeat', action='store_true')
    args = parser.parse_args()
    configure()
    (capture if args.stage == 'capture' else merge)(args)


if __name__ == '__main__':
    main()
