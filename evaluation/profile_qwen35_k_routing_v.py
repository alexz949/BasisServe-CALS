"""Terminal-KL probes for the 32K ALS12 gated-V experiment."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open

from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime, factor_hash
from evaluation.assemble_qwen35_k_routing_v import LAYERS
from evaluation.qwen35_hybrid_common import atomic_save, load_model, sha256


def load_factors(root, schedule, windows_hash):
    bank, hashes = {}, {}
    for layer, rank in zip(LAYERS, schedule, strict=True):
        if rank == 256:
            continue
        path = root/f'l{layer:02d}_r{rank:03d}.pt'
        row = torch.load(path, map_location='cpu', weights_only=True)
        assert row['status'] == 'complete' and row['layer'] == layer and row['rank'] == rank
        assert row['encoder_sweeps'] == 12 and row['encoder_cg'] == 16
        fast_fit = layer >= 19 and rank in (160, 224)
        assert row['decoder_cg'] == (50 if fast_fit else 200)
        if fast_fit:
            assert row['matmul_allow_tf32'] is True
        assert row['wo_compression'] is False and row['windows_sha256'] == windows_hash
        bank[layer] = {key: row[key] for key in ('E_V', 'R_V')}
        hashes[path.name] = sha256(path)
    return bank, hashes


def probe_jobs():
    jobs = [('anchor', [192]*8)]
    for offset, layer in enumerate(LAYERS):
        for rank in (160, 224):
            schedule = [192]*8
            schedule[offset] = rank
            jobs.append((f'l{layer:02d}_r{rank:03d}', schedule))
    return jobs


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=('teacher', 'profile'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--factors', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--indices', required=True, help='Comma-separated fit window indices for KL allocation')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    indices = list(map(int, args.indices.split(',')))
    assert indices and len(indices) == len(set(indices)) and all(0 <= i < 64 for i in indices)
    assert 0 <= args.shard_index < args.num_shards
    windows_path = args.calibration/'windows.safetensors'
    windows_hash = sha256(windows_path)
    with safe_open(windows_path, framework='pt', device='cpu') as stream:
        windows = stream.get_tensor('input_ids')
    assert windows.shape == (80, 32768)
    identity = dict(model_revision=args.model.resolve().name,
        config_sha256=sha256(args.model/'config.json'),
        model_files_sha256={path.name: sha256(path) for path in sorted(args.model.glob('*.safetensors'))})
    assert identity['model_files_sha256']
    common = dict(windows_sha256=windows_hash, profile_indices=indices,
        model_identity=identity, wo_compression=False)
    model = load_model(str(args.model), 'cuda:0')
    if args.stage == 'teacher':
        for index in indices[args.shard_index::args.num_shards]:
            path = args.teacher/f'w{index:03d}.pt'
            if path.exists():
                row = torch.load(path, map_location='cpu', weights_only=True)
                assert all(row[key] == value for key, value in common.items())
                assert row['status'] == 'complete' and row['index'] == index
                continue
            hidden = model.model(windows[index:index+1].cuda(), use_cache=False).last_hidden_state[0]
            assert torch.isfinite(hidden).all()
            atomic_save(path, dict(status='complete', index=index, hidden=hidden.cpu(), **common))
            print(json.dumps(dict(teacher_window=index, complete=True)), flush=True)
        return
    assert args.factors is not None and args.output is not None
    for name, schedule in probe_jobs()[args.shard_index::args.num_shards]:
        bank, hashes = load_factors(args.factors, schedule, windows_hash)
        path = args.output/f'{name}.json'
        if path.exists():
            row = json.loads(path.read_text())
            assert all(row[key] == value for key, value in common.items())
            assert row['source_factors'] == hashes and row['status'] == 'complete'
            continue
        started, values, teacher_hashes = time.monotonic(), [], {}
        with GatedVRuntime(model, bank):
            for index in indices:
                teacher_path = args.teacher/f'w{index:03d}.pt'
                row = torch.load(teacher_path, map_location='cpu', weights_only=True)
                assert all(row[key] == value for key, value in common.items())
                assert row['status'] == 'complete' and row['index'] == index
                reference_hidden = row['hidden']
                hidden = model.model(windows[index:index+1].cuda(), use_cache=False).last_hidden_state[0]
                assert hidden.shape == reference_hidden.shape and torch.isfinite(hidden).all()
                total = 0.0
                for start in range(0, len(hidden), 128):
                    student = model.lm_head(hidden[start:start+128]).float().log_softmax(-1)
                    reference = model.lm_head(reference_hidden[start:start+128].cuda()).float().log_softmax(-1)
                    value = (reference.exp()*(reference-student)).double().sum()
                    assert torch.isfinite(value)
                    total += value.item()
                values.append(total/len(hidden))
                teacher_hashes[teacher_path.name] = sha256(teacher_path)
                print(json.dumps(dict(probe=name, window=index, kl=values[-1])), flush=True)
        atomic_save(path, dict(status='complete', schedule=schedule, window_kl=values,
            mean_kl=sum(values)/len(values), factor_sha256=factor_hash(bank), source_factors=hashes,
            teacher_hashes=teacher_hashes, seconds=time.monotonic()-started,
            command=sys.argv, python=sys.executable, **common))


if __name__ == '__main__':
    main()
