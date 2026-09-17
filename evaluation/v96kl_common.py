"""Artifact checks shared by the V96-KL calibration and LongBench run."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4'
CHECKPOINT = ROOT / 'results/hf/ICLR-results/qwen3-8b/checkpoints/Q3-8B-C1-R96'
CALIBRATION = ROOT / 'results/calibration/v96kl_64x32k'
BANK = ROOT / 'results/checkpoints/v96kl_b16r16'
DATA = ROOT / 'results/datasets/longbench_full'
OUTPUT = ROOT / 'results/evaluation/v96kl_longbench'
OFFICIAL = ROOT / 'results/tools/LongBench/LongBench'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    # Compare the persisted JSON representation on resume (JSON keys are strings).
    value = json.loads(json.dumps(value, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Completed artifacts are immutable. Identical writes are harmless on resume.
    if path.exists():
        assert read_json(path) == value, str(path)
        return
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


def save_tensors(path, tensors):
    path = Path(path)
    if path.exists():
        # A process may stop after its atomic tensor write but before its JSON record.
        # Verify the regenerated payload instead of overwriting a completed file.
        existing = load_file(str(path))
        assert set(existing) == set(tensors), str(path)
        assert all(existing[name].dtype == value.dtype and torch.equal(existing[name], value.cpu())
                   for name, value in tensors.items()), str(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(partial))
    partial.replace(path)


def checkpoint_manifest(checkpoint=CHECKPOINT, model=MODEL):
    manifest = read_json(checkpoint / 'manifest.json')
    assert manifest['status'] == 'complete'
    assert manifest['compression']['method'] == 'c1-two-sided-kl'
    assert manifest['compression']['equivalent_rank_target'] == 96
    ranks = manifest['compression']['layer_ranks']
    assert len(ranks) == 36 and all(len(r) == 8 for r in ranks)
    assert sum(map(sum, ranks)) == 36 * 8 * 96
    assert manifest['model']['config_sha256'] == sha256(model / 'config.json')
    assert manifest['model']['safetensors_index_sha256'] == sha256(model / 'model.safetensors.index.json')
    assert sha256(checkpoint / manifest['artifact']['file']) == manifest['artifact']['sha256']
    assert len(manifest['layers']) == 36
    for i, entry in enumerate(manifest['layers']):
        assert entry['layer'] == i and len(set(entry['ranks'])) == 1
        assert entry['ranks'] == ranks[i]
        assert sha256(checkpoint / entry['file']) == entry['sha256']
    return manifest


def configure():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def code_hashes(names):
    return {name: sha256(ROOT / name) for name in names}
