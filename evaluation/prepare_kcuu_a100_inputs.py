"""Reconstruct only the missing Llama index from pinned, hash-verified HF artifacts.

The published R64-S6-D0 directory contains 32 tensors but no results.json.
This is a local serving manifest, not the unpublished original fit report.
"""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi
from safetensors import safe_open
from evaluation.kcuu_platform import HUB, LLAMA_MODEL, LLAMA_FACTORS


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    repository = 'alexz949/BasisServe-CALS'
    revision = '5a30494ada6aeb5806c7dd7a63e574ebfd163063'
    subdir = 'ICLR-results/llama31-8b-instruct/c1/factor-banks/R64-S6-D0'
    snapshot = HUB / f'models--alexz949--BasisServe-CALS/snapshots/{revision}'
    checkpoint = snapshot / 'ICLR-results/llama31-8b-instruct/checkpoints/L31-8B-I-C1-R64/result.json'
    provenance = json.loads(checkpoint.read_text())
    config = json.loads((LLAMA_MODEL / 'config.json').read_text())
    assert sha(LLAMA_MODEL / 'config.json') == provenance['model_config_sha256']
    assert (config['hidden_size'], config['num_attention_heads'], config['num_key_value_heads'], config['num_hidden_layers']) == (4096, 32, 8, 32)
    paths = [f'{subdir}/layer_{i:03d}.safetensors' for i in range(32)]
    remote = {f.path: f.lfs.sha256 for f in HfApi().get_paths_info(repository, paths, revision=revision)}
    assert set(remote) == set(paths)
    artifacts = {}
    LLAMA_FACTORS.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(paths):
        src = snapshot / name
        checksum = sha(src)
        assert checksum == remote[name]
        with safe_open(str(src), framework='pt', device='cpu') as f:
            assert set(f.keys()) == {'value_coordinate_encoders', 'head_output_decoders'}
            assert f.get_slice('value_coordinate_encoders').get_shape() == [8, 128, 64]
            assert f.get_slice('head_output_decoders').get_shape() == [32, 64, 4096]
            assert all(f.get_slice(k).get_dtype() == 'BF16' for k in f.keys())
        dest = LLAMA_FACTORS / src.name
        if not dest.exists():
            dest.symlink_to(src)
        assert sha(dest) == checksum
        artifacts[str(i)] = dict(file=src.name, sha256=checksum)
    manifest = dict(format='basisserve.llama31_8b_instruct.gqa_c1_joint.v1',
        status='complete', layers=list(range(32)), artifacts=artifacts,
        fit_config=dict(hidden_size=4096, num_query_heads=32, num_physical_kv_heads=8,
                        head_dim=128, num_hidden_layers=32, cache_rank_per_head=64,
                        model_config_sha256=sha(LLAMA_MODEL / 'config.json')),
        source=dict(repository=repository, revision=revision, path=subdir),
        manifest_origin='Locally reconstructed serving index; tensor hashes verified against HF LFS. Not the original fitting report.',
        provenance=dict(checkpoint_result=str(checkpoint), checkpoint_result_sha256=sha(checkpoint),
                        unpublished_original_results_sha256=provenance['factor_sources']['64']['results_sha256']))
    dest = LLAMA_FACTORS / 'results.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == manifest
    else:
        dest.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Verified 32 original R64 factors; local serving manifest:', dest, sha(dest))


if __name__ == '__main__':
    main()
