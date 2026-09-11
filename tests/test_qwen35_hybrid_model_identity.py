import pytest

from evaluation.qwen35_hybrid_common import sha256, verify_model_identity


@pytest.mark.parametrize('mutation', [None, 'config', 'weights', 'extra_shard'])
def test_original_model_identity_rejects_changed_inputs(tmp_path, mutation):
    config = tmp_path / 'config.json'
    weights = tmp_path / 'model.safetensors'
    config.write_text('{"hidden_size": 4096}')
    weights.write_bytes(b'original weight bytes')
    identity = {'config_sha256': sha256(config), 'model_files_sha256': {weights.name: sha256(weights)}}
    if mutation == 'config':
        config.write_text('{"hidden_size": 8192}')
    elif mutation == 'weights':
        weights.write_bytes(b'changed weights with the same file name')
    elif mutation == 'extra_shard':
        (tmp_path / 'extra.safetensors').write_bytes(b'unexpected shard')
    if mutation is None:
        verify_model_identity(tmp_path, identity)
    else:
        with pytest.raises(AssertionError):
            verify_model_identity(tmp_path, identity)
