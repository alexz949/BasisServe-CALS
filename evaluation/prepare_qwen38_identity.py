"""Authenticate the pinned Qwen3.8-27B download and calibration inputs."""
import importlib.metadata
import json
from pathlib import Path

from huggingface_hub import HfApi
from evaluation.qwen35_hybrid_common import atomic_save, sha256


def main():
    root = Path('results/q38_hybrid')
    revision = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    info = HfApi().model_info('Qwen/Qwen3.8-27B', revision=revision, files_metadata=True)
    assert info.sha == revision
    config = json.loads((root / 'model/config.json').read_text())
    tc = config['text_config']
    assert tc['hidden_size'] == 5120 and tc['num_hidden_layers'] == 64
    assert tc['num_attention_heads'] == 24 and tc['num_key_value_heads'] == 4
    assert tc['linear_num_value_heads'] * tc['linear_value_head_dim'] == 6144
    assert tc['num_attention_heads'] * tc['head_dim'] == 6144
    shards = {}
    for sibling in info.siblings:
        if not sibling.rfilename.endswith('.safetensors'):
            continue
        path = root / 'model' / sibling.rfilename
        assert path.stat().st_size == sibling.size
        digest = sha256(path)
        assert digest == sibling.lfs.sha256
        shards[path.name] = digest
        print(f'verified {path.name}', flush=True)
    assert len(shards) == 18
    data = json.loads((root / 'data/manifest.json').read_text())
    assert data['model_config_sha256'] == sha256(root / 'model/config.json')
    assert data['tokenizer_sha256'] == sha256(root / 'model/tokenizer.json')
    assert data['windows_sha256'] == sha256(root / 'data/windows.pt')
    atomic_save(root / 'baseline_summary.json', dict(status='identity_verified',
        model_id='Qwen/Qwen3.8-27B', model_revision=revision,
        config_sha256=data['model_config_sha256'], model_files_sha256=shards,
        windows_sha256=data['windows_sha256'], tokenizer_sha256=data['tokenizer_sha256'],
        chat_template_sha256=sha256(root / 'model/chat_template.jinja'),
        geometry=tc, versions={k: importlib.metadata.version(k) for k in ['torch','transformers','huggingface_hub']},
        note='Identity manifest only; no benchmark score implied. Native attention gate must be audited from executed code.'))


if __name__ == '__main__':
    main()
