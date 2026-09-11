import json
from types import SimpleNamespace

import torch

from evaluation.capture_query_cpqr import collect_query, merge, shard_documents
from evaluation.v96kl_common import sha256


def test_window_shards_cover_only_fit_documents():
    left = shard_documents(64, 2, 0)
    right = shard_documents(64, 2, 1)
    assert not set(left) & set(right)
    assert sorted(left + right) == list(range(64))


def test_hook_saves_post_rope_queries_without_modifying_inputs():
    torch.manual_seed(7)
    module = torch.nn.Module()
    module.head_dim = 128
    module.q_proj = torch.nn.Linear(4, 32 * 128, bias=False).bfloat16()
    module.q_norm = torch.nn.Identity()
    x = torch.randn(1, 512, 4).bfloat16()
    original = x.clone()
    cos = torch.zeros(1, 512, 128, dtype=torch.bfloat16)
    sin = torch.ones_like(cos)
    captured = {}
    with torch.inference_mode():
        collect_query(module, (), dict(hidden_states=x, position_embeddings=(cos, sin)),
                      layer=15, grid=list(range(512)), captured=captured)
        q = module.q_proj(x).reshape(512, 32, 128)
        expected = torch.cat((-q[..., 64:], q[..., :64]), dim=-1)
    assert torch.equal(captured[15], expected)
    assert torch.equal(x, original)


def test_merge_preserves_shards_and_checks_coverage(tmp_path):
    protocol = dict(num_shards=2, fit_document_ids=list(range(4)))
    for shard in range(2):
        folder = tmp_path / f'shard_{shard}'
        folder.mkdir()
        artifacts = {}
        for doc in shard_documents(4, 2, shard):
            path = folder / f'window_{doc:03d}.safetensors'
            path.write_bytes(f'fixture {doc}'.encode())
            artifacts[str(doc)] = dict(file=path.name, sha256=sha256(path))
        (folder / 'manifest.json').write_text(json.dumps(dict(
            status='complete', protocol=protocol, artifacts=artifacts)))
    merge(SimpleNamespace(output_dir=tmp_path, num_shards=2))
    result = json.loads((tmp_path / 'manifest.json').read_text())
    assert set(result['artifacts']) == {'0', '1', '2', '3'}
    assert result['artifacts']['3']['file'] == 'shard_1/window_003.safetensors'
