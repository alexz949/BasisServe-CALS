from copy import deepcopy

import pytest
from transformers import NemotronHConfig

from evaluation.audit_k_routing_fit import attention_payloads


def test_hybrid_audit_resolves_actual_attention_ids_and_rejects_mamba_payloads():
    layers = [17, 38, 49, 60, 86]
    kinds = ['linear_attention' if i % 2 == 0 else 'mlp' for i in range(98)]
    for layer in layers:
        kinds[layer] = 'full_attention'
    config = NemotronHConfig(layers_block_type=kinds, hidden_size=8192,
        num_attention_heads=64, num_key_value_heads=8, head_dim=128)
    ranks = [128, 96, 96, 96, 64]
    identity = dict(attention_layers=layers, layer_ranks=ranks, hkv=8,
        hq=64, hidden_size=8192, head_dim=128)
    manifest = dict(layers=[dict(layer=layer, ranks=[rank] * 8)
        for layer, rank in zip(layers, ranks)])
    # Disk ordering need not equal the numeric layer ID or iteration order.
    manifest['layers'].reverse()
    sources = attention_payloads(identity, manifest, config)
    assert sources[86]['ranks'] == [64] * 8
    assert sources[17]['ranks'] == [128] * 8
    wrong = deepcopy(manifest)
    wrong['layers'][0]['layer'] = 0
    with pytest.raises(AssertionError):
        attention_payloads(identity, wrong, config)
    wrong_identity = deepcopy(identity)
    wrong_identity['attention_layers'][0] = 0
    with pytest.raises(AssertionError):
        attention_payloads(wrong_identity, manifest, config)
    duplicate = deepcopy(manifest)
    duplicate['layers'].append(duplicate['layers'][0])
    with pytest.raises(AssertionError):
        attention_payloads(identity, duplicate, config)
