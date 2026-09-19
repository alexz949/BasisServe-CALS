"""Install audited folded Mamba Wo factors for paired quality evaluation."""
from pathlib import Path

import torch
from safetensors.torch import load_file

from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout, fold_factors_to_dense_weight
from evaluation.v96kl_common import read_json, sha256


@torch.inference_mode()
def fold_into_projection(projection, factors, layout):
    assert set(factors) == {'source_encoders', 'source_decoders'}
    assert tuple(projection.weight.shape) == (layout.output_width, layout.input_width)
    assert projection.bias is None
    encoders = factors['source_encoders'].to(projection.weight.device, dtype=torch.float32)
    decoders = factors['source_decoders'].to(projection.weight.device, dtype=torch.float32)
    assert torch.isfinite(encoders).all() and torch.isfinite(decoders).all()
    weight = fold_factors_to_dense_weight(encoders, decoders, layout)
    assert torch.isfinite(weight).all()
    projection.weight.copy_(weight.to(projection.weight.dtype))


def audit_nemotron_h_wo(config, identity_path, audit_path, directory):
    identity_path, audit_path, directory = map(Path, (identity_path, audit_path, directory))
    identity, audit = read_json(identity_path), read_json(audit_path)
    assert identity['status'] == audit['status'] == 'complete'
    assert config.model_type == 'nemotron_h'
    assert identity['model_config_sha256'] == audit['config_sha256']
    assert identity['mean_rank'] in (64, 96)
    expected = [i for i, kind in enumerate(config.layers_block_type) if kind == 'linear_attention']
    assert expected == [row['layer'] for row in audit['layers'] if row['kind'] == 'linear_attention']
    assert expected
    assert {p.name for p in directory.glob('layer_*.json')} == {f'layer_{i:03d}.json' for i in expected}
    records, common = {}, None
    for layer in expected:
        path = directory / f'layer_{layer:03d}.json'
        record = read_json(path)
        assert record['status'] == 'complete' and record['layer'] == layer
        protocol = record['protocol']
        if common is None:
            common = protocol
        assert protocol == common
        assert protocol['identity_sha256'] == sha256(identity_path)
        assert protocol['audit_sha256'] == sha256(audit_path)
        assert protocol['total_rank'] == identity['hq'] * identity['mean_rank']
        assert protocol['factor_dtype'] == 'bfloat16' and protocol['work_dtype'] == 'float32'
        target = audit['layers'][layer]
        layout = TPSourceWOLayout(input_width=target['input_width'],
            output_width=target['output_width'],
            tp_size=protocol['tp'], source_rank=protocol['source_rank'])
        assert layout.accounting() == record['layout']
        assert layout.retained_ratio_vs_dense_allgather == protocol['retained_ratio']
        assert layout.accounting() == protocol['mamba_reference']
        tensor_path = path.with_suffix('.safetensors')
        assert sha256(tensor_path) == record['sha256']
        records[str(layer)] = dict(manifest_sha256=sha256(path), factors_sha256=record['sha256'])
    return dict(mode='BF16 dense Wo folded from FP32 products of BF16 source factors',
        communication_benchmarked=False, layers=records, protocol=common, source_sha256=sha256(__file__))


@torch.inference_mode()
def install_nemotron_h_wo(model, identity_path, audit_path, directory):
    report = audit_nemotron_h_wo(model.config, identity_path, audit_path, directory)
    protocol = report['protocol']
    for key, record in report['layers'].items():
        layer = int(key)
        projection = model.model.layers[layer].mixer.out_proj
        layout = TPSourceWOLayout(input_width=projection.in_features,
            output_width=projection.out_features,
            tp_size=protocol['tp'], source_rank=protocol['source_rank'])
        path = Path(directory) / f'layer_{layer:03d}.safetensors'
        assert sha256(path) == record['factors_sha256']
        factors = load_file(str(path))
        assert all(t.dtype == torch.bfloat16 for t in factors.values())
        fold_into_projection(projection, factors, layout)
        print('INSTALLED FOLDED MAMBA WO', layer, flush=True)
    return report
