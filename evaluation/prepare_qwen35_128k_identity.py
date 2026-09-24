"""Build the Qwen3.5-9B 128K identity record for the K-routing fitter and evaluator.

The identity binds the original model snapshot, the uniform V192 bank (8 full-attention layers) and the GDN Wo75
Private-AllGather bank, and exports each layer's value coordinate encoder in the per-layer checkpoint layout that
``fit_k_routing_streaming.encoder_for`` reads, so the Base/residual stages can be reused unchanged.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from evaluation.qwen35_hybrid_common import load_bank, verify_model_identity
from basisserve.core.qwen35_gdn_private_ag_runtime import load_qwen35_gdn_private_ag_factors
from evaluation.v96kl_common import sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--v-bank', type=Path, required=True)
    p.add_argument('--gdn-bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    bank = load_bank(args.v_bank)
    assert bank['status'] == 'complete' and bank['method'] == 'uniform' and bank['nominal_v_rank'] == 192
    verify_model_identity(args.model, bank['model_identity'])
    gdn = load_qwen35_gdn_private_ag_factors(args.gdn_bank)
    assert gdn['status'] == 'complete' and len(gdn['layers']) == 24
    assert gdn['protocol']['windows_sha256'] == bank['windows_sha256']
    layers = sorted(int(x) for x in bank['layers'])
    assert layers == [3, 7, 11, 15, 19, 23, 27, 31]
    checkpoint = args.output/'checkpoint'
    checkpoint.mkdir(parents=True, exist_ok=True)
    entries = []
    for layer in layers:
        encoder = bank['layers'][layer]['E_V'].float().contiguous()
        assert encoder.shape == (4, 256, 192)
        path = checkpoint/f'layer_{layer:03d}.safetensors'
        assert not path.exists(), path
        save_file({'value_coordinate_encoders': encoder}, str(path))
        entries.append(dict(layer=layer, file=path.name, sha256=sha256(path), v_rank=int(bank['schedule'][layer])))
    manifest = checkpoint/'manifest.json'
    write_json(manifest, dict(format='basisserve.qwen35.v192_value_encoders.v1', status='complete',
        v_bank_sha256=sha256(args.v_bank), v_factor_sha256=bank['factor_sha256'], layers=entries))
    config = args.model/'config.json'
    text = json.loads(config.read_text()).get('text_config', {})
    identity = dict(status='complete', model=str(args.model), model_config_sha256=sha256(config),
        model_revision=bank['model_identity']['model_revision'], checkpoint=str(checkpoint),
        manifest_sha256=sha256(manifest), head_dim=int(text['head_dim']), hkv=int(text['num_key_value_heads']),
        heads=int(text['num_attention_heads']), attention_layers=layers, v_rank=192,
        v_bank=str(args.v_bank), v_bank_sha256=sha256(args.v_bank),
        gdn_bank=str(args.gdn_bank), gdn_bank_sha256=sha256(args.gdn_bank),
        gdn_protocol={k: str(v) for k, v in gdn['protocol'].items()})
    write_json(args.output/'v192.json', identity)
    print(json.dumps(dict(identity=str(args.output/'v192.json'), layers=layers, head_dim=identity['head_dim'],
        hkv=identity['hkv'], heads=identity['heads'])), flush=True)


if __name__ == '__main__':
    main()
