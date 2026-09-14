"""Bind exact V128 factors and shared 64K moments to the Dense V control."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from evaluation.v96kl_common import configure, read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    configure()
    source, output = args.source_root.resolve(), args.output.resolve()
    cov = read_json(source/'covariance/manifest.json')
    model = Path(cov['model']['path'])
    bank = source/'v_bank_ieee/R128'
    result = read_json(bank/'results.json')
    assert result['status'] == 'complete' and result['layers'] == list(range(32))
    assert result['fit_config']['cache_rank_per_head'] == 128
    output.mkdir(parents=True, exist_ok=True)
    for name in ('moments', 'covariance', 'calibration'):
        path = output/name
        if not path.exists():
            path.symlink_to(source/name, target_is_directory=True)
        assert path.resolve() == source/name
    checkpoint = output/'v128'
    checkpoint.mkdir(exist_ok=True)
    records = []
    for layer in range(32):
        original = bank/f'layer_{layer:03d}.safetensors'
        metadata = read_json(original.with_suffix('.json'))
        tensors = load_file(str(original))
        assert torch.equal(tensors['value_coordinate_encoders'], torch.eye(128, dtype=torch.bfloat16).expand(8,-1,-1))
        payload = load_file(str(source/'covariance'/f'layer_{layer:03d}.safetensors'))
        assert torch.equal(tensors['head_output_decoders'], payload['weight'].T.reshape(32,128,4096))
        tensors['source_ranks'] = torch.full((8,),128,dtype=torch.int32)
        path = checkpoint/original.name
        assert not path.exists()
        save_file(tensors, str(path))
        records.append(dict(layer=layer, ranks=[128]*8, file=path.name, sha256=sha256(path),
            source_sha256=sha256(original)))
        del payload, tensors
    write_json(checkpoint/'manifest.json', dict(status='complete', layers=records,
        value_mode='dense original V and Wo', source_result_sha256=sha256(bank/'results.json')))
    write_json(output/'manifests/v128.json',dict(status='complete', checkpoint=str(checkpoint),
        model=str(model), model_config_sha256=sha256(model/'config.json'),
        manifest_sha256=sha256(checkpoint/'manifest.json'), attention_layers=list(range(32)),
        layer_ranks=[128]*32,mean_rank=128,hq=32,hkv=8,head_dim=128,hidden_size=4096,
        value_mode='dense original V and Wo'))
    print('Verified all 32 identity encoders and original Wo decoders',flush=True)


if __name__ == '__main__':
    main()
