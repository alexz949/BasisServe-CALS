"""Prepare Instruct identity V factors and reuse verified C4 token windows."""
import argparse
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer
from evaluation.v96kl_common import read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = read_json(args.source_root/'manifests/v96.json')
    original = AutoTokenizer.from_pretrained(source['model'], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert original.get_vocab() == tokenizer.get_vocab()
    # C4 windows were tokenized without chat/special tokens. Verify the actual
    # tokenization model, normalization and pre-tokenization before reusing IDs.
    import json
    old, new = (json.loads(t.backend_tokenizer.to_str()) for t in (original, tokenizer))
    for field in ('model', 'normalizer', 'pre_tokenizer'):
        assert old[field] == new[field]
    calibration = args.output/'calibration'
    calibration.mkdir(parents=True, exist_ok=True)
    source_windows = (args.source_root/'calibration/windows.safetensors').resolve()
    manifest = read_json(args.source_root/'calibration/manifest.json')
    assert manifest['sha256'] == sha256(source_windows)
    target = calibration/'windows.safetensors'
    if not target.exists():
        target.symlink_to(source_windows)
    assert target.resolve() == source_windows
    manifest.update(model_config_sha256=sha256(args.model/'config.json'),
        provenance='Same C4 IDs as Base control; verified identical ordinary tokenizer; activations replayed with Instruct',
        source_manifest_sha256=sha256(args.source_root/'calibration/manifest.json'))
    write_json(calibration/'manifest.json', manifest)
    config = read_json(args.model/'config.json')
    assert config['num_hidden_layers'] == 32 and config['hidden_size'] == 4096
    index = read_json(args.model/'model.safetensors.index.json')['weight_map']
    checkpoint = args.output/'v128'
    checkpoint.mkdir(parents=True, exist_ok=True)
    records = []
    for layer in range(32):
        key = f'model.layers.{layer}.self_attn.o_proj.weight'
        with safe_open(args.model/index[key], framework='pt', device='cpu') as f:
            weight = f.get_tensor(key)
        assert weight.dtype == torch.bfloat16 and weight.shape == (4096,4096)
        path = checkpoint/f'layer_{layer:03d}.safetensors'
        tensors = dict(value_coordinate_encoders=torch.eye(128,dtype=torch.bfloat16).expand(8,-1,-1).contiguous(),
            head_output_decoders=weight.T.reshape(32,128,4096).contiguous(),
            source_ranks=torch.full((8,),128,dtype=torch.int32))
        if not path.exists():
            save_file(tensors,str(path))
        else:
            with safe_open(path,framework='pt',device='cpu') as f:
                assert all(torch.equal(f.get_tensor(k),v) for k,v in tensors.items())
        records.append(dict(layer=layer,ranks=[128]*8,file=path.name,sha256=sha256(path)))
    write_json(checkpoint/'manifest.json',dict(status='complete',layers=records,value_mode='dense original V and Wo'))
    write_json(args.output/'manifests/v128.json',dict(status='complete',checkpoint=str(checkpoint),
        model=str(args.model),model_config_sha256=sha256(args.model/'config.json'),
        manifest_sha256=sha256(checkpoint/'manifest.json'),attention_layers=list(range(32)),
        layer_ranks=[128]*32,mean_rank=128,hq=32,hkv=8,head_dim=128,hidden_size=4096,
        value_mode='dense original V and Wo'))
    print('Verified Instruct Dense V/Wo factors and reused C4 tokens',flush=True)


if __name__ == '__main__':
    main()
