"""Check pinned Nemotron checkpoint geometry against the native meta model."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightRenaming,WeightConverter,rename_source_key

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.v96kl_common import configure,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure()
    config=AutoConfig.from_pretrained(args.model,trust_remote_code=False,local_files_only=True)
    assert config.model_type=='nemotron_h'
    with torch.device('meta'):
        model=AutoModelForCausalLM.from_config(config,trust_remote_code=False,
            attn_implementation='sdpa',dtype=torch.bfloat16)
    assert type(model).__module__.startswith('transformers.models.nemotron_h.')
    expected={name:list(value.shape) for name,value in model.state_dict().items()}
    mappings=get_model_conversion_mapping(model)
    renamings=[m for m in mappings if isinstance(m,WeightRenaming)]
    converters=[m for m in mappings if isinstance(m,WeightConverter)]
    index=json.loads((args.model/'model.safetensors.index.json').read_text())
    observed={};dtypes=Counter();header_hashes={};key_mapping={}
    for filename in sorted(set(index['weight_map'].values())):
        path=args.model/filename
        with path.open('rb') as stream:
            size=int.from_bytes(stream.read(8),'little')
            header=stream.read(size)
        header_hashes[filename]=hashlib.sha256(header).hexdigest()
        with safe_open(str(path),framework='pt',device='cpu') as tensors:
            for name in tensors.keys():
                assert index['weight_map'][name]==filename
                target,conversion=rename_source_key(name,renamings,converters,
                    base_model_prefix=model.base_model_prefix,meta_state_dict=expected)
                assert conversion is None, 'This dense checkpoint must require renaming only'
                assert target not in observed
                key_mapping[name]=target
                value=tensors.get_slice(name)
                observed[target]=value.get_shape()
                dtypes[value.get_dtype()]+=1
    missing=sorted(set(expected)-set(observed));extra=sorted(set(observed)-set(expected))
    mismatched={name:dict(model=shape,checkpoint=observed[name])
        for name,shape in expected.items() if name in observed and shape!=observed[name]}
    print('GEOMETRY',len(expected),len(observed),'missing',missing[:8],'extra',extra[:8],
        'mismatched',mismatched,flush=True)
    assert not missing and not extra and not mismatched
    blocks=[]
    for i,layer in enumerate(model.model.layers):
        kind=layer.block_type
        row=dict(layer=i,kind=kind)
        if kind in ('linear_attention','full_attention'):
            name='out_proj' if kind=='linear_attention' else 'o_proj'
            projection=getattr(layer.mixer,name)
            row.update(projection=name,input_width=projection.in_features,
                output_width=projection.out_features)
        blocks.append(row)
    source=Path(inspect.getfile(type(model)))
    write_json(args.output,dict(status='complete',model=str(args.model),
        implementation=type(model).__module__+'.'+type(model).__name__,
        implementation_sha256=sha256(source),config_sha256=sha256(args.model/'config.json'),
        index_sha256=sha256(args.model/'model.safetensors.index.json'),
        checkpoint_header_sha256=header_hashes,tensor_count=len(observed),
        checkpoint_to_native_keys=key_mapping,renaming_implementation_sha256=sha256(Path(inspect.getfile(rename_source_key))),
        checkpoint_dtypes=dict(dtypes),parameter_count=sum(p.numel() for p in model.parameters()),
        layers=blocks,layer_counts=dict(Counter(r['kind'] for r in blocks)),
        numerical_equivalence_tested=False,weights_loaded=False))
    print('NATIVE META MODEL MATCHES CHECKPOINT',dict(Counter(r['kind'] for r in blocks)),flush=True)


if __name__=='__main__':main()
