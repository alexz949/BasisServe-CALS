"""Audit every fitted Nemotron-H Mamba Wo artifact."""
import argparse
import math
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256
from evaluation.install_nemotron_h_wo import audit_nemotron_h_wo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('identity','audit','bank','covariances','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    configure()
    identity=read_json(args.identity)
    config=AutoConfig.from_pretrained(identity['model'],local_files_only=True,
        trust_remote_code=False)
    report=audit_nemotron_h_wo(config,args.identity,args.audit,args.bank)
    protocol=report['protocol']
    covariance_path=args.covariances/'manifest.json'
    covariance=read_json(covariance_path)
    assert covariance['status']=='complete' and covariance['dense_teacher']
    assert covariance['audit_sha256']==sha256(args.audit)
    assert protocol['covariance_manifest_sha256']==sha256(covariance_path)
    assert min(covariance['calibration'][key] for key in ('fit_windows','heldout_windows','sequence_length'))>0
    assert protocol['fit']['encoder_sweeps']==protocol['fit']['minimum_encoder_sweeps']==6
    assert protocol['fit']['encoder_cg_iterations']==16
    assert protocol['tp']==4
    expected={int(key) for key in report['layers']}
    seen=set()
    for shard in range(4):
        complete=read_json(args.bank/f'complete_{shard}.json')
        assert complete['status']=='complete' and complete['protocol']==protocol
        assert complete['layers']==sorted(expected)[shard::4]
        assert not seen.intersection(complete['layers'])
        seen.update(complete['layers'])
    assert seen==expected
    metrics={}
    for layer in sorted(expected):
        path=args.bank/f'layer_{layer:03d}.json'
        record=read_json(path)
        assert record['covariance_sha256']==covariance['artifacts'][str(layer)]['sha256']
        assert record['diagnostics']['encoder_sweeps_completed']==6
        assert 0<=record['selected_sweep']<=6
        values={name:record[name] for name in ('fit_relative_mse','heldout_relative_mse',
            'quantized_fit_relative_mse','quantized_heldout_relative_mse')}
        assert all(math.isfinite(value) and value>=0 for value in values.values())
        with safe_open(str(path.with_suffix('.safetensors')),framework='pt',device='cpu') as tensors:
            target=next(row for row in read_json(args.audit)['layers'] if row['layer']==layer)
            source_width=target['input_width']//protocol['tp']
            source_rank=protocol['source_rank']
            shapes={'source_encoders':(protocol['tp'],source_width,source_rank),
                'source_decoders':(protocol['tp'],source_rank,target['output_width'])}
            assert set(tensors.keys())==set(shapes)
            for name,shape in shapes.items():
                tensor=tensors.get_tensor(name)
                assert tensor.shape==shape and tensor.dtype==torch.bfloat16
                assert torch.isfinite(tensor).all()
                del tensor
        metrics[str(layer)]=values
        print('VERIFIED MAMBA WO',layer,values,flush=True)
    write_json(args.output,dict(status='complete',verified_layers=len(expected),wo=report,metrics=metrics,
        source_sha256=sha256(__file__)))


if __name__=='__main__':
    main()
