"""Audit all 45 fitted Mamba Wo artifacts before paired RULER evaluation."""
import argparse
import math
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256
from evaluation.k_routing_config import routing_config
from evaluation.install_nemotron_h_wo import audit_nemotron_h_wo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('identity','audit','bank','covariances','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    configure()
    identity=read_json(args.identity)
    config=routing_config(identity,rope='native',sequence_length=65536)
    report=audit_nemotron_h_wo(config,args.identity,args.audit,args.bank)
    protocol=report['protocol']
    covariance_path=args.covariances/'manifest.json'
    covariance=read_json(covariance_path)
    assert covariance['status']=='complete' and covariance['dense_teacher']
    assert covariance['audit_sha256']==sha256(args.audit)
    assert protocol['covariance_manifest_sha256']==sha256(covariance_path)
    assert covariance['calibration']['fit_windows']==256
    assert covariance['calibration']['heldout_windows']==64
    assert covariance['calibration']['sequence_length']==2048
    assert protocol['fit']['encoder_sweeps']==protocol['fit']['minimum_encoder_sweeps']==12
    assert protocol['tp']==4 and protocol['source_rank']==1536
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
        assert record['diagnostics']['encoder_sweeps_completed']==12
        assert 0<=record['selected_sweep']<=12
        values={name:record[name] for name in ('fit_relative_mse','heldout_relative_mse',
            'quantized_fit_relative_mse','quantized_heldout_relative_mse')}
        assert all(math.isfinite(value) and value>=0 for value in values.values())
        with safe_open(str(path.with_suffix('.safetensors')),framework='pt',device='cpu') as tensors:
            shapes={'source_encoders':(4,4096,1536),'source_decoders':(4,1536,8192)}
            assert set(tensors.keys())==set(shapes)
            for name,shape in shapes.items():
                tensor=tensors.get_tensor(name)
                assert tensor.shape==shape and tensor.dtype==torch.bfloat16
                assert torch.isfinite(tensor).all()
                del tensor
        metrics[str(layer)]=values
        print('VERIFIED MAMBA WO',layer,values,flush=True)
    write_json(args.output,dict(status='complete',verified_layers=45,wo=report,metrics=metrics,
        source_sha256=sha256(__file__)))


if __name__=='__main__':
    main()
