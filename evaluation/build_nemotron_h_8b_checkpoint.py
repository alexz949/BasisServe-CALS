"""Assemble a self-contained Nemotron-H C1 V and Mamba-Wo checkpoint."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256

FORMAT = 'basisserve.nemotron_h.c1_v_wo.v1'


def _copy_verified(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert sha256(destination) == expected
    else:
        shutil.copy2(source, destination)
        assert sha256(destination) == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--attention', type=Path, required=True)
    parser.add_argument('--wo', type=Path, required=True)
    parser.add_argument('--wo-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); configure()
    audit, identity, attention = read_json(args.audit), read_json(args.identity), read_json(args.attention/'manifest.json')
    wo_audit = read_json(args.wo_audit)
    assert audit['status'] == identity['status'] == attention['status'] == wo_audit['status'] == 'complete'
    assert identity['manifest_sha256'] == sha256(args.attention/'manifest.json')
    assert identity['model_config_sha256'] == audit['config_sha256']
    assert identity['mean_rank'] in (64, 96)
    assert wo_audit['verified_layers'] == len([r for r in audit['layers'] if r['kind']=='linear_attention'])
    assert wo_audit['wo']['protocol']['identity_sha256'] == sha256(args.identity)
    args.output.mkdir(parents=True, exist_ok=True)
    attention_records=[]
    for record in attention['layers']:
        source=args.attention/record['file']; destination=args.output/'attention'/record['file']
        _copy_verified(source,destination,record['sha256'])
        attention_records.append({**record,'file':str(Path('attention')/record['file'])})
    wo_records=[]
    for layer, record in sorted(wo_audit['wo']['layers'].items(),key=lambda item:int(item[0])):
        filename=f'layer_{int(layer):03d}.safetensors'
        source=args.wo/filename; destination=args.output/'mamba_wo'/filename
        _copy_verified(source,destination,record['factors_sha256'])
        wo_records.append(dict(layer=int(layer),file=str(Path('mamba_wo')/filename),
            sha256=record['factors_sha256'],manifest_sha256=record['manifest_sha256']))
    protocol=wo_audit['wo']['protocol']
    mean_rank=int(identity['mean_rank'])
    manifest=dict(status='complete',format=FORMAT,
        model=dict(path=identity['model'],config_sha256=audit['config_sha256'],
            index_sha256=audit['index_sha256'],revision=Path(identity['model']).name),
        compression=dict(method='C1',v_retained_ratio=mean_rank/identity['head_dim'],
            v_equivalent_mean_rank=mean_rank,v_layer_ranks=identity['layer_ranks'],
            attention_layers=identity['attention_layers'],mamba_wo_tp=protocol['tp'],
            mamba_wo_source_rank=protocol['source_rank'],
            full_attention_allgather=protocol['attention_reference'],
            mamba_matches_attention_retained_ratio=True,
            mamba_wo_allgather=protocol['mamba_reference'],
            mamba_wo_dense_allreduce_reduction=protocol['mamba_reference']['reduction_vs_dense_allreduce']),
        calibration=dict(dataset='allenai/c4 train',fit_windows=256,heldout_windows=64,
            sequence_length=2048,attention_als_sweeps=6,attention_encoder_cg_iterations=16,
            mamba_wo_als_sweeps=6,mamba_wo_encoder_cg_iterations=16),
        attention=attention_records,mamba_wo=wo_records,
        provenance=dict(native_audit_sha256=sha256(args.audit),identity_sha256=sha256(args.identity),
            attention_manifest_sha256=sha256(args.attention/'manifest.json'),
            wo_audit_sha256=sha256(args.wo_audit),source_sha256=sha256(__file__)))
    write_json(args.output/'manifest.json',manifest)
    print('CHECKPOINT COMPLETE',args.output,mean_rank,len(attention_records),len(wo_records),flush=True)


if __name__=='__main__':
    main()
