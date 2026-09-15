"""Refit residual rank using verified, unchanged Base0 Fisher statistics."""
import argparse
from pathlib import Path
import torch
from evaluation.fit_k_routing_streaming import verified,save_record
from evaluation.streaming_k_statistics import load_fisher_windows
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_residual_grid
from evaluation.v96kl_common import read_json,write_json,sha256,configure


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['fit','audit'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();configure();r=args.root;source=r/'b0r16';out=r/'b0r20'
    if args.smoke:out=out/'smoke'
    if args.stage=='audit':
        hashes={}
        for layer in range(32):
            _,meta=verified(out/'ours_b0r20'/f'layer_{layer:03d}.safetensors')
            assert meta['layer']==layer and not meta['protocol']['smoke'] and meta['sweeps']==40
            assert meta['protocol']['base_rank']==0 and meta['protocol']['residual_rank']==20
            hashes[str(layer)]=meta['sha256']
        write_json(out/'fit_audit.json',dict(status='complete',bank_sha256=hashes));return
    layer=args.layer;assert 0<=layer<32
    base_path=source/'base'/f'layer_{layer:03d}.safetensors';base,bm=verified(base_path)
    assert bm['protocol']['base_rank']==0 and not bm['protocol']['smoke']
    assert base['left'].shape[-1]==0 and torch.count_nonzero(base['bias'])==0
    stats={};hashes={}
    for split,indices in [('fit',range(1 if args.smoke else 64)),('heldout',range(64,65 if args.smoke else 80))]:
        payloads=[]
        for index in indices:
            path=source/'fisher'/f'l{layer:03d}'/f'w{index:03d}.safetensors'
            payload,meta=verified(path)
            assert meta['protocol']==bm['protocol'] and meta['base_sha256']==sha256(base_path)
            assert meta['window_id']==index and meta['split']==split
            payloads.append(payload);hashes[str(index)]=meta['sha256']
        stats[split]={0:load_fisher_windows(payloads,32,8,128)}
        del payloads
    sweeps=2 if args.smoke else 40
    factors,losses=_fit_residual_grid(stats['fit'],stats['heldout'],residual_ranks=(20,),sweeps=sweeps,
        relative_damping=1e-5,iterative_tolerance=1e-5,iterative_max_iterations=100,device=torch.device('cuda'))
    tensors={f'base_{name}_b0':base[name].float() for name in ('left','right','bias')}
    tensors.update(residual_encoder_b0_r20=factors[(0,20)][0].cpu().float(),residual_query_b0_r20=factors[(0,20)][1].cpu().float())
    assert all(torch.isfinite(t).all() for t in tensors.values())
    protocol=dict(bm['protocol'],residual_rank=20,smoke=args.smoke)
    protocol['source_sha256']={name:sha256(Path(name)) for name in [__file__,'evaluation/streaming_k_statistics.py',
        'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py']}
    if args.smoke:protocol.update(fit_ids=[0],diagnostic_ids=[64])
    save_record(out/'ours_b0r20'/f'layer_{layer:03d}.safetensors',tensors,dict(layer=layer,protocol=protocol,
        identity_sha256=bm['identity_sha256'],base_sha256=sha256(base_path),source_base=str(base_path),
        fisher_sha256=hashes,sweeps=sweeps,pcg_iterations=100,v_rank=128,losses=losses))
    print(dict(layer=layer,losses=losses),flush=True)


if __name__=='__main__':main()
