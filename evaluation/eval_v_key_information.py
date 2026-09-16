"""Held-out local V->pre-RoPE K information with shuffled and PCA controls."""
import argparse
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from evaluation.analyze_qwen3_8b_v80_pre_k_spectra import _moment_from_rows,_empty_moments,_group_moments
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _discover_capture,_load_direct,_rotary_embeddings,_pre_rope_rows
from basisserve.core.c1_v_conditional_k_router import fit_affine_reduced_rank_map

ROOT=Path('results/evaluation/qwen3_v_key_information')
MODEL=Path('/deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4')
C1=Path('results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6')
RANKS=[4,8,16,24,32,48,64,80,128]


def projected(m,f):
    from dataclasses import replace
    return replace(m,input_sum=torch.einsum('gi,gij->gj',m.input_sum,f),input_gram=f.mT@m.input_gram@f,input_target_gram=f.mT@m.input_target_gram)


def error(m,w,b):
    # Exact affine prediction SSE from uncentered sufficient statistics.
    return (m.target_gram.diagonal(dim1=-2,dim2=-1).sum(-1)
        +(w.mT@m.input_gram@w).diagonal(dim1=-2,dim2=-1).sum(-1)
        -2*(w*m.input_target_gram).sum((-2,-1))
        +m.row_count*b.square().sum(-1)+2*((m.input_sum[:,None,:]@w).squeeze(1)*b).sum(-1)
        -2*(m.target_sum*b).sum(-1))


def fit(m,rank):
    maps=[fit_affine_reduced_rank_map(**{k:v for k,v in _group_moments(m,g).items() if k!='target_gram'},rank=rank) for g in range(8)]
    return torch.stack([x.left@x.right for x in maps]),torch.stack([x.bias for x in maps])


@torch.inference_mode()
def layer_run(layer,smoke=False):
    start=time.monotonic();device=torch.device('cuda');cos,sin=_rotary_embeddings(MODEL,sequence=32768,device=device)
    artifacts=json.loads((C1/'results.json').read_text())['artifacts']
    enc=load_file(str(C1/artifacts[str(layer)]['file']))['value_coordinate_encoders'].double()
    totals={};held=[];capture_info={}
    for split,count in [('fit',4 if smoke else 64),('validation',2 if smoke else 16)]:
        path,manifest=_discover_capture(Path('results/calibration'),split=split,layer=layer)
        _,rows=_load_direct(path,manifest,layer);rows=rows[:count]
        assert len(rows)==count
        capture_info[split]=dict(manifest=str(path/'manifest.json'),artifacts=manifest['artifacts'][str(layer)])
        total=_empty_moments(8,128,128);shuffled=_empty_moments(8,128,128)
        rng=torch.Generator(device='cuda').manual_seed(1729+(0 if split=='fit' else 100000))
        # A nonzero cyclic document shift is a derangement; independent random token permutation per document.
        offset=int(torch.randint(1,count,(1,),generator=rng,device='cuda'))
        for doc in range(count):
            x=rows[doc].to(device=device,dtype=torch.float32);v=x[...,:128];k=_pre_rope_rows(x[...,128:],cos,sin)
            source=rows[(doc+offset)%count,...,:128].to(device=device,dtype=torch.float32)
            perm=torch.randperm(32768,generator=rng,device=device);vsh=source[perm]
            m=_moment_from_rows(v.double(),k.double());ms=_moment_from_rows(vsh.double(),k.double())
            total.add_(m);shuffled.add_(ms)
            if split=='validation':held.append((m,ms))
            if doc%8==0:print('PROGRESS',layer,split,doc+1,count,flush=True)
            del x,v,k,source,vsh
        totals[split]=(total,shuffled);del rows
    raw,sh=totals['fit'];c1=projected(raw,enc);c1sh=projected(sh,enc)
    mean=raw.target_sum/raw.row_count
    zero=torch.zeros(8,128,128,dtype=torch.float64)
    den=torch.stack([error(m,zero,mean) for m,_ in held]);assert bool((den>0).all())
    output={};maps={}
    for label,m,f in [('dense_v',raw,None),('c1_v80',c1,enc),('shuffled_dense_v',sh,None),('shuffled_c1_v80',c1sh,enc)]:
        for rank in RANKS:
            if rank>m.input_sum.shape[-1]:continue
            w,b=fit(m,rank)
            if f is not None:w=f@w
            maps[(label,rank)]=(w,b)
            errs=torch.stack([error(ms if label.startswith('shuffled') else hv,w,b) for hv,ms in held])
            assert bool(torch.isfinite(errs).all()) and bool((errs>=-1e-5*den).all())
            output[f'{label}_r{rank}']=dict(sse=errs.tolist(),r2=float((1-errs.sum(0)/den.sum(0)).mean()))
    cov=raw.target_gram-raw.row_count*mean[:,:,None]*mean[:,None,:]
    _,vec=torch.linalg.eigh((cov+cov.mT)/2)
    for rank in RANKS:
        u=vec[:,:,-rank:];w=u@u.mT;b=mean-(mean[:,None,:]@w).squeeze(1)
        # For PCA the observed input is K, never V; this is a reconstruction reference.
        from dataclasses import replace
        errs=torch.stack([error(replace(m,input_sum=m.target_sum,input_gram=m.target_gram,input_target_gram=m.target_gram),w,b) for m,_ in held])
        output[f'k_pca_r{rank}']=dict(sse=errs.tolist(),r2=float((1-errs.sum(0)/den.sum(0)).mean()))
    # Independent direct-row check of the moment SSE on a real held-out window.
    path,manifest=_discover_capture(Path('results/calibration'),split='validation',layer=layer)
    _,rows=_load_direct(path,manifest,layer);x=rows[0].cuda().float();k=_pre_rope_rows(x[...,128:],cos,sin)
    w,b=maps[('dense_v',16)];pred=torch.einsum('tgi,gij->tgj',x[...,:128].double(),w.cuda())+b.cuda()
    direct=(pred-k).square().sum((0,2)).double().cpu();estimated=torch.tensor(output['dense_v_r16']['sse'][0],dtype=torch.float64)
    torch.testing.assert_close(direct,estimated,rtol=2e-4,atol=1e-3)
    return dict(status='complete',layer=layer,fit_windows=4 if smoke else 64,heldout_windows=len(held),denominator=den.tolist(),methods=output,seconds=time.monotonic()-start,captures=capture_info)


def summarize():
    rows=[json.loads((ROOT/f'layer_{i:03d}.json').read_text()) for i in range(36)]
    assert all(r['status']=='complete' and r['fit_windows']==64 and r['heldout_windows']==16 for r in rows)
    den=torch.tensor([r['denominator'] for r in rows],dtype=torch.float64)
    # Resample documents jointly across layers/heads; those are not independent samples.
    rng=torch.Generator().manual_seed(42);idx=torch.randint(16,(2000,16),generator=rng)
    summary={}
    for key in rows[0]['methods']:
        sse=torch.tensor([r['methods'][key]['sse'] for r in rows],dtype=torch.float64)
        boot=(1-sse[:,idx,:].sum(2)/den[:,idx,:].sum(2)).mean((0,2))
        summary[key]=dict(r2=float((1-sse.sum(1)/den.sum(1)).mean()),ci95=torch.quantile(boot,torch.tensor([.025,.975],dtype=torch.float64)).tolist())
    result=dict(status='complete',layers=36,methods=summary)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Local V to pre-RoPE K information','', 'Qwen3-8B-Base; 64x32K fit, 16x32K held-out; 36 layers, eight local KV heads. Fixed C1-V80 checkpoint. All methods use the training K mean as baseline; reported R2 averages per-layer/head ratios. Shuffled controls independently derange windows and permute tokens in each split. Mean-only R2 is exactly zero by definition. PCA is trained on fit K and observes held-out K for reconstruction; it is not a V-only predictor.', '', '| Rank | Dense V | C1-V80 | Shuffled Dense V | Shuffled C1-V80 | K PCA |','|---|---:|---:|---:|---:|---:|']
    for rank in RANKS:
        values=[f"{100*summary[f'{name}_r{rank}']['r2']:.2f}%" if f'{name}_r{rank}' in summary else '—' for name in ['dense_v','c1_v80','shuffled_dense_v','shuffled_c1_v80','k_pca']]
        lines.append('| '+str(rank)+' | '+' | '.join(values)+' |')
    lines+=['','95% paired-window bootstrap intervals (2000 draws) are in summary.json. Per-layer/head/window SSE and denominator are in layer JSON files. Scope is local-head affine readability, not mutual information or all-head V information. Pre-RoPE K is obtained by inverting RoPE on stored post-RoPE captures, matching the earlier experiment. TF32 disabled; FP32 inverse RoPE matching the stored capture convention; FP64 streaming products, accumulation and fitting. Independent direct-row SSE check per layer.', '', 'Environment: basis. Commands: `python -m evaluation.eval_v_key_information --smoke`; `python -m evaluation.eval_v_key_information --shard 0` (0–3); `python -m evaluation.eval_v_key_information --summarize`.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if a.summarize:summarize();return
    for layer in ([0] if a.smoke else range(a.shard,36,4)):
        path=ROOT/('smoke.json' if a.smoke else f'layer_{layer:03d}.json')
        if path.exists() and not a.smoke:continue
        row=layer_run(layer,a.smoke);path.write_text(json.dumps(row,indent=2)+'\n');print('COMPLETE',layer,row['seconds'],flush=True)

if __name__=='__main__':main()
