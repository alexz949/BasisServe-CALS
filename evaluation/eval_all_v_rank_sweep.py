"""Per-target-head rank sweep with all same-token V heads as input."""
import argparse,json,time,csv
from pathlib import Path
import torch
from safetensors.torch import load_file
from evaluation.eval_v_key_information import MODEL,C1,RANKS
from evaluation.eval_all_v_key_information import LOCAL,ROOT as CEILING
from evaluation.analyze_qwen3_8b_all_group_v_pre_k import _empty_moments,_moments_from_rows
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _discover_capture,_load_direct,_rotary_embeddings,_pre_rope_rows
ROOT=Path('results/evaluation/qwen3_all_v_rank_sweep')


def fit(m,rep):
    n=m.row_count;mx=getattr(m,rep+'_sum')/n;my=m.target_sum/n
    gram=getattr(m,rep+'_gram')-n*mx[:,None]*mx[None,:]
    cross=getattr(m,rep+'_target_gram')-n*mx[:,None,None]*my[None]
    eig,u=torch.linalg.eigh((gram+gram.mT)/2)
    cutoff=torch.finfo(gram.dtype).eps*len(mx)*eig.abs().max()
    roots=torch.where(eig>cutoff,eig.clamp_min(torch.finfo(eig.dtype).tiny).rsqrt(),0.)
    inv=(u*roots[None])@u.mT
    left=[];right=[]
    for g in range(8):
        a,s,b=torch.linalg.svd(inv@cross[:,g],full_matrices=False)
        left.append((inv@a)*s[None]);right.append(b)
    return torch.stack(left).cuda(),torch.stack(right).cuda(),mx.cuda(),my.cuda()


def evaluate(m,rep,model):
    l,r,mx,my=model;n=m.row_count
    sx=getattr(m,rep+'_sum').cuda();sy=m.target_sum.cuda()
    xx=getattr(m,rep+'_gram').cuda()-sx[:,None]*mx[None,:]-mx[:,None]*sx[None,:]+n*mx[:,None]*mx[None,:]
    xy=getattr(m,rep+'_target_gram').cuda()-sx[:,None,None]*my[None]-mx[:,None,None]*sy[None]+n*mx[:,None,None]*my[None]
    den=m.target_gram.diagonal(dim1=-2,dim2=-1).sum(-1).cuda()-2*(sy*my).sum(-1)+n*my.square().sum(-1)
    # Orthogonal decoder rows make the rank-r prediction norm diagonal in coordinates.
    norm=(l*(xx@l)).sum(1)
    dot=(l*(xy.permute(1,0,2)@r.mT)).sum(1)
    errors=den[:,None]+(norm-2*dot).cumsum(-1)
    return errors[:,torch.tensor([x-1 for x in RANKS],device='cuda')].mT.cpu(),den.cpu()


@torch.inference_mode()
def run(layer,smoke):
    start=time.monotonic();nf,nv=(4,2) if smoke else (64,16)
    cos,sin=_rotary_embeddings(MODEL,sequence=32768,device=torch.device('cuda'))
    artifact=json.loads((C1/'results.json').read_text())['artifacts'][str(layer)]['file']
    enc=load_file(str(C1/artifact))['value_coordinate_encoders'].cuda().double()
    def load(split,count):
        root,manifest=_discover_capture(Path('results/calibration'),split=split,layer=layer)
        _,rows=_load_direct(root,manifest,layer);assert len(rows)>=count
        return rows[:count]
    def values(row):
        v=row[...,:128].cuda().double();c=torch.einsum('tgi,gir->tgr',v,enc)
        k=_pre_rope_rows(row[...,128:].cuda().float(),cos,sin).double()
        return v,c,k
    total=_empty_moments(8,128,80);rows=load('fit',nf)
    for i in range(nf):
        v,c,k=values(rows[i]);total.add_(_moments_from_rows(v,c,k))
        if i%8==0:print('PROGRESS',layer,'fit',i+1,nf,flush=True)
    models={rep:fit(total,rep) for rep in ['raw','c1']};del rows
    rows=load('validation',nv);errors={'raw':[],'c1':[]};dens=[]
    for i in range(nv):
        v,c,k=values(rows[i]);m=_moments_from_rows(v,c,k)
        for rep in ['raw','c1']:
            err,den=evaluate(m,rep,models[rep]);errors[rep].append(err)
            assert bool(torch.isfinite(err).all()) and bool((err>=-den[None]*1e-7).all())
            if i==0:
                l,r,mx,my=models[rep];x=(v if rep=='raw' else c).reshape(32768,-1)
                direct=[]
                for g in range(8):
                    pred=((x-mx)@l[g,:,:16])@r[g,:16]+my[g]
                    direct.append((pred-k[:,g]).square().sum())
                torch.testing.assert_close(torch.stack(direct).cpu(),err[RANKS.index(16)],rtol=1e-7,atol=.01)
        dens.append(den)
        if i%4==0:print('PROGRESS',layer,'heldout',i+1,nv,flush=True)
    den=torch.stack(dens);methods={}
    for rep,errs in errors.items():
        allerr=torch.stack(errs)
        for j,rank in enumerate(RANKS):
            e=allerr[:,j];methods[f'all_{rep}_r{rank}']=dict(sse=e.tolist(),r2=float((1-e.sum(0)/den.sum(0)).mean()))
    if not smoke:
        old=json.loads((CEILING/f'layer_{layer:03d}.json').read_text())
        torch.testing.assert_close(den,torch.tensor(old['denominator'],dtype=torch.float64),rtol=1e-7,atol=.01)
        for rep in ['raw','c1']:
            torch.testing.assert_close(torch.tensor(methods[f'all_{rep}_r128']['sse'],dtype=torch.float64),torch.tensor(old['methods'][f'all_{rep}']['sse'],dtype=torch.float64),rtol=1e-7,atol=.01)
    return dict(status='complete',layer=layer,fit_windows=nf,heldout_windows=nv,denominator=den.tolist(),methods=methods,seconds=time.monotonic()-start)


def summarize():
    rows=[json.loads((ROOT/f'layer_{i:03d}.json').read_text()) for i in range(36)]
    assert all(r['status']=='complete' and r['fit_windows']==64 and r['heldout_windows']==16 for r in rows)
    local=json.loads((LOCAL/'summary.json').read_text())['methods'];den=torch.tensor([r['denominator'] for r in rows],dtype=torch.float64)
    idx=torch.randint(16,(2000,16),generator=torch.Generator().manual_seed(42));summary={}
    for name in rows[0]['methods']:
        e=torch.tensor([r['methods'][name]['sse'] for r in rows],dtype=torch.float64)
        boot=(1-e[:,idx,:].sum(2)/den[:,idx,:].sum(2)).mean((0,2))
        summary[name]=dict(r2=float((1-e.sum(1)/den.sum(1)).mean()),ci95=torch.quantile(boot,torch.tensor([.025,.975],dtype=torch.float64)).tolist())
    (ROOT/'summary.json').write_text(json.dumps(dict(status='complete',methods=summary),indent=2)+'\n')
    lines=['# All-group V per-K-head rank sweep','', 'Qwen3-8B-Base,64x32K fit and16x32K held-out,36 layers. Input: all eight V groups concatenated, raw1024 or C1-V640. Each K128 group gets a separately fitted rank-r affine map; rank16 means up to8x16=128 total coordinates, not a shared16-dimensional layer code. Training K mean baseline; average layer/head held-out R2. FP64 moments/fitting/evaluation; FP32 inverse RoPE matches previous captures. No new model forward.', '', '| Rank per K head | Local Dense V | All Dense V | Local V80 | All V80 |','|---|---:|---:|---:|---:|']
    for rank in RANKS:
        lv=f"{100*local[f'c1_v80_r{rank}']['r2']:.2f}%" if rank<=80 else '—'
        lines.append(f"| {rank} | {100*local[f'dense_v_r{rank}']['r2']:.2f}% | {100*summary[f'all_raw_r{rank}']['r2']:.2f}% | {lv} | {100*summary[f'all_c1_r{rank}']['r2']:.2f}% |")
    lines+=['','Rank16 direct-row SSE checked independently on the first held-out document for all heads and both inputs in each layer. Rank128 SSE matches the prior unrestricted all-group run per window/head. All baseline denominators match. CI95 uses2000 jointly resampled held-out windows. This measures reconstruction information, not routing quality or TP communication cost.','', 'Environment: basis. Commands: `python -m evaluation.eval_all_v_rank_sweep --smoke`; `python -m evaluation.eval_all_v_rank_sweep --shard 0` (0–3); `python -m evaluation.eval_all_v_rank_sweep --summarize`.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    with (ROOT/'layers.csv').open('w') as f:
        w=csv.writer(f);keys=list(summary);w.writerow(['layer']+keys)
        for r in rows:w.writerow([r['layer']]+[r['methods'][k]['r2'] for k in keys])
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if a.summarize:summarize();return
    for layer in ([0] if a.smoke else range(a.shard,36,4)):
        path=ROOT/('smoke.json' if a.smoke else f'layer_{layer:03d}.json')
        if path.exists() and not a.smoke:continue
        result=run(layer,a.smoke);path.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',layer,result['seconds'],flush=True)

if __name__=='__main__':main()
