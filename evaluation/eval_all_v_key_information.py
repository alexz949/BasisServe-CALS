"""All eight V groups -> each K group, with matched held-out local controls."""
import argparse
import json
from pathlib import Path
import time
from dataclasses import replace
import torch
from safetensors.torch import load_file
from evaluation.eval_v_key_information import MODEL,C1
from evaluation.analyze_qwen3_8b_all_group_v_pre_k import _empty_moments,_moments_from_rows,_fit_map
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _discover_capture,_load_direct,_rotary_embeddings,_pre_rope_rows

ROOT=Path('results/evaluation/qwen3_all_v_key_information')
LOCAL=Path('results/evaluation/qwen3_v_key_information')


@torch.inference_mode()
def run(layer,smoke):
    start=time.monotonic();nf,nv=(4,2) if smoke else (64,16)
    cos,sin=_rotary_embeddings(MODEL,sequence=32768,device=torch.device('cuda'))
    artifact=json.loads((C1/'results.json').read_text())['artifacts'][str(layer)]['file']
    enc=load_file(str(C1/artifact))['value_coordinate_encoders'].cuda().double()
    def load(split,count):
        path,manifest=_discover_capture(Path('results/calibration'),split=split,layer=layer)
        _,rows=_load_direct(path,manifest,layer);assert len(rows)>=count
        return rows[:count]
    def values(row):
        v=row[...,:128].cuda().double();c=torch.einsum('tgi,gir->tgr',v,enc)
        return v,c
    rows=load('fit',nf);total=_empty_moments(8,128,80)
    raw_cross=torch.zeros(1024,8,128,dtype=torch.float64);c1_cross=torch.zeros(640,8,128,dtype=torch.float64)
    rng=torch.Generator(device='cuda').manual_seed(1729);offset=int(torch.randint(1,nf,(1,),device='cuda',generator=rng))
    for i in range(nf):
        v,c=values(rows[i]);k=_pre_rope_rows(rows[i,...,128:].cuda().float(),cos,sin).double()
        total.add_(_moments_from_rows(v,c,k))
        sv,sc=values(rows[(i+offset)%nf]);perm=torch.randperm(32768,device='cuda',generator=rng)
        raw_cross+=(sv[perm].reshape(32768,1024).mT@k.reshape(32768,1024)).reshape(1024,8,128).cpu()
        c1_cross+=(sc[perm].reshape(32768,640).mT@k.reshape(32768,1024)).reshape(640,8,128).cpu()
        if i%8==0:print('PROGRESS',layer,'fit',i+1,nf,flush=True)
    shuffled=replace(total,raw_target_gram=raw_cross,c1_target_gram=c1_cross)
    maps={}
    for prefix,m in [('all',total),('shuffled_all',shuffled)]:
        for rep in ['raw','c1']:
            fitted,_=_fit_map(m,representation=rep)
            maps[f'{prefix}_{rep}']=(fitted.weight.reshape(-1,1024).cuda(),fitted.bias.cuda())
    mean=(total.target_sum/total.row_count).cuda();del rows
    rows=load('validation',nv);rng=torch.Generator(device='cuda').manual_seed(101729)
    offset=int(torch.randint(1,nv,(1,),device='cuda',generator=rng));den=[];errors={k:[] for k in maps}
    for i in range(nv):
        v,c=values(rows[i]);k=_pre_rope_rows(rows[i,...,128:].cuda().float(),cos,sin).double()
        sv,sc=values(rows[(i+offset)%nv]);perm=torch.randperm(32768,device='cuda',generator=rng)
        inputs=dict(all_raw=v.reshape(32768,1024),all_c1=c.reshape(32768,640),shuffled_all_raw=sv[perm].reshape(32768,1024),shuffled_all_c1=sc[perm].reshape(32768,640))
        den.append((k-mean).square().sum((0,2)).cpu())
        for name,(w,b) in maps.items():
            pred=(inputs[name]@w).reshape(32768,8,128)+b
            errors[name].append((pred-k).square().sum((0,2)).cpu())
        if i%4==0:print('PROGRESS',layer,'heldout',i+1,nv,flush=True)
    den=torch.stack(den);methods={}
    for name,arr in errors.items():
        sse=torch.stack(arr);assert bool(torch.isfinite(sse).all())
        methods[name]=dict(sse=sse.tolist(),r2=float((1-sse.sum(0)/den.sum(0)).mean()))
    if not smoke:
        local=json.loads((LOCAL/f'layer_{layer:03d}.json').read_text());ld=torch.tensor(local['denominator'],dtype=torch.float64)
        torch.testing.assert_close(den,ld,rtol=1e-7,atol=.01)
        methods['local_raw']=local['methods']['dense_v_r128'];methods['local_c1']=local['methods']['c1_v80_r80']
    return dict(status='complete',layer=layer,fit_windows=nf,heldout_windows=nv,denominator=den.tolist(),methods=methods,seconds=time.monotonic()-start)


def summarize():
    rows=[json.loads((ROOT/f'layer_{i:03d}.json').read_text()) for i in range(36)]
    assert all(r['status']=='complete' and r['fit_windows']==64 for r in rows)
    den=torch.tensor([r['denominator'] for r in rows],dtype=torch.float64)
    rng=torch.Generator().manual_seed(42);idx=torch.randint(16,(2000,16),generator=rng);summary={}
    for name in rows[0]['methods']:
        sse=torch.tensor([r['methods'][name]['sse'] for r in rows],dtype=torch.float64)
        boot=(1-sse[:,idx,:].sum(2)/den[:,idx,:].sum(2)).mean((0,2))
        summary[name]=dict(r2=float((1-sse.sum(1)/den.sum(1)).mean()),ci95=torch.quantile(boot,torch.tensor([.025,.975],dtype=torch.float64)).tolist())
    (ROOT/'summary.json').write_text(json.dumps(dict(status='complete',methods=summary),indent=2)+'\n')
    names=['local_raw','all_raw','local_c1','all_c1','shuffled_all_raw','shuffled_all_c1']
    lines=['# All-group V to pre-RoPE K information','', 'Qwen3-8B-Base,64x32K fit and16x32K held-out; all36 layers. All eight same-token V heads concatenated: raw1024 or C1 latent640. Unrestricted affine map to each K128 head (no Base16 constraint). FP64 products, fitting and direct held-out SSE; FP32 inverse RoPE matches prior capture convention. Training K mean baseline, layer/head macro-average R2. Local control denominator verified identical to the prior experiment. Shuffled controls independently derange windows and permute tokens in each split. No future/neighboring token or K input in paired predictors.', '', '| Layer | Local Dense V | All Dense V | Local V80 | All V80 | Shuffled all Dense | Shuffled all V80 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+str(r['layer'])+' | '+' | '.join(f"{100*r['methods'][k]['r2']:.2f}%" for k in names)+' |')
    lines.append('| **Mean** | '+' | '.join(f"**{100*summary[k]['r2']:.2f}%**" for k in names)+' |')
    lines+=['','2000 paired-window bootstrap intervals are saved in summary.json. Window/layer/head SSE is retained in layer files. This measures affine readability of K, not mutual information or routing recall.','', 'Environment: basis. Commands: `python -m evaluation.eval_all_v_key_information --smoke`; `python -m evaluation.eval_all_v_key_information --shard 0` (0–3); `python -m evaluation.eval_all_v_key_information --summarize`.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if a.summarize:summarize();return
    for layer in ([0] if a.smoke else range(a.shard,36,4)):
        path=ROOT/('smoke.json' if a.smoke else f'layer_{layer:03d}.json')
        if path.exists() and not a.smoke:continue
        row=run(layer,a.smoke);path.write_text(json.dumps(row,indent=2)+'\n');print('COMPLETE',layer,row['seconds'],flush=True)

if __name__=='__main__':main()
