"""Base4-only query controls with identical causal support constraints."""
import math
import torch
from safetensors import safe_open
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation.llama_sink_recent_routing import page_support
from evaluation.v96kl_common import read_json,sha256

ARMS=('base4_correct','base4_shuffled','base4_mean','base4_norm','random','exact_k')


def load_queries(root,layer,windows_sha):
    pool=[];mean=None
    for index in range(80):
        p=root/'b0r16/fisher'/f'l{layer:03d}'/f'w{index:03d}.safetensors'
        meta=read_json(p.with_suffix('.json'))
        assert meta['status']=='complete' and meta['protocol']['windows_sha256']==windows_sha
        assert meta['window_id']==index and meta['split']==('fit' if index<64 else 'heldout')
        # Read only saved queries, not the large Fisher Gram tensor.
        with safe_open(str(p),framework='pt',device='cpu') as f:q=f.get_tensor('queries')
        assert q.shape==(32,64 if index<64 else 32,128) and torch.isfinite(q).all()
        if index<64:
            part=q.double().mean(1)
            mean=part if mean is None else mean+part
        else:pool.append(q)
    return torch.stack(pool), (mean/64).float()


def random_support(length,kv,budget,device,seed):
    if length<=budget:
        ids=torch.arange(length,device=device).expand(1,kv,-1)
        return ids,torch.ones_like(ids,dtype=torch.bool)
    historical=length-64;pages=math.ceil(historical/32);count=(budget-64)//32
    generator=torch.Generator(device=device).manual_seed(seed)
    chosen=[]
    for _ in range(kv):
        chosen.append(torch.cat((torch.zeros(1,device=device,dtype=torch.long),torch.randperm(pages-1,device=device,generator=generator)[:count-1]+1)).sort().values)
    ids=(torch.stack(chosen)[None,:,:,None]*32+torch.arange(32,device=device)).flatten(-2)
    recent=torch.arange(historical,length,device=device).expand(1,kv,-1)
    valid=torch.cat((ids<historical,torch.ones_like(recent,dtype=torch.bool)),-1)
    return torch.cat((ids,recent),-1),valid


@torch.inference_mode()
def query_metrics(value,key,queries,positions,cos,sin,factors,pool,mean_query,window,layer,budget=1024):
    batch,kv,length,dim=key.shape;heads=queries.shape[1];groups=heads//kv
    assert batch==1 and queries.shape==(1,heads,32,dim)
    torch.testing.assert_close(queries[0].float().cpu(),pool[window-64],atol=0,rtol=0)
    # Use the same head and position in the next held-out window (fixed derangement).
    shuffled=pool[(window-64+1)%16].to(key.device)
    mean_query=mean_query.to(key.device)
    base=build_conditional_routing_sidecar(value,key,base_left=factors['base_left_b4'],
        base_right=factors['base_right_b4'],base_bias=factors['base_bias_b4'],
        residual_encoder=torch.empty(kv,dim,0,device=key.device,dtype=key.dtype),cos=cos,sin=sin)
    base=base.float()
    reports={a:{k:0. for k in ('attention_mass','non_sink_attention_mass','page_recall','routed_page_recall')} for a in ARMS}
    counts={a:{m:0 for m in ('page_recall','routed_page_recall')} for a in ARMS}
    for offset,position in enumerate(positions):
        n=position+1;q=queries[:,:,offset].float()
        logits=(q.reshape(batch,kv,groups,dim)@key[:,:,:n].float().transpose(-1,-2))*dim**-.5
        probs=logits.softmax(-1);non_sink_logits=logits.clone();non_sink_logits[...,:32]=-torch.inf
        non_sink=non_sink_logits.softmax(-1)
        supports={}
        for name,query in [('base4_correct',q),('base4_shuffled',shuffled[:,offset][None]),('base4_mean',mean_query[None])]:
            scores=(query.reshape(batch,kv,groups,dim)@base[:,:,:n].transpose(-1,-2))*dim**-.5
            supports[name]=page_support(scores,budget=budget)
        static=base[:,:,:n].norm(dim=-1)[:,:,None].expand(batch,kv,groups,n)
        supports['base4_norm']=page_support(static,budget=budget)
        supports['random']=random_support(n,kv,budget,key.device,42+layer*100000+(window-64)*100+offset)
        supports['exact_k']=page_support(logits,budget=budget)
        page_count=math.ceil(n/32)
        def pages(ids,valid,routed=False):
            selected=torch.zeros(batch,kv,page_count,device=key.device,dtype=torch.int32)
            if routed:valid=valid & (ids>=32) & (ids<n-64)
            selected.scatter_add_(-1,(ids//32).clamp_max(page_count-1),valid.int())
            return selected>0
        exact_pages=pages(*supports['exact_k']);exact_routed=pages(*supports['exact_k'],routed=True)
        for name,(ids,valid) in supports.items():
            assert (valid.sum(-1)<=budget).all()
            for metric,reference,routed in [('page_recall',exact_pages,False),('routed_page_recall',exact_routed,True)]:
                selected=pages(ids,valid,routed);denominator=reference.sum(-1)
                eligible=denominator>0
                reports[name][metric]+=float((((selected&reference).sum(-1)/denominator.clamp_min(1))*eligible).sum())
                counts[name][metric]+=int(eligible.sum())
            gather=ids.clamp_max(position)[:,:,None].expand(batch,kv,groups,-1)
            mask=valid[:,:,None].expand_as(gather)
            for metric,prob in [('attention_mass',probs),('non_sink_attention_mass',non_sink)]:
                mass=prob.gather(-1,gather).masked_fill(~mask,0).sum(-1)
                assert torch.isfinite(mass).all() and mass.min()>=0 and mass.max()<=1.00001
                reports[name][metric]+=float(mass.mean())/len(positions)
    for name in ARMS:
        for metric in counts[name]:
            assert counts[name][metric]>0
            reports[name][metric]/=counts[name][metric]
    return reports
