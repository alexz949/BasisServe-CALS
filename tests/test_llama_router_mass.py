import torch
from evaluation.diagnose_llama_router_mass import window_mass


def test_full_rank_residual_matches_exact_mass_and_short_context_is_complete():
    torch.manual_seed(19)
    dim,heads,kv,length=16,4,2,193
    value=torch.randn(1,kv,length,dim)
    key=torch.randn_like(value)
    positions=[64,128,192]
    queries=torch.randn(1,heads,len(positions),dim)
    banks={rank:{f'base_left_b{rank}':torch.randn(kv,dim,rank)*.1,
        f'base_right_b{rank}':torch.randn(kv,rank,dim)*.1,
        f'base_bias_b{rank}':torch.zeros(kv,dim) if rank==0 else torch.randn(kv,dim)*.1,
        f'residual_encoder_b{rank}_r16':torch.eye(dim).expand(kv,-1,-1),
        f'residual_query_b{rank}_r16':torch.eye(dim).expand(heads,-1,-1)} for rank in (0,4,16)}
    cos=torch.ones(1,length,dim);sin=torch.zeros_like(cos)
    mass=window_mass(value,key,queries,positions,cos,sin,banks,budget=96)
    for arm in ('b0r16','b4r16','b16r16'):
        for metric in mass[arm]:
            assert abs(mass[arm][metric]-mass['exact_k'][metric])<1e-6
    short=window_mass(value,key,queries[:,:,:1],positions[:1],cos,sin,banks,budget=96)
    for report in short.values():
        for value in report.values():assert abs(value-1)<1e-6


def test_output_error_matches_masked_attention_and_original_wo():
    from evaluation.llama_sink_recent_routing import page_support
    torch.manual_seed(31)
    dim,heads,kv,length=16,4,2,193
    value=torch.randn(1,kv,length,dim)
    key=torch.randn_like(value)
    q=torch.randn(1,heads,1,dim)
    wo=torch.randn(25,heads*dim)
    cos=torch.ones(1,length,dim);sin=torch.zeros_like(cos)
    report=window_mass(value,key,q,[length-1],cos,sin,{},budget=96,output_weight=wo)['exact_k']
    logits=(q[:,:,0].reshape(1,kv,heads//kv,dim)@key.transpose(-1,-2))*dim**-.5
    ids,valid=page_support(logits,budget=96)
    mask=torch.zeros(1,kv,length,dtype=torch.bool)
    for h in range(kv):mask[0,h,ids[0,h,valid[0,h]]]=True
    dense=torch.einsum('bght,bgtd->bghd',logits.softmax(-1),value)
    sparse=torch.einsum('bght,bgtd->bghd',logits.masked_fill(~mask[:,:,None],-torch.inf).softmax(-1),value)
    delta=sparse-dense
    expected={'output_squared_error':delta.double().square().sum(),
        'output_energy':dense.double().square().sum(),
        'wo_squared_error':(delta.flatten(1)@wo.T).double().square().sum(),
        'wo_energy':(dense.flatten(1)@wo.T).double().square().sum()}
    for metric,reference in expected.items():
        torch.testing.assert_close(torch.tensor(report[metric],dtype=torch.float64),reference,atol=1e-6,rtol=1e-5)
    assert report['output_squared_error']>0
    short=window_mass(value,key,q,[64],cos,sin,{},budget=96,output_weight=wo)['exact_k']
    assert short['output_squared_error']<1e-10 and short['wo_squared_error']<1e-9
