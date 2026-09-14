"""Two dense replays for C1/Base moments and packed Page-Fisher; no raw capture."""
import argparse
import gc
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from basisserve.core.c1_v_conditional_k_router import AffineReducedRankMap
from basisserve.core.query_position_sampling import candidate_positions, select_stratified_query_positions
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from evaluation.streaming_k_statistics import RawBaseMoments, base_from_moments, base_mse, packed_fisher, load_fisher_windows
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_residual_grid


def layer_file(root, stage, layer):
    return root/stage/f'layer_{layer:03d}.safetensors'


def save_record(path, tensors, metadata):
    save_tensors(path, {k:v.cpu().contiguous() for k,v in tensors.items()})
    write_json(path.with_suffix('.json'), dict(status='complete', sha256=sha256(path), **metadata))


def verified(path):
    meta = read_json(path.with_suffix('.json'))
    assert meta['status']=='complete' and meta['sha256']==sha256(path)
    return load_file(str(path)), meta


def encoder_for(identity, layer):
    checkpoint = Path(identity['checkpoint'])
    assert sha256(checkpoint/'manifest.json')==identity['manifest_sha256']
    entry = read_json(checkpoint/'manifest.json')['layers'][layer]
    path = checkpoint/entry['file']
    assert sha256(path)==entry['sha256']
    return load_file(str(path))['value_coordinate_encoders']


def restore_base(payload):
    return {16:tuple(AffineReducedRankMap(payload['left'][g],payload['right'][g],payload['bias'][g])
        for g in range(len(payload['left'])))}


@torch.inference_mode()
def replay(args, identity, windows, layers, protocol, *, fisher=False):
    model = AutoModelForCausalLM.from_pretrained(identity['model'], dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda()
    assert model.config.model_type=='llama'
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    config=model.config
    heads,groups=config.num_attention_heads,config.num_key_value_heads
    dim=config.hidden_size//heads
    grid=candidate_positions(args.sequence_length)
    stats_cos,stats_sin=model.model.rotary_emb(torch.empty(1,device='cuda',dtype=torch.float32),
        torch.arange(args.sequence_length,device='cuda')[None])
    moments,covariances,candidates,base_payloads,selections = {},{},{},{},{}
    active={}
    handles=[]
    for layer in layers:
        module=model.model.layers[layer].self_attn
        if fisher:
            base_payloads[layer],meta=verified(layer_file(args.output,'base',layer))
            assert meta['protocol']==protocol and meta['identity_sha256']==sha256(args.identity)
            _,selection_meta=verified(layer_file(args.output,'moments',layer))
            selections[layer]=selection_meta['selections']
        else:
            moments[layer]={s:RawBaseMoments(groups,dim) for s in ('fit','heldout')}
            covariances[layer]={s:torch.zeros(heads*dim,heads*dim,device='cuda',dtype=torch.float32)
                for s in ('fit','heldout')}
            candidates[layer]=torch.empty(args.fit_count,len(grid),heads,dim,dtype=torch.bfloat16)
        def capture(attention,positional,kwargs,layer=layer):
            split,index=active['split'],active['index']
            x=kwargs['hidden_states']; n=x.shape[1]
            k=attention.k_proj(x).view(1,n,groups,dim)
            v=attention.v_proj(x).view(1,n,groups,dim)
            if not fisher:
                moments[layer][split].update(v[0],k[0],args.chunk_rows)
                if split=='fit':
                    q=attention.q_proj(x).view(1,n,heads,dim)
                    q,_=apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),*kwargs['position_embeddings'])
                    candidates[layer][index].copy_(q[0,:,grid].transpose(0,1).cpu())
            else:
                positions=selections[layer][split]['selected_positions']
                q=attention.q_proj(x).view(1,n,heads,dim)
                q,k=apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),*kwargs['position_embeddings'])
                rows=torch.cat((v,k.transpose(1,2)),-1)
                stats,diagnostics=build_multi_query_statistics(q[:,:,positions].transpose(1,2),rows,
                    query_positions=positions,cos=stats_cos,sin=stats_sin,
                    value_encoder=base_payloads[layer]['encoder'],base_maps=restore_base(base_payloads[layer]),
                    page_size=32,excluded_prefix_pages=1,device=torch.device('cuda'))
                path=args.output/'fisher'/f'l{layer:03d}'/f'w{index:03d}.safetensors'
                save_record(path,packed_fisher(stats[16]),dict(protocol=protocol,layer=layer,
                    window_id=index,split=split,base_sha256=sha256(layer_file(args.output,'base',layer)),
                    diagnostics=diagnostics,storage='symmetric upper triangle FP32'))
        handles.append(module.register_forward_pre_hook(capture,with_kwargs=True))
        if not fisher:
            def covariance(module,positional,layer=layer):
                rows=positional[0].reshape(-1,heads*dim)
                result=covariances[layer][active['split']]
                for start in range(0,len(rows),args.chunk_rows):
                    z=rows[start:start+args.chunk_rows].float()
                    result.addmm_(z.mT,z)
            handles.append(module.o_proj.register_forward_pre_hook(covariance))
    for split,indices in (('fit',range(args.fit_count)),('heldout',range(64,64+args.diagnostic_count))):
        for index in indices:
            active.update(split=split,index=index)
            # One window's hidden states exist at a time; never allocate [N,T,H].
            output=model.model(windows[index:index+1].long().cuda(),use_cache=False)
            assert torch.isfinite(output.last_hidden_state).all()
            del output
            print(dict(stage='fisher' if fisher else 'moments',window=index,layers=layers),flush=True)
    for handle in handles:handle.remove()
    if not fisher:
        for layer in layers:
            selections={}
            for split,qpb in (('fit',16),('heldout',8)):
                selected,_,_=select_stratified_query_positions(candidates[layer],grid,
                    context_length=args.sequence_length,num_bins=4,queries_per_bin=qpb)
                selections[split]=selected
            tensors={f'{split}_{name}':value for split,m in moments[layer].items() for name,value in m.tensors().items()}
            save_record(layer_file(args.output,'moments',layer),tensors,
                dict(protocol=protocol,layer=layer,selections=selections,selection_fit_only=True))
            payload={f'{split}_covariance':covariances[layer][split].cpu()/moments[layer][split].count
                for split in ('fit','heldout')}
            payload['weight']=model.model.layers[layer].self_attn.o_proj.weight.detach().cpu()
            save_record(layer_file(args.output,'covariance',layer),payload,dict(protocol=protocol,layer=layer))
            del candidates[layer],covariances[layer],moments[layer]
    del model
    gc.collect();torch.cuda.empty_cache()


def fit_bases(args,identity,layers,protocol):
    for layer in layers:
        payload,meta=verified(layer_file(args.output,'moments',layer))
        assert meta['protocol']==protocol
        split_moments={s:{k.removeprefix(s+'_'):v for k,v in payload.items() if k.startswith(s+'_')}
            for s in ('fit','heldout')}
        encoder=encoder_for(identity,layer)
        bases=base_from_moments(split_moments['fit'],encoder)
        tensors={name:torch.stack([getattr(m,name) for m in bases[16]]) for name in ('left','right','bias')}
        tensors['encoder']=encoder
        save_record(layer_file(args.output,'base',layer),tensors,dict(protocol=protocol,layer=layer,
            identity_sha256=sha256(args.identity),moments_sha256=sha256(layer_file(args.output,'moments',layer)),
            metrics={s:base_mse(m,encoder,bases[16]) for s,m in split_moments.items()}))


def fit_residuals(args,identity,layers,protocol):
    for layer in layers:
        base,meta=verified(layer_file(args.output,'base',layer))
        assert meta['protocol']==protocol and meta['identity_sha256']==sha256(args.identity)
        groups,dim=base['encoder'].shape[:2]
        stats={}
        for split,indices in (('fit',range(args.fit_count)),('heldout',range(64,64+args.diagnostic_count))):
            payloads=[]
            for index in indices:
                path=args.output/'fisher'/f'l{layer:03d}'/f'w{index:03d}.safetensors'
                payload,record=verified(path)
                assert record['protocol']==protocol and record['split']==split and record['window_id']==index
                assert record['base_sha256']==sha256(layer_file(args.output,'base',layer))
                payloads.append(payload)
            heads=payloads[0]['queries'].shape[0]
            stats[split]={16:load_fisher_windows(payloads,heads,groups,dim)}
            del payloads
        factors,losses=_fit_residual_grid(stats['fit'],stats['heldout'],residual_ranks=(16,),
            sweeps=args.sweeps,relative_damping=1e-5,iterative_tolerance=1e-5,
            iterative_max_iterations=args.pcg_iterations,device=torch.device('cuda'))
        tensors={f'base_{name}_b16':base[name].float() for name in ('left','right','bias')}
        tensors.update(residual_encoder_b16_r16=factors[(16,16)][0].cpu().float(),
            residual_query_b16_r16=factors[(16,16)][1].cpu().float())
        assert all(torch.isfinite(t).all() for t in tensors.values())
        save_record(layer_file(args.output,'ours_b16r16',layer),tensors,
            dict(protocol=protocol,layer=layer,v_rank=base['encoder'].shape[-1],identity_sha256=sha256(args.identity),losses=losses,
                base_sha256=sha256(layer_file(args.output,'base',layer)),sweeps=args.sweeps,pcg_iterations=args.pcg_iterations))
        del stats,factors
        gc.collect();torch.cuda.empty_cache()
        print(dict(stage='fit complete',layer=layer,losses=losses),flush=True)


def assemble_covariance(args,identity,protocol):
    config=read_json(Path(identity['model'])/'config.json')
    layers=list(range(config['num_hidden_layers']))
    artifacts={}
    for layer in layers:
        path=layer_file(args.output,'covariance',layer)
        payload,meta=verified(path)
        assert meta['protocol']==protocol
        artifacts[str(layer)]=dict(file=path.name,sha256=sha256(path))
    write_json(args.output/'covariance'/'manifest.json',dict(format='basisserve.attention_o_proj_covariances.v1',
        schema_version=1,model=dict(path=identity['model'],config_sha256=identity['model_config_sha256'],
            model_type='llama',attention_type='gqa',num_hidden_layers=len(layers),
            num_attention_heads=config['num_attention_heads'],num_key_value_heads=config['num_key_value_heads'],
            head_dim=config['hidden_size']//config['num_attention_heads'],hidden_size=config['hidden_size']),
        layers=layers,artifacts=artifacts,calibration=dict(fit_windows=args.fit_count,heldout_windows=args.diagnostic_count,
            window_count=args.fit_count+args.diagnostic_count,sequence_length=args.sequence_length,
            positions_per_window=args.sequence_length,fit_rows=args.fit_count*args.sequence_length,
            heldout_rows=args.diagnostic_count*args.sequence_length,
            rows_per_layer=(args.fit_count+args.diagnostic_count)*args.sequence_length,
            storage='normalized_covariance_sufficient_statistics',
            windows_sha256=protocol['windows_sha256']),protocol=protocol))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=('moments','base','fisher','fit','assemble-covariance','all'))
    for name in ('identity','windows','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layers',required=True)
    p.add_argument('--sequence-length',type=int,default=65536)
    p.add_argument('--fit-count',type=int,default=64)
    p.add_argument('--diagnostic-count',type=int,default=16)
    p.add_argument('--chunk-rows',type=int,default=2048)
    p.add_argument('--sweeps',type=int,default=40)
    p.add_argument('--pcg-iterations',type=int,default=100)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();configure()
    identity=read_json(args.identity)
    assert identity['status']=='complete'
    assert sha256(Path(identity['model'])/'config.json')==identity['model_config_sha256']
    windows=load_file(str(args.windows))['input_ids']
    wm=read_json(args.windows.with_name('manifest.json'))
    assert wm['sha256']==sha256(args.windows) and wm['model_config_sha256']==identity['model_config_sha256']
    assert wm['fit_ids']==list(range(64)) and wm['validation_ids']==list(range(64,80))
    assert windows.shape[0]==80 and 4096<=args.sequence_length<=windows.shape[1]
    if not args.smoke:
        assert not wm.get('test_only', False)
        assert windows.shape[1]==args.sequence_length and args.fit_count==64 and args.diagnostic_count==16
        assert args.sweeps==40 and args.pcg_iterations==100
    assert 0<args.fit_count<=64 and 0<args.diagnostic_count<=16 and args.chunk_rows>0
    windows=windows[:,:args.sequence_length]
    layers=[int(x) for x in args.layers.split(',')]
    assert len(set(layers))==len(layers) and set(layers)<=set(identity['attention_layers'])
    protocol=dict(format='basisserve.k_router.streaming.v1',model_config_sha256=identity['model_config_sha256'],
        windows_sha256=sha256(args.windows),windows_manifest_sha256=sha256(args.windows.with_name('manifest.json')),
        sequence_length=args.sequence_length,fit_ids=list(range(args.fit_count)),
        diagnostic_ids=list(range(64,64+args.diagnostic_count)),fit_queries=64,diagnostic_queries=32,
        base_rank=16,residual_rank=16,page_size=32,excluded_prefix_pages=1,chunk_rows=args.chunk_rows,
        smoke=args.smoke,teacher='native dense BF16 SDPA, model.model, use_cache=False',
        base_moments='FP64 raw moments and encoder transform; no bitwise claim versus projected FP32 accumulation',
        covariance='FP32 chunked normalized o_proj input second moment',
        source_sha256={name:sha256(Path(name)) for name in ('evaluation/fit_k_routing_streaming.py',
            'evaluation/streaming_k_statistics.py','evaluation/fit_qwen3_8b_q8_fisher_residual.py',
            'basisserve/core/query_position_sampling.py','basisserve/core/c1_v_conditional_k_router.py')})
    stages=('moments','base','fisher','fit') if args.stage=='all' else (args.stage,)
    for stage in stages:
        if stage=='moments':replay(args,identity,windows,layers,protocol)
        elif stage=='base':fit_bases(args,identity,layers,protocol)
        elif stage=='fisher':replay(args,identity,windows,layers,protocol,fisher=True)
        elif stage=='fit':fit_residuals(args,identity,layers,protocol)
        else:assemble_covariance(args,identity,protocol)


if __name__=='__main__':
    main()
