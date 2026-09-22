"""Llama V96 RULER baselines (Full-K, LRQK, ShadowKV, Loki) on frozen prompts shared with the
B16R16/Full-K evaluator; a copy of eval_k_routing_ruler with a --prompts input, a Loki arm and
per-baseline support knobs. Paired full-K, exact-page, LRQK, ShadowKV and Base16/R16 RULER evaluation."""
import argparse
import inspect
import json
import math
from pathlib import Path
import shlex
import sys
import time
from types import MethodType

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256
from evaluation.k_routing_config import routing_config, validate_residual_fisher_support
from evaluation.ruler_v1 import parse_tasks, ruler_prompt, sample_score
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.checkpoint.gqa_vo_nemotron_h import nemotron_h_c1_attention
from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk_adapter
from basisserve.checkpoint.c1_shadowkv_qwen3 import C1ShadowKVCache, install_c1_shadowkv
from basisserve.core.c1_lrqk import LRQKConfig
from basisserve.core.c1_lrqk import LRQKState
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar, conditional_routing_query_projector
from basisserve.core.c1_conditional_page_attention import c1_conditional_page_topk_attention
from basisserve.kernels.compressed_v_decode_attention import (
    compressed_v_decode_attention_triton, compressed_v_prefill_attention,
)

ARMS = ('full','lrqk','shadowkv','loki')
# Set from the CLI in main(); the formal Llama runs used LRQK top-832 + recent 64 and Loki top-856 + recent 0.
LRQK_TOPK, LOKI_TOPK, LOKI_RECENT = 832, 856, 0
DEFAULT_TASK_NAMES = (
    'niah_single_1','niah_single_2','niah_single_3','niah_multikey_1',
    'niah_multikey_2','niah_multiquery','niah_multivalue','vt','fwe','qa_1','qa_2',
)


def native_protocol(args, identity, config):
    from transformers.models.nemotron_h import modeling_nemotron_h as native
    from evaluation.install_nemotron_h_wo import audit_nemotron_h_wo

    assert args.native_audit is not None and args.full_smoke is not None and args.wo_bank is not None
    audit, smoke = read_json(args.native_audit), read_json(args.full_smoke)
    assert audit['status'] == smoke['status'] == 'complete'
    assert audit['config_sha256'] == identity['model_config_sha256']
    assert audit['index_sha256'] == sha256(Path(identity['model'])/'model.safetensors.index.json')
    assert audit['implementation_sha256'] == sha256(inspect.getfile(native.NemotronHForCausalLM))
    assert smoke['audit_sha256'] == sha256(args.native_audit)
    assert smoke['full_model_tested'] and smoke['verified_tensor_count'] == audit['tensor_count'] == 577
    assert smoke['device_guard_sha256'] == sha256(ROOT/'evaluation/nemotron_h_runtime.py')
    assert smoke['guarded_mamba_layers'] == [i for i,k in enumerate(config.layers_block_type) if k=='linear_attention']
    assert identity['attention_layers'] == [i for i,k in enumerate(config.layers_block_type) if k=='full_attention']
    wo=audit_nemotron_h_wo(config,args.identity,args.native_audit,args.wo_bank)
    wo_audit_path=args.wo_bank.parent/'manifests/wo_audit.json'
    wo_audit=read_json(wo_audit_path)
    assert wo_audit['status']=='complete' and wo_audit['verified_layers']==45 and wo_audit['wo']==wo
    return dict(audit_sha256=sha256(args.native_audit), full_smoke_sha256=sha256(args.full_smoke),
        device_guard_sha256=smoke['device_guard_sha256'], guarded_mamba_layers=smoke['guarded_mamba_layers'],
        wo=wo,wo_audit_sha256=sha256(wo_audit_path))


def install_native_runtime(model, args, protocol):
    import causal_conv1d
    import mamba_ssm
    from transformers.models.nemotron_h import modeling_nemotron_h as native
    from evaluation.install_nemotron_h_wo import install_nemotron_h_wo
    from evaluation.nemotron_h_runtime import install_mamba_device_guards
    from evaluation.nemotron_h_scan_layout import install_contiguous_dt_scan, install_chunked_mamba_norms

    for name, package in (('mamba2_chunk_scan','mamba_ssm'),
            ('mamba2_selective_state_update','mamba_ssm'),
            ('causal_conv1d_fn','causal_conv1d'),('causal_conv1d_update','causal_conv1d')):
        implementation=inspect.getclosurevars(getattr(native,name)).nonlocals['implementation']
        assert implementation.__module__.startswith(package)
    assert install_mamba_device_guards(model)==protocol['guarded_mamba_layers']
    assert install_nemotron_h_wo(model,args.identity,args.native_audit,args.wo_bank)==protocol['wo']
    install_contiguous_dt_scan(native)
    install_chunked_mamba_norms(model)


def load_evaluation_model(identity, config):
    options=dict(config=config,dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=True,trust_remote_code=False)
    count=torch.cuda.device_count()
    assert count>0
    if count==1:
        model=AutoModelForCausalLM.from_pretrained(identity['model'],**options).cuda().eval()
    else:
        # Leave room on each card for the dense K cache, compressed V, and prefill.
        memory={i:f'{int(torch.cuda.get_device_properties(i).total_memory/2**30)-12}GiB'
                for i in range(count)}
        model=AutoModelForCausalLM.from_pretrained(identity['model'],device_map='balanced',
            max_memory=memory,**options).eval()
        assert all(str(device) not in ('cpu','disk') for device in model.hf_device_map.values())
    if config.model_type=='llama':
        from evaluation.chunked_prefill_mlp import install_chunked_prefill_mlps
        install_chunked_prefill_mlps(model)
    return model


class RoutingCache(DynamicCache):
    def __init__(self, config):
        super().__init__(config=config)
        self.sidecars = {}
        self.statistics = {}


@torch.inference_mode()
def routing_forward(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, **kwargs):
    batch, length, _ = hidden_states.shape
    assert batch == 1
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    q = self.q_norm(self.q_proj(hidden_states).view(batch,length,self.num_attention_heads,self.head_dim)).transpose(1,2)
    pre = self.k_norm(self.k_proj(hidden_states).view(batch,length,self.num_key_value_heads,self.head_dim)).transpose(1,2)
    v = self.v_proj(hidden_states).view(batch,length,self.num_key_value_heads,self.value_head_dim).transpose(1,2)
    cos,sin = position_embeddings
    q,k = apply_rotary_pos_emb(q,pre,cos,sin)
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or length == 1
    assert attention_mask is None or length == 1
    if self._routing_arm == 'ours':
        t = self._routing_factors
        if self._routing_base_rank==0:
            current=torch.einsum('bhtd,hdr->bhtr',k,t['residual_encoder'].to(k.dtype))
        else:
            current = build_conditional_routing_sidecar(v,k,base_left=t['base_left'],
                base_right=t['base_right'],base_bias=t['base_bias'],
                residual_encoder=t['residual_encoder'],cos=cos,sin=sin)
        past_key_values.sidecars[self.layer_idx] = (torch.cat((past_key_values.sidecars[self.layer_idx],current),2)
            if previous else current)
    elif self._routing_arm == 'loki':
        from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
        current = build_routing_sidecar(k, self._loki_projector)
        past_key_values.sidecars[self.layer_idx] = (torch.cat((past_key_values.sidecars[self.layer_idx], current), 2)
            if previous else current)
    k,v = past_key_values.update(k,v,self.layer_idx)
    if previous == 0:
        output = compressed_v_prefill_attention(q,k,v,scale=self.scaling)
    elif self._routing_arm == 'full':
        output = compressed_v_decode_attention_triton(q,k,v,scale=self.scaling)
    elif self._routing_arm == 'loki':
        from basisserve.core.c1_loki_attention import c1_loki_recent_decode
        result = c1_loki_recent_decode(q, k, v, self._loki_projector,
            past_key_values.sidecars[self.layer_idx], top_k=LOKI_TOPK, recent_tokens=LOKI_RECENT, scale=self.scaling)
        output = result.output
        past_key_values.statistics[self.layer_idx] = result.statistics
    else:
        if self._routing_arm == 'exact_sparse':
            # FP32 identity routing gives an exact-QK Page-mass oracle with the same GQA rule.
            sidecar = k.float()
            projector = torch.eye(self.head_dim,device=q.device).expand(self.num_attention_heads,-1,-1)
            route_q, route_k, route_v = q.float(),k.float(),v.float()
        else:
            sidecar = past_key_values.sidecars[self.layer_idx]
            projector = self._routing_projector
            route_q,route_k,route_v = q,k,v
        if self.config.model_type == 'llama':
            from evaluation.llama_sink_recent_routing import page_support
            from basisserve.kernels.split_indexed_attention import split_indexed_attention
            if attention_mask is not None:
                assert attention_mask.shape[-2] == 1
                assert bool(attention_mask.all()) if attention_mask.dtype == torch.bool else bool((attention_mask == 0).all())
            group_heads = self.num_attention_heads // self.num_key_value_heads
            codes = torch.einsum('bhqd,hdr->bhqr', q.float(), projector.float())
            codes = codes[:, :, 0].reshape(batch, self.num_key_value_heads, group_heads, -1)
            scores = (codes @ sidecar.float().transpose(-1, -2)) * self.scaling
            ids, valid = page_support(scores)
            selected = ids.masked_fill(~valid, -1).repeat_interleave(group_heads, dim=1)
            output = split_indexed_attention(q, k, v, selected, scale=self.scaling)
            past_key_values.statistics[self.layer_idx] = dict(
                selected_tokens_mean=float(valid.sum(-1).float().mean()), sink_tokens=32,
                recent_tokens=64, token_budget=2048)
        else:
            result = c1_conditional_page_topk_attention(route_q,route_k,route_v,sidecar,projector,
                page_size=32,exact_token_budget=2048,pinned_prefix_pages=1,scale=self.scaling,
                query_block_size=1,attention_mask=attention_mask,collect_statistics=True)
            output = result.output.to(v.dtype)
            past_key_values.statistics[self.layer_idx] = result.statistics
    # Release the full prefill Q buffer before allocating the output projection.
    del q, pre
    output=output.transpose(1,2).contiguous().reshape(batch,length,-1)
    return self.o_proj(output),None


@torch.inference_mode()
def install(model, checkpoint, manifest, arm, bank, *, dense_v=False):
    config = model.config
    assert config.model_type in ('llama','qwen3','nemotron_h')
    hq,hkv,hidden = config.num_attention_heads,config.num_key_value_heads,config.hidden_size
    sources = {record['layer']:record for record in manifest['layers']}
    targets = c1_attention_layers(model)
    assert len(sources)==len(manifest['layers']) and set(sources)=={i for i,_ in targets}
    for index,m in targets:
        layer,record = model.model.layers[index],sources[index]
        if config.model_type == 'llama':
            m.q_norm,m.k_norm = torch.nn.Identity(),torch.nn.Identity()
            m.num_key_value_groups = hq//hkv
            m.sliding_window = None
        dim = m.head_dim
        if dense_v:
            rank = dim
            e = torch.eye(dim, device=m.v_proj.weight.device, dtype=m.v_proj.weight.dtype).expand(hkv, -1, -1)
            d = m.o_proj.weight.detach().T.reshape(hq, dim, hidden)
        else:
            t = load_file(str(checkpoint/record['file']))
            e,d = t['value_coordinate_encoders'],t['head_output_decoders']
            rank = record['ranks'][0]
            assert t['source_ranks'].tolist()==record['ranks']
        assert e.shape==(hkv,dim,rank) and d.shape==(hq,rank,hidden)
        assert torch.isfinite(e).all() and torch.isfinite(d).all()
        device=m.v_proj.weight.device
        weight=torch.bmm(e.to(device).float().mT,m.v_proj.weight.float().reshape(hkv,dim,hidden)).reshape(hkv*rank,hidden)
        decoder=d.to(device).float().permute(2,0,1).reshape(hidden,hq*rank)
        factory = nemotron_h_c1_attention if config.model_type=='nemotron_h' else GQATiedVOQwen3Attention
        replacement=factory(m,v_proj_compressed_weight=weight,
            o_decoder_weight=decoder,attention_backend='triton',value_coordinate_encoder=e)
        if config.model_type=='nemotron_h':
            layer.mixer=replacement
        else:
            layer.self_attn=replacement
        if arm in ('full','exact_sparse','ours','loki'):
            replacement._routing_arm=arm
            if arm == 'loki':
                replacement._loki_projector = bank[record['layer']]['projector'].to(device=device, dtype=torch.bfloat16)
            if arm=='ours':
                payload=bank[record['layer']]
                base_keys=[k for k in payload if k.startswith('base_left_b')]
                residual_keys=[k for k in payload if k.startswith('residual_encoder_b')]
                assert len(base_keys)==len(residual_keys)==1
                base_tag=base_keys[0].removeprefix('base_left_')
                residual_tag=residual_keys[0].removeprefix('residual_encoder_')
                assert residual_tag.startswith(base_tag+'_r')
                replacement._routing_factors={
                    **{name:payload[name+'_'+base_tag].to(device) for name in ('base_left','base_right','base_bias')},
                    **{name:payload[name+'_'+residual_tag].to(device) for name in ('residual_encoder','residual_query')}}
                replacement._routing_base_rank=int(base_tag[1:])
                residual_query=replacement._routing_factors['residual_query']
                replacement._routing_projector=(residual_query if replacement._routing_base_rank==0 else
                    conditional_routing_query_projector(residual_query)).to(device)
            replacement.forward=MethodType(routing_forward,replacement)
    if arm=='lrqk':
        lrqk_adapter.LRQKState=LRQKState
        lrqk_adapter.install_c1_lrqk(model,LRQKConfig(topk=LRQK_TOPK))
    elif arm=='shadowkv':
        install_c1_shadowkv(model)
    model.eval()


def inputs(args, tokenizer, *, task_names, samples_per_task):
    identity=read_json(args.identity)
    runtime_config=routing_config(identity,rope=args.rope,sequence_length=args.sequence_length)
    checkpoint=Path(identity['checkpoint'])
    manifest=read_json(checkpoint/'manifest.json')
    assert sha256(checkpoint/'manifest.json')==identity['manifest_sha256']
    assert sha256(Path(identity['model'])/'config.json')==identity['model_config_sha256']
    for record in manifest['layers']:
        assert sha256(checkpoint/record['file'])==record['sha256']
    data=read_json(args.data/'manifest.json')
    task_names = tuple(task_names)
    assert task_names and len(set(task_names)) == len(task_names)
    assert samples_per_task > 0
    assert data['status']=='complete' and data['protocol']['samples_per_task']==samples_per_task
    assert data['protocol']['sequence_length']==args.sequence_length
    assert data['tokenizer_config_sha256']==sha256(Path(identity['model'])/'tokenizer_config.json')
    assert set(task_names).issubset(data['protocol']['tasks'])
    tasks=parse_tasks(','.join(task_names))
    assert len(tasks)==len(task_names)
    # Frozen prompts written by eval_llama_cal128 prepare (chat template, generation prompt, official
    # answer prefix): every arm, including the B16R16/Full-K arms run by that evaluator, sees identical tokens.
    frozen=read_json(args.prompts/'prompts.json')
    assert frozen['status']=='complete' and frozen['identity_sha256']==sha256(args.identity)
    assert frozen['data_sha256']==sha256(args.data/'manifest.json')
    assert frozen['tokens_sha256']==sha256(args.prompts/'prompts.safetensors')
    tokens=load_file(str(args.prompts/'prompts.safetensors'))
    by_task={t.name:t for t in tasks}
    rows=[]
    for row in frozen['rows']:
        if row['task'] not in by_task: continue
        task=by_task[row['task']]
        ids=tokens[str(row['index'])].tolist()
        assert len(ids)==row['input_tokens'] and len(ids)+task.tokens_to_generate<=args.sequence_length
        rows.append(dict(index=row['index'],task=task.name,ordinal=row['ordinal'],input_ids=ids,input_sha256=row['input_sha256'],
            answers=row['answers'],match_type=task.match_type,maximum_tokens=task.tokens_to_generate))
    assert len(rows)==len(tasks)*samples_per_task and [r['index'] for r in rows]==list(range(len(rows)))
    bank,bank_hashes={},{}
    if args.dense_v:
        assert identity['layer_ranks'] == [identity['head_dim']] * len(identity['attention_layers'])
    if args.arm=='loki' or args.stage in ('summarize','audit-smoke'):
        pca=read_json(args.loki/'manifest.json')
        assert pca['status']=='complete' and pca['rank']==32 and not pca['smoke']
        assert pca['model_config_sha256']==identity['model_config_sha256']
        assert pca['runtime']=='post-RoPE Q/K projection without mean subtraction'
        assert [row['layer'] for row in pca['layers']]==identity['attention_layers']
        for entry in pca['layers']:
            path=args.loki/entry['file']
            assert sha256(path)==entry['sha256']
            projector=load_file(str(path))['projector']
            assert projector.shape==(identity['hkv'],identity['head_dim'],32) and torch.isfinite(projector).all()
            bank[entry['layer']]=dict(projector=projector);bank_hashes[str(entry['layer'])]=entry['sha256']
    config_record=runtime_config.to_dict()
    if runtime_config.model_type=='nemotron_h':
        config_record['time_step_limit']=[str(x) if math.isinf(x) else x for x in runtime_config.time_step_limit]
    spec=dict(identity_sha256=sha256(args.identity),data_sha256=sha256(args.data/'manifest.json'),
        runtime_config=config_record,rope=args.rope,
        sequence_length=args.sequence_length,samples=len(rows),samples_per_task=samples_per_task,
        task_names=list(task_names),dtype='bfloat16',generation='greedy, native EOS, official caps',
        prefill=('full causal FlashAttention-2 on all arms' if args.dense_v else
                 'full causal FlashAttention-2 for equal QKV widths, C1 Triton for compact V'),rank_schedule=identity['layer_ranks'],
        exact_sparse='FP32 exact-QK normalized Page32 mass, GQA max, pinned page0 within B2048',
        ours='native Base16/Residual16 Page32 GQA max, pinned page0 within B2048',
        prompts_sha256=sha256(args.prompts/'prompts.json'),
        input_template='frozen prompts from eval_llama_cal128 prepare: chat template, generation prompt, official answer prefix',
        loki=dict(pca_rank=32,topk_per_query_head=LOKI_TOPK,recent=LOKI_RECENT,calibration='dense model, post-RoPE K PCA, 32 x 128K C4',
                  manifest_sha256=sha256(args.loki/'manifest.json') if args.loki else None),
        lrqk=dict(rank=32,topk_per_query_head=LRQK_TOPK,recent=64,prefill_iterations=2,decode_iterations=2,tolerance=0.01,seed=0,state_dtype='bfloat16',solve_dtype='float32'),
        shadowkv=dict(rank=160,chunk=8,routed=2048,outlier_chunks=48,extra_support='native local and generated tokens'),
        source_sha256={n:sha256(ROOT/n) for n in ('evaluation/eval_k_routing_ruler.py',
            'evaluation/k_routing_config.py',
            'basisserve/core/c1_conditional_page_attention.py','basisserve/core/c1_v_conditional_k_router.py',
            'basisserve/core/c1_lrqk.py','basisserve/core/c1_shadowkv.py',
            'basisserve/checkpoint/c1_lrqk_qwen3.py','basisserve/checkpoint/c1_shadowkv_qwen3.py',
            'basisserve/kernels/compressed_v_decode_attention.py','evaluation/eval_longbench_lrqk_fp32route.py',
            'basisserve/core/c1_loki_attention.py','basisserve/core/c1_k_routing_sidecar.py')})
    if runtime_config.model_type=='nemotron_h':
        spec['native']=native_protocol(args,identity,runtime_config)
        for name in ('evaluation/install_nemotron_h_wo.py','evaluation/nemotron_h_runtime.py',
                     'evaluation/nemotron_h_scan_layout.py',
                     'evaluation/chunked_prefill_mlp.py',
                'basisserve/checkpoint/gqa_vo_nemotron_h.py','basisserve/checkpoint/gqa_vo_qwen3.py',
                'basisserve/checkpoint/c1_attention_layers.py','basisserve/core/tp_source_wo_c1.py'):
            spec['source_sha256'][name]=sha256(ROOT/name)
    else:
        assert args.native_audit is None and args.full_smoke is None and args.wo_bank is None
    if runtime_config.model_type == 'llama':
        spec['prefill_mlp_chunk_tokens'] = 1024
        spec['exact_sparse'] = 'FP32 exact-QK Page32 mass, GQA max; sink32 + recent64 inside hard B2048'
        spec['ours'] = 'Base16/Residual16 Page32 mass, GQA max; sink32 + recent64 inside hard B2048'
        for name in ('evaluation/llama_sink_recent_routing.py', 'basisserve/kernels/split_indexed_attention.py',
                     'evaluation/chunked_prefill_mlp.py'):
            spec['source_sha256'][name] = sha256(ROOT/name)
    spec['value_mode'] = 'dense original V and Wo' if args.dense_v else 'allocated C1 V'
    if args.chat_template:
        spec['input_template'] = 'tokenizer.apply_chat_template user message, add_generation_prompt=True'
    if args.official_lrqk:
        assert args.dense_v
        spec['lrqk'].update(tolerance=0.01, implementation='upstream unchanged cache', cpu_offload=True,
            initialization='upstream global RNG randn', weights=[1,1,1,1])
        for name in ('evaluation/official_lrqk_state.py','external/LRQK/lrqk_attention.py',
                     'external/LRQK/cpp_kernel/__init__.py','external/LRQK/cpp_kernel/take_along_dim_grouped.cpp'):
            spec['source_sha256'][name]=sha256(ROOT/name)
    if args.dense_v:
        spec['rank_schedule'] = [identity['head_dim']] * len(identity['attention_layers'])
    spec=json.loads(json.dumps(spec,allow_nan=False))
    return identity,manifest,rows,bank,bank_hashes,spec


@torch.inference_mode()
def generate(model,tokenizer,row,arm,cap):
    torch.manual_seed(0)
    cache=(lrqk_adapter.C1LRQKCache(config=model.config) if arm=='lrqk' else
           C1ShadowKVCache(model.config) if arm=='shadowkv' else RoutingCache(model.config))
    input_device=model.get_input_embeddings().weight.device
    tokens=torch.tensor([row['input_ids']],device=input_device)
    out=model(input_ids=tokens,past_key_values=cache,use_cache=True,logits_to_keep=1)
    assert torch.isfinite(out.logits).all()
    first=out.logits[0,-1].float().cpu();ids=[int(first.argmax())]
    del out,tokens
    eos=model.config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id}
    while len(ids)<cap and ids[-1] not in eos:
        mask=torch.ones(1,1,1,cache.get_seq_length()+1,device=input_device,dtype=torch.bool)
        if model.config.model_type=='nemotron_h':
            mask={'full_attention':mask,'linear_attention':None}
        out=model(input_ids=torch.tensor([[ids[-1]]],device=input_device),past_key_values=cache,
            attention_mask=mask,use_cache=True,logits_to_keep=1)
        assert torch.isfinite(out.logits).all()
        ids.append(int(out.logits[0,-1].argmax()));del out
    stats=[]
    for i,m in c1_attention_layers(model):
        k,v=cache.layers[i].keys,cache.layers[i].values
        assert k.shape==(1,m.num_key_value_heads,len(row['input_ids'])+len(ids)-1,m.head_dim)
        assert v.shape==(*k.shape[:-1],m.value_head_dim)
        if arm=='lrqk':stats.append(cache.lrqk_states[i].statistics(m.num_key_value_heads))
        elif arm=='shadowkv':stats.append(cache.shadow_states[i].statistics())
        else:stats.append(cache.statistics.get(i,{}))
    return ids,first,stats,ids[-1] in eos


def summarize(args,rows,spec,tokenizer,bank_hashes,identity):
    results={}
    config=read_json(Path(identity['model'])/'config.json')
    eos=config['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id}
    for arm in ARMS:
        results[arm]=[]
        for row in rows:
            saved=read_json(args.output/arm/'evaluate'/f"sample_{row['index']:03d}.json")
            assert saved['status']=='complete' and saved['protocol']==spec and saved['sample']==row
            assert saved['bank_sha256']==(bank_hashes if arm=='loki' else {})
            r=saved['result'];ids=r['ids']
            assert 0<len(ids)<=row['maximum_tokens'] and not any(t in eos for t in ids[:-1])
            assert r['stopped']==(ids[-1] in eos) and (r['stopped'] or len(ids)==row['maximum_tokens'])
            assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['prediction']
            assert sample_score(r['prediction'],row['answers'],row['match_type'])==r['score']
            results[arm].append(r)
        assert all(a['ids'][0]==b['ids'][0] for a,b in zip(results['full'],results[arm],strict=True))
    tasks={t:{a:100*sum(r['score'] for r,row in zip(rs,rows,strict=True) if row['task']==t)/8
              for a,rs in results.items()} for t in dict.fromkeys(r['task'] for r in rows)}
    means={a:100*sum(r['score'] for r in rs)/len(rs) for a,rs in results.items()}
    write_json(args.output/'summary.json',dict(status='complete',verified_predictions=len(ARMS)*len(rows),
        protocol=spec,means=means,tasks=tasks,results=results,bank_sha256=bank_hashes,
        refit64k_required=None if args.dense_v else means['full']-means['ours']>4))
    lines=[f"# RULER {spec['sequence_length']//1024}K: "+spec['value_mode'],'',
        f'{len(rows)} prompts; shared V setting across arms.', '',
        '| Task | '+' | '.join(ARMS)+' |','|---|'+'---:|'*len(ARMS)]
    for t,values in [*tasks.items(),('Mean',means)]:
        lines.append('| '+t+' | '+' | '.join(f'{values[a]:.4f}' for a in ARMS)+' |')
    path=args.output/'summary.md';text='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print('VERIFIED',means,flush=True)


def audit_smoke(args,rows,spec,bank_hashes):
    first={}
    for arm in ARMS:
        for index in (0,56):
            smoke_dir='smoke_mlp1024_deviceguard' if 'prefill_mlp_chunk_tokens' in spec else 'smoke'
            saved=read_json(args.output/arm/smoke_dir/f'sample_{index:03d}.json')
            assert saved['status']=='complete' and saved['protocol']==spec
            assert saved['sample']==rows[index]
            assert saved['bank_sha256']==(bank_hashes if arm=='loki' else {})
            result=saved['result']
            assert 0<len(result['ids'])<=4 and (result['stopped'] or len(result['ids'])==4)
            if arm=='full':first[index]=result['ids'][0]
            assert first[index]==result['ids'][0]
            if arm!='full':
                assert len(result['ids'])>1, 'Smoke must exercise sparse decode'
    write_json(args.output/'smoke_audit.json',dict(status='complete',protocol=spec,
        verified=2*len(ARMS),first_token_agreement=True,bank_sha256=bank_hashes))
    print('ALL FIVE ARM SMOKES VERIFIED',flush=True)


def main():
    global ARMS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['smoke','audit-smoke','evaluate','summarize'])
    p.add_argument('--arm',choices=ARMS,default='full')
    p.add_argument('--dense-v', action='store_true')
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequence-length',type=int,default=65536)
    p.add_argument('--rope',choices=('native','yarn2'),required=True)
    p.add_argument('--native-audit',type=Path)
    p.add_argument('--full-smoke',type=Path)
    p.add_argument('--wo-bank',type=Path)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=4)
    p.add_argument('--official-lrqk',action='store_true')
    p.add_argument('--prompts',type=Path,required=True,help='directory with frozen prompts.json/prompts.safetensors')
    p.add_argument('--loki',type=Path,help='Loki PCA directory (manifest.json + layer files)')
    p.add_argument('--lrqk-topk',type=int,default=832)
    p.add_argument('--loki-topk',type=int,default=856)
    p.add_argument('--loki-recent',type=int,default=0)
    p.add_argument('--chat-template',action='store_true')
    args=p.parse_args();configure();torch.manual_seed(0)
    if args.dense_v:
        ARMS = ('full', 'lrqk', 'shadowkv', 'loki')
        assert args.arm in ARMS
    global LRQK_TOPK, LOKI_TOPK, LOKI_RECENT
    LRQK_TOPK, LOKI_TOPK, LOKI_RECENT = args.lrqk_topk, args.loki_topk, args.loki_recent
    identity=read_json(args.identity)
    tokenizer=AutoTokenizer.from_pretrained(identity['model'],local_files_only=True)
    identity,manifest,rows,bank,bank_hashes,spec=inputs(
        args, tokenizer, task_names=DEFAULT_TASK_NAMES,
        samples_per_task=read_json(args.data/'manifest.json')['protocol']['samples_per_task']
    )
    if args.stage=='summarize':
        summarize(args,rows,spec,tokenizer,bank_hashes,identity)
        return
    if args.stage=='audit-smoke':
        audit_smoke(args,rows,spec,bank_hashes)
        return
    if args.stage=='evaluate':
        gate=read_json(args.output/'smoke_audit.json')
        assert gate['status']=='complete' and gate['protocol']==spec
    config=routing_config(identity,rope=args.rope,sequence_length=args.sequence_length)
    if config.model_type=='nemotron_h':
        import causal_conv1d
        import mamba_ssm
    model=load_evaluation_model(identity,config)
    if config.model_type=='nemotron_h':
        install_native_runtime(model,args,spec['native'])
    install(model,Path(identity['checkpoint']),manifest,args.arm,bank,dense_v=args.dense_v)
    if args.official_lrqk and args.arm=='lrqk':
        from evaluation.official_lrqk_state import OfficialLRQKState
        lrqk_adapter.LRQKState=OfficialLRQKState
        for _,module in c1_attention_layers(model):
            module._lrqk_official=True
    selected=[rows[0],rows[56]] if args.stage=='smoke' else rows[args.shard_index::args.num_shards]
    for row in selected:
        stage_dir='smoke_mlp1024_deviceguard' if args.stage=='smoke' and 'prefill_mlp_chunk_tokens' in spec else args.stage
        path=args.output/args.arm/stage_dir/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved=read_json(path)
            assert saved['status']=='complete' and saved['protocol']==spec and saved['sample']==row
            assert saved['bank_sha256']==bank_hashes
            continue
        cap=min(4,row['maximum_tokens']) if args.stage=='smoke' else row['maximum_tokens']
        print('START',args.arm,row['index'],row['task'],len(row['input_ids']),flush=True)
        started=time.monotonic()
        for device in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            ids,first,stats,stopped=generate(model,tokenizer,row,args.arm,cap)
            if args.stage=='smoke':
                again,other,_,_=generate(model,tokenizer,row,args.arm,cap)
                assert ids==again
                torch.testing.assert_close(first,other,atol=0,rtol=0)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        result=dict(ids=ids,prediction=prediction,score=sample_score(prediction,row['answers'],row['match_type']),
            stopped=stopped,routing=stats,seconds=time.monotonic()-started,
            peak_gib_by_device={str(i):torch.cuda.max_memory_allocated(i)/2**30
                                for i in range(torch.cuda.device_count())})
        write_json(path,dict(status='complete',sample=row,result=result,protocol=spec,bank_sha256=bank_hashes,
            command=shlex.join(sys.argv),python=sys.executable,gpu=torch.cuda.get_device_name(0),
            device_map={str(k):str(v) for k,v in getattr(model,'hf_device_map',{'':0}).items()}))
        print('COMPLETE',args.arm,row['index'],result['score'],result['seconds'],flush=True)


if __name__=='__main__':
    main()
