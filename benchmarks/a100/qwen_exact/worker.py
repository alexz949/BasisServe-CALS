import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer
from evaluation import eval_llama_exact_sparse_page4 as e
from evaluation import eval_k_routing_ruler as runtime
from evaluation import llama_b16r16_k_offload_p1 as offload
from evaluation import eval_llama_cal128_p1 as matched
from evaluation.k_routing_config import routing_config
from evaluation.ruler_v1 import sample_score
ROOT=Path('/mnt/basiskv/qwen-exact')
ARCHIVE=Path('/home/sesterce/BasisServe-CALS/results/q3-8b-post-v96-retrievalmix-ruler128k')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):e.atomic_json(p,r)

def prepare():
    a=read(ROOT/'assets.json');identity=read(a['identity']);manifest=read(Path(a['checkpoint'])/'manifest.json')
    assert sha(Path(a['checkpoint'])/'manifest.json')==identity['manifest_sha256']
    assert sha(Path(a['model'])/'config.json')==identity['model_config_sha256']
    for entry in manifest['layers']:assert sha(Path(a['checkpoint'])/entry['file'])==entry['sha256']
    data=read(Path(a['data'])/'manifest.json');protocol=read(ARCHIVE/'protocols/b16r16_page4_fisher.json')
    assert sha(Path(a['data'])/'manifest.json')==protocol['data_sha256']
    tokenizer=AutoTokenizer.from_pretrained(a['model'],local_files_only=True)
    assert sha(Path(a['model'])/'tokenizer_config.json')==data['tokenizer_config_sha256']
    archived=[json.loads(s) for s in (ARCHIVE/'predictions/b16r16_page4_fisher.jsonl').read_text().splitlines()]
    original={r['sample']['index']:r['sample'] for r in archived};assert len(original)==1100
    rows=[];tokens={}
    for task in matched.TASKS:
        path=Path(a['data'])/task/'validation.jsonl';assert sha(path)==data['artifacts'][task]['sha256']
        sources=[json.loads(s) for s in path.read_text().splitlines()];assert len(sources)==100
        for ordinal,source in enumerate(sources):
            idx=len(rows);row=original[idx]
            assert row['task']==task and row['ordinal']==ordinal and row['answers']==source['outputs']
            text=tokenizer.apply_chat_template([dict(role='user',content=source['input'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)+str(source.get('answer_prefix',''))
            ids=torch.tensor(tokenizer(text,add_special_tokens=False)['input_ids'],dtype=torch.int32)
            assert ids.numel()==row['input_tokens'] and hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest()==row['input_sha256']
            assert ids.numel()+row['maximum_tokens']<=131072
            tokens[str(idx)]=ids;rows.append(row)
        print('PREPARED',task,len(rows),flush=True)
    save_file(tokens,str(ROOT/'prompts.safetensors'));save(ROOT/'prompts.json',rows)
    files=[Path(a['identity']),Path(a['checkpoint'])/'manifest.json',Path(a['data'])/'manifest.json',ROOT/'prompts.safetensors',ROOT/'prompts.json',Path(__file__)]
    files += [Path(a['checkpoint'])/x['file'] for x in manifest['layers']]
    files += list(Path(a['model']).glob('*.json'))+list(Path(a['model']).glob('*.safetensors'))
    files += [e.ROOT/n for n in set(matched.SOURCES)|{'basisserve/core/exact_page_routing.py','evaluation/eval_llama_exact_sparse_page4.py'}]
    config=dict(prefill='torch FlashAttention with zero-padded V96, matching Qwen paper protocol',assets=a,rope=protocol['rope'],rope_parameters=protocol['runtime_config']['rope_parameters'],page_size=4,historical=2048,recent=64,sink=0,layer_count=36,chat_template_kwargs={'enable_thinking':False},files={str(p):sha(p) for p in files})
    save(ROOT/'configuration.json',dict(configuration=config,hash=e.fingerprint(config)))
    print('PREPARE COMPLETE: all 1100 input hashes match archived BasisKV',flush=True)

def audit(saved,row,key,tokenizer):
    assert saved['status']=='complete' and saved['sample']==row and saved['configuration_hash']==key
    r=saved['result'];ids=r['ids'];eos={151645}
    assert saved['eos_ids']==sorted(eos) and saved['first_argmax']==ids[0]
    assert 0<len(ids)<=row['maximum_tokens'] and not any(t in eos for t in ids[:-1])
    assert r['stopped']==(ids[-1] in eos) and (r['stopped'] or len(ids)==row['maximum_tokens'])
    assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['prediction']
    assert sample_score(r['prediction'],row['answers'],row['match_type'])==r['score']
    matched.verify_dispatch(r['kernel_calls'],'ours',36,len(ids))
    if len(ids)>1:
        assert len(r['routing'])==36
        for s in r['routing']:
            assert s['sink_tokens']==0 and s['recent_tokens']==64 and s['token_budget']==2112
            assert s['maximum_support']<=2112 and s['score_method']=='exact_full_dimensional_qk_fp32'
            steps=len(ids)-1
            assert s['decode_steps']==steps and s['key_tokens_scanned']==steps*row['input_tokens']+steps*(steps+1)//2

def run(args):
    assert torch.cuda.device_count()==1
    e.settings();e.configure_deterministic_evaluation();torch.set_num_threads(2)
    c=read(ROOT/'configuration.json');key=c['hash'];cfg=c['configuration'];assert e.fingerprint(cfg)==key
    for p,digest in cfg['files'].items():assert sha(p)==digest,p
    if args.stage=='evaluate':assert read(ROOT/'smoke.json')['status']=='passed' and read(ROOT/'smoke.json')['configuration_hash']==key
    a=cfg['assets'];identity=read(a['identity']);identity.update(model=a['model'],checkpoint=a['checkpoint'])
    config=routing_config(identity,rope=cfg['rope'],sequence_length=131072)
    assert config.rope_parameters==cfg['rope_parameters'] and config.num_hidden_layers==36
    tokenizer=AutoTokenizer.from_pretrained(a['model'],local_files_only=True)
    rows=read(ROOT/'prompts.json');tokens=load_file(str(ROOT/'prompts.safetensors'))
    model=runtime.load_evaluation_model(identity,config)
    runtime.install(model,Path(a['checkpoint']),read(Path(a['checkpoint'])/'manifest.json'),'full',{})
    counts=matched.instrument_kernels();reference={}
    original_flash=runtime.flash_prefill_attention
    def tracked_flash(*args,**kwargs):
        counts['prefill']+=1
        return original_flash(*args,**kwargs)
    runtime.compressed_v_prefill_attention=tracked_flash
    selected=[0,800] if args.stage=='smoke' else list(range(args.worker,1100,8))
    if args.stage=='smoke':
        for idx in selected:
            _,first,_,_=runtime.generate(model,tokenizer,dict(rows[idx],input_ids=tokens[str(idx)].tolist()),'full',1)
            reference[idx]=hashlib.sha256(first.numpy().tobytes()).hexdigest();print('FULL PREFILL',idx,flush=True)
    offload.install(model);runtime.RoutingCache=e.ExactRoutingCache
    for layer in model.model.layers:
        layer.self_attn._routing_arm='ours';layer.self_attn._exact_page4=True;layer.self_attn._exact_chunk_size=4096
    for idx in selected:
        row=rows[idx];path=ROOT/args.stage/f'sample_{idx:04d}.json'
        if path.exists() and read(path)['status']=='complete':audit(read(path),row,key,tokenizer);continue
        save(path,dict(status='running',sample=row,configuration_hash=key));print('START',idx,row['task'],flush=True)
        counts.clear();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        ids,first,stats,stopped=runtime.generate(model,tokenizer,dict(row,input_ids=tokens[str(idx)].tolist()),'ours',row['maximum_tokens'])
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        saved=dict(status='complete',sample=row,configuration_hash=key,eos_ids=[151645],first_argmax=int(first.argmax()),prefill_logits_sha256=hashlib.sha256(first.numpy().tobytes()).hexdigest(),gpu=torch.cuda.get_device_name(),result=dict(ids=ids,prediction=prediction,stopped=stopped,score=sample_score(prediction,row['answers'],row['match_type']),routing=stats,kernel_calls=dict(counts),seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        audit(saved,row,key,tokenizer)
        if args.stage=='smoke':assert saved['prefill_logits_sha256']==reference[idx], 'Full-K and exact prefill mismatch'
        save(path,saved);print('DONE',idx,saved['result']['score'],saved['result']['seconds'],flush=True)
    if args.stage=='smoke':save(ROOT/'smoke.json',dict(status='passed',configuration_hash=key,samples=selected,prefill_bitwise_match=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','smoke','evaluate']);parser.add_argument('--worker',type=int,default=0);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:
        with torch.inference_mode():run(args)
