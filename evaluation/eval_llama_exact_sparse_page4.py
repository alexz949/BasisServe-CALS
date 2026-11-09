#!/usr/bin/env python3
"""Matched exact-score Page4 RULER reference on the frozen Llama V96 model."""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from basisserve.core.rank_exchange_audit import atomic_json, file_hash, fingerprint
from evaluation import eval_llama_cal128_p1 as matched
from evaluation import eval_k_routing_ruler as runtime
from evaluation import llama_b16r16_k_offload_p1 as offload
from evaluation.deterministic_evaluation import configure_deterministic_evaluation, NUMERICAL_POLICY
from evaluation.k_routing_config import routing_config
from evaluation.ruler_v1 import sample_score

REVISION = '63a35906f354cc239339465db97fab136ccdb1de'
MODEL_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
CACHE = Path.home()/'.cache/huggingface/hub'
ASSETS = CACHE/'models--alexz949--BasisServe-CALS/snapshots'/REVISION
MODEL = CACHE/'models--meta-llama--Llama-3.1-8B-Instruct/snapshots'/MODEL_REVISION
CHECKPOINT = ASSETS/'checkpoints/attention_c1/llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix'
IDENTITY = ASSETS/'calibration/llama31-8b-instruct-128k/identity/B.json'
PROMPTS = ASSETS/'evaluation/llama31-8b-instruct-ruler128k-1100-seed42'
ARCHIVE = ROOT/'results/l31-v96-retrievalmix-ruler1100'
SMOKE_IDS = (0,800)
PAPER = ROOT/'BasisKVICLR2027 (32).pdf'
PAPER_TASKS = ('niah_single_1','niah_single_2','niah_multikey_1','niah_multikey_2',
    'niah_multiquery','niah_multivalue','qa_1','qa_2','vt','fwe')


def read(path):
    return json.loads(Path(path).read_text())


def settings():
    matched.PAGE_SIZE, matched.PINNED_PAGES, matched.OURS_BUDGET = 4,0,2112
    offload.PAGE_SIZE, offload.PINNED_PAGES, offload.BUDGET = 4,0,2112


def archive_rows():
    rows = [json.loads(line) for line in (ARCHIVE/'predictions/b16r16_page4_fisher.jsonl').read_text().splitlines()]
    assert len(rows)==1100
    return {row['sample']['index']:row for row in rows}


def source_files():
    return list(dict.fromkeys([*matched.SOURCES, 'evaluation/eval_llama_exact_sparse_page4.py',
        'basisserve/core/exact_page_routing.py', 'basisserve/core/c1_conditional_page_attention.py',
        'basisserve/core/rank_exchange_audit.py', 'tests/test_exact_page_routing.py',
        'evaluation/package_llama31_8b_c1_checkpoint.py']))


def recover_original_checkpoint_manifest(output, checkpoint, identity):
    # The publisher expanded the metadata. Recreate the original packager's exact
    # serialization and verify its historical digest, including every factor hash.
    fit_hash=file_hash(CHECKPOINT/'fit_results.json')
    assert fit_hash==checkpoint['fit_results_sha256']
    original=dict(status='complete',format='basisserve.llama31_8b_instruct.c1_v96_manual_package.v1',
        source_results_sha256=fit_hash,layers=checkpoint['layers'])
    payload=json.dumps(original,indent=1).encode()
    assert hashlib.sha256(payload).hexdigest()==identity['manifest_sha256']
    path=output/'provenance/original_checkpoint_manifest.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==payload
    else:
        path.write_bytes(payload)
    return path


def prepare(args):
    identity = read(IDENTITY)
    checkpoint = read(CHECKPOINT/'manifest.json')
    prompts = read(PROMPTS/'prompts.json')
    original = read(ARCHIVE/'protocols/b16r16_page4_fisher.json')
    assert file_hash(IDENTITY)==original['identity_sha256']==prompts['identity_sha256']
    assert file_hash(PROMPTS/'prompts.json')==original['prompts_sha256']
    assert file_hash(PROMPTS/'prompts.safetensors')==prompts['tokens_sha256']
    recovered=recover_original_checkpoint_manifest(args.output,checkpoint,identity)
    assert file_hash(MODEL/'config.json')==identity['model_config_sha256']
    assert file_hash(MODEL/'model.safetensors.index.json')==checkpoint['model']['safetensors_index_sha256']
    assert file_hash(MODEL/'tokenizer.json')==checkpoint['model']['tokenizer_sha256']
    assert checkpoint['model']['revision']==MODEL_REVISION
    assert (identity['hq'],identity['hkv'],identity['head_dim'])==(32,8,128)
    assert identity['layer_ranks']==[96]*32
    assert len(prompts['rows'])==1100 and Counter(r['task'] for r in prompts['rows'])==dict.fromkeys(matched.TASKS,100)
    tensors = load_file(str(PROMPTS/'prompts.safetensors'))
    archived = archive_rows()
    for row in prompts['rows']:
        ids = tensors[str(row['index'])]
        assert hashlib.sha256(ids.numpy().tobytes()).hexdigest()==row['input_sha256']
        assert ids.numel()==row['input_tokens'] and ids.numel()+row['maximum_tokens']<=131072
        assert archived[row['index']]['sample']==row
    versions = {name:importlib.metadata.version(name) for name in ('torch','transformers','triton')}
    assert all(versions[k].split('+')[0]==v for k,v in original['versions'].items()), versions
    files = [IDENTITY,CHECKPOINT/'manifest.json',CHECKPOINT/'fit_results.json',recovered,
             PROMPTS/'prompts.json',PROMPTS/'prompts.safetensors',
             ARCHIVE/'predictions/b16r16_page4_fisher.jsonl',ARCHIVE/'protocols/b16r16_page4_fisher.json',PAPER]
    for row in checkpoint['layers']:
        path = CHECKPOINT/row['file']
        assert file_hash(path)==row['sha256']
        files.append(path)
    for i in range(32):
        bank = CHECKPOINT/'router_page4_b16r16'
        record = read(bank/f'layer_{i:03d}.json')
        tensor = bank/f'layer_{i:03d}.safetensors'
        assert file_hash(tensor)==record['sha256']==archived[0]['bank_sha256'][str(i)]
        files.extend([tensor,bank/f'layer_{i:03d}.json'])
    index = read(MODEL/'model.safetensors.index.json')
    files += [MODEL/name for name in sorted(set(index['weight_map'].values()) | {'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json','generation_config.json'})]
    paper_tasks = list(PAPER_TASKS)
    configuration = dict(asset_revision=REVISION,model_revision=MODEL_REVISION,
        identity=str(IDENTITY),model=str(MODEL),checkpoint=str(CHECKPOINT),prompts=str(PROMPTS),
        checkpoint_manifest_equivalence=dict(original_sha256=identity['manifest_sha256'],
            published_sha256=file_hash(CHECKPOINT/'manifest.json'),
            recovered_original=str(recovered),
            proof='Exact original-manifest SHA256 recovered using original packager serialization, published fit-results hash, and all 32 factor-file hashes'),
        frozen_original_protocol=original, versions=versions, numerical_policy=NUMERICAL_POLICY,
        files=[dict(path=str(p),sha256=file_hash(p)) for p in files],
        sources={name:file_hash(ROOT/name) for name in source_files()},
        original_source_differences=[name for name,h in original['source_sha256'].items() if file_hash(ROOT/name)!=h],
        exact_score=dict(arithmetic='FP32 dot products of stored BF16 Q/K',chunk_size=args.chunk_size,
                         key_placement=args.key_placement,global_normalization=True,page_size=4,
                         history_budget=2048,recent=64,sink=0,maximum_unique_support=2112),
        sample_count=1100, tasks=list(matched.TASKS),paper_tasks=paper_tasks,
        paper_subset_status='recovered_from_Table_2_before_new_scores',
        paper_subset_source=dict(path=str(PAPER),sha256=file_hash(PAPER),table=2,
            page=8,excluded_task='niah_single_3',aggregation='unweighted task mean'),
        bootstrap=dict(seed=20260925,replicates=10000,unit='paired samples resampled independently within each task'),
        smoke_ids=list(SMOKE_IDS),generation='greedy; original EOS; original per-task caps; no prompt truncation')
    key = fingerprint(configuration)
    path = args.output/'configuration.json'
    if path.exists():
        assert read(path)['hash']==key, 'Existing output has a different configuration'
    atomic_json(path,dict(hash=key,configuration=configuration))
    print(json.dumps(dict(configuration_hash=key,samples=1100,shards=4,per_shard=275,paper_tasks=paper_tasks)),flush=True)


def load_configuration(args, verify=False):
    saved = read(args.output/'configuration.json')
    config,key = saved['configuration'],saved['hash']
    assert fingerprint(config)==key
    for name,digest in config['sources'].items():
        assert file_hash(ROOT/name)==digest, f'Code changed since prepare: {name}'
    for name,version in config['versions'].items():
        assert importlib.metadata.version(name)==version
    if verify:
        for entry in config['files']:
            assert file_hash(entry['path'])==entry['sha256'], f"Asset mismatch: {entry['path']}"
    return config,key


class ExactRoutingCache(offload.KOffloadRoutingCache):
    placement = 'cuda'

    def __init__(self, config):
        super().__init__(config)
        self.exact_key_placement = self.placement


def set_arm(model, arm, config):
    exact = arm=='exact_sparse_page4'
    runtime.RoutingCache = ExactRoutingCache if exact else offload.KOffloadRoutingCache
    ExactRoutingCache.placement = config['exact_score']['key_placement']
    for layer in model.model.layers:
        layer.self_attn._exact_page4 = exact
        layer.self_attn._exact_chunk_size = config['exact_score']['chunk_size']


def load_model(config):
    assert torch.cuda.device_count()==1, 'Use CUDA_VISIBLE_DEVICES for one complete model per worker'
    identity = read(IDENTITY)
    identity.update(model=config['model'],checkpoint=config['checkpoint'])
    manifest = read(CHECKPOINT/'manifest.json')
    args = SimpleNamespace(bank=CHECKPOINT/'router_page4_b16r16',identity=IDENTITY,allow_bank_sink_mismatch=False)
    bank,hashes = matched.bank_for(args,identity,manifest,'ours')
    model = runtime.load_evaluation_model(identity,routing_config(identity,rope='native',sequence_length=131072))
    runtime.install(model,CHECKPOINT,manifest,'ours',bank)
    offload.install(model)
    return model,hashes


def audit_result(saved,row,key,hashes,tokenizer,arm):
    assert saved['status']=='complete' and saved['configuration_hash']==key and saved['sample']==row
    assert saved['bank_sha256']==hashes and saved['arm']==arm
    result = saved['result']; ids = result['ids']; eos = set(saved['eos_ids'])
    assert eos=={128001,128008,128009}
    assert 0<len(ids)<=row['maximum_tokens'] and not any(t in eos for t in ids[:-1])
    assert result['stopped']==(ids[-1] in eos)
    assert result['stopped'] or len(ids)==row['maximum_tokens']
    assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==result['prediction']
    assert sample_score(result['prediction'],row['answers'],row['match_type'])==result['score']
    matched.verify_dispatch(result['kernel_calls'],'ours',32,len(ids))
    if len(ids)>1:
        assert len(result['routing'])==32
        for stat in result['routing']:
            assert stat['sink_tokens']==0 and stat['recent_tokens']==64 and stat['token_budget']==2112
            assert stat['selected_tokens_mean']<=2112
            if arm=='exact_sparse_page4':
                assert stat['score_method']=='exact_full_dimensional_qk_fp32' and stat['maximum_support']<=2112
                assert stat['decode_steps']==len(ids)-1
                steps=len(ids)-1
                assert stat['key_tokens_scanned']==steps*row['input_tokens']+steps*(steps+1)//2
    return result


def generate_one(model,tokenizer,row,tokens,arm,key,hashes,counts,path):
    atomic_json(path,dict(status='running',sample=row,arm=arm,configuration_hash=key))
    complete = False
    started = time.perf_counter()
    try:
        ids_tensor = tokens[str(row['index'])]
        assert hashlib.sha256(ids_tensor.numpy().tobytes()).hexdigest()==row['input_sha256']
        counts.clear()
        torch.cuda.reset_peak_memory_stats()
        ids,first,stats,stopped = runtime.generate(model,tokenizer,dict(row,input_ids=ids_tensor.tolist()),'ours',row['maximum_tokens'])
        prediction = tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        result = dict(ids=ids,prediction=prediction,stopped=stopped,
            score=sample_score(prediction,row['answers'],row['match_type']),routing=stats,kernel_calls=dict(counts),
            seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        eos = model.config.eos_token_id
        eos = sorted(set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id})
        saved = dict(status='complete',configuration_hash=key,sample=row,arm=arm,bank_sha256=hashes,
            result=result,eos_ids=eos,first_argmax=int(first.argmax()),
            prefill_logits_sha256=hashlib.sha256(first.numpy().tobytes()).hexdigest(),gpu=torch.cuda.get_device_name())
        audit_result(saved,row,key,hashes,tokenizer,arm)
        atomic_json(path,saved)
        complete = True
        print('COMPLETE',arm,row['index'],row['task'],result['score'],result['seconds'],flush=True)
        return saved
    finally:
        if not complete:
            atomic_json(path,dict(status='failed',configuration_hash=key,sample=row,arm=arm,
                seconds=time.perf_counter()-started,error=str(sys.exc_info()[1]),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def run(args,config,key):
    if args.stage=='evaluate':
        gate = read(args.output/'smoke.json')
        assert gate['status']=='passed' and gate['configuration_hash']==key
    model,hashes = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    rows = read(PROMPTS/'prompts.json')['rows']
    tokens = load_file(str(PROMPTS/'prompts.safetensors'))
    counts = matched.instrument_kernels()
    if args.stage=='smoke':
        selected = [rows[i] for i in SMOKE_IDS]
        arms = ['proxy_page4','exact_sparse_page4']
    else:
        selected = rows[args.shard::args.shards]
        arms = [args.arm]
    results = {}
    for arm in arms:
        set_arm(model,arm,config)
        results[arm] = []
        for row in selected:
            path = args.output/arm/args.stage/f"sample_{row['index']:04d}.json"
            if path.exists() and read(path)['status']=='complete':
                assert args.resume
                saved = read(path)
                audit_result(saved,row,key,hashes,tokenizer,arm)
            else:
                print('START',arm,row['index'],row['task'],row['input_tokens'],flush=True)
                saved = generate_one(model,tokenizer,row,tokens,arm,key,hashes,counts,path)
            results[arm].append(saved)
    if args.stage=='smoke':
        archived = archive_rows()
        matched_ids = [r['result']['ids']==archived[r['sample']['index']]['result']['ids'] for r in results['proxy_page4']]
        assert all(a['prefill_logits_sha256']==b['prefill_logits_sha256'] for a,b in zip(results['proxy_page4'],results['exact_sparse_page4'])), 'Prefill differs between exact and proxy'
        assert all(len(r['result']['ids'])>1 for arm in results.values() for r in arm)
        atomic_json(args.output/'smoke.json',dict(status='passed',configuration_hash=key,
            sample_ids=list(SMOKE_IDS),full_generation_caps=True,prefill_logits_bitwise_equal=True,
            proxy_matches_archived_per_sample=matched_ids,
            require_matched_proxy_rerun=not all(matched_ids)))


def bootstrap(differences,seed,replicates):
    rng = np.random.default_rng(seed)
    estimates = np.zeros(replicates)
    for values in differences:
        values = np.asarray(values,dtype=np.float64)
        estimates += values[rng.integers(0,len(values),size=(replicates,len(values)))].mean(1)/len(differences)
    return np.quantile(estimates,[.025,.975]).tolist()


def summarize(args,config,key):
    rows = read(PROMPTS/'prompts.json')['rows']; archived = archive_rows()
    gate = read(args.output/'smoke.json')
    assert gate['configuration_hash']==key and gate['status']=='passed'
    tokenizer = AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    hashes = archived[0]['bank_sha256']
    paired, tasks = [], {}
    for row in rows:
        path = args.output/'exact_sparse_page4/evaluate'/f"sample_{row['index']:04d}.json"
        assert path.is_file(), f'Missing sample: {path}'
        exact = audit_result(read(path),row,key,hashes,tokenizer,'exact_sparse_page4')
        if gate['require_matched_proxy_rerun']:
            proxy = audit_result(read(args.output/'proxy_page4/evaluate'/path.name),row,key,hashes,tokenizer,'proxy_page4')
        else:
            saved = archived[row['index']]
            assert saved['sample']==row and saved['bank_sha256']==hashes and saved['status']=='complete'
            proxy = saved['result']
            assert sample_score(proxy['prediction'],row['answers'],row['match_type'])==proxy['score']
        paired.append(dict(task=row['task'],sample_id=row['ordinal'],index=row['index'],prompt_hash=row['input_sha256'],
            exact=100*exact['score'],proxy=100*proxy['score'],difference=100*(exact['score']-proxy['score']),
            exact_seconds=exact['seconds'],peak_allocated_bytes=exact['peak_allocated_bytes']))
    for task in config['tasks']:
        selected = [r for r in paired if r['task']==task]
        assert len(selected)==100
        tasks[task] = {name:float(np.mean([r[name] for r in selected])) for name in ('exact','proxy','difference')}
    result = dict(status='complete',configuration_hash=key,samples=1100,tasks=tasks,
        proxy_source='new_matched_run' if gate['require_matched_proxy_rerun'] else 'archive_verified_by_smoke',
        paper_subset_status=config['paper_subset_status'],bootstrap=config['bootstrap'],
        cost=dict(exact_seconds=sum(r['exact_seconds'] for r in paired),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in paired)))
    for label,subset in [('all_11',config['tasks']),('paper_10',config['paper_tasks'])]:
        if subset:
            result[label] = dict(tasks=subset,**{name:float(np.mean([tasks[t][name] for t in subset])) for name in ('exact','proxy','difference')},
                paired_95_percent_interval=bootstrap([[r['difference'] for r in paired if r['task']==t] for t in subset],
                    config['bootstrap']['seed'],config['bootstrap']['replicates']))
    atomic_json(args.output/'summary.json',result)
    atomic_json(args.output/'paired_samples.json',paired)
    with (args.output/'paired_samples.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    lines=['# Llama-3.1-8B-Instruct exact-score Page4 RULER reference','',
        '1100 frozen prompts; V96; historical budget 2048 + recent64; no sink; independent greedy generation.',
        '', '| Task | Page-Fisher | Exact QK | Exact − proxy |','|---|---:|---:|---:|']
    for task,r in tasks.items():
        lines.append(f"| {task} | {r['proxy']:.4f} | {r['exact']:.4f} | {r['difference']:+.4f} |")
    for label in ('all_11','paper_10'):
        if label in result:
            r=result[label]
            lines += ['',f"{label}: proxy {r['proxy']:.4f}; exact {r['exact']:.4f}; difference {r['difference']:+.4f}; paired 95% interval {r['paired_95_percent_interval']}."]
    if 'paper_10' not in result:
        lines += ['', 'The paper 10-task subset has not been identified in the available records; no task was excluded based on the new scores.']
    lines += ['',f"Proxy source: {result['proxy_source']}. Exact-run cost: {result['cost']}.",
        'Exact refers to full-dimensional QK scoring, not optimal page selection or an accuracy upper bound. A confidence interval including zero does not establish equivalence.']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','evaluate','summarize'])
    p.add_argument('--output',type=Path,default=ROOT/'runs/l31-exact-page4')
    p.add_argument('--arm',choices=['exact_sparse_page4','proxy_page4'],default='exact_sparse_page4')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--chunk-size',type=int,default=16384)
    p.add_argument('--key-placement',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    assert 0<=args.shard<args.shards and args.chunk_size>0
    settings();configure_deterministic_evaluation();torch.set_num_threads(2)
    if args.stage=='prepare':
        prepare(args);return
    config,key=load_configuration(args,verify=args.stage in ('smoke','evaluate'))
    if args.stage=='summarize':summarize(args,config,key)
    else:
        with torch.inference_mode():run(args,config,key)


if __name__=='__main__':main()
