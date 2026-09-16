"""Paired RULER evaluation in the measured native K-offload runtime."""
import argparse
import hashlib
import importlib.util
import importlib.machinery
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from evaluation.ruler_v1 import sample_score, ruler_prompt, TASK_BY_NAME
from benchmarks.system.numa_memory import bind_host_allocations,slurm_gpu_numa
from benchmarks.system.two_stage_router import compile_fine
from benchmarks.system.local_kernel_candidates import compile_slots
from benchmarks.system.two_stage_runtime import install

ROOT=Path('results/evaluation/two_stage_ruler64k_330')
DATA=ROOT/'data'
COUNT=330
PREVIOUS=Path('/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k_densev/eval64k')


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize():
    rows=[read(p) for p in sorted((ROOT/'samples').glob('sample_*.json'))]
    assert len(rows)==COUNT and [r['index'] for r in rows]==list(range(COUNT))
    tasks=list(dict.fromkeys(r['task'] for r in rows))
    for r in rows:
        assert r['full']['first_token']==r['two']['first_token']
        for arm in ['full','two']:
            assert r[arm]['score']==sample_score(r[arm]['prediction'],r['answers'],r['match_type'])
            assert r[arm]['finite_logits']
    means={arm:100*statistics.mean(r[arm]['score'] for r in rows) for arm in ['full','two']}
    per_task={task:{arm:100*statistics.mean(r[arm]['score'] for r in rows if r['task']==task) for arm in ['full','two']} for task in tasks}
    changed=[dict(index=r['index'],task=r['task'],full=r['full']['score'],two=r['two']['score']) for r in rows if r['full']['score']!=r['two']['score']]
    result=dict(status='complete',samples=COUNT,means=means,tasks=per_task,changed_scores=changed,
        same_generated_sequences=sum(r['full']['ids']==r['two']['ids'] for r in rows),protocol=read(ROOT/'protocol.json'))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'# Two-stage RULER 64K, {COUNT} prompts','',
        'Model: Llama-3.1-8B base; four independent L40S workers, one GPU per prompt. BF16; full FlashAttention prefill, original Dense V and Wo, CPU K offload, vector missing-K fetch and Triton slot attention. Greedy generation, native EOS, original per-task caps. Both arms use identical token IDs from the frozen official dataset.', '',
        'Full arm: current 8-warp B16R16 full scan. Two arm: exact post-RoPE Page32 min/max, 512 total candidates per KV head including sink/recent pages, B16R16 candidate scan and candidate normalization. Both use sink32 + recent64 within hard2048. The two-stage branch never scans all routing codes during decode.', '',
        '| Task | Full B16R16 | Two-stage | Difference (pp) |','|---|---|---|---|']
    for task,v in per_task.items():lines.append(f'| {task} | {v["full"]:.2f} | {v["two"]:.2f} | {v["two"]-v["full"]:+.2f} |')
    lines += [f'| **All {COUNT}** | **{means["full"]:.2f}** | **{means["two"]:.2f}** | **{means["two"]-means["full"]:+.2f}** |','',
        f'Changed scores: {len(changed)}; identical generated sequences: {result["same_generated_sequences"]}/{COUNT}. All paired prefill first tokens match; all evaluated logits finite. Per-request min/max and slot states are reset. Runtime numerical behavior differs from the older Transformers evaluations, so their scores are not used as the paired baseline.', '',
        'Environment: basis. Commands:','', '```bash',
        'python -m evaluation.eval_two_stage_ruler --smoke',
        'python -m evaluation.eval_two_stage_ruler --shard 0 --shards 4',
        '# Other workers use --shard 1, 2, 3.',
        'python -m evaluation.eval_two_stage_ruler --summarize','```','',
        'All raw per-sample predictions and source hashes are retained in samples/. No refit, production default change, commit, or push.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(result,indent=2)[:4500],flush=True)


@torch.inference_mode()
def main():
    global ROOT, DATA, COUNT
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--smoke',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    ROOT=a.output;DATA=a.data
    data=read(DATA/'manifest.json');assert data['status']=='complete'
    names=data['protocol']['tasks'];per_task=data['protocol']['samples_per_task'];COUNT=len(names)*per_task
    assert len(names)==11 and data['protocol']['sequence_length']==65536
    ROOT.mkdir(parents=True,exist_ok=True)
    if a.summarize:summarize();return
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    hardware=read(Path('results/system_benchmarks/l40s/hardware.json'));bind_host_allocations(slurm_gpu_numa(hardware)['numa_node'])
    previous=read(PREVIOUS/'summary.json');identity=read(Path('/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k_densev/manifests/v128.json'))
    model=identity['model'];assert previous['protocol']['sequence_length']==65536
    bank=Path('/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k_densev/ours_b16r16')
    assert all(digest(bank/f'layer_{i:03d}.safetensors')==previous['bank_sha256'][str(i)] for i in range(32))
    full,fine=compile_fine(ROOT/'source');slots=compile_slots(ROOT/'source/slots',True);install(full,fine,slots)
    upstream=Path('/home/zhangal/BasisServe-CALS-runs/shadow_native/ShadowKV')
    sys.path[:0]=[str(upstream),str(upstream.parent/'deps')]
    spec=importlib.machinery.ModuleSpec('models',loader=None,is_package=True)
    package=importlib.util.module_from_spec(spec);package.__path__=[str(upstream/'models')];sys.modules['models']=package
    torch.ops.load_library(str(Path(importlib.util.find_spec('vllm').origin).parent/'_C.abi3.so'))
    from models.llama import Llama
    from benchmarks.system.native_basis_cache import basis_llama_class
    llm=basis_llama_class(Llama,16)(model_name=model,device='cuda:0',batch_size=1,max_length=65536,
        attn_mode='basis16',sparse_budget=2048,rank=160,chunk_size=8,minference=False)
    llm.kv_cache.validate=a.smoke
    config=read(Path(model)/'config.json');eos=config['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])|{llm.tokenizer.eos_token_id}
    protocol=dict(model=model,sequence_length=65536,samples=COUNT,samples_per_task=per_task,data_sha256=digest(DATA/'manifest.json'),budget=2048,sink_tokens=32,recent_tokens=64,
        candidates=512,base_rank=16,residual_rank=16,dense_v=True,wo_compressed=False,greedy=True,environment='basis',
        previous_protocol=previous['protocol'],bank_sha256=previous['bank_sha256'],
        code_sha256={str(p):digest(p) for p in [Path(__file__),Path('benchmarks/system/two_stage_runtime.py'),Path('benchmarks/system/two_stage_router.py'),Path('benchmarks/system/register_router_body.cuh')]})
    if a.smoke:(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    else:assert read(ROOT/'protocol.json')==protocol
    indices=[0,3*per_task] if a.smoke else list(range(a.shard,COUNT,a.shards))
    sources={}
    for name in names:
        path=DATA/name/'validation.jsonl';assert digest(path)==data['artifacts'][name]['sha256']
        sources[name]=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(sources[name])==per_task
    folder=ROOT/('smoke' if a.smoke else 'samples');folder.mkdir(parents=True,exist_ok=True)
    for index in indices:
        dest=folder/f'sample_{index:03d}.json'
        if dest.exists():
            assert read(dest)['status']=='complete';continue
        name=names[index//per_task];ordinal=index%per_task;source=sources[name][ordinal];task=TASK_BY_NAME[name]
        source_path=DATA/name/'validation.jsonl'
        sample=dict(index=index,task=name,ordinal=ordinal,answers=source['outputs'],match_type=task.match_type,
            maximum_tokens=task.tokens_to_generate,input_ids=llm.tokenizer(ruler_prompt(source),add_special_tokens=True)['input_ids'])
        assert sample['index']==index and len(sample['input_ids'])+sample['maximum_tokens']<=65536
        ids=torch.tensor([sample['input_ids']],device='cuda',dtype=torch.long)
        record={k:v for k,v in sample.items() if k!='input_ids'}
        record.update(source_sha256=digest(source_path),input_length=ids.shape[-1])
        for arm in ['full','two']:
            llm.kv_cache.mode=arm
            torch.cuda.synchronize();start=time.perf_counter();logits=llm.batch_prefill(ids)
            torch.cuda.synchronize();prefill=time.perf_counter()-start
            assert bool(torch.isfinite(logits).all())
            generated=[int(logits[0,-1].argmax())];first=generated[0];checks=[]
            start=time.perf_counter()
            while len(generated)<sample['maximum_tokens'] and generated[-1] not in eos:
                token=torch.tensor([[generated[-1]]],device='cuda',dtype=torch.long)
                logits=llm.inference(token,llm.get_ctx(token));checks.append(torch.isfinite(logits).all())
                generated.append(int(logits[0,-1].argmax()))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            assert not checks or bool(torch.stack(checks).all())
            prediction=llm.tokenizer.decode(generated,skip_special_tokens=True)
            record[arm]=dict(ids=generated,first_token=first,prediction=prediction,
                score=sample_score(prediction,sample['answers'],sample['match_type']),stopped=generated[-1] in eos,
                finite_logits=True,prefill_seconds=prefill,decode_seconds=elapsed)
            print('PROGRESS',index,sample['task'],arm,'score',record[arm]['score'],'tokens',len(generated),'seconds',round(prefill+elapsed,2),flush=True)
        assert record['full']['first_token']==record['two']['first_token']
        record['status']='complete';dest.write_text(json.dumps(record,indent=2)+'\n')
    print('COMPLETE',a.shard,'smoke',a.smoke,flush=True)

if __name__=='__main__':main()
