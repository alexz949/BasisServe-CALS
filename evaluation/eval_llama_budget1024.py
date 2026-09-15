"""1024-budget LRQK/Base+Residual comparison against the completed Dense V baseline."""
import argparse
from functools import partial
from pathlib import Path
import shlex
import sys
import time
import torch
from transformers import AutoTokenizer
from evaluation import eval_k_routing_ruler as common
from evaluation import llama_sink_recent_routing as support
from evaluation.official_lrqk_state import OfficialLRQKState


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['smoke','audit','evaluate','summarize'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['lrqk','ours'],default='ours')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--base-rank',type=int,default=16)
    p.add_argument('--arms',nargs='+',choices=['lrqk','ours'],default=['lrqk','ours'])
    args=p.parse_args();r=args.root
    assert args.base_rank in (4,16) and len(set(args.arms))==len(args.arms)
    experiment=r if args.base_rank==16 else r/f'b{args.base_rank}r16'
    out=experiment/'eval128k_b1024'
    baseline=common.read_json(r/'eval128k_fa2/summary.json')
    assert baseline['status']=='complete' and baseline['verified_predictions']==352
    a=argparse.Namespace(identity=r/'manifests/v128.json',data=r/'ruler128k',bank=experiment/f'ours_b{args.base_rank}r16',
        sequence_length=131072,rope='native',dense_v=True,official_lrqk=True,chat_template=True,
        native_audit=None,full_smoke=None,wo_bank=None,arm='ours',stage='summarize')
    identity=common.read_json(a.identity)
    tokenizer=AutoTokenizer.from_pretrained(identity['model'],local_files_only=True)
    identity,manifest,rows,bank,hashes,spec=common.inputs(a,tokenizer)
    spec['ours']=f'Base{args.base_rank}/Residual16 Page32 GQA max; sink32 + recent64 inside hard B1024'
    spec['evaluated_arms']=args.arms
    for layer,payload in bank.items():
        assert set(payload)=={f'base_{name}_b{args.base_rank}' for name in ('left','right','bias')} | {f'residual_{name}_b{args.base_rank}_r16' for name in ('encoder','query')}
        assert payload[f'base_left_b{args.base_rank}'].shape[-1]==args.base_rank
        audit=common.read_json(a.bank/f'layer_{layer:03d}.json')
        assert audit['protocol']['base_rank']==args.base_rank and audit['protocol']['residual_rank']==16
    spec['lrqk']['topk_per_query_head']=1024
    spec['baseline_sha256']=common.sha256(r/'eval128k_fa2/summary.json')
    spec['source_sha256']['evaluation/eval_llama_budget1024.py']=common.sha256(Path(__file__))
    if args.base_rank==4:
        gate=common.read_json(experiment/'manifests/fit_audit.json')
        assert gate['status']=='complete' and gate['bank_sha256']==hashes
    dense={i:common.read_json(r/'eval128k_fa2/full/evaluate'/f'sample_{i:03d}.json') for i in range(88)}
    def verify(d,row):
        assert d['status']=='complete' and d['protocol']==spec and d['sample']==row
        assert row==dense[row['index']]['sample']
        ids=d['result']['ids'];assert ids and ids[0]==dense[row['index']]['result']['ids'][0]
        assert len(ids)<=row['maximum_tokens']
        text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert text==d['result']['prediction']
        assert common.sample_score(text,row['answers'],row['match_type'])==d['result']['score']
    if args.stage in ['audit','summarize']:
        stage='smoke' if args.stage=='audit' else 'evaluate'
        selected=[rows[0],rows[56]] if stage=='smoke' else rows
        results={}
        for arm in args.arms:
            results[arm]=[]
            for row in selected:
                d=common.read_json(out/arm/stage/f"sample_{row['index']:03d}.json")
                verify(d,row);assert d['bank_sha256']==(hashes if arm=='ours' else {})
                results[arm].append(d['result'])
        comparison={}
        if stage=='evaluate' and args.base_rank==4:
            reference=common.read_json(r/'eval128k_b1024/summary.json')
            assert reference['status']=='complete' and reference['verified_predictions']==176
            assert reference['protocol']['data_sha256']==spec['data_sha256']
            assert reference['protocol']['baseline_sha256']==spec['baseline_sha256']
            paired=[]
            for row,result in zip(rows,results['ours']):
                old=common.read_json(r/'eval128k_b1024/ours/evaluate'/f"sample_{row['index']:03d}.json")
                assert old['status']=='complete' and old['sample']==row
                paired.append(dict(index=row['index'],task=row['task'],b4r16=result['score'],b16r16=old['result']['score']))
            comparison=dict(b16r16_mean=reference['means']['ours'],reference_sha256=common.sha256(r/'eval128k_b1024/summary.json'),paired=paired)
        common.write_json(out/('smoke_audit.json' if stage=='smoke' else 'summary.json'),
            dict(status='complete',protocol=spec,verified_predictions=len(args.arms)*len(selected),
                 means={arm:100*sum(x['score'] for x in ds)/len(ds) for arm,ds in results.items()},results=results,comparison=comparison))
        print('VERIFIED',stage,flush=True);return
    if args.stage=='evaluate':
        gate=common.read_json(out/'smoke_audit.json');assert gate['status']=='complete' and gate['protocol']==spec
    assert args.arm in args.arms
    common.configure();torch.manual_seed(0)
    support.page_support=partial(support.page_support,budget=1024)
    common.LRQKConfig=partial(common.LRQKConfig,topk=1024)
    model=common.load_evaluation_model(identity,common.routing_config(identity,rope='native',sequence_length=131072))
    common.install(model,Path(identity['checkpoint']),manifest,args.arm,bank,dense_v=True)
    if args.arm=='lrqk':
        common.lrqk_adapter.LRQKState=OfficialLRQKState
        for _,m in common.c1_attention_layers(model):m._lrqk_official=True
    selected=[rows[0],rows[56]] if args.stage=='smoke' else rows[args.shard_index::4]
    for row in selected:
        path=out/args.arm/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():verify(common.read_json(path),row);continue
        cap=4 if args.stage=='smoke' else row['maximum_tokens']
        print('START',args.arm,row['index'],flush=True);started=time.monotonic()
        ids,first,stats,stopped=common.generate(model,tokenizer,row,args.arm,cap)
        if args.stage=='smoke':
            again,other,_,_=common.generate(model,tokenizer,row,args.arm,cap)
            assert ids==again;torch.testing.assert_close(first,other,atol=0,rtol=0)
        if args.arm=='ours':
            for stat in stats:
                stat['token_budget']=1024;assert stat['selected_tokens_mean']<=1024
        else:
            assert all(stat['topk']==1024 and stat['selected_per_query_head']==1088 for stat in stats)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        d=dict(status='complete',sample=row,protocol=spec,bank_sha256=hashes if args.arm=='ours' else {},
            command=shlex.join(sys.argv),python=sys.executable,result=dict(ids=ids,prediction=prediction,
            score=common.sample_score(prediction,row['answers'],row['match_type']),stopped=stopped,routing=stats,seconds=time.monotonic()-started))
        verify(d,row);common.write_json(path,d);print('COMPLETE',args.arm,row['index'],d['result']['score'],flush=True)


if __name__=='__main__':main()
