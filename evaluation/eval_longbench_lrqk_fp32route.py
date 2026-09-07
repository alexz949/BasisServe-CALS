"""FP16 C1 payload/model with FP32 LRQK state; reuse the completed FP16 full-K control."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basisserve.core import c1_lrqk as core
from basisserve.checkpoint import c1_lrqk_qwen3 as adapter
from evaluation import eval_longbench_lrqk_fp16 as driver
from evaluation.fit_qwen3_8b_residual_kl_bank import write_json,sha256


class FP32RoutingState(core.LRQKState):
    @torch.inference_mode()
    def __init__(self,q,k,config,layer=0):
        super().__init__(q.float(),k.float(),config,layer)

    @torch.inference_mode()
    def decode(self,q,k,v,scale):
        assert q.shape[2]==1 and k.shape[2]==v.shape[2]==self.length+1
        assert self.selected.max()<self.length and self.ak.shape[2]==self.length
        active_k=core.gather_heads(k,self.selected)
        active_ak=self.ak.gather(2,self.selected[...,None].expand(-1,-1,-1,self.config.rank))
        current_k=k[:,:,-1:].repeat_interleave(q.shape[1]//k.shape[1],dim=1)
        self.bq,self.bk,qcode,kcode=core.decode_factors(self.bq,active_ak,self.bk,
            active_k.float(),q.float(),current_k.float(),self.config.decode_iterations,self.config.tolerance)
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in (self.bq,self.bk,qcode,kcode))
        self.ak=torch.cat((self.ak,kcode),dim=2)
        self.length+=1
        self.steps+=1
        self.selected=core.select_tokens(qcode,self.ak,self.config)
        output=core.selected_attention(q,k,v,self.selected,scale)
        assert output.dtype==v.dtype and torch.isfinite(output).all()
        return output


def summarize(model,output):
    args=SimpleNamespace(model=model,num_shards=4,
        data_dir=ROOT/'results/datasets/longbench_c1_32k',c1_results=ROOT/'results/evaluation/longbench_c1_32k',
        dense_results=ROOT/'results/evaluation/longbench_dense_32k',
        c1_checkpoint=ROOT/'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6')
    rows,_,_,_,_,scorer=driver.inputs(args)
    tokenizer=driver.AutoTokenizer.from_pretrained(model,local_files_only=True)
    eos=json.loads((model/'config.json').read_text())['eos_token_id']
    eos=set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id}
    records={}; protocols={}; hashes={}
    for arm in ('full','k1152','k1280'):
        root=ROOT/'results/evaluation/longbench_lrqk_fp16' if arm=='full' else output
        records[arm]=[]; hashes[arm]={}
        for row in rows:
            path=root/arm/'evaluate'/f"sample_{row['index']:03d}.json"
            saved=json.loads(path.read_text()); r=saved['result']; ids=r['generated_token_ids']
            protocols.setdefault(arm,saved['protocol'])
            assert saved['status']=='complete' and saved['arm']==arm and saved['protocol']==protocols[arm]
            assert r['sample']==row and 0<len(ids)==r['generated_tokens']<=row['maximum_tokens']
            assert r['stopped_on_eos']==(ids[-1] in eos) and not any(i in eos for i in ids[:-1])
            assert r['stopped_on_eos'] or len(ids)==row['maximum_tokens']
            assert r['prediction']==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert r['score']==driver.score_prediction(scorer,row['task'],r['prediction'],row['answers'],row['all_classes'])
            records[arm].append(r); hashes[arm][str(row['index'])]=sha256(path)
        for shard in range(4):
            done=json.loads((root/arm/'evaluate'/f'shard_{shard}.json').read_text())
            assert done['status']=='complete' and done['protocol']==protocols[arm]
            assert done['indices']==list(range(shard,192,4))
    for arm in ('k1152','k1280'):
        comparable=dict(protocols[arm]); comparable['source']=dict(comparable['source'])
        precision=comparable['source'].pop('routing_precision')
        assert precision==dict(storage='float32',score_scan='float32',wrapper_sha256=sha256(Path(__file__)))
        assert comparable==protocols['full']
        assert all(a['generated_token_ids'][0]==b['generated_token_ids'][0]
                   for a,b in zip(records['full'],records[arm],strict=True))
    tasks={t:{a:100*sum(r['score'] for r in rs if r['sample']['task']==t)/32
              for a,rs in records.items()} for t in driver.TASKS}
    means={a:sum(t[a] for t in tasks.values())/len(tasks) for a in records}
    budgets={}
    for arm in ('k1152','k1280'):
        u=torch.tensor([v for r in records[arm] for s in r['routing']
                        for group in s['physical_union_per_kv_group'] for v in group],dtype=torch.float64)
        budgets[arm]=dict(scope='Final decode step per prompt, all layers/groups; not all-step mean',
            mean=u.mean().item(),minimum=u.min().item(),maximum=u.max().item(),
            p50=u.quantile(.5).item(),p90=u.quantile(.9).item(),p95=u.quantile(.95).item(),
            fraction_above_2048=(u>2048).double().mean().item())
    write_json(output/'result.json',dict(status='complete',protocols=protocols,tasks=tasks,means=means,
        physical_budget=budgets,records=records,input_record_sha256=hashes))
    write_json(output/'audit.json',dict(status='complete',verified=576,result_sha256=sha256(output/'result.json')))
    lines=['# FP16 C1 with FP32 LRQK routing','',
        'Same192 LongBench prompts. FP16 model and exact K/C1-V96 cache; FP32 routing factors/codes/score scan.',
        'C1-V96 memory-efficient SDPA prefill. Existing completed FP16 full-K control reused with per-record hashes.',
        'No clipping or optimizer change. Neither LRQK budget is a hard shared B2048 cap.','',
        '| Task | Full K | k1152 | k1280 |','|---|---:|---:|---:|']
    for t,v in [*tasks.items(),('Mean',means)]:
        lines.append('| '+t+' | '+' | '.join(f'{x:.4f}' for x in v.values())+' |')
    lines+=['','## Physical union','','```json',json.dumps(budgets,indent=2),'```','']
    (output/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(means),flush=True)


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--sample',type=int)
    known,remaining=p.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    output=ROOT/'results/evaluation/longbench_lrqk_fp32route'
    if '--output-dir' not in sys.argv: sys.argv+=['--output-dir',str(output)]
    else: output=Path(sys.argv[sys.argv.index('--output-dir')+1])
    if sys.argv[sys.argv.index('--stage')+1]=='summarize':
        assert known.sample is None
        summarize(Path(sys.argv[sys.argv.index('--model')+1]),output)
        return
    original=driver.inputs
    def inputs(a):
        values=list(original(a))
        values[4]['routing_precision']=dict(storage='float32',score_scan='float32',wrapper_sha256=sha256(Path(__file__)))
        if known.sample is not None:
            values[0]=[r for r in values[0] if r['index']==known.sample]
            assert len(values[0])==1 and a.shard_index==0
        return tuple(values)
    driver.inputs=inputs
    adapter.LRQKState=FP32RoutingState
    driver.main()


if __name__=='__main__':
    main()
