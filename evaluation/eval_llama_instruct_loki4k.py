"""Add rank-32 Loki to the frozen Instruct Dense-V RULER comparison."""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from types import MethodType
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from evaluation import eval_k_routing_ruler as common
from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.v96kl_common import configure, read_json, write_json, sha256
from basisserve.core.c1_loki_attention import c1_loki_pca_topk_attention
from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar


@torch.inference_mode()
def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    b,n,_=hidden_states.shape
    q=self.q_norm(self.q_proj(hidden_states).view(b,n,32,128)).transpose(1,2)
    k=self.k_norm(self.k_proj(hidden_states).view(b,n,8,128)).transpose(1,2)
    v=self.v_proj(hidden_states).view(b,n,8,128).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
    previous=past_key_values.get_seq_length(self.layer_idx)
    assert previous==0 or n==1
    codes=build_routing_sidecar(k,self._loki_basis)
    past_key_values.sidecars[self.layer_idx]=(torch.cat((past_key_values.sidecars[self.layer_idx],codes),2)
        if previous else codes)
    k,v=past_key_values.update(k,v,self.layer_idx)
    if previous==0:
        out=common.compressed_v_prefill_attention(q,k,v,scale=self.scaling)
    else:
        assert attention_mask is None or (attention_mask.dtype==torch.bool and bool(attention_mask.all()))
        result=c1_loki_pca_topk_attention(q,k,v,self._loki_basis,self._loki_basis,
            top_k=2048,scale=self.scaling,query_block_size=1,
            routing_sidecar=past_key_values.sidecars[self.layer_idx],collect_statistics=False)
        out=result.output
        past_key_values.statistics[self.layer_idx]=dict(topk_per_query_head=2048,rank=32)
    return self.o_proj(out.transpose(1,2).reshape(b,n,-1)),None


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['smoke','evaluate','summarize'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sequence-length',type=int,choices=[65536,131072],default=65536)
    args=p.parse_args();configure()
    r=args.root; evaluation=r/{65536:'eval64k',131072:'eval128k_fa2'}[args.sequence_length]
    out=r/f'eval{args.sequence_length//1024}k_loki128x4k'; basis=r/'loki128x4k'
    identity=read_json(r/'manifests/v128.json')
    baseline=read_json(evaluation/'summary.json')
    assert baseline['status']=='complete' and baseline['verified_predictions']==352
    assert baseline['protocol']['sequence_length']==args.sequence_length
    bm=read_json(basis/'manifest.json')
    assert bm['status']=='complete' and bm['model_config_sha256']==identity['model_config_sha256']
    for rec in bm['layers']:assert sha256(basis/rec['file'])==rec['sha256']
    spec=dict(baseline_sha256=sha256(evaluation/'summary.json'),pca_sha256=sha256(basis/'manifest.json'),
        sequence_length=args.sequence_length,calibration_sequence_length=4096,calibration_windows=128,prefill_mlp_chunk_tokens=1024,
        prefill='full causal FlashAttention-2',
        rank=32,topk_per_query_head=2048,extra_recent=0,extra_sink=0,physical_gqa_union='uncapped',
        source_sha256={name:sha256(Path(name)) for name in ['evaluation/eval_llama_instruct_loki4k.py',
            'evaluation/eval_k_routing_ruler.py','evaluation/chunked_prefill_mlp.py','basisserve/core/c1_loki_attention.py',
            'basisserve/kernels/indexed_sparse_decode_attention.py']})
    rows=[read_json(evaluation/'full/evaluate'/f'sample_{i:03d}.json')['sample'] for i in range(88)]
    tokenizer=AutoTokenizer.from_pretrained(identity['model'],local_files_only=True)
    if args.stage=='summarize':
        results=[]
        for row in rows:
            d=read_json(out/'evaluate'/f"sample_{row['index']:03d}.json")
            assert d['status']=='complete' and d['protocol']==spec and d['sample']==row
            result=d['result'];ids=result['ids']
            assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==result['prediction']
            assert common.sample_score(result['prediction'],row['answers'],row['match_type'])==result['score']
            results.append(result)
        tasks={t:100*sum(d['score'] for d,row in zip(results,rows) if row['task']==t)/8 for t in baseline['tasks']}
        summary=dict(status='complete',verified_predictions=88,protocol=spec,mean=100*sum(d['score'] for d in results)/88,tasks=tasks)
        write_json(out/'summary.json',summary);print(summary,flush=True);return
    if args.stage=='evaluate':
        for i in [0,56]:
            d=read_json(out/'smoke'/f'sample_{i:03d}.json')
            assert d['status']=='complete' and d['protocol']==spec
    assert 0<=args.shard_index<2
    assert torch.cuda.device_count()==2
    model=common.load_evaluation_model(identity,AutoConfig.from_pretrained(identity['model'],local_files_only=True))
    common.install(model,Path(identity['checkpoint']),read_json(Path(identity['checkpoint'])/'manifest.json'),'full',{},dense_v=True)
    for layer,module in common.c1_attention_layers(model):
        module._loki_basis=load_file(str(basis/f'layer_{layer:03d}.safetensors'))['projector'].to(module.q_proj.weight.device)
        module.forward=MethodType(forward,module)
    selected=[rows[0],rows[56]] if args.stage=='smoke' else rows[args.shard_index::2]
    for row in selected:
        path=out/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            d=read_json(path);assert d['status']=='complete' and d['protocol']==spec;continue
        cap=4 if args.stage=='smoke' else row['maximum_tokens']
        print('START',row['index'],row['task'],flush=True);start=time.monotonic()
        ids,first,stats,stopped=common.generate(model,tokenizer,row,'full',cap)
        if args.stage=='smoke':
            again,_,_,_=common.generate(model,tokenizer,row,'full',cap);assert ids==again
        pred=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        result=dict(ids=ids,prediction=pred,stopped=stopped,routing=stats,
            score=common.sample_score(pred,row['answers'],row['match_type']),seconds=time.monotonic()-start)
        write_json(path,dict(status='complete',sample=row,result=result,protocol=spec,
            command=shlex.join(sys.argv),python=sys.executable))
        print('COMPLETE',row['index'],result['score'],result['seconds'],flush=True)


if __name__=='__main__':main()
