"""Isolate ShadowKV reconstruction and C1 value compression on frozen FWE."""
import argparse
from pathlib import Path
import sys
import time
import shlex
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.v96kl_common import MODEL,CHECKPOINT,configure,checkpoint_manifest,read_json,write_json,sha256
from evaluation.eval_longbench_c1_twosided_denseprefill import install
from evaluation.eval_v96kl_longbench import generate
from evaluation.eval_ruler_kl96 import shadow_generate
from evaluation.ruler_v1 import sample_score
from basisserve.checkpoint import c1_shadowkv_qwen3 as adapter
from basisserve.core.c1_shadowkv import C1ShadowKVState,gather_group


class MeasuredState(C1ShadowKVState):
    exact=False

    def __init__(self,pre,post,cos,sin):
        super().__init__(pre,post,cos,sin)
        self.exact_prompt_key=post
        self.diagnostic={}

    def reconstruct(self,ids):
        return gather_group(self.exact_prompt_key,ids) if self.exact else super().reconstruct(ids)

    def decode(self,q,current,v,scale):
        output=super().decode(q,current,v,scale)
        if self.steps==1:
            routed=self.select(q)
            reconstructed=super().reconstruct(routed)
            exact=gather_group(self.exact_prompt_key,routed)
            groups=q.shape[1]//exact.shape[1]
            exact_all=torch.cat((self.exact_prompt_key,current),2).repeat_interleave(groups,1)
            scores=q.float()@exact_all.float().mT*scale
            mass=scores.softmax(-1)
            ids=self.selected_ids.repeat_interleave(groups,1)
            recall=mass[:,:,0].gather(-1,ids).sum(-1)
            ek=exact.repeat_interleave(groups,1).float()
            rk=reconstructed.repeat_interleave(groups,1).float()
            self.diagnostic=dict(
                key_relative_mse=float((reconstructed.float()-exact.float()).square().sum()/exact.float().square().sum()),
                selected_exact_mass_mean=float(recall.mean()),selected_exact_mass_min=float(recall.min()),
                routed_logit_rmse=float(((q.float()@(rk-ek).mT*scale).square().mean()).sqrt()),
            )
        return output

    def statistics(self):
        return {**super().statistics(),**self.diagnostic}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['repeat','exact_k','dense_full','dense_shadow'],required=True)
    args=p.parse_args();configure();torch.manual_seed(0)
    root=ROOT/'results/evaluation/ruler_kl96_seed42'
    out=ROOT/'results/evaluation/shadowkv_fwe_diagnostic'/args.arm
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True,
                                            attn_implementation='sdpa').cuda().eval()
    manifest=checkpoint_manifest(CHECKPOINT,MODEL)
    originals,modules=install(model,CHECKPOINT,manifest)
    if args.arm.startswith('dense_'):
        for original,module in zip(originals,modules,strict=True):
            module.v_proj=original.v_proj
            module.o_proj=original.o_proj
            module.value_head_dim=128
    del originals
    if args.arm!='dense_full':
        MeasuredState.exact=args.arm=='exact_k'
        adapter.C1ShadowKVState=MeasuredState
        adapter.install_c1_shadowkv(model)
    protocol=dict(arm=args.arm,dtype='bfloat16',indices=list(range(64,72)),
        reference_sha256=sha256(root/'result.json'),checkpoint_sha256=sha256(CHECKPOINT/'manifest.json'),
        code_sha256={f:sha256(ROOT/f) for f in ['evaluation/diagnose_shadowkv_fwe.py',
            'basisserve/core/c1_shadowkv.py','basisserve/checkpoint/c1_shadowkv_qwen3.py']})
    for i in range(64,72):
        reference=read_json(root/'shadowkv/evaluate'/f'sample_{i:03d}.json')
        row=reference['sample'];assert row['task']=='fwe'
        path=out/f'sample_{i:03d}.json'
        if path.exists():
            assert read_json(path)['protocol']==protocol
            continue
        tokens=torch.tensor(row['input_ids']);started=time.monotonic()
        if args.arm=='dense_full':
            ids,first,stats,stopped=generate(model,tokenizer,tokens,row,'full',None,row['maximum_tokens'])
        else:
            ids,first,stats,stopped=shadow_generate(model,tokenizer,tokens,row['maximum_tokens'])
        if args.arm in ('repeat','exact_k'):
            assert ids[0]==reference['result']['ids'][0]
        if args.arm=='repeat': assert ids==reference['result']['ids']
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        score=sample_score(prediction,row['answers'],row['match_type'])
        write_json(path,dict(status='complete',protocol=protocol,sample=row,
            result=dict(ids=ids,prediction=prediction,score=score,routing=stats,stopped=stopped,
                        seconds=time.monotonic()-started),command=shlex.join(sys.argv),python=sys.executable))
        print(args.arm,i,score,repr(prediction),flush=True)


if __name__=='__main__':main()
