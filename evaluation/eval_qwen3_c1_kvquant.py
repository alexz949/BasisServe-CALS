"""KVQuant NUQ4 numerical adaptation for Qwen3 C1 Value coordinates; no packed cache."""
import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.eval_qwen3_8b_iclr_quality import install_c1_allocation
from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss, _token_ids
from evaluation.v96kl_common import configure, read_json, sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4')
UPSTREAM=ROOT/'external/KVQuant'


def quantized_output(upstream,quantizers,indices,name,output,heads,head_dim):
    shape=output.shape; flat=output.reshape(-1,heads*head_dim)
    data=flat.index_select(-1,indices[name]).float()
    hi,lo,lut=quantizers[name]
    hi=hi.flatten().to(data.device); lo=lo.flatten().to(data.device)
    dynamic=name.endswith('.v'); axis=-1 if dynamic else 0
    mask=(upstream.get_outliers_dynamic(data,channel=-1,thresh=0.99) if dynamic else
          upstream.get_outliers(data,channel=0,outlier_threshold_upper=hi,outlier_threshold_lower=lo))
    quant=upstream.quant_fn_nuq_recon(data,bits=4,qchannel=axis,dynamicquantization=dynamic,
        include_sparse=True,outlier_mask=mask,maxval=hi,minval=lo,lut=lut,first_few_fp16=-1)
    assert torch.isfinite(quant).all()
    result=flat.clone(); result[:,indices[name]]=quant.to(result.dtype)
    return result.reshape(shape)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--arm',choices=['Dense','C1-R64','C1-R96'],required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--audit-seqlen',type=int,default=256)
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args()
    configure(); torch.set_num_threads(2)
    source=UPSTREAM/'quant/kvquant/simquant_module_quantizer.py'
    spec=importlib.util.spec_from_file_location('kvquant_official',source)
    upstream=importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    out=ROOT/'results/q3-kvquant'/('smoke' if args.smoke else 'full')/args.arm
    output_name=('baseline.json' if args.baseline_only else
                 f'audit-{args.audit_seqlen}.json' if args.audit else 'result.json')
    assert not (out/output_name).exists()
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').eval().cuda()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    manifest=None
    if args.arm!='Dense':
        bank=ROOT/'ICLR-results/qwen3-8b/checkpoints'/f'Q3-8B-{args.arm}'
        manifest=read_json(bank/'manifest.json')
        assert manifest['model']['config_sha256']==sha256(MODEL/'config.json')
        assert install_c1_allocation(model,bank,manifest) is not None
    for parameter in model.parameters(): parameter.requires_grad_(False)
    head_dim=model.config.head_dim; heads=model.config.num_key_value_heads
    modules={}; indices={}
    for i,layer in enumerate(model.model.layers):
        modules[f'{i}.k']=layer.self_attn.k_norm
        modules[f'{i}.v']=layer.self_attn.v_proj
        ranks=manifest['compression']['layer_ranks'][i] if manifest else [head_dim]*heads
        indices[f'{i}.k']=torch.arange(heads*head_dim,device='cuda')
        indices[f'{i}.v']=torch.tensor([h*head_dim+j for h,r in enumerate(ranks) for j in range(r)],device='cuda')
    seqlen=args.audit_seqlen if args.audit else (256 if args.smoke else 2048)
    def ppl(batch_size=1,max_samples=None):
        with torch.inference_mode():
            return _eval_ppl_fp32_loss(model,tokenizer,dataset='wikitext2',split='test',
                seqlen=seqlen,batch_size=batch_size,
                max_samples=2 if args.smoke and max_samples is None else max_samples,max_tokens=None)
    if args.baseline_only:
        assert not args.smoke and not args.audit
        baseline=ppl(batch_size=2)
        write_json(out/'baseline.json',dict(status='complete',arm=args.arm,baseline=baseline,
            model=str(MODEL),checkpoint_manifest_sha256=sha256(bank/'manifest.json') if manifest else None,
            protocol=dict(quantization='none',evaluation_split='WT2 test',
                evaluation_length=2048,batch_size=2),source_sha256=sha256(Path(__file__))))
        print('BASELINE_COMPLETE',args.arm,baseline,flush=True)
        return
    if args.audit:
        assert args.smoke
        quantizers=torch.load(out/'quantizers.pt',map_location='cpu',weights_only=False)
        results={}
        for mode in ('baseline1','baseline2','identity','k','v','kv','baseline3'):
            handles=[]
            for name,module in modules.items():
                if mode=='identity':
                    handles.append(module.register_forward_hook(lambda m,x,y:y.clone()))
                elif name[-1] in mode and mode in ('k','v','kv'):
                    handles.append(module.register_forward_hook(lambda m,x,y,name=name:
                        quantized_output(upstream,quantizers,indices,name,y,heads,head_dim)))
            results[mode]=ppl()
            print('AUDIT',mode,results[mode],flush=True)
            for handle in handles: handle.remove()
        write_json(out/f'audit-{seqlen}.json',results)
        return
    baseline=ppl()
    records={name:[] for name in modules}; handles=[]
    def capture(name,module,inputs,output):
        flat=output.reshape(-1,heads*head_dim)
        entry=[flat.index_select(-1,indices[name]).detach().float().cpu(),None]
        records[name].append(entry)
        def gradient(grad):
            entry[1]=grad.reshape(-1,heads*head_dim).index_select(-1,indices[name]).float().square().cpu()
        output.register_hook(gradient)
    for name,module in modules.items():
        handles.append(module.register_forward_hook(lambda m,x,y,name=name:capture(name,m,x,y)))
    train=_token_ids(tokenizer,'wikitext2','train',None).reshape(-1)
    length=128 if args.smoke else 2048
    generator=torch.Generator().manual_seed(0)
    starts=torch.randint(0,train.numel()-length,(1 if args.smoke else 16,),generator=generator).tolist()
    for i,start in enumerate(starts):
        tokens=train[start:start+length][None].cuda()
        embedded=model.get_input_embeddings()(tokens).detach().requires_grad_(True)
        logits=model(inputs_embeds=embedded,use_cache=False).logits
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
            tokens[:,1:].reshape(-1),reduction='sum')
        loss.backward()
        print('FISHER',i,float(loss.detach()),flush=True)
        del embedded,logits,loss,tokens
    for handle in handles: handle.remove()
    quantizers={}
    for name in modules:
        data=torch.cat([r[0] for r in records[name]])
        fisher=torch.cat([r[1] for r in records[name]])
        assert torch.isfinite(data).all() and torch.isfinite(fisher).all() and fisher.sum()>0
        fisher=fisher/fisher.mean()
        fitter=SimpleNamespace(out=data,perchannel=True,qchannel=0 if name.endswith('.k') else -1,
            bits=4,nsamples=len(starts))
        quantizers[name]=upstream.SimQuant.quantize(fitter,include_sparse=True,
            sparsity_threshold=0.99,nuq=True,fisher=fisher,first_few_fp16=-1)
        del records[name]
        print('CODEBOOK',name,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'quantizers.pt').exists()
    torch.save(quantizers,out/'quantizers.pt')
    handles=[]
    def quantize(name,module,inputs,output):
        return quantized_output(upstream,quantizers,indices,name,output,heads,head_dim)
    for name,module in modules.items():
        handles.append(module.register_forward_hook(lambda m,x,y,name=name:quantize(name,m,x,y)))
    quantized=ppl()
    write_json(out/'result.json',dict(status='complete',arm=args.arm,smoke=args.smoke,
        baseline=baseline,quantized=quantized,model=str(MODEL),
        checkpoint_manifest_sha256=sha256(bank/'manifest.json') if manifest else None,
        protocol=dict(bits=4,sparsity_threshold=0.99,first_few_fp16=-1,rotation='none',
            k='after k_norm before RoPE; static per-channel',
            v='active latent coordinates only; dynamic per-token across KV heads',
            fisher='squared activation gradient of summed next-token CE; arm-specific',
            calibration_split='WT2 train',calibration_length=length,calibration_starts=starts,
            evaluation_split='WT2 test',evaluation_length=seqlen,packed_cache=False),
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip(),
        upstream_sha256=sha256(source),source_sha256=sha256(Path(__file__))))
    print('COMPLETE',args.arm,baseline,quantized,flush=True)


if __name__=='__main__': main()
