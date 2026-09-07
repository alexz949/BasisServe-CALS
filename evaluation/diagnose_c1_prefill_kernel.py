"""Read-only numerical diagnosis of actual C1 prefill tensors and trajectories."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.kernels.compressed_v_decode_attention import compressed_v_prefill_attention
from evaluation.eval_longbench_c1_denseprefill import activate
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json

INDICES=[0,32,64,96,128,160,119,175]


def positions(length):
    points=[0,1,30,31,32,33,62,63,64,65,127,128,255,256,
        length//4,length//2,3*length//4,length-65,length-64,length-33,length-32,length-2,length-1]
    return sorted(set(p for p in points if 0<=p<length))


def error(actual,expected):
    a,b=actual.float(),expected.float()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:return dict(finite=False)
    delta=a-b
    return dict(finite=True,relative_l2=float(delta.norm()/b.norm().clamp_min(1e-30)),
        rms=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()),
        reference_rms=float(b.square().mean().sqrt()))


def flash(query,key,value,*,scale=None):
    rank=value.shape[-1]
    padded=F.pad(value,(0,query.shape[-1]-rank))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output=F.scaled_dot_product_attention(query,key,padded,dropout_p=0.,
            is_causal=True,enable_gqa=True,scale=scale)
    return output[...,:rank].contiguous()


def fp32_rows(query,key,value,points,scale):
    group=query.shape[1]//key.shape[1]
    idx=torch.tensor(points,device=query.device)
    mask=torch.arange(key.shape[-2],device=query.device)[None,:]<=idx[:,None]
    pieces=[]
    for head in range(query.shape[1]):
        scores=(query[:,head,idx].float() @ key[:,head//group].float().transpose(-1,-2))*scale
        prob=scores.masked_fill(~mask,-torch.inf).softmax(-1)
        pieces.append(prob @ value[:,head//group].float())
    return torch.stack(pieces,dim=1)


def decoded_rows(attention,latent):
    flattened=latent.transpose(1,2).reshape(latent.shape[0],latent.shape[2],-1).float()
    return F.linear(flattened,attention.o_proj.weight.float(),None)


def kl(teacher,candidate):
    lp=teacher.float().log_softmax(-1);lq=candidate.float().log_softmax(-1)
    return float((lp.exp()*(lp-lq)).sum())


@torch.inference_mode()
def prefill(model,tokens):
    cache=RoutingDynamicCache()
    output=model(input_ids=tokens.long()[None].to('cuda:0'),past_key_values=cache,
        use_cache=True,logits_to_keep=1)
    logits=output.logits[0,-1].float().cpu()
    assert torch.isfinite(logits).all()
    del output,cache
    torch.cuda.synchronize()
    return logits


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('smoke','probe','summarize'),required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6')
    p.add_argument('--data-dir',type=Path,default=ROOT/'results/datasets/longbench_c1_32k')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/evaluation/c1_prefill_kernel')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    settings=dict(format='basisserve.c1_prefill_kernel_diagnosis.v1',model=str(args.model.resolve()),
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha256(args.checkpoint/'results.json'),
        dataset_manifest_sha256=sha256(args.data_dir/'manifest.json'),sample_indices=INDICES,
        sampling='first saved prompt from each of six tasks plus shortest and longest; diagnostic, not accuracy estimate',
        reference='forced PyTorch Flash-SDPA, causal GQA, zero-pad V80 to128 and slice output',
        independent_reference='explicit per-head FP32 QK, causal softmax and PV at block-boundary/terminal Q positions',
        trajectories='real Triton C1 prefill; all-layer reference C1 prefill; original dense prefill',
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/diagnose_c1_prefill_kernel.py',
            'basisserve/kernels/compressed_v_decode_attention.py','basisserve/checkpoint/gqa_vo_qwen3.py')},
        torch=torch.__version__,dtype='bfloat16',tf32=False)
    if args.stage=='summarize':
        results=[]
        for index in INDICES:
            record=json.loads((args.output_dir/f'sample_{index:03d}.json').read_text())
            assert record['status']=='complete' and record['protocol']==settings and record['sample']['index']==index
            assert len(record['layers'])==36
            results.append(record)
        for shard in range(args.num_shards):
            record=json.loads((args.output_dir/f'shard_{shard}.json').read_text())
            assert record['status']=='complete' and record['indices']==INDICES[shard::args.num_shards]
        layer_rows=[r for sample in results for r in sample['layers']]
        maxima={field:max(r[field]['relative_l2'] for r in layer_rows) for field in
            ['triton_vs_flash_all','triton_vs_fp32_sampled','flash_vs_fp32_sampled','decoded_triton_vs_fp32','decoded_flash_vs_fp32']}
        changes=sum(r['terminal']['triton_top1']!=r['terminal']['reference_top1'] for r in results)
        report=dict(status='complete',protocol=settings,samples=results,max_relative_l2=maxima,
            triton_reference_top1_disagreements=changes,records=len(layer_rows),command=shlex.join(sys.argv),python=sys.executable)
        write_json(args.output_dir/'result.json',report)
        lines=['# C1 prefill kernel numerical diagnosis','',
            'Eight fixed LongBench prompts,36 layers each; uniform C1-V80. Read-only diagnosis: production kernel and C1 factors unchanged. No benchmark accuracy run.','',
            '| Sample | Task | Tokens | Triton top1 | Reference-C1 top1 | Dense top1 | KL(dense,Triton) | KL(dense,reference) |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in results:
            t=r['terminal'];s=r['sample']
            lines.append(f"| {s['index']} | {s['task']} | {s['prompt_tokens']} | {t['triton_top1']} | {t['reference_top1']} | {t['dense_top1']} | {t['dense_to_triton_kl']:.6g} | {t['dense_to_reference_kl']:.6g} |")
        lines+=['','Maximum relative L2 errors over288 layer/input pairs:','','```json',json.dumps(maxima,indent=2),'```','',
            f'Triton/reference-C1 terminal top1 disagreements: {changes}/8.',
            'Both full C1 trajectories use identical factors and differ only in the prefill attention kernel. The local three-way comparisons use exactly the same Q/K/V tensors on the original Triton trajectory.',
            'Flash reference pads only the Value feature dimension with zeros; Q/K scale and causal mask remain unchanged. Independent sampled-query reference uses explicit FP32 operations with TF32 disabled. Decoded-output errors use the same C1 output projection in FP32.',
            'Errors and logits diagnose numerical behavior, not downstream task accuracy or all possible inputs. No generation, refitting, kernel modification, or RULER rerun is included.','']
        (args.output_dir/'summary.md').write_text('\n'.join(lines))
        print(json.dumps(dict(max_relative_l2=maxima,top1_disagreements=changes),indent=2),flush=True)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L40S'
    if args.stage=='smoke':
        torch.manual_seed(73);records=[]
        for length in [129,4097]:
            q=torch.randn(1,32,length,128,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(1,8,length,128,device='cuda',dtype=torch.bfloat16)
            v=torch.randn(1,8,length,80,device='cuda',dtype=torch.bfloat16)
            a=compressed_v_prefill_attention(q,k,v);b=flash(q,k,v);pts=positions(length)
            c=fp32_rows(q,k,v,pts,128**-.5)
            records.append(dict(length=length,triton_flash=error(a,b),triton_fp32=error(a[:,:,pts],c),flash_fp32=error(b[:,:,pts],c)))
        write_json(args.output_dir/'smoke.json',dict(status='complete',protocol=settings,records=records,command=shlex.join(sys.argv),python=sys.executable))
        print(json.dumps(records,indent=2),flush=True);return
    rows=json.loads((args.data_dir/'samples.json').read_text());tokens=load_file(str(args.data_dir/'tokens.safetensors'))
    dataset=json.loads((args.data_dir/'manifest.json').read_text())
    assert sha256(args.data_dir/'tokens.safetensors')==dataset['tokens_sha256']
    assert sha256(args.data_dir/'samples.json')==dataset['samples_sha256']
    old=json.loads((ROOT/'results/evaluation/longbench_c1_32k/result.json').read_text())
    assert settings['checkpoint_sha256']==old['protocol']['c1_manifest_sha256']
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda:0').eval()
    original=[l.self_attn for l in model.model.layers]
    install_qwen3_gqa_vo_als_export(model,args.checkpoint,attention_backend='triton');model.eval()
    compressed=[l.self_attn for l in model.model.layers]
    assigned=INDICES[args.shard_index::args.num_shards]
    for index in assigned:
        path=args.output_dir/f'sample_{index:03d}.json'
        if path.exists():
            saved=json.loads(path.read_text());assert saved['status']=='complete' and saved['protocol']==settings
            continue
        row=rows[index];tensor=tokens[f'sample_{index:03d}'];layer_records=[]
        assert len(tensor)==row['prompt_tokens']
        started=time.monotonic();torch.cuda.reset_peak_memory_stats();activate(model,compressed)
        def observe(query,key,value,*,scale=None):
            layer=len(layer_records);assert layer<36
            selected_scale=query.shape[-1]**-.5 if scale is None else scale
            a=compressed_v_prefill_attention(query,key,value,scale=selected_scale)
            b=flash(query,key,value,scale=selected_scale)
            pts=positions(query.shape[-2]);c=fp32_rows(query,key,value,pts,selected_scale)
            ar,br=a[:,:,pts],b[:,:,pts]
            decoded=decoded_rows(compressed[layer],c)
            record=dict(layer=layer,shape=list(query.shape),value_shape=list(value.shape),
                strides=dict(q=list(query.stride()),k=list(key.stride()),v=list(value.stride())),query_positions=pts,
                triton_vs_flash_all=error(a,b),triton_vs_fp32_sampled=error(ar,c),flash_vs_fp32_sampled=error(br,c),
                decoded_triton_vs_fp32=error(decoded_rows(compressed[layer],ar),decoded),
                decoded_flash_vs_fp32=error(decoded_rows(compressed[layer],br),decoded),
                per_query=[dict(position=p,triton=error(ar[:,:,i],c[:,:,i]),flash=error(br[:,:,i],c[:,:,i])) for i,p in enumerate(pts)])
            layer_records.append(record)
            write_json(args.output_dir/f'local_{index:03d}_{layer:02d}.json',record)
            print(f'sample={index} layer={layer} latent_rel={record["triton_vs_fp32_sampled"].get("relative_l2")} decoded_rel={record["decoded_triton_vs_fp32"].get("relative_l2")}',flush=True)
            return a
        with patch('basisserve.checkpoint.gqa_vo_qwen3.compressed_v_prefill_attention',new=observe):
            triton_logits=prefill(model,tensor)
        assert len(layer_records)==36
        assert int(triton_logits.argmax())==old['records'][index]['arms']['full_exact_k']['generated_token_ids'][0]
        with patch('basisserve.checkpoint.gqa_vo_qwen3.compressed_v_prefill_attention',new=flash):
            reference_logits=prefill(model,tensor)
        activate(model,original);dense_logits=prefill(model,tensor)
        terminal=dict(triton_top1=int(triton_logits.argmax()),reference_top1=int(reference_logits.argmax()),dense_top1=int(dense_logits.argmax()),
            triton_to_reference_kl=kl(triton_logits,reference_logits),dense_to_triton_kl=kl(dense_logits,triton_logits),
            dense_to_reference_kl=kl(dense_logits,reference_logits),logit_error=error(triton_logits,reference_logits))
        for key in ['triton_top1','reference_top1','dense_top1']:terminal[key+'_text']=tokenizer.decode([terminal[key]])
        write_json(path,dict(status='complete',protocol=settings,sample=row,layers=layer_records,terminal=terminal,
            elapsed_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            command=shlex.join(sys.argv),python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f'sample={index} COMPLETE {json.dumps(terminal)}',flush=True)
    write_json(args.output_dir/f'shard_{args.shard_index}.json',dict(status='complete',indices=assigned,protocol=settings))


if __name__=='__main__':main()
