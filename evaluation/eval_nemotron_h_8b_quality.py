"""Evaluate Nemotron-H dense or C1 V+Wo on the Qwen3-8B quality protocol."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from safetensors.torch import load_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basisserve.checkpoint.gqa_vo_nemotron_h import nemotron_h_c1_attention
from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout, fold_factors_to_dense_weight
from evaluation.build_nemotron_h_8b_checkpoint import FORMAT as CHECKPOINT_FORMAT
from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss
from evaluation.eval_qwen3_32b_c1_c4_ppl_shard import _evaluate_document_ppl
from evaluation.eval_qwen3_8b_iclr_quality import TASKS, summarize_commonsense, _write_json
from evaluation.nemotron_h_runtime import install_mamba_device_guards
from evaluation.v96kl_common import configure, read_json, sha256

FORMAT='basisserve.nemotron_h.quality.v1'


def _quality_windows(path, model):
    manifest=read_json(path.parent/'manifest.json')
    protocol=manifest['protocol']
    assert manifest['status']=='complete' and manifest['sha256']==sha256(path)
    assert protocol['model_config_sha256']==sha256(model/'config.json')
    assert protocol['split']=='validation' and protocol['samples']==128
    assert protocol['sequence_length']==2048
    tokens=load_file(str(path))['input_ids'].long()
    assert tokens.shape==(128,2048)
    return tokens,dict(path=str(path),sha256=sha256(path),
        manifest_sha256=sha256(path.parent/'manifest.json'),protocol=protocol)


@torch.inference_mode()
def _install(model, checkpoint):
    manifest=read_json(checkpoint/'manifest.json')
    assert manifest['status']=='complete' and manifest['format']==CHECKPOINT_FORMAT
    assert manifest['model']['config_sha256']==sha256(Path(manifest['model']['path'])/'config.json')
    installed=[]
    for record in manifest['mamba_wo']:
        path=checkpoint/record['file']; assert sha256(path)==record['sha256']
        factors=load_file(str(path)); layer=model.model.layers[record['layer']]
        projection=layer.mixer.out_proj
        protocol=manifest['compression']
        layout=TPSourceWOLayout(input_width=projection.in_features,
            output_width=projection.out_features,tp_size=protocol['mamba_wo_tp'],
            source_rank=protocol['mamba_wo_source_rank'])
        weight=fold_factors_to_dense_weight(
            factors['source_encoders'].to(projection.weight.device,dtype=torch.float32),
            factors['source_decoders'].to(projection.weight.device,dtype=torch.float32),layout)
        projection.weight.copy_(weight.to(projection.weight.dtype))
        installed.append(dict(layer=record['layer'],kind='linear_attention',file=record['file']))
        del factors,weight
    hq=model.config.num_attention_heads;hkv=model.config.num_key_value_heads
    hidden=model.config.hidden_size
    for record in manifest['attention']:
        path=checkpoint/record['file'];assert sha256(path)==record['sha256']
        factors=load_file(str(path));layer=model.model.layers[record['layer']]
        mixer=layer.mixer;rank=record['ranks'][0]
        assert record['ranks']==[rank]*hkv
        encoder=factors['value_coordinate_encoders']
        decoder=factors['head_output_decoders']
        assert encoder.shape==(hkv,mixer.head_dim,rank) and decoder.shape==(hq,rank,hidden)
        device=mixer.v_proj.weight.device
        compressed=torch.bmm(encoder.to(device).float().mT,
            mixer.v_proj.weight.float().reshape(hkv,mixer.head_dim,hidden)).reshape(hkv*rank,hidden)
        o_decoder=decoder.to(device).float().permute(2,0,1).reshape(hidden,hq*rank)
        layer.mixer=nemotron_h_c1_attention(mixer,
            v_proj_compressed_weight=compressed,o_decoder_weight=o_decoder,
            value_coordinate_encoder=encoder,attention_backend='sdpa')
        installed.append(dict(layer=record['layer'],kind='full_attention',rank=rank,file=record['file']))
        del factors,compressed,o_decoder
    return manifest,installed


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--checkpoint',required=True,help="Checkpoint directory or 'dense'")
    parser.add_argument('--c4-windows',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--lm-eval-batch-size',type=int,default=8)
    parser.add_argument('--expected-gpu-count',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();configure();torch.set_num_threads(4)
    assert torch.cuda.device_count()==args.expected_gpu_count
    model_path=args.model.resolve();started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa',
        device_map='balanced',max_memory={i:torch.cuda.get_device_properties(i).total_memory-8*2**30
            for i in range(args.expected_gpu_count)}).eval()
    guarded=install_mamba_device_guards(model)
    checkpoint_manifest=None;installation=[]
    if args.checkpoint!='dense':
        checkpoint_manifest,installation=_install(model,Path(args.checkpoint))
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
    sequences,c4_provenance=_quality_windows(args.c4_windows,model_path)
    if args.smoke:
        input_device=model.get_input_embeddings().weight.device
        logits=model(sequences[:1,:128].to(input_device),use_cache=False,
            logits_to_keep=1).logits
        assert torch.isfinite(logits).all()
        result=dict(status='complete',format=FORMAT,smoke=True,run_id=args.run_id,
            checkpoint=args.checkpoint,installed=installation,
            guarded_mamba_layers=guarded,logits_shape=list(logits.shape),
            logits_max_abs=float(logits.abs().max()),command=shlex.join(sys.argv))
        args.output.parent.mkdir(parents=True,exist_ok=True);_write_json(args.output,result)
        print('QUALITY SMOKE COMPLETE',args.run_id,flush=True)
        return
    model.config.use_cache=False
    wiki=_eval_ppl_fp32_loss(model,tokenizer,dataset='wikitext2',split='test',
        seqlen=2048,batch_size=args.batch_size,max_samples=None,max_tokens=None)
    print('WIKITEXT2',wiki['ppl'],flush=True)
    c4=_evaluate_document_ppl(model,sequences,batch_size=args.batch_size,label=args.run_id)
    print('C4',c4['ppl'],flush=True)
    model.config.use_cache=True
    lm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=args.lm_eval_batch_size,
        max_length=4096,add_bos_token=False)
    raw=lm_eval.simple_evaluate(model=lm,tasks=list(TASKS),num_fewshot=0,
        task_manager=TaskManager(),log_samples=False)
    summary=summarize_commonsense(raw,TASKS);assert summary is not None
    tasks,average=summary
    result=dict(status='complete',format=FORMAT,run_id=args.run_id,
        model=dict(path=str(model_path),config_sha256=sha256(model_path/'config.json')),
        checkpoint=('dense' if args.checkpoint=='dense' else dict(path=str(Path(args.checkpoint).resolve()),
            manifest_sha256=sha256(Path(args.checkpoint)/'manifest.json'),compression=checkpoint_manifest['compression'])),
        protocol=dict(wikitext2='full test corpus, 2048-token windows, FP32 loss',
            c4='128 document-disjoint validation windows, 2048 tokens, FP32 loss',
            tasks=list(TASKS),num_fewshot=0,max_length=4096,
            metric_selection='acc_norm when present, otherwise acc'),
        metrics=dict(wikitext2_ppl=wiki['ppl'],c4_validation_128_ppl=c4['ppl'],
            average_accuracy=average,task_accuracy=tasks),
        details=dict(wikitext2=wiki,c4=c4,commonsense=raw,c4_windows=c4_provenance,
            installed=installation,guarded_mamba_layers=guarded),
        environment=dict(conda_environment=os.environ.get('CONDA_DEFAULT_ENV'),python=sys.version,
            torch=torch.__version__,transformers=importlib.metadata.version('transformers'),
            datasets=importlib.metadata.version('datasets'),lm_eval=importlib.metadata.version('lm-eval'),
            cuda_devices=[torch.cuda.get_device_name(i) for i in range(args.expected_gpu_count)],
            peak_cuda_allocated_bytes={str(i):torch.cuda.max_memory_allocated(i)
                for i in range(args.expected_gpu_count)}),
        command=shlex.join(sys.argv),seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True);_write_json(args.output,result)
    print('QUALITY COMPLETE',args.run_id,wiki['ppl'],c4['ppl'],average,flush=True)


if __name__=='__main__':
    main()
