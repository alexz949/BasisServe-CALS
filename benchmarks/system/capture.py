"""Capture real dense-model inputs to one attention layer for system benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from evaluation.chunked_prefill_mlp import ChunkedTokenwise
from benchmarks.system.common import metadata,save


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank);torch.set_num_threads(2)
    length=4096 if a.smoke else [16384,32768,65536,131072][rank]
    count=1 if a.smoke else 8
    identity=json.loads((a.root/'manifests/v96.json').read_text())
    model=AutoModelForCausalLM.from_pretrained(identity['model'],dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',local_files_only=True).cuda().eval()
    # Native decoder forwards; only tokenwise operations are blocked for memory.
    for layer in model.model.layers:
        layer.input_layernorm=ChunkedTokenwise(layer.input_layernorm)
        layer.post_attention_layernorm=ChunkedTokenwise(layer.post_attention_layernorm)
        layer.mlp=ChunkedTokenwise(layer.mlp)
    factors=load_file(str(a.root/'v96/selected_factors/layer_003.safetensors'),device=f'cuda:{rank}')
    changed=load_file(str(a.output/'routing_basis/layer_003.safetensors'),device=f'cuda:{rank}')
    assert torch.all(factors['source_ranks']==96)
    weight=model.model.layers[3].self_attn.v_proj.weight.reshape(8,128,4096).float()
    folded={name:torch.bmm(enc.float().transpose(1,2),weight).reshape(8*96,4096).bfloat16()
            for name,enc in [('value',factors['value_coordinate_encoders']),('value_transformed',changed['value_coordinate_encoders'])]}
    del factors,changed,weight
    windows=load_file(str(a.root/'calibration/windows.safetensors'))['input_ids']
    folder=a.output/('capture_smoke' if a.smoke else 'capture')/f't{length}'
    folder.mkdir(parents=True,exist_ok=True)
    meta=metadata()
    for sample in range(count):
        path=folder/f'sample{sample}.safetensors';report=folder/f'sample{sample}.json'
        if report.exists():
            previous=json.loads(report.read_text());assert previous['length']==length and path.exists()
            continue
        assert not path.exists()
        ids=torch.cat((windows[64+sample],windows[72+sample]))[:length][None].cuda()
        hidden=model.model.embed_tokens(ids)
        positions=torch.arange(length,device='cuda')[None]
        cos,sin=model.model.rotary_emb(hidden,positions)
        # Layer 3 has an actual selected V rank of 96, not an invented uniform rank.
        for layer in model.model.layers[:3]:
            hidden=layer(hidden,position_embeddings=(cos,sin),attention_mask=None,use_cache=False)
        normalized=model.model.layers[3].input_layernorm(hidden);del hidden
        attn=model.model.layers[3].self_attn
        tensors={name:torch.empty(1,heads,length,128,dtype=torch.bfloat16) for name,heads in [('q',32),('k',8),('pre_k',8),('raw_v',8)]}
        tensors.update({name:torch.empty(1,8,length,96,dtype=torch.bfloat16) for name in folded})
        for left in range(0,length,2048):
            right=min(left+2048,length);block=normalized[:,left:right]
            q=attn.q_proj(block).view(1,right-left,32,128).transpose(1,2)
            pre=attn.k_proj(block).view(1,right-left,8,128).transpose(1,2)
            v=attn.v_proj(block).view(1,right-left,8,128).transpose(1,2)
            for name,projection in folded.items():
                coordinates=torch.nn.functional.linear(block,projection).view(1,right-left,8,96).transpose(1,2)
                tensors[name][:,:,left:right].copy_(coordinates)
            q,k=apply_rotary_pos_emb(q,pre,cos[:,left:right],sin[:,left:right])
            for name,value in [('q',q),('k',k),('pre_k',pre),('raw_v',v)]:tensors[name][:,:,left:right].copy_(value)
        del normalized
        tensors.update(cos=cos.cpu().contiguous(),sin=sin.cpu().contiguous(),input_ids=ids.cpu())
        save_file(tensors,str(path));del tensors
        save(report,dict(metadata=meta,model=identity['model'],layer=3,length=length,sample=sample,
                         diagnostic_window_ids=[64+sample,72+sample],dense_prefix_layers=[0,1,2],
                         capture='native dense input to layer3; full post-RoPE Q/K, pre-RoPE K, raw V',
                         file_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()))
        print(dict(length=length,sample=sample,status='captured'),flush=True)

if __name__=='__main__':main()
