"""Paired KL96 payload extension of the frozen Llama Base exact-V experiment."""
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import eval_llama31_shadow as base
from evaluation.v96kl_common import read_json,sha256

CHECKPOINT=ROOT/'results/hf/ICLR-results/llama31-8b/checkpoints/L31-8B-C1-R96'
original_inputs=base.inputs


def inputs(tokenizer):
    rows,spec=original_inputs(tokenizer)
    manifest=read_json(CHECKPOINT/'manifest.json')
    assert manifest['model']['revision']==base.MODEL.name
    assert manifest['model']['config_sha256']==sha256(base.MODEL/'config.json')
    assert manifest['model']['safetensors_index_sha256']==sha256(base.MODEL/'model.safetensors.index.json')
    assert manifest['compression']['allocation']=='two_sided_factorized_terminal_kl_alpha1'
    assert manifest['compression']['equivalent_rank_target']==96
    assert sha256(CHECKPOINT/'result.json')==manifest['artifact']['sha256']
    assert len(manifest['layers'])==32
    for layer in manifest['layers']:
        assert sha256(CHECKPOINT/layer['file'])==layer['sha256']
    spec.update(value='HF two-sided KL C1 average V96, per-layer latent and refitted output decoder',
                checkpoint_sha256=sha256(CHECKPOINT/'manifest.json'),
                layer_ranks=manifest['compression']['layer_ranks'])
    spec['code_sha256']['evaluation/eval_llama31_shadow_v96.py']=sha256(Path(__file__))
    return rows,spec


class Loader:
    @staticmethod
    def from_pretrained(*args,**kwargs):
        model=AutoModelForCausalLM.from_pretrained(*args,**kwargs)
        manifest=read_json(CHECKPOINT/'manifest.json')
        for layer,record in zip(model.model.layers,manifest['layers'],strict=True):
            m=layer.self_attn;t=load_file(str(CHECKPOINT/record['file']))
            rank=record['ranks'][0]
            assert t['source_ranks'].tolist()==record['ranks']
            encoder=t['value_coordinate_encoders'];decoder=t['head_output_decoders']
            assert encoder.shape==(8,128,rank) and decoder.shape==(32,rank,4096)
            assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
            assert m.v_proj.bias is None and m.o_proj.bias is None
            weight=torch.bmm(encoder.float().transpose(1,2),m.v_proj.weight.float().reshape(8,128,4096)).reshape(8*rank,4096)
            output=decoder.float().permute(2,0,1).reshape(4096,32*rank)
            dtype=m.v_proj.weight.dtype
            m.v_proj=torch.nn.Linear(4096,8*rank,bias=False,dtype=dtype)
            m.o_proj=torch.nn.Linear(32*rank,4096,bias=False,dtype=dtype)
            with torch.no_grad():
                m.v_proj.weight.copy_(weight);m.o_proj.weight.copy_(output)
            m.value_head_dim=rank
        return model


@torch.inference_mode()
def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,cache_position=None,**kwargs):
    batch,length,_=hidden_states.shape
    assert batch==1 and isinstance(past_key_values,base.C1ShadowKVCache)
    q=self.q_proj(hidden_states).view(batch,length,self.config.num_attention_heads,self.head_dim).transpose(1,2)
    pre=self.k_proj(hidden_states).view(batch,length,self.config.num_key_value_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(hidden_states).view(batch,length,self.config.num_key_value_heads,self.value_head_dim).transpose(1,2)
    cos,sin=position_embeddings
    q,k=apply_rotary_pos_emb(q,pre,cos,sin)
    previous=past_key_values.get_seq_length(self.layer_idx)
    assert previous==0 or length==1
    if attention_mask is not None:
        expected=torch.arange(previous+length,device=q.device)[None,:]<=(previous+torch.arange(length,device=q.device)[:,None])
        valid=attention_mask if attention_mask.dtype==torch.bool else attention_mask==0
        assert torch.equal(valid.expand(batch,1,length,previous+length)[0,0],expected)
    current=k
    k,v=past_key_values.update(k,v,self.layer_idx,{'cos':cos,'sin':sin,'cache_position':cache_position})
    if previous==0:
        if self.shadow_enabled:
            past_key_values.shadow_states[self.layer_idx]=base.C1ShadowKVState(pre,k,cos,sin)
        output=base.compressed_v_prefill_attention(q,k,v,scale=self.scaling)
    elif self.shadow_enabled:
        output=past_key_values.shadow_states[self.layer_idx].decode(q,current,v,self.scaling)
    else:
        groups=q.shape[1]//k.shape[1]
        output=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(groups,1),
                    v.repeat_interleave(groups,1),scale=self.scaling,dropout_p=0,is_causal=False)
    return self.o_proj(output.transpose(1,2).contiguous().reshape(batch,length,-1)),None


if __name__=='__main__':
    base.inputs=inputs
    base.forward=forward
    base.AutoModelForCausalLM=Loader
    base.OUTPUT=ROOT/'results/evaluation/llama31_base_shadow_v96'
    base.main()
