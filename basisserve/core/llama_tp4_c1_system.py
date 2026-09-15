"""Llama TP4 C1 serving with transformed split V and exact local/mapped K."""
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
import torch
from torch import nn
from safetensors.torch import load_file
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,rotate_half
from basisserve.core.llama_tp4_k_offload import decoder,LlamaTP4KCacheAttention
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from basisserve.kernels.mapped_host_paged_attention import (
    mapped_host_bf16_empty,append_mapped_host_key,mapped_host_paged_attention,gpu_paged_attention,
    conditional_router_query_code,conditional_router_page_lse,select_fixed_group_max_pages_cuda)
from evaluation.chunked_prefill_mlp import ChunkedTokenwise


class LlamaC1SystemAttention(nn.Module):
    def __init__(self,original,*,mode,batch,capacity,root,basis_root,rank,communicator,profile=False):
        super().__init__()
        assert mode in ('c1','sparse_local','offload')
        self.mode=mode;self.layer_idx=original.layer_idx;self.scaling=original.scaling
        self.q_proj=original.q_proj;self.k_proj=original.k_proj
        self.capacity=capacity;self.length=0;self.batch=batch;self.profile=profile
        self.communicator=communicator
        transformed=load_file(str(Path(basis_root)/f'layer_{self.layer_idx:03d}.safetensors'))
        encoder=transformed['value_coordinate_encoders'][rank*2:(rank+1)*2].cuda()
        self.width=encoder.shape[-1];r=self.width
        weight=original.v_proj.weight
        assert weight.shape==(256,4096) and original.v_proj.bias is None,weight.shape
        local_weight=weight.detach().reshape(2,128,4096)
        # Fold the real transformed C1 encoder into the model's local V projection.
        folded=torch.einsum('hdr,hdm->hrm',encoder.float(),local_weight.float()).reshape(2*r,4096).bfloat16()
        self.v_proj=nn.Linear(4096,2*r,bias=False,device='cuda',dtype=torch.bfloat16)
        self.v_proj.weight=nn.Parameter(folded,requires_grad=False)
        self.register_buffer('global_decoder',transformed['head_output_decoders'].reshape(32*r,4096).cuda().bfloat16())
        self.register_buffer('base_cache',torch.empty(batch,2,capacity,16,device='cuda',dtype=torch.bfloat16),persistent=False)
        self.register_buffer('tail_cache',torch.empty(batch,2,capacity,r-16,device='cuda',dtype=torch.bfloat16),persistent=False)
        assert self.base_cache.untyped_storage().data_ptr()!=self.tail_cache.untyped_storage().data_ptr()
        if mode=='offload':self.host_key=mapped_host_bf16_empty(batch=batch,kv_heads=2,capacity=capacity)
        else:self.register_buffer('key_cache',torch.empty(batch,2,capacity,128,device='cuda',dtype=torch.bfloat16),persistent=False)
        if mode!='c1':
            f=load_file(str(Path(root)/'ours_b16r16'/f'layer_{self.layer_idx:03d}.safetensors'))
            self.factors={name:t[rank*(8 if name.startswith('residual_query') else 2):(rank+1)*(8 if name.startswith('residual_query') else 2)].cuda().bfloat16().contiguous() for name,t in f.items()}
            self.register_buffer('residual_cache',torch.empty_like(self.base_cache),persistent=False)
            self.query_code=torch.empty(batch,2,4,16,device='cuda',dtype=torch.bfloat16)
        self.workspace=torch.empty(batch*8,128,r+2,device='cuda',dtype=torch.float32)
        self.plan=StaticRaggedPlan.from_source_widths((8*r,)*4)
        self.decode_ag=communicator.prepare_uniform(self.plan,tokens=batch,dtype=torch.bfloat16,backend='uniform_nccl')
        # The attention kernel supports feature-strided output: write directly to
        # the AllGather source slot instead of repacking a second output buffer.
        self.decode_output=self.decode_ag.local_feature_major_view_fast().T.view(batch,8,1,r)
        self.nccl_input_bytes=0;self.logical_k_read_bytes=0;self.selected_counts=[]

    def mark(self,name):return torch.cuda.nvtx.range(name) if self.profile else nullcontext()

    def project_output(self,attention,*,decode=False):
        b,h,n,r=attention.shape;tokens=b*n
        prepared=self.decode_ag if decode else self.communicator.prepare_uniform(self.plan,tokens=tokens,dtype=attention.dtype,backend='uniform_nccl')
        if not decode:
            with self.mark('low-rank coordinate layout'):
                prepared.local_feature_major_view_fast().copy_(attention.permute(1,3,0,2).reshape(h*r,tokens))
        with self.mark('NCCL AllGather'):full=prepared.gather_inplace_fast()
        with self.mark('decoder'):out=full.T@self.global_decoder
        if decode:self.nccl_input_bytes+=tokens*8*r*2
        return out.reshape(b,n,4096)

    @torch.inference_mode()
    def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,**kwargs):
        assert past_key_values is None and attention_mask is None
        x=hidden_states;b,n,_=x.shape;start=self.length;end=start+n
        assert end<=self.capacity and (start==0 or n==1)
        cos,sin=position_embeddings;r=self.width
        key=torch.empty(b,2,n,128,device=x.device,dtype=x.dtype)
        padded_value=torch.zeros(b,2,n,128,device=x.device,dtype=x.dtype) if start==0 else None
        for left in range(0,n,2048):
            right=min(left+2048,n);block=x[:,left:right]
            pre=self.k_proj(block).view(b,right-left,2,128).transpose(1,2)
            _,post=apply_rotary_pos_emb(pre,pre,cos[:,left:right],sin[:,left:right])
            key[:,:,left:right]=post
            with self.mark('low-rank encoder'):
                coordinates=self.v_proj(block).view(b,right-left,2,r).transpose(1,2)
            self.base_cache[:,:,start+left:start+right]=coordinates[...,:16]
            self.tail_cache[:,:,start+left:start+right]=coordinates[...,16:]
            if padded_value is not None:padded_value[:,:,left:right,:r]=coordinates
            if self.mode!='c1':
                f=self.factors
                predicted=(coordinates[...,:16]@f['base_right_b16']+f['base_bias_b16'][None,:,None]).bfloat16()
                predicted=(predicted*cos[:,None,left:right]+rotate_half(predicted)*sin[:,None,left:right]).bfloat16()
                self.residual_cache[:,:,start+left:start+right]=(post-predicted)@f['residual_encoder_b16_r16']
        self.length=end
        if self.mode=='offload':append_mapped_host_key(self.host_key,key,start=start)
        else:self.key_cache[:,:,start:end]=key
        for left in range(0,n,2048):
            right=min(left+2048,n)
            q=self.q_proj(x[:,left:right]).view(b,right-left,8,128).transpose(1,2)
            q,_=apply_rotary_pos_emb(q,q,cos[:,left:right],sin[:,left:right])
            if start==0:
                attention=flash_attn_func(q.transpose(1,2),key[:,:,:right].transpose(1,2),padded_value[:,:,:right].transpose(1,2),
                    softmax_scale=self.scaling,causal=True).transpose(1,2)[...,:r]
                x[:,left:right]=self.project_output(attention)
                continue
            if self.mode=='c1' or end<=2048:
                page_size=32;ids=torch.arange((end+31)//32,device=x.device).expand(b,2,-1).contiguous()
                actual=end
            else:
                page_size=1;historical=end-64;f=self.factors
                with self.mark('query preprocessing'):
                    conditional_router_query_code(q,f['residual_query_b16_r16'],self.query_code)
                with self.mark('route'):
                    logs=conditional_router_page_lse(q,self.base_cache[:,:,:historical],self.residual_cache[:,:,:historical],
                        base_right=f['base_right_b16'],base_bias=f['base_bias_b16'],residual_query=f['residual_query_b16_r16'],
                        rope_cos=self.rope_cos[:historical],rope_sin=self.rope_sin[:historical],scale=self.scaling,
                        query_code=self.query_code,query_code_prepared=True)
                with self.mark('top-k/page selection'):
                    pages=select_fixed_group_max_pages_cuda(logs,pages_per_kv_head=62,pinned_prefix_pages=1,force_current_page=False)
                    historical_ids=(pages[...,None]*32+torch.arange(32,device=x.device)).flatten(-2)
                    historical_ids=historical_ids.masked_fill(historical_ids>=historical,-1)
                    ids=torch.cat((historical_ids,torch.arange(historical,end,device=x.device).expand(b,2,-1)),-1)
                # Keep counts on GPU until the measurement interval finishes.
                actual=(ids>=0).sum(-1)
            self.selected_counts.append(actual)
            function=mapped_host_paged_attention if self.mode=='offload' else gpu_paged_attention
            source=self.host_key if self.mode=='offload' else self.key_cache
            label='mapped K read + exact QK + softmax/PV (fused)' if self.mode=='offload' else 'exact QK + softmax/PV (fused)'
            with self.mark(label):
                attention=function(source,q,self.tail_cache,ids,sequence_length=end,page_size=page_size,splits=32,
                    value_prefix=self.base_cache,workspace=self.workspace,output=self.decode_output,scale=self.scaling)
            x[:,left:right]=self.project_output(attention,decode=True)
        return x,None


def install(model,*,mode,batch,capacity,root,basis_root,rank,profile=False):
    assert model.config.model_type=='llama' and model.config.num_hidden_layers==32
    communicator=None if mode=='dense' else FeatureRaggedCommunicator.from_distributed(device=torch.device('cuda',rank))
    model.model.norm=ChunkedTokenwise(model.model.norm);modules=[]
    for layer in model.model.layers:
        if mode=='dense':
            attention=LlamaTP4KCacheAttention(layer.self_attn,mode='dense',batch=batch,capacity=capacity,root=root,rank=rank)
        else:
            attention=LlamaC1SystemAttention(layer.self_attn,mode=mode,batch=batch,capacity=capacity,
                root=root,basis_root=basis_root,rank=rank,communicator=communicator,profile=profile)
        layer.self_attn=attention;layer.input_layernorm=ChunkedTokenwise(layer.input_layernorm)
        layer.forward=MethodType(decoder,layer);modules.append(attention)
    return modules,communicator
