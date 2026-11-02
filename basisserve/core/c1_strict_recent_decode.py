"""Triton C1 routing/decode with sink32 and two recent physical Page32s."""

import torch
from basisserve.kernels.indexed_sparse_decode_attention import (
    gqa_page32_log_mass_triton, gqa_indexed_sparse_decode_attention_triton,
)


def strict_recent_token_ids(page_log_mass, length, *, budget=2048):
    """One physical support per KV group; disjoint complete historical pages."""
    batch, groups, heads_per_group, pages = page_log_mass.shape
    assert pages == (length+31)//32 and budget >= 96 and budget%32 == 0
    device = page_log_mass.device
    if length <= budget:
        return torch.arange(length,device=device).view(1,1,-1).expand(batch,groups,-1)
    historical_end = pages-2
    historical_count = (budget-96)//32
    eligible = torch.arange(pages,device=device)
    eligible = (eligible > 0) & (eligible < historical_end)
    mass = torch.softmax(page_log_mass.masked_fill(~eligible,-torch.inf),dim=-1)
    scores = mass.amax(dim=2).masked_fill(~eligible,-torch.inf)
    selected_pages = torch.topk(scores,historical_count,dim=-1,sorted=False).indices
    historical = (selected_pages[...,None]*32+torch.arange(32,device=device)).flatten(-2)
    sink = torch.arange(32,device=device).view(1,1,-1).expand(batch,groups,-1)
    recent = torch.arange(historical_end*32,length,device=device).view(1,1,-1).expand(batch,groups,-1)
    return torch.cat((sink,historical,recent),dim=-1).contiguous()


def c1_strict_recent_decode(query,key,value,sidecar,residual_query,*,scale):
    assert query.shape[2] == 1
    q = query[:,:,0]
    code = torch.cat((q,torch.einsum('bhd,hdr->bhr',q,residual_query)),dim=-1).contiguous()
    log_mass = gqa_page32_log_mass_triton(code,sidecar,scale=scale)
    physical_ids = strict_recent_token_ids(log_mass,key.shape[2])
    ids = physical_ids.repeat_interleave(query.shape[1]//key.shape[1],dim=1)
    output = gqa_indexed_sparse_decode_attention_triton(query,key,value,ids,scale=scale)
    return output, physical_ids
