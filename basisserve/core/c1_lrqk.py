"""LRQK factor equations with resident exact K / C1-V payload (quality reference).

Factor equations adapted from tenghuilee/LRQK, caf16293db2e4423a84ab2e895bacf64479f1eb7.
Copyright (c) 2025 tenghuilee. MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Uses the upstream lambda=1 equations, including analytic-gradient B updates.
Corrected reference cache semantics: active K/AK refer to the same PREVIOUS
token IDs; recent tokens are the exact latest suffix, without ring-index gaps.
No CPU offload, autograd, Adam, page selection, or GQA-union budget enforcement.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LRQKConfig:
    rank: int = 32
    topk: int = 2048
    recent: int = 64
    prefill_iterations: int = 2
    decode_iterations: int = 2
    tolerance: float = 1e-8
    seed: int = 0
    prefill_backend: str = 'triton'

    def __post_init__(self):
        assert min(self.rank, self.topk, self.recent, self.prefill_iterations, self.decode_iterations) > 0
        assert self.tolerance >= 0 and self.prefill_backend in ('triton', 'sdpa')


def _solve(a, b, *, left=True):
    result, info = torch.linalg.solve_ex(a, b, left=left)
    assert (info == 0).all() and torch.isfinite(result).all()
    return result


def prefill_factors(q, k, aq, ak, iterations, tolerance):
    """Upstream _lrqk_prefill_inv_w1, FP32; inputs include fixed initialization."""
    assert q.dtype == k.dtype == aq.dtype == ak.dtype == torch.float32
    for _ in range(iterations):
        qq = aq.transpose(-1,-2) @ aq
        kk = ak.transpose(-1,-2) @ ak
        bq = _solve(qq, aq.transpose(-1,-2) @ q)
        bk = _solve(kk, ak.transpose(-1,-2) @ k)
        new_aq = q @ _solve(kk + bq @ bq.transpose(-1,-2),
            k.transpose(-1,-2) @ ak + bq.transpose(-1,-2), left=False)
        qq = new_aq.transpose(-1,-2) @ new_aq
        new_ak = k @ _solve(qq + bk @ bk.transpose(-1,-2),
            q.transpose(-1,-2) @ new_aq + bk.transpose(-1,-2), left=False)
        delta = max(F.mse_loss(aq,new_aq), F.mse_loss(ak,new_ak))
        aq, ak = new_aq, new_ak
        if delta < tolerance:
            break
    bq = _solve(aq.transpose(-1,-2) @ aq, aq.transpose(-1,-2) @ q)
    bk = _solve(ak.transpose(-1,-2) @ ak, ak.transpose(-1,-2) @ k)
    return aq, bq, ak, bk


def _gradient_step(db, code):
    return db.square().sum(dim=(2,3),keepdim=True) / (
        (code @ db).square().sum(dim=(2,3),keepdim=True) + 1e-6)


def decode_factors(bq, ak, bk, active_k, q, k, iterations, tolerance):
    """Upstream _lrqk_decode_inv_w1, including manual B_Q/B_K gradient steps."""
    kb = k @ bk.transpose(-1,-2)
    bbk = bk @ bk.transpose(-1,-2)
    rhs = q @ bq.transpose(-1,-2) + (q @ active_k.transpose(-1,-2)) @ ak
    normal = bq @ bq.transpose(-1,-2) + ak.transpose(-1,-2) @ ak
    qk = q @ k.transpose(-1,-2)
    new_k = _solve(bbk, kb, left=False)
    for _ in range(max(iterations,1)):
        new_q = _solve(normal + new_k.transpose(-1,-2) @ new_k,
            rhs + qk @ new_k, left=False)
        updated_k = _solve(bbk + new_q.transpose(-1,-2) @ new_q,
            kb + qk @ new_q, left=False)
        delta = F.mse_loss(new_k,updated_k)
        new_k = updated_k
        if delta < tolerance:
            break
    dbq = new_q.transpose(-1,-2) @ (new_q @ bq - q)
    dbk = new_k.transpose(-1,-2) @ (new_k @ bk - k)
    return bq - _gradient_step(dbq,new_q)*dbq, bk - _gradient_step(dbk,new_k)*dbk, new_q, new_k


def gather_heads(tensor, ids):
    """Gather per-query-head token IDs directly from physical KV storage."""
    batch, heads, _ = ids.shape
    assert heads % tensor.shape[1] == 0
    mapping = torch.arange(heads,device=tensor.device) // (heads//tensor.shape[1])
    return tensor[torch.arange(batch,device=tensor.device)[:,None,None], mapping[None,:,None], ids]


def select_tokens(qcode, kcode, config):
    batch, heads, length, _ = kcode.shape
    tail = min(length,config.recent)
    historical = length-tail
    count = min(config.topk,historical)
    if count == historical:
        old = torch.arange(historical,device=kcode.device).expand(batch,heads,-1)
    else:
        scores = (kcode[:,:,:historical] @ qcode.transpose(-1,-2)).squeeze(-1)
        assert torch.isfinite(scores).all()
        old = scores.topk(count,dim=-1).indices.sort(dim=-1).values
    recent = torch.arange(historical,length,device=kcode.device).expand(batch,heads,-1)
    return torch.cat((old,recent),dim=-1)


def selected_attention(q, k, v, ids, scale):
    selected_k, selected_v = gather_heads(k,ids), gather_heads(v,ids)
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and v.shape[-1] <= q.shape[-1]:
        from basisserve.core.compact_v_flash import compact_v_flash_attention
        return compact_v_flash_attention(q, selected_k, selected_v, scale=scale)
    return F.scaled_dot_product_attention(q,selected_k,selected_v,scale=scale,dropout_p=0,is_causal=False)


class LRQKState:
    @torch.inference_mode()
    def __init__(self, q, k, config, layer=0):
        assert q.shape[2] == k.shape[2] and q.shape[2] >= config.rank
        assert q.shape[1] % k.shape[1] == 0 and q.shape[-1] >= config.rank
        self.config = config
        self.length = k.shape[2]
        self.steps = 0
        generator = torch.Generator(device=q.device).manual_seed(config.seed+layer)
        shape = (*q.shape[:-1],config.rank)
        aq = torch.randn(shape,device=q.device,dtype=torch.float32,generator=generator)
        ak = torch.randn(shape,device=q.device,dtype=torch.float32,generator=generator)
        repeated_k = k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
        factors = prefill_factors(q.float(),repeated_k.float(),aq,ak,
            config.prefill_iterations,config.tolerance)
        aq,self.bq,self.ak,self.bk = [t.to(q.dtype) for t in factors]
        assert all(torch.isfinite(t).all() for t in (aq,self.bq,self.ak,self.bk))
        self.selected = select_tokens(aq[:,:,-1:],self.ak,config)

    @torch.inference_mode()
    def decode(self,q,k,v,scale):
        assert q.shape[2] == 1 and k.shape[2] == v.shape[2] == self.length+1
        assert self.selected.max() < self.length and self.ak.shape[2] == self.length
        active_k = gather_heads(k,self.selected)
        active_ak = self.ak.gather(2,self.selected[...,None].expand(-1,-1,-1,self.config.rank))
        current_k = k[:,:,-1:].repeat_interleave(q.shape[1]//k.shape[1],dim=1)
        factors = decode_factors(self.bq.float(),active_ak.float(),self.bk.float(),
            active_k.float(),q.float(),current_k.float(),self.config.decode_iterations,self.config.tolerance)
        self.bq,self.bk,qcode,kcode = [t.to(q.dtype) for t in factors]
        assert all(torch.isfinite(t).all() for t in (self.bq,self.bk,qcode,kcode))
        self.ak = torch.cat((self.ak,kcode),dim=2)
        self.length += 1
        self.steps += 1
        self.selected = select_tokens(qcode,self.ak,self.config)
        output = selected_attention(q,k,v,self.selected,scale)
        assert torch.isfinite(output).all()
        return output

    def statistics(self,kv_heads):
        batch,heads,count = self.selected.shape
        grouped = self.selected.reshape(batch,kv_heads,-1)
        ordered = grouped.sort(dim=-1).values
        union = 1+(ordered[...,1:] != ordered[...,:-1]).sum(-1)
        return dict(length=self.length,decode_steps=self.steps,selected_per_query_head=count,
            physical_union_per_kv_group=union.cpu().tolist(),
            key_code_shape=list(self.ak.shape),key_code_bytes=self.ak.numel()*self.ak.element_size())
