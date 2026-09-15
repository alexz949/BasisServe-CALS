"""Shared split-V exact selected attention with staged or fused mapped CPU K."""
import torch
from basisserve.core.c1_k_offload import PinnedCPUExactKeyPageStore,PreparedExactKeyPageFetch
from basisserve.kernels.mapped_host_paged_attention import mapped_host_paged_attention


class SplitValueKFetch:
    def __init__(self,router,host_key):
        assert host_key.device.type=='cpu' and host_key.is_pinned()
        self.router=router;self.host_key=host_key
        b,h,t,d=host_key.shape;budget=router.budget
        self.page_ids=torch.empty(b,h,budget//32,dtype=torch.int64,device='cuda')
        self.store=PinnedCPUExactKeyPageStore(host_key)
        self.fetch=PreparedExactKeyPageFetch(self.store,pages_per_head=budget//32,page_size=32,device=router.q.device)
        self.staged_key=self.fetch.destination.view(b,h,budget,128)
        self.q=router.q.reshape(b,h,4,128).float()
        self.k32=torch.empty(b,h,budget,128,device='cuda',dtype=torch.float32)
        self.logits=torch.empty(b,h,4,budget,device='cuda',dtype=torch.float32)
        self.probability=torch.empty_like(self.logits)
        self.selected_base=torch.empty(b,h,budget,16,device='cuda',dtype=torch.bfloat16)
        width=16+router.tail.shape[-1]
        self.selected_tail=torch.empty(b,h,budget,width-16,device='cuda',dtype=torch.bfloat16)
        self.v32=torch.empty(b,h,budget,width,device='cuda',dtype=torch.float32)
        self.out32=torch.empty(b,h,4,width,device='cuda',dtype=torch.float32)
        self.output=torch.empty(b,h*4,1,width,device='cuda',dtype=torch.bfloat16)
        self.workspace=torch.empty(b*h*4,128,width+2,device='cuda',dtype=torch.float32)
        self.update_pages()
        assert torch.equal((self.page_ids[...,None]*32+torch.arange(32,device='cuda')).flatten(-2),router.ids)
        assert self.fetch.requested_bytes==b*h*budget*128*2
        assert self.staged_key.numel()<self.host_key.numel()

    def update_pages(self):
        torch.div(self.router.ids[...,::32],32,rounding_mode='floor',out=self.page_ids)

    def staged_fetch(self):
        self.update_pages();return self.fetch(self.page_ids)

    def exact_qk(self):
        self.k32.copy_(self.staged_key)
        torch.matmul(self.q,self.k32.transpose(-1,-2),out=self.logits)
        self.logits.mul_(128**-.5)

    def softmax_pv(self):
        ids=self.router.ids
        torch.gather(self.router.base,2,ids[...,None].expand(*ids.shape,16),out=self.selected_base)
        torch.gather(self.router.tail,2,ids[...,None].expand(*ids.shape,self.router.tail.shape[-1]),out=self.selected_tail)
        self.v32[...,:16].copy_(self.selected_base);self.v32[...,16:].copy_(self.selected_tail)
        torch.softmax(self.logits,-1,out=self.probability)
        torch.matmul(self.probability,self.v32,out=self.out32)
        self.output.copy_(self.out32.reshape_as(self.output))
        return self.output

    def staged_attention(self):
        self.staged_fetch();self.exact_qk();return self.softmax_pv()

    def mapped_attention(self):
        self.update_pages()
        return mapped_host_paged_attention(self.host_key,self.router.q,self.router.tail,self.page_ids,
            sequence_length=self.router.length,page_size=32,splits=32,value_prefix=self.router.base,
            workspace=self.workspace,output=self.output)

    def validate(self):
        actual=self.staged_attention().clone();torch.cuda.synchronize()
        ids=self.router.ids.cpu()
        reference_key=self.host_key.gather(2,ids[...,None].expand(*ids.shape,128)).cuda().float()
        assert torch.equal(reference_key,self.k32)
        logits=(self.q@reference_key.transpose(-1,-2))*128**-.5
        expected=(logits.softmax(-1)@self.v32).reshape_as(actual)
        torch.testing.assert_close(actual.float(),expected,rtol=.015,atol=.004)
        fused=self.mapped_attention().clone()
        torch.testing.assert_close(fused.float(),expected,rtol=.015,atol=.004)
        return dict(staged_exact_keys_equal=True,staged_reference_equal=True,mapped_reference_equal=True,
            no_full_gpu_k_buffer=True,gpu_k_staging_bytes=self.staged_key.numel()*2,
            full_cpu_k_bytes=self.host_key.numel()*2,
            mapped_rel_mse=float((fused.float()-expected).square().sum()/expected.square().sum()))
