"""Observe PyTorch tensor collectives for one untimed/warmup model step."""
import torch.distributed as dist


class CollectiveAudit:
    def __init__(self):
        self.records=[]
        self.original={name:getattr(dist,name) for name in ['all_reduce','all_gather','all_gather_into_tensor']}

    def record(self,name,tensor):
        n=tensor.numel()*tensor.element_size()
        self.records.append(dict(operation=name,shape=list(tensor.shape),dtype=str(tensor.dtype),
            input_bytes=n,output_bytes=n if name=='all_reduce' else 4*n,
            analytical_bus_bytes=n*(1.5 if name=='all_reduce' else 3)))

    def start(self):
        def reduce(tensor,*args,**kwargs):
            self.record('all_reduce',tensor)
            return self.original['all_reduce'](tensor,*args,**kwargs)
        def gather(output,tensor,*args,**kwargs):
            self.record('all_gather',tensor)
            return self.original['all_gather'](output,tensor,*args,**kwargs)
        def gather_into(output,tensor,*args,**kwargs):
            self.record('all_gather_into_tensor',tensor)
            return self.original['all_gather_into_tensor'](output,tensor,*args,**kwargs)
        dist.all_reduce=reduce;dist.all_gather=gather;dist.all_gather_into_tensor=gather_into

    def stop(self):
        for name,function in self.original.items():setattr(dist,name,function)
