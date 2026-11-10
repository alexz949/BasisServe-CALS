"""Inspect each vLLM worker after the Llama generation smoke."""

import torch


class LlamaSmokeWorker:
    def llama_smoke_statistics(self):
        from vllm.compilation.counter import compilation_counter

        torch.cuda.synchronize()
        result = dict(
            rank=torch.distributed.get_rank(), device=torch.cuda.get_device_name(),
            cudagraph_captures=compilation_counter.num_cudagraph_captured,
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        )
        model = self.get_model()
        if hasattr(model, "joint_statistics"):
            result["joint"] = model.joint_statistics()
        if hasattr(model, "deployment_statistics"):
            result["deployment"] = model.deployment_statistics()
        return result
