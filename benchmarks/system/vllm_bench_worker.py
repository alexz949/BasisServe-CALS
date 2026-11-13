"""Per-rank boundary snapshots, whole-step events and separate profile runs."""

import os
from pathlib import Path
import resource

import torch
from vllm.distributed import get_tp_group


def storage_bytes(tensors):
    allocations = {}
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.numel():
            storage = tensor.untyped_storage()
            allocations[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(allocations.values())


class LlamaBenchWorker:
    def bench_install(self):
        self.bench_events = None
        self.bench_pending = None
        self.bench_support = None
        self.bench_index = 0
        original_execute = self.execute_model
        original_sample = self.sample_tokens

        def execute(scheduler_output):
            enabled = self.bench_events is not None and scheduler_output.total_num_scheduled_tokens > 0
            if enabled:
                assert self.bench_pending is None
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                self.bench_pending = (begin, end)
                begin.record()
            result = original_execute(scheduler_output)
            if enabled and result is not None:
                finish_step()
            return result

        def finish_step():
            begin, end = self.bench_pending
            end.record()
            self.bench_events.append((begin, end))
            self.bench_pending = None
            if self.bench_support is not None:
                assert self.bench_index < self.bench_support.shape[0]
                for layer_index, layer in enumerate(self.get_model().model.layers):
                    self.bench_support[self.bench_index, layer_index].copy_(layer.self_attn.attn.joint_slots.counts)
            self.bench_index += 1

        def sample(grammar_output):
            result = original_sample(grammar_output)
            if self.bench_pending is not None:
                finish_step()
            return result

        self.execute_model, self.sample_tokens = execute, sample
        return self.bench_snapshot()

    def bench_begin(self, support_steps=0):
        torch.cuda.synchronize()
        self.bench_events, self.bench_pending, self.bench_index = [], None, 0
        self.bench_support = None
        model = self.get_model()
        if support_steps and bool(getattr(model, "router_hashes", [])):
            shape = model.model.layers[0].self_attn.attn.joint_slots.counts.shape
            self.bench_support = torch.empty((support_steps, 32, *shape), dtype=torch.int32, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        return self.bench_snapshot()

    def bench_end(self):
        torch.cuda.synchronize()
        assert self.bench_pending is None
        result = self.bench_snapshot()
        result["step_cuda_ms"] = [begin.elapsed_time(end) for begin, end in self.bench_events]
        result["support_counts"] = None if self.bench_support is None else self.bench_support[:self.bench_index].cpu().tolist()
        self.bench_events, self.bench_support = None, None
        return result

    def bench_snapshot(self):
        from vllm.compilation.counter import compilation_counter
        import pynvml

        torch.cuda.synchronize()
        model = self.get_model()
        group = get_tp_group()
        layers = [layer.self_attn for layer in model.model.layers]
        caches = [layer.attn.kv_cache for layer in layers]
        cache_tensors = [tensor for cache in caches for tensor in (cache if isinstance(cache, list) else [cache])]
        routed = bool(getattr(model, "router_hashes", []))
        host, work, epochs, slots, rope = [], [], [], [], []
        if routed:
            for layer in layers:
                attn = layer.attn
                if attn.joint_cache is not None:
                    host.append(attn.joint_cache.host)
                    epochs.append(attn.joint_cache.epoch)
                slots.append(attn.joint_slots.keys)
                rope.append(attn.joint_rope)
                work.extend(t for state in (attn.joint_router, attn.joint_slots)
                            for t in vars(state).values() if isinstance(t, torch.Tensor))
        pynvml.nvmlInit()
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[torch.cuda.current_device()]
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible))
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        gpu_process_bytes = next((int(p.usedGpuMemory) for p in processes if p.pid == os.getpid()), None)
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        communicator = group.device_communicator
        custom = communicator.ca_comm
        return dict(
            rank=group.rank_in_group, pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            model_parameter_storage_bytes=storage_bytes(model.parameters()),
            model_buffer_storage_bytes=storage_bytes(model.buffers()),
            gpu_cache_pool_bytes=storage_bytes(cache_tensors), host_key_pool_bytes=storage_bytes(host),
            gpu_slot_keys_bytes=storage_bytes(slots), gpu_slot_router_workspace_bytes=storage_bytes(work),
            gpu_epoch_bytes=storage_bytes(epochs), gpu_rope_bytes=storage_bytes(rope),
            gpu_base_copy_bytes=0 if routed else None,
            process_rss_bytes=int(status["VmRSS"].split()[0]) * 1024,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            process_vm_lck_bytes=int(status["VmLck"].split()[0]) * 1024,
            pinned_capacity_scope="known host-K pools; VmLck is not a CUDA registration total",
            nvml_sampled_process_bytes=gpu_process_bytes,
            nvml_sampled_device_bytes=int(pynvml.nvmlDeviceGetMemoryInfo(handle).used),
            cudagraph_captures=compilation_counter.num_cudagraph_captured,
            custom_allreduce_enabled=custom is not None and not custom.disabled,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            attention_impl=type(layers[0].attn.impl).__name__,
            # Logical bytes of each layer's view into the shared cache pool.
            gpu_cache_layer_bytes=[sum(t.numel() * t.element_size() for t in (cache if isinstance(cache, list) else [cache]))
                                   for cache in caches],
            # Physical layout of each layer's cache, including any padding of the stored width.
            gpu_cache_layer_shapes=[[list(t.shape) for t in (cache if isinstance(cache, list) else [cache])]
                                    for cache in caches],
            deployment=model.deployment_statistics() if hasattr(model, "deployment_statistics") else None,
        )
