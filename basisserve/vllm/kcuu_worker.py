"""KCuu worker: semantic prefill graph exclusion and CPU step audit.

Only the experiment explicitly selecting this worker installs this dispatcher.
No vLLM package files or global methods are modified.
"""
from vllm.config import CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker.gpu_worker import Worker
from basisserve.core.tp4_step import CURRENT_OUTPUT_STEP, classify_step


class KCuuDispatcher(CudagraphDispatcher):
    def __init__(self, config, runner):
        super().__init__(config)
        self.runner = runner
        self.scheduled_tokens = None
        self.step_records = []

    def dispatch(self, num_tokens, uniform_decode=False, has_lora=False,
                 num_active_loras=0, valid_modes=None, invalid_modes=None):
        record = None
        if self.scheduled_tokens is not None:
            batch = self.runner.input_batch
            ids = batch.req_ids
            lengths = [self.scheduled_tokens.get(req, 0) for req in ids]
            # Recompute can extend into already generated history, beyond the prompt.
            flags = [int(batch.num_computed_tokens_cpu[i]) < int(batch.num_prompt_tokens[i])
                     or length > 1
                     or int(batch.num_computed_tokens_cpu[i]) < self.runner.requests[req].num_tokens - 1
                     for i, (req, length) in enumerate(zip(ids, lengths))]
            step = classify_step(lengths, flags)
            CURRENT_OUTPUT_STEP.set(step)
            if step.prefill_requests:
                valid_modes = {CUDAGraphMode.NONE}
                uniform_decode = False
            record = dict(kind=step.kind, **vars(step), input_tokens=num_tokens)
        result = super().dispatch(num_tokens, uniform_decode, has_lora, num_active_loras,
                                  valid_modes, invalid_modes)
        if record is not None:
            record.update(graph_mode=str(result[0]), padded_tokens=result[1].num_tokens)
            self.step_records.append(record)
        return result


class KCuuWorker(Worker):
    def init_device(self):
        super().init_device()
        assert not self.use_v2_model_runner
        assert not self.vllm_config.scheduler_config.async_scheduling
        assert self.vllm_config.parallel_config.pipeline_parallel_size == 1
        self.model_runner.cudagraph_dispatcher = KCuuDispatcher(self.vllm_config, self.model_runner)

    def execute_model(self, scheduler_output):
        dispatcher = self.model_runner.cudagraph_dispatcher
        CURRENT_OUTPUT_STEP.set(None)
        dispatcher.scheduled_tokens = scheduler_output.num_scheduled_tokens
        result = super().execute_model(scheduler_output)
        dispatcher.scheduled_tokens = None
        CURRENT_OUTPUT_STEP.set(None)
        return result

    def basisserve_step_records(self):
        return list(self.model_runner.cudagraph_dispatcher.step_records)

    def basisserve_reset_audit(self):
        self.model_runner.cudagraph_dispatcher.step_records.clear()
        if hasattr(self.get_model(), 'c1_boundary'):
            self.get_model().c1_boundary.route_counts.zero_()

    def basisserve_begin_measurement(self):
        import torch
        self.basisserve_reset_audit()
        torch.cuda.reset_peak_memory_stats()

    def basisserve_measurement_stats(self):
        import torch
        from vllm.distributed import get_tp_group
        cache = self.model_runner.kv_cache_config
        return dict(rank=get_tp_group().rank_in_group,
                    actual_max_num_seqs=self.vllm_config.scheduler_config.max_num_seqs,
                    actual_max_num_batched_tokens=self.vllm_config.scheduler_config.max_num_batched_tokens,
                    actual_max_model_len=self.vllm_config.model_config.max_model_len,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    memory_source='torch.cuda peak statistics reset after warmup/before this repetition',
                    total_device_memory_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                    kv_blocks=cache.num_blocks,
                    kv_groups=[dict(block_size=g.kv_cache_spec.block_size,
                                    token_capacity=cache.num_blocks * g.kv_cache_spec.block_size,
                                    layers=len(g.layer_names)) for g in cache.kv_cache_groups],
                    steps=self.basisserve_step_records())

    def basisserve_capture_logits(self):
        model = self.get_model()
        model.first_logits = None
        model.capture_first_logits = True

    def basisserve_save_logits(self, path):
        from safetensors.torch import save_file
        from vllm.distributed import get_tp_group
        model = self.get_model()
        if get_tp_group().rank_in_group == 0:
            assert model.first_logits is not None
            save_file({'logits': model.first_logits.cpu().contiguous()}, path)
        model.capture_first_logits = False
        model.first_logits = None
