"""Native vLLM scheduling, with a prefill barrier only for decode experiments."""

from vllm.v1.core.sched.scheduler import Scheduler


class CohortScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_ready_decodes = False
        self.held_requests = []
        self.trial_steps = []
        self.record_steps = False

    def release_decodes(self):
        self.running.extend(self.held_requests)
        self.held_requests.clear()
        self.hold_ready_decodes = False
        assert len(self.running) <= self.max_num_running_reqs

    def schedule(self, throttle_prefills=False):
        if self.hold_ready_decodes:
            ready = [r for r in self.running if r.num_computed_tokens >= r.num_prompt_tokens]
            self.held_requests.extend(ready)
            self.running = [r for r in self.running if r.num_computed_tokens < r.num_prompt_tokens]
        result = super().schedule(throttle_prefills)
        if self.record_steps:
            rows = []
            for req_id, count in result.num_scheduled_tokens.items():
                request = self.requests[req_id]
                before = request.num_computed_tokens - count
                rows.append(dict(request_id=req_id, scheduled_tokens=count,
                                 cache_tokens_before=before, cache_tokens_after=before + count,
                                 is_prefill=before < request.num_prompt_tokens))
            pool = self.kv_cache_manager.block_pool
            self.trial_steps.append(dict(
                requests=rows, running=len(self.running), held=len(self.held_requests),
                preempted_ids=sorted(result.preempted_req_ids or []),
                occupied_pool_blocks=pool.num_gpu_blocks - pool.get_num_free_blocks(),
            ))
        return result
