"""CPU scheduler semantics shared by routing and its validation."""
from dataclasses import dataclass
from contextvars import ContextVar

# Set by the experiment worker's scheduler dispatch; consumed by its metadata builder.
CURRENT_OUTPUT_STEP = ContextVar('basisserve_output_step', default=None)


@dataclass(frozen=True)
class OutputStep:
    prefill_requests: int
    decode_requests: int
    prefill_tokens: int
    decode_tokens: int

    @property
    def kind(self):
        if self.prefill_requests:
            return 'mixed' if self.decode_requests else 'prefill'
        return 'decode' if self.decode_requests else 'empty'

    def collective(self, mode):
        assert mode in ('ag', 'hybrid', 'ar')
        return 'ar' if mode == 'ar' or (mode == 'hybrid' and self.prefill_requests) else 'ag'


def classify_step(query_lengths, is_prefilling):
    assert len(query_lengths) == len(is_prefilling)
    prefill = [int(n) for n, flag in zip(query_lengths, is_prefilling) if n > 0 and flag]
    decode = [int(n) for n, flag in zip(query_lengths, is_prefilling) if n > 0 and not flag]
    # Speculative decoding is outside this experiment.
    assert all(n == 1 for n in decode)
    return OutputStep(len(prefill), len(decode), sum(prefill), sum(decode))
