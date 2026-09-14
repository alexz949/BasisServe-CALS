import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from evaluation.host_staged_dispatch import host_staged_send_to_device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires two GPUs')
@pytest.mark.parametrize('source,target', [(0, 1), (1, 0)])
def test_host_transfer_preserves_values_and_skipped_fields(source, target):
    for length in [128256, 130843 * 4096]:
        x = torch.arange(length, device=f'cuda:{source}', dtype=torch.int64).remainder(127).to(torch.bfloat16)
        for value in [x, x[::2]]:
            result = host_staged_send_to_device(value, target)
            assert result.device == torch.device(f'cuda:{target}')
            assert torch.equal(result.cpu(), value.cpu())
        del x
    logits = torch.randn(1, 1, 128256, device=f'cuda:{source}', dtype=torch.bfloat16)
    cache = object()
    output = CausalLMOutputWithPast(logits=logits, past_key_values=cache)
    copied = host_staged_send_to_device(output, target, skip_keys=['past_key_values'])
    assert isinstance(copied, CausalLMOutputWithPast)
    assert copied.past_key_values is cache
    assert torch.equal(copied.logits.cpu(), logits.cpu())
