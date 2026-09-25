from types import SimpleNamespace

import torch

from benchmarks.system.bench_tp1_capacity import prefill_row


def test_prefill_admission_writes_independent_batch_rows():
    value = torch.zeros(2, 8, 16, 128)
    host_key = torch.zeros_like(value)
    layer = SimpleNamespace(value_cache=value, host_key=host_key, key_cache=None, length=99)
    for row in range(2):
        with prefill_row([layer], row):
            assert layer.length == 0
            assert layer.value_cache.shape[0] == 1
            layer.value_cache.fill_(row + 1)
            layer.host_key.fill_(row + 3)
            layer.length = 16
        assert layer.value_cache is value and layer.host_key is host_key
    assert torch.all(value[0] == 1) and torch.all(value[1] == 2)
    assert torch.all(host_key[0] == 3) and torch.all(host_key[1] == 4)
    assert layer.length == 16
