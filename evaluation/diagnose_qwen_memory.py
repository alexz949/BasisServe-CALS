"""Trace the frozen two-GPU evaluator without changing its numerical protocol."""
import sys
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'results/k_routing_fit/qwen3_32b/sources_2gpu'))
import torch
from evaluation import eval_k_routing_ruler as evaluator

original_install = evaluator.install


def traced_install(model, *args, **kwargs):
    original_install(model, *args, **kwargs)
    before = [torch.cuda.memory_allocated(i) / 2**30
              for i in range(torch.cuda.device_count())]
    collected = gc.collect()
    after = [torch.cuda.memory_allocated(i) / 2**30
             for i in range(torch.cuda.device_count())]
    print('POST_INSTALL_GC', collected, 'before_GiB', before,
          'after_GiB', after, flush=True)
    print('DEVICE_MAP', model.hf_device_map, flush=True)
    for device in range(torch.cuda.device_count()):
        size = sum(p.numel() * p.element_size() for p in model.parameters()
                   if p.device.index == device)
        print('WEIGHTS_GIB', device, size / 2**30, flush=True)
    for index, layer in enumerate(model.model.layers):
        def trace(module, inputs, index=index):
            if inputs[0].shape[1] > 1:
                print('PREFILL_LAYER', index,
                      {i: round(torch.cuda.memory_allocated(i) / 2**30, 3)
                       for i in range(torch.cuda.device_count())}, flush=True)
        layer.register_forward_pre_hook(trace)


evaluator.install = traced_install
if __name__ == '__main__':
    evaluator.main()
