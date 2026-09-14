"""Compare compact and expanded Page32 routing at the 64K Qwen geometry."""
import importlib.util
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch


def load(name, folder):
    path = ROOT / 'results/k_routing_fit/qwen3_32b' / folder / 'basisserve/core/c1_conditional_page_attention.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load('expanded_page_reference', 'sources_2gpu')
new = load('compact_page_reference', 'sources_compact')
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
device = parser.parse_args().device
print('device', device, flush=True)
torch.backends.cuda.matmul.allow_tf32 = False
for dtype in (torch.float32, torch.bfloat16):
    generator = torch.Generator(device=device).manual_seed(42)
    def random(shape):
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    q, k, v = random((1,64,1,128)), random((1,8,65537,128)), random((1,8,65537,96))
    sidecar, projector = random((1,8,65537,144)), random((64,128,144)) / 128**0.5
    results, pages = [], []
    for module in (old, new):
        original = module._selected_pages
        def capture(*args, original=original, **kwargs):
            selected = original(*args, **kwargs)
            pages.append(selected[0].clone())
            return selected
        module._selected_pages = capture
        results.append(module.c1_conditional_page_topk_attention(q,k,v,sidecar,projector,
            page_size=32,exact_token_budget=2048,pinned_prefix_pages=1,
            scale=128**-0.5,query_block_size=1,collect_statistics=True))
        module._selected_pages = original
    error = ((results[0].output.float()-results[1].output.float()).square().sum()
             / results[0].output.float().square().sum()).sqrt()
    same_pages = torch.equal(pages[0], pages[1])
    print(str(dtype), 'relative_output_RMSE', float(error),
          'identical_pages', same_pages, flush=True)
    assert same_pages and results[0].statistics == results[1].statistics
    assert error < 0.01
