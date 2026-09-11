"""Check the numerical error and actual benefit before choosing TF32x3."""
import json
import torch
from basisserve.kernels.fp32_matmul import fp32_tf32x3_mm
from evaluation.qwen35_hybrid_common import atomic_save


def timed(fn):
    fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        value = fn()
    stop.record()
    stop.synchronize()
    return value, start.elapsed_time(stop) / 5


def main():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    records = []
    for m, k, n in [(37, 131, 71), (2048, 4096, 4096)]:
        a, b = torch.randn(m, k, device='cuda'), torch.randn(k, n, device='cuda')
        reference = (a.double() @ b.double()).float()
        actual, fast_ms = timed(lambda: fp32_tf32x3_mm(a, b))
        baseline, torch_ms = timed(lambda: a @ b)
        relative = float((actual - reference).norm() / reference.norm())
        baseline_relative = float((baseline - reference).norm() / reference.norm())
        assert relative < 2e-6
        row = dict(shape=[m, k, n], relative_l2_error=relative, torch_relative_l2_error=baseline_relative,
            tf32x3_ms=fast_ms, torch_ms=torch_ms, speedup=torch_ms / fast_ms)
        records.append(row)
        print(json.dumps(row), flush=True)
    atomic_save('results/q35_hybrid/matmul_benchmark.json', records)


if __name__ == '__main__':
    main()
