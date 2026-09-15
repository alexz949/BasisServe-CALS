"""GPU-event timing of a prebuilt CUDA graph; allocations are captured once."""
import statistics
import torch


def measure(function,warmup=100,iterations=500):
    for _ in range(warmup):function()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):function()
    for _ in range(warmup):graph.replay()
    torch.cuda.synchronize()
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for a,b in zip(starts,ends):
        a.record();graph.replay();b.record()
    torch.cuda.synchronize()
    values=[a.elapsed_time(b)*1000 for a,b in zip(starts,ends)]
    ordered=sorted(values)
    return dict(raw_us=values,mean_us=statistics.mean(values),p50_us=statistics.median(values),
                p95_us=ordered[max(0,int(.95*len(values)+.999)-1)],stddev_us=statistics.pstdev(values),
                warmup=warmup,iterations=iterations,timing='CUDA events around one graph replay; graph allocations outside timing')


def measure_host_sequence(function,warmup=100,iterations=500):
    """Time a real CPU-dependent fetch, keeping its D2H synchronization visible."""
    import time
    for _ in range(warmup):function()
    torch.cuda.synchronize()
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for a,b in zip(starts,ends):a.record();b.record()
    torch.cuda.synchronize()
    wall=[]
    for a,b in zip(starts,ends):
        begin=time.perf_counter();a.record();function();b.record();b.synchronize()
        wall.append((time.perf_counter()-begin)*1e6)
    values=[a.elapsed_time(b)*1000 for a,b in zip(starts,ends)]
    return dict(raw_us=values,mean_us=statistics.mean(values),p50_us=statistics.median(values),
                p95_us=sorted(values)[max(0,int(.95*len(values)+.999)-1)],stddev_us=statistics.pstdev(values),
                raw_wall_us=wall,wall_p50_us=statistics.median(wall),warmup=warmup,iterations=iterations,
                timing='CUDA timeline including required host-dependent gaps; separate synchronized wall time; no CUDA graph')
