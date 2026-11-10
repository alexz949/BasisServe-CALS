"""Copy-bandwidth part of check_interconnect.sh (single process, all visible GPUs).

Measures host<->device bandwidth per GPU, aggregate host->device bandwidth when 1, 2 and all GPUs
copy at once (flat aggregate = the GPUs share one upstream link), and the peer-to-peer copy matrix.
GPU indices are CUDA ordinals within CUDA_VISIBLE_DEVICES.
"""
import json
import time

import torch

BIG, SMALL = 256 << 20, 256 << 10
n = torch.cuda.device_count()


def timed(fn, iters):
    for _ in range(3):
        fn()
    for d in range(n):
        torch.cuda.synchronize(d)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    for d in range(n):
        torch.cuda.synchronize(d)
    return (time.perf_counter() - start) / iters


hosts = [torch.empty(BIG, dtype=torch.uint8).pin_memory() for _ in range(n)]
devs = [torch.empty(BIG, dtype=torch.uint8, device=i) for i in range(n)]
out = dict(gpus=n, h2d_GBs={}, d2h_GBs={}, concurrent_h2d_aggregate_GBs={}, peer_access={}, p2p_GBs={}, p2p_256KiB_us={})
for i in range(n):
    out["h2d_GBs"][i] = round(BIG / timed(lambda: devs[i].copy_(hosts[i], non_blocking=True), 20) / 1e9, 2)
    out["d2h_GBs"][i] = round(BIG / timed(lambda: hosts[i].copy_(devs[i], non_blocking=True), 20) / 1e9, 2)
for k in sorted({1, 2, n}):
    def copy_k():
        for i in range(k):
            devs[i].copy_(hosts[i], non_blocking=True)
    out["concurrent_h2d_aggregate_GBs"][k] = round(k * BIG / timed(copy_k, 10) / 1e9, 2)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        key = f"{i}->{j}"
        out["peer_access"][key] = torch.cuda.can_device_access_peer(i, j)
        out["p2p_GBs"][key] = round(BIG / timed(lambda: devs[j].copy_(devs[i]), 20) / 1e9, 2)
        src, dst = devs[i][:SMALL], devs[j][:SMALL]
        out["p2p_256KiB_us"][key] = round(timed(lambda: dst.copy_(src), 200) * 1e6, 1)
print("RESULT copy", json.dumps(out), flush=True)
