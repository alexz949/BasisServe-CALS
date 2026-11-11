"""CPU placement shared by every V2Q8 arm: a process is pinned to its GPU's NUMA node and, because two GPUs share
each node on this host, to every other logical CPU of it (`slot` picks which half). torch intra-op threads and
OpenMP (set OMP_NUM_THREADS to the same count in the launcher) use all of them. The same policy for every arm keeps
CPU-side work (LRQK's miss gather, host-side bookkeeping) from being throttled for one method and not another."""

import os
from pathlib import Path

import torch


def bind_to_gpu_node(device_index, slot):
    properties = torch.cuda.get_device_properties(device_index)
    device = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
    node = max(int(Path(f"/sys/bus/pci/devices/{device}/numa_node").read_text()), 0)
    cpus = []
    for item in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
        left, _, right = item.partition("-")
        cpus.extend(range(int(left), int(right or left) + 1))
    assigned = cpus[slot % 2::2]
    os.sched_setaffinity(0, assigned)
    torch.set_num_threads(len(assigned))
    return dict(numa_node=node, cpus=assigned, torch_threads=torch.get_num_threads(),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"))
