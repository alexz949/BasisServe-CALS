# L40S TP8 Machine Topology

Recorded on 2026-09-23 UTC for the Llama-3.1-8B TP8 decode experiments.
The commands below ran directly on the benchmark machine; no conda
environment is required for these system commands. Slurm is not configured
in this environment.

## Hardware

- GPUs: 8 x NVIDIA L40S, 46,068 MiB per GPU.
- NVIDIA driver: 580.178.04.
- CPUs: 2 x AMD EPYC 9354 32-Core Processor.
- CPU topology: 64 physical cores, 128 logical CPUs, 2 NUMA nodes.
- NUMA 0: CPUs `0-31,64-95`; GPUs 0, 1, 2, 3.
- NUMA 1: CPUs `32-63,96-127`; GPUs 4, 5, 6, 7.
- No NVLink connections are reported in the topology matrix.
- P2P read and write capability is `OK` for every distinct GPU pair.

## GPU Inventory

Command:

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version --format=csv
```

```text
index, name, pci.bus_id, memory.total [MiB], driver_version
0, NVIDIA L40S, 00000000:03:00.0, 46068 MiB, 580.178.04
1, NVIDIA L40S, 00000000:04:00.0, 46068 MiB, 580.178.04
2, NVIDIA L40S, 00000000:63:00.0, 46068 MiB, 580.178.04
3, NVIDIA L40S, 00000000:64:00.0, 46068 MiB, 580.178.04
4, NVIDIA L40S, 00000000:83:00.0, 46068 MiB, 580.178.04
5, NVIDIA L40S, 00000000:84:00.0, 46068 MiB, 580.178.04
6, NVIDIA L40S, 00000000:E3:00.0, 46068 MiB, 580.178.04
7, NVIDIA L40S, 00000000:E4:00.0, 46068 MiB, 580.178.04
```

## Connectivity

Command: `nvidia-smi topo -m`. Output below has terminal styling removed
and whitespace normalized.

```text
      GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7 NIC0 CPU Affinity   NUMA Affinity GPU NUMA ID
GPU0  X    PIX  NODE NODE SYS  SYS  SYS  SYS  NODE 0-31,64-95     0             N/A
GPU1  PIX  X    NODE NODE SYS  SYS  SYS  SYS  NODE 0-31,64-95     0             N/A
GPU2  NODE NODE X    PIX  SYS  SYS  SYS  SYS  NODE 0-31,64-95     0             N/A
GPU3  NODE NODE PIX  X    SYS  SYS  SYS  SYS  NODE 0-31,64-95     0             N/A
GPU4  SYS  SYS  SYS  SYS  X    PIX  NODE NODE SYS  32-63,96-127   1             N/A
GPU5  SYS  SYS  SYS  SYS  PIX  X    NODE NODE SYS  32-63,96-127   1             N/A
GPU6  SYS  SYS  SYS  SYS  NODE NODE X    PIX  SYS  32-63,96-127   1             N/A
GPU7  SYS  SYS  SYS  SYS  NODE NODE PIX  X    SYS  32-63,96-127   1             N/A
NIC0  NODE NODE NODE NODE SYS  SYS  SYS  SYS  X
```

- `X`: self.
- `PIX`: traverses at most one PCIe bridge.
- `NODE`: traverses PCIe and the interconnect between PCIe host bridges
  within one NUMA node.
- `SYS`: traverses PCIe and the interconnect between NUMA nodes.
- `NIC0`: `rocep33s0f0`.

## Peer Access

Commands:

```bash
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
```

Both commands returned the following matrix:

```text
      GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
GPU0  X    OK   OK   OK   OK   OK   OK   OK
GPU1  OK   X    OK   OK   OK   OK   OK   OK
GPU2  OK   OK   X    OK   OK   OK   OK   OK
GPU3  OK   OK   OK   X    OK   OK   OK   OK
GPU4  OK   OK   OK   OK   X    OK   OK   OK
GPU5  OK   OK   OK   OK   OK   X    OK   OK
GPU6  OK   OK   OK   OK   OK   OK   X    OK
GPU7  OK   OK   OK   OK   OK   OK   OK   X
```

## CPU NUMA Mapping

Command: `lscpu`. Relevant fields:

```text
CPU(s):                 128
Model name:             AMD EPYC 9354 32-Core Processor
Thread(s) per core:      2
Core(s) per socket:      32
Socket(s):              2
NUMA node(s):           2
NUMA node0 CPU(s):      0-31,64-95
NUMA node1 CPU(s):      32-63,96-127
```

## Interpretation

The closest GPU pairs are (0, 1), (2, 3), (4, 5), and (6, 7), each marked
`PIX`. TP8 spans both NUMA nodes and therefore includes `SYS` paths.
P2P capability does not measure bandwidth or latency, and does not imply
that NCCL uses a particular transport. No bandwidth or latency measurements
are included in this record.
