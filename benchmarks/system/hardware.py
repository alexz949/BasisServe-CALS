"""Capture actual GPU topology without interpreting Slurm allocation as idleness."""
import argparse
from pathlib import Path
from benchmarks.system.common import command, metadata, save


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    commands=[['nvidia-smi','-L'],['nvidia-smi','topo','-m'],['nvidia-smi','-q'],
              ['numactl','--hardware'],['lscpu'],['nvcc','--version'],
              ['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max','--format=csv']]
    outputs=[command(c) for c in commands]
    buses=command(['nvidia-smi','--query-gpu=index,pci.bus_id','--format=csv,noheader'])
    topology=[]
    for line in buses['stdout'].splitlines():
        index,bus=[v.strip() for v in line.split(',')]
        fields=bus.lower().split(':');bus=f'{int(fields[0],16):04x}:{fields[1]}:{fields[2]}'
        root=Path('/sys/bus/pci/devices')/bus
        topology.append(dict(gpu=int(index),pci_bus=bus,
            numa_node=int((root/'numa_node').read_text()),local_cpu_list=(root/'local_cpulist').read_text().strip()))
    save(a.output/'hardware.json',dict(metadata=metadata(),commands=outputs,gpu_numa=topology,
         tools=command(['bash','-c','command -v nsys; command -v ncu; command -v numactl'])))
    text='\n\n'.join('$ '+' '.join(r['command'])+'\n'+r['stdout']+r['stderr'] for r in outputs)
    path=a.output/'hardware.txt';assert not path.exists();path.write_text(text)
    print(topology,flush=True)

if __name__=='__main__':main()
