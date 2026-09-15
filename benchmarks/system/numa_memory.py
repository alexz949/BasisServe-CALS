"""Bind future host allocations, then verify their actual sampled page locations."""
import ctypes
import os
from pathlib import Path
import torch


def slurm_gpu_numa(hardware,local_rank=0):
    """Resolve Slurm's physical GPU IDs before any CUDA allocator initialization."""
    physical=[int(index) for index in os.environ['SLURM_JOB_GPUS'].split(',')]
    selected=physical[local_rank]
    return next(row for row in hardware['gpu_numa'] if row['gpu']==selected)


def bind_host_allocations(node):
    library=ctypes.CDLL('libnuma.so.1',use_errno=True)
    library.numa_available.restype=ctypes.c_int
    assert library.numa_available()>=0
    library.numa_parse_nodestring.argtypes=[ctypes.c_char_p]
    library.numa_parse_nodestring.restype=ctypes.c_void_p
    library.numa_set_membind.argtypes=[ctypes.c_void_p]
    library.numa_bitmask_free.argtypes=[ctypes.c_void_p]
    mask=library.numa_parse_nodestring(str(node).encode());assert mask
    library.numa_set_membind(mask)
    library.numa_bitmask_free(mask)
    return library


def audit_host_tensor(tensor,node,library,samples=1024):
    assert tensor.device.type=='cpu' and tensor.is_contiguous()
    pinned=bool(tensor.is_pinned());assert pinned
    page=os.sysconf('SC_PAGE_SIZE');first=tensor.data_ptr()//page*page
    last=(tensor.data_ptr()+tensor.numel()*tensor.element_size()-1)//page*page
    count=(last-first)//page+1
    indices=sorted({i*(count-1)//max(1,min(samples,count)-1) for i in range(min(samples,count))})
    addresses=(ctypes.c_void_p*len(indices))(*(first+i*page for i in indices))
    status=(ctypes.c_int*len(indices))()
    library.numa_move_pages.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    library.numa_move_pages.restype=ctypes.c_int
    rc=library.numa_move_pages(0,len(indices),addresses,None,status,0)
    record=dict(pinned=pinned,expected_numa_node=node,query_returncode=rc,errno=ctypes.get_errno() if rc else 0,
                sampled_pages=len(indices),page_nodes=list(status) if rc==0 else None,
                numa_maps=Path('/proc/self/numa_maps').read_text())
    assert rc==0,{k:v for k,v in record.items() if k!='numa_maps'}
    assert all(v==node for v in status),dict(expected=node,observed_nodes=sorted(set(status)),sampled_pages=len(indices))
    return record
