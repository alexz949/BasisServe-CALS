"""Build the register-consumed MMA router against the current serving baseline."""
import hashlib
from pathlib import Path
from torch.utils.cpp_extension import load
from benchmarks.system.local_kernel_candidates import write_source


def compile_register(root, rank, warps):
    src = Path('basisserve/kernels/csrc')
    text = (src/'conditional_router_page32.cu').read_text()
    if warps:
        left=text.index('#if defined(__CUDA_ARCH__)',text.index('__global__ void conditional_router_page_lse_kernel'))
        right=text.index('\n#endif',left)
        text=text[:left]+Path('benchmarks/system/register_router_body.cuh').read_text()+text[right+len('\n#endif'):]
        start=text.index('  constexpr int shared_bytes =',text.index('at::Tensor conditional_router_page_lse_cuda'))
        end=text.index('  conditional_router_page_lse_kernel\n',start)
        text=text[:start]+'  constexpr int shared_bytes = 0;\n'+text[end:]
        text=text.replace('base_code.size(0) * base_code.size(1) * pages),\n         kThreads,',
            f'base_code.size(0) * base_code.size(1) * ((pages+{warps//2}-1)/{warps//2})),\n         {warps*32},')
    folder=root/f'b{rank}_w{warps}';folder.mkdir(parents=True,exist_ok=True)
    write_source(folder/'conditional_router_page32.cu',text)
    for name in ['mapped_host_paged_attention.cpp','mapped_host_paged_attention.cu']:
        write_source(folder/name,(src/name).read_text())
    digest=hashlib.sha256(text.encode()).hexdigest()[:10]
    return load(name=f'register_router_{digest}_b{rank}_w{warps}',
        sources=[str(folder/name) for name in ['mapped_host_paged_attention.cpp','mapped_host_paged_attention.cu','conditional_router_page32.cu']],
        extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','--use_fast_math','--ptxas-options=-v',
            '-DBASIS_VALUE_DIM=80','-DBASIS_GQA=4','-DBASIS_PAGE_SIZE=32',
            f'-DBASIS_BASE_RANK={rank}',f'-DBASIS_RESIDUAL_RANK={rank}',f'-DREGISTER_WARPS={warps}'])
