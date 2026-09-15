"""Isolated clock64 instrumentation of the existing fused Page-LSE kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import torch
from torch.utils.cpp_extension import load
import benchmarks.system.native_basis_cache as basis
from basisserve.kernels.mapped_host_paged_attention import conditional_router_query_code


def build_sources(root):
    original=Path('basisserve/kernels/csrc');folder=root/'source';folder.mkdir(parents=True,exist_ok=True)
    s=(original/'conditional_router_page32.cu').read_text()
    left=s.index('__global__ void conditional_router_page_lse_kernel(')
    right=s.index('__global__ void conditional_router_append_decode_kernel(',left)
    kernel=s[left:right].replace('    float scale) {','    float scale, int64_t* cycles) {',1)
    needle='  const int64_t page_start = page * kPageSize;'
    kernel=kernel.replace(needle,needle+'\n  __shared__ unsigned long long stamps[7];\n  if (thread == 0) stamps[0] = clock64();',1)
    parts=kernel.split('  __syncthreads();')
    assert len(parts)==6
    kernel=parts[0]+''.join('  __syncthreads();\n  if (thread == 0) stamps['+str(i)+'] = clock64();'+part for i,part in enumerate(parts[1:],1))
    kernel=kernel.replace('\n#endif\n}', '''
  __syncthreads();
  if (thread == 0) {
    stamps[6] = clock64();
    for (int stage = 0; stage < 6; ++stage)
      cycles[row * 6 + stage] = stamps[stage + 1] - stamps[stage];
  }
#endif
}''',1)
    s=s[:left]+kernel+s[right:]
    s=s.replace('}  // namespace','}  // namespace\nstatic at::Tensor router_cycle_buffer;\nat::Tensor get_router_cycles() { return router_cycle_buffer; }',1)
    s=s.replace('  conditional_router_page_lse_kernel\n      <<<','  router_cycle_buffer = at::empty({base_code.size(0) * base_code.size(1) * pages, 6}, base_code.options().dtype(at::kLong));\n  conditional_router_page_lse_kernel\n      <<<',1)
    s=s.replace('          static_cast<float>(scale));','          static_cast<float>(scale), router_cycle_buffer.mutable_data_ptr<int64_t>());',1)
    (folder/'conditional_router_page32.cu').write_text(s)
    cpp=(original/'mapped_host_paged_attention.cpp').read_text()
    cpp=cpp.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {','at::Tensor get_router_cycles();\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {\n  module.def("get_router_cycles", &get_router_cycles);',1)
    (folder/'mapped_host_paged_attention.cpp').write_text(cpp)
    (folder/'mapped_host_paged_attention.cu').write_text((original/'mapped_host_paged_attention.cu').read_text())
    return [str(folder/name) for name in ['mapped_host_paged_attention.cpp','mapped_host_paged_attention.cu','conditional_router_page32.cu']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=[8,16],required=True);a=p.parse_args()
    root=Path('results/system_benchmarks/router_cycles');root.mkdir(parents=True,exist_ok=True)
    sources=build_sources(root)
    r=a.rank
    ext=load(name=f'basis_router_clock_b{r}_r{r}',sources=sources,
        extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','--use_fast_math',
            '-DBASIS_VALUE_DIM=80','-DBASIS_GQA=4','-DBASIS_PAGE_SIZE=32',f'-DBASIS_BASE_RANK={r}',f'-DBASIS_RESIDUAL_RANK={r}'])
    original=basis.conditional_router_page_lse;rows=[];seen=set()
    names=['load_base_and_right','reconstruct_K','bias_round_RoPE','load_query_and_query_code','QK_dot','residual_dot_and_LSE']
    def measured(q,b,res,**kw):
        result=original(q,b,res,**kw)
        identity=kw['base_right'].data_ptr()
        if identity in seen:return result
        seen.add(identity)
        code=torch.empty(q.shape[0],b.shape[1],4,r,device='cuda',dtype=torch.bfloat16)
        conditional_router_query_code(q,kw['residual_query'],code)
        out=torch.empty_like(result)
        args=(q,b,res,kw['base_right'],kw['base_bias'],kw['residual_query'],kw['rope_cos'],kw['rope_sin'],code,out,kw['scale'],True)
        ext.conditional_router_page_lse(*args)
        torch.testing.assert_close(out,result,rtol=0,atol=0)
        cycle_runs=[];profile_times=[];normal_times=[]
        for _ in range(5):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();ext.conditional_router_page_lse(*args);end.record();end.synchronize()
            profile_times.append(begin.elapsed_time(end));cycle_runs.append(ext.get_router_cycles().double().mean(0).cpu())
            begin.record();original(q,b,res,**dict(kw,query_code=code,query_code_prepared=True,output=out));end.record();end.synchronize()
            normal_times.append(begin.elapsed_time(end))
        mean=torch.stack(cycle_runs).mean(0);fraction=mean/mean.sum()
        rows.append(dict(layer=len(rows),mean_cycles_per_block=dict(zip(names,mean.tolist())),
            block_cycle_fraction=dict(zip(names,fraction.tolist())),raw_instrumented_ms=profile_times,raw_original_ms=normal_times,
            bitwise_output_equal=True))
        print('ROUTER_CYCLES',r,len(rows)-1,dict(zip(names,[round(v,4) for v in fraction.tolist()])),flush=True)
        return result
    basis.conditional_router_page_lse=measured
    sys.argv=['bench_shadow_native','--method',f'basis{r}','--length','65536','--batch','1','--output',str(root)]
    runpy.run_module('benchmarks.system.bench_shadow_native',run_name='__main__')
    assert len(rows)==32
    report=dict(rank=r,layers=rows,source_sha256={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
        mean_block_cycle_fraction={k:statistics.mean(row['block_cycle_fraction'][k] for row in rows) for k in names},
        original_mean_ms_per_layer=statistics.mean(statistics.mean(row['raw_original_ms']) for row in rows),
        instrumented_mean_ms_per_layer=statistics.mean(statistics.mean(row['raw_instrumented_ms']) for row in rows),
        scope='Real first decode query in all32 layers,64K batch1,5 repeated samples. clock64 per-block stage fractions include stalls and barriers; not additive device wall-time attribution. Clock instrumentation perturbs scheduling; compare instrumented/original CUDA event times. Formal serving timings unchanged.')
    (root/f'b{r}_r{r}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
