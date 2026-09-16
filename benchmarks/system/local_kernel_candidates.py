"""Build isolated A/B candidates without changing serving cache policy."""
import hashlib
from pathlib import Path
import subprocess
from torch.utils.cpp_extension import load


REFERENCE_COMMIT = '0d1c847d7799983dcd309ec2b9f4e486cfc6b526'


def reference_source(name):
    return subprocess.check_output(['git', 'show', f'{REFERENCE_COMMIT}:basisserve/kernels/csrc/{name}'], text=True)


def write_source(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def compile_slots(root, vector):
    source = reference_source('persistent_key_slots.cu')
    if vector:
        start = source.index('  int i=blockIdx.x*blockDim.x+threadIdx.x;', source.index('__global__ void fetch_missing'))
        end = source.index('\n}', start)
        source = source[:start] + '''  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rows*budget*16)return;
  int selected=i/16,vector=i%16,row=selected/budget,token=missing[selected];
  if(token>=0){
    const uint4* src=reinterpret_cast<const uint4*>(host)+(static_cast<int64_t>(row)*capacity+token)*16+vector;
    uint4* dst=reinterpret_cast<uint4*>(cache)+(static_cast<int64_t>(row)*budget+slots[selected])*16+vector;
    *dst=*src;
  }''' + source[end:]
        source = source.replace('(rows*budget*128+255)/256', '(rows*budget*16+255)/256')
    vectors = 16 if vector else 128
    source = source.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("refresh",&refresh);}', f'''
void fetch_only(int64_t pointer,const at::Tensor& cache,const at::Tensor& missing,const at::Tensor& slots,int capacity){{
  c10::cuda::CUDAGuard guard(cache.device());
  int rows=cache.size(0)*cache.size(1),budget=cache.size(2);
  fetch_missing<<<(rows*budget*{vectors}+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>(
    reinterpret_cast<const __nv_bfloat16*>(pointer),reinterpret_cast<__nv_bfloat16*>(cache.data_ptr<at::BFloat16>()),
    missing.data_ptr<int>(),slots.data_ptr<int64_t>(),capacity,budget,rows);
  assert(cudaGetLastError()==cudaSuccess);
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){{m.def("refresh",&refresh);m.def("fetch_only",&fetch_only);}}
''')
    path = root / ('slots_vector.cu' if vector else 'slots_scalar.cu')
    write_source(path, source)
    digest = hashlib.sha256(source.encode()).hexdigest()[:10]
    return load(name=f'local_slots_{digest}', sources=[str(path)],
                extra_cflags=['-O3', '-std=c++17'], extra_cuda_cflags=['-O3', '-std=c++17'])


def compile_router(root, rank, reference=False):
    src = Path('basisserve/kernels/csrc')
    text = reference_source('conditional_router_page32.cu')
    original = text
    start = text.index('  for (int index = thread; index < kPageSize * kHalfHeadDim;', text.index('__global__ void conditional_router_page_lse_kernel'))
    end = text.index('  for (int head = warp;', start)
    fused = '''  // Reconstruction above is unchanged. Each warp handles a token and shares
  // its rounded RoPE coordinates across the actual query heads.
  __nv_bfloat16* shared_query_code = shared_scratch;
  for (int index = thread; index < kQueriesPerKv*kResidualRank; index += kThreads)
    shared_query_code[index] = *reinterpret_cast<const __nv_bfloat16*>(query_code +
      (batch*kv_heads*kQueriesPerKv+kv_head*kQueriesPerKv)*kResidualRank+index);
  for (int page_token=warp; page_token<kPageSize; page_token+=kThreads/kWarpSize) {
    const int64_t token=page_start+page_token;
    float key_first[2],key_second[2];
#pragma unroll
    for(int piece=0;piece<2;++piece){
      int feature=lane+piece*32;
      float a=round_bfloat16(round_bfloat16(shared_accumulator[page_token*kQueryKeyDim+feature])+
          static_cast<float>(base_bias[kv_head*kQueryKeyDim+feature]));
      float b=round_bfloat16(round_bfloat16(shared_accumulator[page_token*kQueryKeyDim+feature+64])+
          static_cast<float>(base_bias[kv_head*kQueryKeyDim+feature+64]));
      float c=token<tokens?static_cast<float>(rope_cos[token*rope_stride_token+feature]):0.f;
      float s=token<tokens?static_cast<float>(rope_sin[token*rope_stride_token+feature]):0.f;
      key_first[piece]=round_bfloat16(round_bfloat16(a*c)-round_bfloat16(b*s));
      key_second[piece]=round_bfloat16(round_bfloat16(b*c)+round_bfloat16(a*s));
    }
#pragma unroll
    for(int head=0;head<kQueriesPerKv;++head){
      const int64_t qbase=batch*query_stride_batch+(kv_head*kQueriesPerKv+head)*query_stride_head;
      float dot=0.f;
#pragma unroll
      for(int piece=0;piece<2;++piece){
        int feature=lane+piece*32;
        dot=fmaf(key_first[piece],static_cast<float>(query[qbase+feature]),dot);
        dot=fmaf(key_second[piece],static_cast<float>(query[qbase+feature+64]),dot);
      }
      dot=warp_sum(dot);
      if(lane==0)shared_scores[head*kPageSize+page_token]=dot;
    }
  }
  __syncthreads();

'''
    text = text[:start] + fused + text[end:]
    # Keep the WMMA reconstruction layout; remove unused Q and rotated-K storage.
    text = text.replace('constexpr int kSharedScratchElements =\n    kSharedQueryElements + kQueriesPerKv * kResidualRank;',
                        'constexpr int kSharedScratchElements = kQueriesPerKv * kResidualRank;')
    text = text.replace('  auto* shared_key = shared_right + kBasePadded * kQueryKeyDim;\n', '')
    text = text.replace(' + kBasePadded * kQueryKeyDim + kPageSize * kQueryKeyDim);', ' + kBasePadded * kQueryKeyDim);')
    if reference: text = original
    folder = root / f'{"reference_" if reference else ""}router_b{rank}'; folder.mkdir(parents=True, exist_ok=True)
    write_source(folder / 'conditional_router_page32.cu', text)
    for name in ['mapped_host_paged_attention.cpp', 'mapped_host_paged_attention.cu']:
        write_source(folder / name, reference_source(name))
    digest = hashlib.sha256(text.encode()).hexdigest()[:10]
    return load(name=f'local_router_{digest}_b{rank}',
        sources=[str(folder / name) for name in ['mapped_host_paged_attention.cpp','mapped_host_paged_attention.cu','conditional_router_page32.cu']],
        extra_cflags=['-O3', '-std=c++17'], extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
            '-DBASIS_VALUE_DIM=80', '-DBASIS_GQA=4', '-DBASIS_PAGE_SIZE=32',
            f'-DBASIS_BASE_RANK={rank}', f'-DBASIS_RESIDUAL_RANK={rank}'])
