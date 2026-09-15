#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cassert>

__global__ void fetch_keys(const __nv_bfloat16* host, const __nv_bfloat16* old_key,
                          __nv_bfloat16* next_key, const __nv_bfloat16* values,
                          __nv_bfloat16* next_value, const int64_t* old_ids,
                          const int64_t* ids, const int* slots, int* hits,
                          int capacity, int budget, int rows, bool reuse) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rows*budget*128)return;
  int feature=i%128, selected=i/128, row=selected/budget;
  int64_t token=ids[selected];
  bool valid=token>=0 && token<capacity;
  int slot=valid && reuse ? slots[row*capacity+token] : -1;
  bool hit=slot>=0 && slot<budget && old_ids[row*budget+slot]==token;
  __nv_bfloat16 key=__float2bfloat16(0),value=__float2bfloat16(0);
  if(valid){
    key=hit ? old_key[(row*budget+slot)*128+feature] : host[(row*capacity+token)*128+feature];
    value=values[(row*capacity+token)*128+feature];
  }
  next_key[i]=key;next_value[i]=value;
  if(feature==0 && hit)atomicAdd(hits+row,1);
}

__global__ void update_slots(const int64_t* ids,int* slots,int capacity,int budget,int count){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count){int64_t token=ids[i];if(token>=0 && token<capacity)slots[(i/budget)*capacity+token]=i%budget;}
}

void refresh(int64_t pointer,const at::Tensor& old_key,const at::Tensor& next_key,
             const at::Tensor& values,const at::Tensor& next_value,const at::Tensor& old_ids,
             const at::Tensor& ids,const at::Tensor& slots,const at::Tensor& hits,bool reuse){
  c10::cuda::CUDAGuard guard(next_key.device());
  int capacity=values.size(2),budget=ids.size(2),rows=ids.size(0)*ids.size(1);
  assert(values.is_contiguous() && ids.is_contiguous() && old_ids.is_contiguous());
  assert(values.scalar_type()==at::kBFloat16 && ids.scalar_type()==at::kLong);
  assert(next_key.numel()==rows*budget*128 && old_key.sizes()==next_key.sizes());
  auto stream=c10::cuda::getCurrentCUDAStream().stream();
  fetch_keys<<<(rows*budget*128+255)/256,256,0,stream>>>(
    reinterpret_cast<const __nv_bfloat16*>(pointer),
    reinterpret_cast<const __nv_bfloat16*>(old_key.data_ptr<at::BFloat16>()),
    reinterpret_cast<__nv_bfloat16*>(next_key.data_ptr<at::BFloat16>()),
    reinterpret_cast<const __nv_bfloat16*>(values.data_ptr<at::BFloat16>()),
    reinterpret_cast<__nv_bfloat16*>(next_value.data_ptr<at::BFloat16>()),
    old_ids.data_ptr<int64_t>(),ids.data_ptr<int64_t>(),slots.data_ptr<int>(),hits.data_ptr<int>(),capacity,budget,rows,reuse);
  update_slots<<<(rows*budget+255)/256,256,0,stream>>>(ids.data_ptr<int64_t>(),slots.data_ptr<int>(),capacity,budget,rows*budget);
  assert(cudaGetLastError()==cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("refresh",&refresh);}
