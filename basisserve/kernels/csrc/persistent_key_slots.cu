#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cassert>
#include <cstdint>

__global__ void plan_slots(const int64_t* ids,int* lookup,int64_t* resident,
                          int64_t* selected_slots,int* missing,int* counts,
                          int capacity,int budget,bool reuse){
  __shared__ uint8_t used[2048];
  __shared__ int free_slots[2048],requested[2048];
  __shared__ int free_count,miss_count,hit_count,valid_count;
  int row=blockIdx.x,thread=threadIdx.x;
  if(thread==0){free_count=0;miss_count=0;hit_count=0;valid_count=0;}
  for(int i=thread;i<budget;i+=blockDim.x)used[i]=0;
  __syncthreads();
  for(int i=thread;i<budget;i+=blockDim.x){
    int64_t token=ids[row*budget+i];
    int slot=token>=0 && token<capacity && reuse ? lookup[row*capacity+token] : -1;
    bool hit=slot>=0 && slot<budget && resident[row*budget+slot]==token;
    requested[i]=hit?slot:-1;
    if(hit){used[slot]=1;atomicAdd(&hit_count,1);}
    if(token>=0 && token<capacity)atomicAdd(&valid_count,1);
  }
  __syncthreads();
  for(int i=thread;i<budget;i+=blockDim.x){
    if(!used[i]){
      int index=atomicAdd(&free_count,1);
      free_slots[index]=i;
    }
  }
  __syncthreads();
  for(int i=thread;i<budget;i+=blockDim.x){
    int64_t token=ids[row*budget+i];int slot=requested[i];int miss=-1;
    if(token>=0 && token<capacity){
      if(slot<0){slot=free_slots[atomicAdd(&miss_count,1)];miss=token;}
      resident[row*budget+slot]=token;lookup[row*capacity+token]=slot;
    }else slot=-1;
    selected_slots[row*budget+i]=slot;missing[row*budget+i]=miss;
  }
  if(thread==0){counts[row*2]=hit_count;counts[row*2+1]=valid_count;}
}

__global__ void fetch_missing(const __nv_bfloat16* host,__nv_bfloat16* cache,
                             const int* missing,const int64_t* slots,
                             int capacity,int budget,int rows){
  constexpr int kVectorsPerKey=16;
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=rows*budget*kVectorsPerKey)return;
  int selected=index/kVectorsPerKey,vector=index%kVectorsPerKey;
  int row=selected/budget,token=missing[selected];
  if(token>=0){
    int64_t slot=slots[selected];
    const uint4* host_vector=reinterpret_cast<const uint4*>(
        host+(static_cast<int64_t>(row)*capacity+token)*128);
    uint4* cache_vector=reinterpret_cast<uint4*>(
        cache+(static_cast<int64_t>(row)*budget+slot)*128);
    cache_vector[vector]=host_vector[vector];
  }
}

void refresh(int64_t pointer,const at::Tensor& cache,const at::Tensor& ids,
             const at::Tensor& resident,const at::Tensor& lookup,const at::Tensor& slots,
             const at::Tensor& missing,const at::Tensor& counts,bool reuse){
  c10::cuda::CUDAGuard guard(cache.device());
  int rows=ids.size(0)*ids.size(1),budget=ids.size(2),capacity=lookup.size(2);
  assert(budget<=2048 && budget>0 && cache.size(3)==128);
  assert(cache.is_contiguous() && ids.is_contiguous());
  assert(cache.numel()==rows*budget*128 && resident.numel()==rows*budget);
  assert(slots.numel()==rows*budget && missing.numel()==rows*budget);
  auto stream=c10::cuda::getCurrentCUDAStream().stream();
  plan_slots<<<rows,256,0,stream>>>(ids.data_ptr<int64_t>(),lookup.data_ptr<int>(),
      resident.data_ptr<int64_t>(),slots.data_ptr<int64_t>(),
      missing.data_ptr<int>(),counts.data_ptr<int>(),capacity,budget,reuse);
  constexpr int kThreads=256,kVectorsPerKey=16;
  int vectors=rows*budget*kVectorsPerKey;
  fetch_missing<<<(vectors+kThreads-1)/kThreads,kThreads,0,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(pointer),
      reinterpret_cast<__nv_bfloat16*>(cache.data_ptr<at::BFloat16>()),
      missing.data_ptr<int>(),slots.data_ptr<int64_t>(),capacity,budget,rows);
  assert(cudaGetLastError()==cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("refresh",&refresh);}
