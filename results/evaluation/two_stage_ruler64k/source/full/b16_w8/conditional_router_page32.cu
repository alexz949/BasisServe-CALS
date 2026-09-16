#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>
#include <mma.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kQueriesPerKv = BASIS_GQA;
constexpr int kQueryKeyDim = 128;
constexpr int kValueRank = BASIS_VALUE_DIM;
constexpr int kBaseRank = BASIS_BASE_RANK;
constexpr int kResidualRank = BASIS_RESIDUAL_RANK;
constexpr int kPageSize = BASIS_PAGE_SIZE;
constexpr int kMaxPages = 8192;
constexpr int kMaxSelectedPages = 128;
constexpr int kBasePadded = ((kBaseRank + 15) / 16 > 0 ? (kBaseRank + 15) / 16 : 1) * 16;
constexpr int kHalfHeadDim = kQueryKeyDim / 2;
constexpr int kSharedQueryElements = 16 * kQueryKeyDim;
constexpr int kSharedScratchElements = kQueriesPerKv * kResidualRank;
constexpr int kSharedAccumulatorElements = kPageSize * kQueryKeyDim;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float round_bfloat16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void residual_query_code_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ residual_query,
    c10::BFloat16* __restrict__ query_code,
    int64_t query_heads,
    int64_t query_stride_batch,
    int64_t query_stride_head) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = row / query_heads;
  const int64_t query_head = row % query_heads;
  const int64_t query_base =
      batch * query_stride_batch + query_head * query_stride_head;

#pragma unroll
  for (int residual = 0; residual < kResidualRank; ++residual) {
    float partial = 0.0f;
#pragma unroll
    for (int feature = lane; feature < kQueryKeyDim; feature += kWarpSize) {
      partial = fmaf(
          static_cast<float>(query[query_base + feature]),
          static_cast<float>(
              residual_query[
                  (query_head * kQueryKeyDim + feature) * kResidualRank +
                  residual]),
          partial);
    }
    const float result = warp_sum(partial);
    if (lane == 0) {
      query_code[row * kResidualRank + residual] =
          static_cast<c10::BFloat16>(result);
    }
  }
}

__global__ void conditional_router_page_lse_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ base_code,
    const c10::BFloat16* __restrict__ residual_code,
    const c10::BFloat16* __restrict__ base_right,
    const c10::BFloat16* __restrict__ base_bias,
    const c10::BFloat16* __restrict__ rope_cos,
    const c10::BFloat16* __restrict__ rope_sin,
    const c10::BFloat16* __restrict__ query_code,
    float* __restrict__ output,
    int64_t kv_heads,
    int64_t tokens,
    int64_t pages,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t base_stride_batch,
    int64_t base_stride_head,
    int64_t base_stride_token,
    int64_t residual_stride_batch,
    int64_t residual_stride_head,
    int64_t residual_stride_token,
    int64_t rope_stride_token,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_query,
    float scale) {
// Experimental Page32 router: each warp owns 16 tokens. MMA D register pairs
// are adjacent RoPE coordinates, consumed before the next output tile.
  static_assert(kPageSize == 32 && (kBaseRank == 8 || kBaseRank == 16));
  constexpr int W = REGISTER_WARPS, T = W*16, P = T/32;
  __shared__ __align__(16) __nv_bfloat16 sb[T*kBaseRank];
  __shared__ __align__(16) __nv_bfloat16 sr[128*kBaseRank];
  __shared__ __nv_bfloat16 sq[kQueriesPerKv*128], bias[128];
  __shared__ __nv_bfloat16 rq[kQueriesPerKv*kResidualRank];
  __shared__ float scores[kQueriesPerKv*T];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, g=lane/4, u=lane%4;
  int64_t groups=(pages+P-1)/P, group=blockIdx.x%groups;
  int64_t kvrow=blockIdx.x/groups, kv_head=kvrow%kv_heads, batch=kvrow/kv_heads;
  int64_t start=group*T;
  for(int i=tid;i<T*kBaseRank;i+=W*32){
    int t=i/kBaseRank,r=i%kBaseRank;
    sb[i]=start+t<tokens ? static_cast<__nv_bfloat16>(base_code[batch*base_stride_batch+kv_head*base_stride_head+(start+t)*base_stride_token+r]) : __float2bfloat16(0.f);
  }
  // Coalesced reads of the original factor; transpose/pair only inside shared.
  for(int i=tid;i<kBaseRank*128;i+=W*32){
    int r=i/128,d=i%128,paircol=(d%64)*2+d/64;
    sr[paircol*kBaseRank+r]=static_cast<__nv_bfloat16>(base_right[kv_head*kBaseRank*128+i]);
  }
  for(int i=tid;i<kQueriesPerKv*128;i+=W*32){
    int h=i/128,d=i%128;
    sq[i]=static_cast<__nv_bfloat16>(query[batch*query_stride_batch+(kv_head*kQueriesPerKv+h)*query_stride_head+d]);
  }
  for(int i=tid;i<128;i+=W*32)bias[i]=static_cast<__nv_bfloat16>(base_bias[kv_head*128+i]);
  for(int i=tid;i<kQueriesPerKv*kResidualRank;i+=W*32)
    rq[i]=static_cast<__nv_bfloat16>(query_code[(batch*kv_heads+kv_head)*kQueriesPerKv*kResidualRank+i]);
  __syncthreads();
  int t0=warp*16+g,t1=t0+8;
  unsigned a0=*reinterpret_cast<unsigned*>(sb+t0*kBaseRank+u*2);
  unsigned a1=*reinterpret_cast<unsigned*>(sb+t1*kBaseRank+u*2);
#if BASIS_BASE_RANK == 16
  unsigned a2=*reinterpret_cast<unsigned*>(sb+t0*kBaseRank+u*2+8);
  unsigned a3=*reinterpret_cast<unsigned*>(sb+t1*kBaseRank+u*2+8);
#endif
  float sums[2][kQueriesPerKv]={};
#pragma unroll 1
  for(int j=0;j<16;++j){
    unsigned b0=*reinterpret_cast<unsigned*>(sr+(j*8+g)*kBaseRank+u*2);
    float d0,d1,d2,d3;
#if BASIS_BASE_RANK == 8
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
      : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
      : "r"(a0),"r"(a1),"r"(b0),"f"(0.f));
#else
    unsigned b1=*reinterpret_cast<unsigned*>(sr+(j*8+g)*kBaseRank+u*2+8);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(0.f));
#endif
    int f=j*4+u;
    float lo[2]={d0,d2}, hi[2]={d1,d3};
#pragma unroll
    for(int v=0;v<2;++v){
      int64_t token=start+t0+v*8;
      float a=round_bfloat16(round_bfloat16(lo[v])+__bfloat162float(bias[f]));
      float b=round_bfloat16(round_bfloat16(hi[v])+__bfloat162float(bias[f+64]));
      float c=token<tokens?static_cast<float>(rope_cos[token*rope_stride_token+f]):0.f;
      float s=token<tokens?static_cast<float>(rope_sin[token*rope_stride_token+f]):0.f;
      float x=round_bfloat16(round_bfloat16(a*c)-round_bfloat16(b*s));
      float y=round_bfloat16(round_bfloat16(b*c)+round_bfloat16(a*s));
#pragma unroll
      for(int h=0;h<kQueriesPerKv;++h){
        sums[v][h]=fmaf(x,__bfloat162float(sq[h*128+f]),sums[v][h]);
        sums[v][h]=fmaf(y,__bfloat162float(sq[h*128+f+64]),sums[v][h]);
      }
    }
  }
#pragma unroll
  for(int v=0;v<2;++v){
#pragma unroll
    for(int h=0;h<kQueriesPerKv;++h){
      float z=sums[v][h];
      z+=__shfl_xor_sync(0xffffffffu,z,1,4);
      z+=__shfl_xor_sync(0xffffffffu,z,2,4);
      if(u==0)scores[h*T+t0+v*8]=z;
    }
  }
  __syncthreads();
  for(int item=warp;item<P*kQueriesPerKv;item+=W){
    int p=item/kQueriesPerKv,h=item%kQueriesPerKv;
    int64_t page=group*P+p,token=start+p*32+lane;
    float score=-CUDART_INF_F;
    if(token<tokens){
      float r=0.f;
#pragma unroll
      for(int f=0;f<kResidualRank;++f)
        r=fmaf(__bfloat162float(rq[h*kResidualRank+f]),static_cast<float>(residual_code[batch*residual_stride_batch+kv_head*residual_stride_head+token*residual_stride_token+f]),r);
      score=round_bfloat16(round_bfloat16(round_bfloat16(scores[h*T+p*32+lane])+round_bfloat16(r))*scale);
    }
    float maximum=warp_max(score);
    float sum=warp_sum(__expf(score-maximum));
    if(lane==0 && page<pages)output[batch*output_stride_batch+kv_head*output_stride_head+h*output_stride_query+page]=maximum+__logf(sum);
  }

}

__global__ void conditional_router_append_decode_kernel(
    const c10::BFloat16* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    const c10::BFloat16* __restrict__ base_left,
    const c10::BFloat16* __restrict__ base_right,
    const c10::BFloat16* __restrict__ base_bias,
    const c10::BFloat16* __restrict__ residual_encoder,
    const c10::BFloat16* __restrict__ rope_cos,
    const c10::BFloat16* __restrict__ rope_sin,
    c10::BFloat16* __restrict__ value_cache,
    c10::BFloat16* __restrict__ base_cache,
    c10::BFloat16* __restrict__ residual_cache,
    c10::BFloat16* __restrict__ rope_cos_cache,
    c10::BFloat16* __restrict__ rope_sin_cache,
    int64_t kv_heads,
    int64_t start,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_cache_stride_batch,
    int64_t value_cache_stride_head,
    int64_t value_cache_stride_token,
    int64_t base_cache_stride_batch,
    int64_t base_cache_stride_head,
    int64_t base_cache_stride_token,
    int64_t residual_cache_stride_batch,
    int64_t residual_cache_stride_head,
    int64_t residual_cache_stride_token,
    int64_t rope_cache_stride_token,
    bool write_rope) {
  __shared__ __align__(16) __nv_bfloat16 shared_value[kValueRank];
  __shared__ __align__(16) __nv_bfloat16 shared_base[kBaseRank > 0 ? kBaseRank : 1];
  __shared__ __align__(16) __nv_bfloat16 shared_residual[kQueryKeyDim];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpSize;
  const int lane = thread % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t kv_head = row % kv_heads;
  const int64_t batch = row / kv_heads;
  const int64_t value_input_base =
      batch * value_stride_batch + kv_head * value_stride_head;
  const int64_t key_input_base =
      batch * key_stride_batch + kv_head * key_stride_head;

  for (int thread = threadIdx.x; thread < kValueRank; thread += kThreads) {
    const c10::BFloat16 loaded = value[value_input_base + thread];
    shared_value[thread] =
        *reinterpret_cast<const __nv_bfloat16*>(value + value_input_base + thread);
    value_cache[
        batch * value_cache_stride_batch + kv_head * value_cache_stride_head +
        start * value_cache_stride_token + thread] = loaded;
  }
  __syncthreads();

  if (thread < kBaseRank) {
    float accumulator = 0.0f;
#pragma unroll
    for (int feature = 0; feature < kValueRank; ++feature) {
      accumulator = fmaf(
          __bfloat162float(shared_value[feature]),
          static_cast<float>(
              base_left[
                  (kv_head * kValueRank + feature) * kBaseRank + thread]),
          accumulator);
    }
    const c10::BFloat16 result = static_cast<c10::BFloat16>(accumulator);
    shared_base[thread] = __float2bfloat16_rn(accumulator);
    base_cache[
        batch * base_cache_stride_batch + kv_head * base_cache_stride_head +
        start * base_cache_stride_token + thread] = result;
  }
  __syncthreads();

  if (thread < kHalfHeadDim) {
    float first_accumulator = 0.0f;
    float second_accumulator = 0.0f;
#pragma unroll
    for (int base = 0; base < kBaseRank; ++base) {
      const float coordinate = __bfloat162float(shared_base[base]);
      first_accumulator = fmaf(
          coordinate,
          static_cast<float>(
              base_right[
                  (kv_head * kBaseRank + base) * kQueryKeyDim + thread]),
          first_accumulator);
      second_accumulator = fmaf(
          coordinate,
          static_cast<float>(
              base_right[
                  (kv_head * kBaseRank + base) * kQueryKeyDim +
                  thread + kHalfHeadDim]),
          second_accumulator);
    }
    const float first_pre = round_bfloat16(
        round_bfloat16(first_accumulator) +
        static_cast<float>(base_bias[kv_head * kQueryKeyDim + thread]));
    const float second_pre = round_bfloat16(
        round_bfloat16(second_accumulator) +
        static_cast<float>(
            base_bias[
                kv_head * kQueryKeyDim + thread + kHalfHeadDim]));
    const float cosine = static_cast<float>(rope_cos[thread]);
    const float sine = static_cast<float>(rope_sin[thread]);
    const float first_post = round_bfloat16(
        round_bfloat16(first_pre * cosine) -
        round_bfloat16(second_pre * sine));
    const float second_post = round_bfloat16(
        round_bfloat16(second_pre * cosine) +
        round_bfloat16(first_pre * sine));
    const float first_residual = round_bfloat16(
        static_cast<float>(key[key_input_base + thread]) - first_post);
    const float second_residual = round_bfloat16(
        static_cast<float>(
            key[key_input_base + thread + kHalfHeadDim]) - second_post);
    shared_residual[thread] = __float2bfloat16_rn(first_residual);
    shared_residual[thread + kHalfHeadDim] =
        __float2bfloat16_rn(second_residual);
  }
  if (write_rope && row == 0 && thread < kHalfHeadDim) {
    rope_cos_cache[start * rope_cache_stride_token + thread] =
        rope_cos[thread];
    rope_sin_cache[start * rope_cache_stride_token + thread] =
        rope_sin[thread];
  }
  __syncthreads();

  for (int residual = warp; residual < kResidualRank; residual += kThreads / kWarpSize) {
    float partial = 0.0f;
#pragma unroll
    for (int feature = lane; feature < kQueryKeyDim; feature += kWarpSize) {
      partial = fmaf(
          __bfloat162float(shared_residual[feature]),
          static_cast<float>(
              residual_encoder[
                  (kv_head * kQueryKeyDim + feature) * kResidualRank + residual]),
          partial);
    }
    const float result = warp_sum(partial);
    if (lane == 0) {
      residual_cache[
          batch * residual_cache_stride_batch +
          kv_head * residual_cache_stride_head +
          start * residual_cache_stride_token + residual] =
          static_cast<c10::BFloat16>(result);
    }
  }
}

__global__ void fill_all_page_ids_kernel(
    int64_t* __restrict__ output,
    int64_t rows,
    int pages) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  for (int page = static_cast<int>(threadIdx.x); page < pages;
       page += kThreads) {
    output[row * pages + page] = page;
  }
}

template <int ItemsPerThread>
__global__ void select_fixed_group_max_pages_kernel(
    const float* __restrict__ page_log_mass,
    int64_t* __restrict__ output,
    int64_t kv_heads,
    int pages,
    int selected_count,
    int prefix_count,
    int fixed_count,
    bool force_current_page,
    int64_t input_stride_batch,
    int64_t input_stride_head,
    int64_t input_stride_query) {
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  using BlockSort =
      cub::BlockRadixSort<float, kThreads, ItemsPerThread, int>;
  union TemporaryStorage {
    typename BlockReduce::TempStorage reduce;
    typename BlockSort::TempStorage sort;
  };
  __shared__ TemporaryStorage temporary;
  __shared__ float head_maximum[kQueriesPerKv];
  __shared__ float head_inverse_sum[kQueriesPerKv];
  __shared__ int selected_pages[kMaxSelectedPages];

  const int thread = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = row / kv_heads;
  const int64_t kv_head = row % kv_heads;
  const int64_t input_base =
      batch * input_stride_batch + kv_head * input_stride_head;
  const int current_page = pages - 1;

  if (thread < prefix_count) {
    selected_pages[thread] = thread;
  }
  if (thread == 0 && fixed_count > prefix_count) {
    selected_pages[prefix_count] = current_page;
  }
  __syncthreads();

#pragma unroll
  for (int head = 0; head < kQueriesPerKv; ++head) {
    float local_maximum = -CUDART_INF_F;
    for (int page = thread; page < pages; page += kThreads) {
      const bool fixed = page < prefix_count ||
          (force_current_page && page == current_page);
      if (!fixed) {
        local_maximum = fmaxf(
            local_maximum,
            page_log_mass[
                input_base + head * input_stride_query + page]);
      }
    }
    const float maximum =
        BlockReduce(temporary.reduce).Reduce(local_maximum, cub::Max());
    if (thread == 0) {
      head_maximum[head] = maximum;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int page = thread; page < pages; page += kThreads) {
      const bool fixed = page < prefix_count ||
          (force_current_page && page == current_page);
      if (!fixed) {
        local_sum += __expf(
            page_log_mass[
                input_base + head * input_stride_query + page] -
            head_maximum[head]);
      }
    }
    const float sum = BlockReduce(temporary.reduce).Sum(local_sum);
    if (thread == 0) {
      head_inverse_sum[head] = 1.0f / sum;
    }
    __syncthreads();
  }

  float scores[ItemsPerThread];
  int page_ids[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int page = thread * ItemsPerThread + item;
    page_ids[item] = page;
    const bool fixed = page < prefix_count ||
        (force_current_page && page == current_page);
    float group_score = -CUDART_INF_F;
    if (page < pages && !fixed) {
      group_score = 0.0f;
#pragma unroll
      for (int head = 0; head < kQueriesPerKv; ++head) {
        const float probability = __expf(
            page_log_mass[
                input_base + head * input_stride_query + page] -
            head_maximum[head]) *
            head_inverse_sum[head];
        group_score = fmaxf(group_score, probability);
      }
    }
    scores[item] = group_score;
  }

  BlockSort(temporary.sort).SortDescending(scores, page_ids);
  const int remaining = selected_count - fixed_count;
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int rank = thread * ItemsPerThread + item;
    if (rank < remaining) {
      selected_pages[fixed_count + rank] = page_ids[item];
    }
  }
  __syncthreads();

  if (thread < kMaxSelectedPages && thread >= selected_count) {
    selected_pages[thread] = INT_MAX;
  }
  __syncthreads();
  for (int width = 2; width <= kMaxSelectedPages; width *= 2) {
    for (int stride = width / 2; stride > 0; stride /= 2) {
      if (thread < kMaxSelectedPages) {
        const int partner = thread ^ stride;
        if (partner > thread) {
          const int left = selected_pages[thread];
          const int right = selected_pages[partner];
          const bool ascending = (thread & width) == 0;
          if ((ascending && left > right) ||
              (!ascending && left < right)) {
            selected_pages[thread] = right;
            selected_pages[partner] = left;
          }
        }
      }
      __syncthreads();
    }
  }

  if (thread < selected_count) {
    output[row * selected_count + thread] = selected_pages[thread];
  }
}

}  // namespace

at::Tensor conditional_router_query_code_cuda(
    const at::Tensor& query, const at::Tensor& residual_query,
    const at::Tensor& query_code) {
  assert(query.is_cuda() && residual_query.device() == query.device() && query_code.device() == query.device());
  assert(query.scalar_type() == at::kBFloat16 && residual_query.scalar_type() == at::kBFloat16 && query_code.scalar_type() == at::kBFloat16);
  assert(query.dim() == 4 && query.size(2) == 1 && query.size(3) == kQueryKeyDim && query.stride(3) == 1);
  assert(residual_query.sizes() == at::IntArrayRef({query.size(1), kQueryKeyDim, kResidualRank}));
  assert(query_code.numel() == query.size(0) * query.size(1) * kResidualRank);
  assert(residual_query.is_contiguous() && query_code.is_contiguous());
  c10::cuda::CUDAGuard guard(query.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  residual_query_code_kernel
      <<<static_cast<unsigned int>(query.size(0) * query.size(1)),
         kWarpSize,
         0,
         stream>>>(
          query.const_data_ptr<c10::BFloat16>(),
          residual_query.const_data_ptr<c10::BFloat16>(),
          query_code.mutable_data_ptr<c10::BFloat16>(),
          query.size(1),
          query.stride(0),
          query.stride(1));
  assert(cudaGetLastError() == cudaSuccess);
  return query_code;
}

at::Tensor conditional_router_page_lse_cuda(
    const at::Tensor& query,
    const at::Tensor& base_code,
    const at::Tensor& residual_code,
    const at::Tensor& base_right,
    const at::Tensor& base_bias,
    const at::Tensor& residual_query,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    const at::Tensor& query_code,
    const at::Tensor& output,
    double scale, bool query_code_prepared) {
  assert(query.is_cuda() && base_code.is_cuda() && residual_code.is_cuda());
  assert(base_right.is_cuda() && base_bias.is_cuda());
  assert(residual_query.is_cuda() && rope_cos.is_cuda() && rope_sin.is_cuda());
  assert(query_code.is_cuda() && output.is_cuda());
  assert(query.scalar_type() == at::kBFloat16);
  assert(base_code.scalar_type() == at::kBFloat16);
  assert(residual_code.scalar_type() == at::kBFloat16);
  assert(base_right.scalar_type() == at::kBFloat16);
  assert(base_bias.scalar_type() == at::kBFloat16);
  assert(residual_query.scalar_type() == at::kBFloat16);
  assert(rope_cos.scalar_type() == at::kBFloat16);
  assert(rope_sin.scalar_type() == at::kBFloat16);
  assert(query_code.scalar_type() == at::kBFloat16);
  assert(output.scalar_type() == at::kFloat);
  assert(query.dim() == 4 && query.size(2) == 1 &&
         query.size(3) == kQueryKeyDim);
  assert(base_code.dim() == 4 && base_code.size(3) == kBaseRank);
  assert(residual_code.dim() == 4 &&
         residual_code.size(3) == kResidualRank);
  assert(base_code.size(0) == query.size(0));
  assert(base_code.size(1) * kQueriesPerKv == query.size(1));
  assert(residual_code.sizes() ==
         at::IntArrayRef(
             {base_code.size(0),
              base_code.size(1),
              base_code.size(2),
              kResidualRank}));
  assert(base_right.sizes() ==
         at::IntArrayRef({base_code.size(1), kBaseRank, kQueryKeyDim}));
  assert(base_bias.sizes() ==
         at::IntArrayRef({base_code.size(1), kQueryKeyDim}));
  assert(residual_query.sizes() ==
         at::IntArrayRef({query.size(1), kQueryKeyDim, kResidualRank}));
  assert(rope_cos.sizes() ==
         at::IntArrayRef({base_code.size(2), kHalfHeadDim}));
  assert(rope_sin.sizes() == rope_cos.sizes());
  const int64_t pages =
      (base_code.size(2) + kPageSize - 1) / kPageSize;
  assert(query_code.sizes() ==
         at::IntArrayRef(
             {base_code.size(0),
              base_code.size(1),
              kQueriesPerKv,
              kResidualRank}));
  assert(output.sizes() ==
         at::IntArrayRef(
             {base_code.size(0), base_code.size(1), kQueriesPerKv, pages}));
  assert(query.stride(3) == 1 && base_code.stride(3) == 1);
  assert(residual_code.stride(3) == 1);
  assert(base_right.is_contiguous());
  assert(base_bias.is_contiguous() && residual_query.is_contiguous());
  assert(rope_cos.stride(1) == 1 && rope_sin.stride(1) == 1);
  assert(query_code.is_contiguous() && output.stride(3) == 1);
  assert(std::isfinite(scale) && scale > 0.0);

  c10::cuda::CUDAGuard guard(query.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  if (!query_code_prepared) conditional_router_query_code_cuda(query, residual_query, query_code);
  constexpr int shared_bytes = 0;
  conditional_router_page_lse_kernel
      <<<static_cast<unsigned int>(
             base_code.size(0) * base_code.size(1) * ((pages+4-1)/4)),
         256,
         shared_bytes,
         stream>>>(
          query.const_data_ptr<c10::BFloat16>(),
          base_code.const_data_ptr<c10::BFloat16>(),
          residual_code.const_data_ptr<c10::BFloat16>(),
          base_right.const_data_ptr<c10::BFloat16>(),
          base_bias.const_data_ptr<c10::BFloat16>(),
          rope_cos.const_data_ptr<c10::BFloat16>(),
          rope_sin.const_data_ptr<c10::BFloat16>(),
          query_code.const_data_ptr<c10::BFloat16>(),
          output.mutable_data_ptr<float>(),
          base_code.size(1),
          base_code.size(2),
          pages,
          query.stride(0),
          query.stride(1),
          base_code.stride(0),
          base_code.stride(1),
          base_code.stride(2),
          residual_code.stride(0),
          residual_code.stride(1),
          residual_code.stride(2),
          rope_cos.stride(0),
          output.stride(0),
          output.stride(1),
          output.stride(2),
          static_cast<float>(scale));
  assert(cudaGetLastError() == cudaSuccess);
  return output;
}

void conditional_router_append_decode_cuda(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& base_left,
    const at::Tensor& base_right,
    const at::Tensor& base_bias,
    const at::Tensor& residual_encoder,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    const at::Tensor& value_cache,
    const at::Tensor& base_cache,
    const at::Tensor& residual_cache,
    const at::Tensor& rope_cos_cache,
    const at::Tensor& rope_sin_cache,
    int64_t start,
    bool write_rope) {
  assert(key.is_cuda() && value.is_cuda());
  assert(base_left.is_cuda() && base_right.is_cuda() && base_bias.is_cuda());
  assert(residual_encoder.is_cuda() && rope_cos.is_cuda() && rope_sin.is_cuda());
  assert(value_cache.is_cuda() && base_cache.is_cuda());
  assert(residual_cache.is_cuda());
  assert(rope_cos_cache.is_cuda() && rope_sin_cache.is_cuda());
  assert(key.scalar_type() == at::kBFloat16);
  assert(value.scalar_type() == at::kBFloat16);
  assert(base_left.scalar_type() == at::kBFloat16);
  assert(base_right.scalar_type() == at::kBFloat16);
  assert(base_bias.scalar_type() == at::kBFloat16);
  assert(residual_encoder.scalar_type() == at::kBFloat16);
  assert(rope_cos.scalar_type() == at::kBFloat16);
  assert(rope_sin.scalar_type() == at::kBFloat16);
  assert(value_cache.scalar_type() == at::kBFloat16);
  assert(base_cache.scalar_type() == at::kBFloat16);
  assert(residual_cache.scalar_type() == at::kBFloat16);
  assert(rope_cos_cache.scalar_type() == at::kBFloat16);
  assert(rope_sin_cache.scalar_type() == at::kBFloat16);
  assert(key.dim() == 4 && key.size(2) == 1 &&
         key.size(3) == kQueryKeyDim);
  assert(value.sizes() ==
         at::IntArrayRef({key.size(0), key.size(1), 1, kValueRank}));
  assert(base_left.sizes() ==
         at::IntArrayRef({key.size(1), kValueRank, kBaseRank}));
  assert(base_right.sizes() ==
         at::IntArrayRef({key.size(1), kBaseRank, kQueryKeyDim}));
  assert(base_bias.sizes() ==
         at::IntArrayRef({key.size(1), kQueryKeyDim}));
  assert(residual_encoder.sizes() ==
         at::IntArrayRef({key.size(1), kQueryKeyDim, kResidualRank}));
  assert(rope_cos.sizes() == at::IntArrayRef({1, kHalfHeadDim}));
  assert(rope_sin.sizes() == rope_cos.sizes());
  assert(value_cache.dim() == 4 && value_cache.size(0) == key.size(0));
  assert(value_cache.size(1) == key.size(1) &&
         value_cache.size(3) == kValueRank);
  const int64_t capacity = value_cache.size(2);
  assert(base_cache.sizes() ==
         at::IntArrayRef(
             {key.size(0), key.size(1), capacity, kBaseRank}));
  assert(residual_cache.sizes() ==
         at::IntArrayRef(
             {key.size(0), key.size(1), capacity, kResidualRank}));
  assert(rope_cos_cache.sizes() ==
         at::IntArrayRef({capacity, kHalfHeadDim}));
  assert(rope_sin_cache.sizes() == rope_cos_cache.sizes());
  assert(start >= 0 && start < capacity);
  assert(key.stride(3) == 1 && value.stride(3) == 1);
  assert(value_cache.stride(3) == 1 && base_cache.stride(3) == 1);
  assert(residual_cache.stride(3) == 1);
  assert(rope_cos.stride(1) == 1 && rope_sin.stride(1) == 1);
  assert(rope_cos_cache.stride(1) == 1 && rope_sin_cache.stride(1) == 1);
  assert(base_left.is_contiguous() && base_right.is_contiguous());
  assert(base_bias.is_contiguous() && residual_encoder.is_contiguous());

  c10::cuda::CUDAGuard guard(key.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(key.get_device()).stream();
  conditional_router_append_decode_kernel
      <<<static_cast<unsigned int>(key.size(0) * key.size(1)),
         kThreads,
         0,
         stream>>>(
          key.const_data_ptr<c10::BFloat16>(),
          value.const_data_ptr<c10::BFloat16>(),
          base_left.const_data_ptr<c10::BFloat16>(),
          base_right.const_data_ptr<c10::BFloat16>(),
          base_bias.const_data_ptr<c10::BFloat16>(),
          residual_encoder.const_data_ptr<c10::BFloat16>(),
          rope_cos.const_data_ptr<c10::BFloat16>(),
          rope_sin.const_data_ptr<c10::BFloat16>(),
          value_cache.mutable_data_ptr<c10::BFloat16>(),
          base_cache.mutable_data_ptr<c10::BFloat16>(),
          residual_cache.mutable_data_ptr<c10::BFloat16>(),
          rope_cos_cache.mutable_data_ptr<c10::BFloat16>(),
          rope_sin_cache.mutable_data_ptr<c10::BFloat16>(),
          key.size(1),
          start,
          key.stride(0),
          key.stride(1),
          value.stride(0),
          value.stride(1),
          value_cache.stride(0),
          value_cache.stride(1),
          value_cache.stride(2),
          base_cache.stride(0),
          base_cache.stride(1),
          base_cache.stride(2),
          residual_cache.stride(0),
          residual_cache.stride(1),
          residual_cache.stride(2),
          rope_cos_cache.stride(0),
          write_rope);
  assert(cudaGetLastError() == cudaSuccess);
}

at::Tensor select_fixed_group_max_pages_cuda(
    const at::Tensor& page_log_mass,
    const at::Tensor& output,
    int64_t pages_per_kv_head,
    int64_t pinned_prefix_pages,
    bool force_current_page) {
  assert(page_log_mass.is_cuda() && output.is_cuda());
  assert(page_log_mass.device() == output.device());
  assert(page_log_mass.scalar_type() == at::kFloat);
  assert(output.scalar_type() == at::kLong);
  assert(page_log_mass.dim() == 4);
  assert(page_log_mass.size(2) == kQueriesPerKv);
  const int64_t pages = page_log_mass.size(3);
  assert(pages > 0 && pages <= kMaxPages);
  assert(pages_per_kv_head > 0);
  const int64_t selected_count = std::min(pages_per_kv_head, pages);
  assert(selected_count <= kMaxSelectedPages);
  const int64_t prefix_count = std::min(pinned_prefix_pages, pages);
  assert(prefix_count >= 0);
  const int64_t fixed_count = prefix_count +
      ((force_current_page && pages - 1 >= prefix_count) ? 1 : 0);
  assert(fixed_count <= selected_count);
  assert(output.sizes() ==
         at::IntArrayRef(
             {page_log_mass.size(0),
              page_log_mass.size(1),
              selected_count}));
  assert(page_log_mass.stride(3) == 1 && output.is_contiguous());

  c10::cuda::CUDAGuard guard(page_log_mass.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
                                  page_log_mass.get_device())
                                  .stream();
  const int64_t rows = page_log_mass.size(0) * page_log_mass.size(1);
  if (selected_count == pages) {
    fill_all_page_ids_kernel
        <<<static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
            output.mutable_data_ptr<int64_t>(), rows, pages);
  } else if (pages <= kThreads * 8) {
    select_fixed_group_max_pages_kernel<8>
        <<<static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
            page_log_mass.const_data_ptr<float>(),
            output.mutable_data_ptr<int64_t>(),
            page_log_mass.size(1),
            pages,
            selected_count,
            prefix_count,
            fixed_count,
            force_current_page,
            page_log_mass.stride(0),
            page_log_mass.stride(1),
            page_log_mass.stride(2));
  } else if (pages <= kThreads * 16) {
    select_fixed_group_max_pages_kernel<16>
        <<<static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
            page_log_mass.const_data_ptr<float>(),
            output.mutable_data_ptr<int64_t>(),
            page_log_mass.size(1),
            pages,
            selected_count,
            prefix_count,
            fixed_count,
            force_current_page,
            page_log_mass.stride(0),
            page_log_mass.stride(1),
            page_log_mass.stride(2));
  } else {
    select_fixed_group_max_pages_kernel<32>
        <<<static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
            page_log_mass.const_data_ptr<float>(),
            output.mutable_data_ptr<int64_t>(),
            page_log_mass.size(1),
            pages,
            selected_count,
            prefix_count,
            fixed_count,
            force_current_page,
            page_log_mass.stride(0),
            page_log_mass.stride(1),
            page_log_mass.stride(2));
  }
  assert(cudaGetLastError() == cudaSuccess);
  return output;
}
