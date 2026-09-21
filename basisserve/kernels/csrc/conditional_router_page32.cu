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

#if !defined(BASIS_DISABLE_REGISTER_ROUTER) && \
    BASIS_PAGE_SIZE == 32 && BASIS_GQA == 4 && \
    (BASIS_BASE_RANK == 8 || BASIS_BASE_RANK == 16)
// Page32/GQA4 decode router. Each warp owns 16 tokens and consumes the MMA
// accumulator from registers before advancing to the next output tile. This
// avoids materializing the reconstructed K128 page in FP32 shared memory.
__global__ void conditional_router_page_lse_register_kernel(
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int kRegisterWarps = 8;
  constexpr int kTokensPerBlock = kRegisterWarps * 16;
  constexpr int kPagesPerBlock = kTokensPerBlock / kPageSize;
  __shared__ __align__(16) __nv_bfloat16 shared_base[
      kTokensPerBlock * kBaseRank];
  __shared__ __align__(16) __nv_bfloat16 shared_right[
      kQueryKeyDim * kBaseRank];
  __shared__ __nv_bfloat16 shared_query[
      kQueriesPerKv * kQueryKeyDim];
  __shared__ __nv_bfloat16 shared_bias[kQueryKeyDim];
  __shared__ __nv_bfloat16 shared_residual_query[
      kQueriesPerKv * kResidualRank];
  __shared__ float shared_scores[kQueriesPerKv * kTokensPerBlock];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpSize;
  const int lane = thread % kWarpSize;
  const int lane_group = lane / 4;
  const int lane_in_group = lane % 4;
  const int64_t page_groups =
      (pages + kPagesPerBlock - 1) / kPagesPerBlock;
  const int64_t page_group = static_cast<int64_t>(blockIdx.x) % page_groups;
  const int64_t kv_row = static_cast<int64_t>(blockIdx.x) / page_groups;
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t token_start = page_group * kTokensPerBlock;

  for (int index = thread; index < kTokensPerBlock * kBaseRank;
       index += kRegisterWarps * kWarpSize) {
    const int token_in_block = index / kBaseRank;
    const int rank = index % kBaseRank;
    const int64_t token = token_start + token_in_block;
    shared_base[index] = token < tokens
        ? static_cast<__nv_bfloat16>(
              base_code[batch * base_stride_batch +
                        kv_head * base_stride_head +
                        token * base_stride_token + rank])
        : __float2bfloat16_rn(0.0f);
  }
  // Keep global reads coalesced, then transpose and RoPE-pair in shared.
  for (int index = thread; index < kBaseRank * kQueryKeyDim;
       index += kRegisterWarps * kWarpSize) {
    const int rank = index / kQueryKeyDim;
    const int feature = index % kQueryKeyDim;
    const int paired_feature = (feature % kHalfHeadDim) * 2 +
        feature / kHalfHeadDim;
    shared_right[paired_feature * kBaseRank + rank] =
        static_cast<__nv_bfloat16>(
            base_right[kv_head * kBaseRank * kQueryKeyDim + index]);
  }
  for (int index = thread; index < kQueriesPerKv * kQueryKeyDim;
       index += kRegisterWarps * kWarpSize) {
    const int query_head = index / kQueryKeyDim;
    const int feature = index % kQueryKeyDim;
    shared_query[index] = static_cast<__nv_bfloat16>(
        query[batch * query_stride_batch +
              (kv_head * kQueriesPerKv + query_head) * query_stride_head +
              feature]);
  }
  for (int index = thread; index < kQueryKeyDim;
       index += kRegisterWarps * kWarpSize) {
    shared_bias[index] = static_cast<__nv_bfloat16>(
        base_bias[kv_head * kQueryKeyDim + index]);
  }
  for (int index = thread;
       index < kQueriesPerKv * kResidualRank;
       index += kRegisterWarps * kWarpSize) {
    shared_residual_query[index] = static_cast<__nv_bfloat16>(
        query_code[(batch * kv_heads + kv_head) *
                       kQueriesPerKv * kResidualRank +
                   index]);
  }
  __syncthreads();

  const int token_zero = warp * 16 + lane_group;
  const int token_one = token_zero + 8;
  const unsigned left_zero = *reinterpret_cast<unsigned*>(
      shared_base + token_zero * kBaseRank + lane_in_group * 2);
  const unsigned left_one = *reinterpret_cast<unsigned*>(
      shared_base + token_one * kBaseRank + lane_in_group * 2);
#if BASIS_BASE_RANK == 16
  const unsigned left_zero_high = *reinterpret_cast<unsigned*>(
      shared_base + token_zero * kBaseRank + lane_in_group * 2 + 8);
  const unsigned left_one_high = *reinterpret_cast<unsigned*>(
      shared_base + token_one * kBaseRank + lane_in_group * 2 + 8);
#endif
  float dot_sums[2][kQueriesPerKv] = {};
#pragma unroll 1
  for (int feature_group = 0; feature_group < 16; ++feature_group) {
    const unsigned right_low = *reinterpret_cast<unsigned*>(
        shared_right +
        (feature_group * 8 + lane_group) * kBaseRank +
        lane_in_group * 2);
    float accumulator_zero_low;
    float accumulator_zero_high;
    float accumulator_one_low;
    float accumulator_one_high;
#if BASIS_BASE_RANK == 8
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
        : "=f"(accumulator_zero_low), "=f"(accumulator_zero_high),
          "=f"(accumulator_one_low), "=f"(accumulator_one_high)
        : "r"(left_zero), "r"(left_one), "r"(right_low), "f"(0.0f));
#else
    const unsigned right_high = *reinterpret_cast<unsigned*>(
        shared_right +
        (feature_group * 8 + lane_group) * kBaseRank +
        lane_in_group * 2 + 8);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(accumulator_zero_low), "=f"(accumulator_zero_high),
          "=f"(accumulator_one_low), "=f"(accumulator_one_high)
        : "r"(left_zero), "r"(left_one),
          "r"(left_zero_high), "r"(left_one_high),
          "r"(right_low), "r"(right_high), "f"(0.0f));
#endif
    const int feature = feature_group * 4 + lane_in_group;
    const float reconstructed_low[2] = {
        accumulator_zero_low, accumulator_one_low};
    const float reconstructed_high[2] = {
        accumulator_zero_high, accumulator_one_high};
#pragma unroll
    for (int token_pair = 0; token_pair < 2; ++token_pair) {
      const int64_t token = token_start + token_zero + token_pair * 8;
      const float biased_low = round_bfloat16(
          round_bfloat16(reconstructed_low[token_pair]) +
          __bfloat162float(shared_bias[feature]));
      const float biased_high = round_bfloat16(
          round_bfloat16(reconstructed_high[token_pair]) +
          __bfloat162float(shared_bias[feature + kHalfHeadDim]));
      const float cosine = token < tokens
          ? static_cast<float>(rope_cos[token * rope_stride_token + feature])
          : 0.0f;
      const float sine = token < tokens
          ? static_cast<float>(rope_sin[token * rope_stride_token + feature])
          : 0.0f;
      const float rotated_low = round_bfloat16(
          round_bfloat16(biased_low * cosine) -
          round_bfloat16(biased_high * sine));
      const float rotated_high = round_bfloat16(
          round_bfloat16(biased_high * cosine) +
          round_bfloat16(biased_low * sine));
#pragma unroll
      for (int query_head = 0; query_head < kQueriesPerKv; ++query_head) {
        dot_sums[token_pair][query_head] = fmaf(
            rotated_low,
            __bfloat162float(
                shared_query[query_head * kQueryKeyDim + feature]),
            dot_sums[token_pair][query_head]);
        dot_sums[token_pair][query_head] = fmaf(
            rotated_high,
            __bfloat162float(
                shared_query[query_head * kQueryKeyDim + feature +
                             kHalfHeadDim]),
            dot_sums[token_pair][query_head]);
      }
    }
  }
#pragma unroll
  for (int token_pair = 0; token_pair < 2; ++token_pair) {
#pragma unroll
    for (int query_head = 0; query_head < kQueriesPerKv; ++query_head) {
      float score = dot_sums[token_pair][query_head];
      score += __shfl_xor_sync(0xffffffffu, score, 1, 4);
      score += __shfl_xor_sync(0xffffffffu, score, 2, 4);
      if (lane_in_group == 0) {
        shared_scores[query_head * kTokensPerBlock +
                      token_zero + token_pair * 8] = score;
      }
    }
  }
  __syncthreads();

  for (int item = warp; item < kPagesPerBlock * kQueriesPerKv;
       item += kRegisterWarps) {
    const int page_in_block = item / kQueriesPerKv;
    const int query_head = item % kQueriesPerKv;
    const int64_t page = page_group * kPagesPerBlock + page_in_block;
    const int64_t token =
        token_start + page_in_block * kPageSize + lane;
    float score = -CUDART_INF_F;
    if (token < tokens) {
      float residual_score = 0.0f;
#pragma unroll
      for (int residual = 0; residual < kResidualRank; ++residual) {
        residual_score = fmaf(
            __bfloat162float(
                shared_residual_query[query_head * kResidualRank + residual]),
            static_cast<float>(
                residual_code[batch * residual_stride_batch +
                              kv_head * residual_stride_head +
                              token * residual_stride_token + residual]),
            residual_score);
      }
      score = round_bfloat16(
          round_bfloat16(
              round_bfloat16(
                  shared_scores[query_head * kTokensPerBlock +
                                page_in_block * kPageSize + lane]) +
              round_bfloat16(residual_score)) *
          scale);
    }
    const float maximum = warp_max(score);
    const float sum = warp_sum(__expf(score - maximum));
    if (lane == 0 && page < pages) {
      output[batch * output_stride_batch +
             kv_head * output_stride_head +
             query_head * output_stride_query + page] =
          maximum + __logf(sum);
    }
  }
#endif
}
#endif

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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(16) unsigned char storage[];
  auto* shared_accumulator = reinterpret_cast<float*>(storage);
  auto* shared_scores = shared_accumulator + kSharedAccumulatorElements;
  auto* shared_scratch = reinterpret_cast<__nv_bfloat16*>(shared_scores + 16 * kPageSize);
  auto* shared_base = shared_scratch + ((kSharedScratchElements + 7) / 8) * 8;
  auto* shared_right = shared_base + kPageSize * kBasePadded;

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpSize;
  const int lane = thread % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t page = row % pages;
  const int64_t kv_row = row / pages;
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t page_start = page * kPageSize;

  for (int index = thread; index < kPageSize * kBasePadded;
       index += kThreads) {
    const int page_token = index / kBasePadded;
    const int feature = index % kBasePadded;
    const int64_t token = page_start + page_token;
    __nv_bfloat16 loaded = __float2bfloat16_rn(0.0f);
    if (token < tokens && feature < kBaseRank) {
      const int64_t source = batch * base_stride_batch +
          kv_head * base_stride_head + token * base_stride_token + feature;
      loaded = *reinterpret_cast<const __nv_bfloat16*>(base_code + source);
    }
    shared_base[index] = loaded;
  }
  for (int index = thread; index < kBasePadded * kQueryKeyDim; index += kThreads) {
    shared_right[index] = index / kQueryKeyDim < kBaseRank
        ? *reinterpret_cast<const __nv_bfloat16*>(base_right + kv_head * kBaseRank * kQueryKeyDim + index)
        : __float2bfloat16_rn(0.0f);
  }
  __syncthreads();

  {
    using namespace nvcuda;
    const int output_column = warp * 16;
#pragma unroll
    for (int output_row = 0; output_row < kPageSize; output_row += 16) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
      wmma::fragment<
          wmma::matrix_a,
          16,
          16,
          16,
          __nv_bfloat16,
          wmma::row_major>
          left;
      wmma::fragment<
          wmma::matrix_b,
          16,
          16,
          16,
          __nv_bfloat16,
          wmma::row_major>
          right;
      wmma::fill_fragment(accumulator, 0.0f);
      for (int inner = 0; inner < kBasePadded; inner += 16) {
        wmma::load_matrix_sync(left, shared_base + output_row * kBasePadded + inner, kBasePadded);
        wmma::load_matrix_sync(right, shared_right + inner * kQueryKeyDim + output_column, kQueryKeyDim);
        wmma::mma_sync(accumulator, left, right, accumulator);
      }
      wmma::store_matrix_sync(
          shared_accumulator + output_row * kQueryKeyDim + output_column,
          accumulator,
          kQueryKeyDim,
          wmma::mem_row_major);
    }
  }
  __syncthreads();

  // Reconstruction above is unchanged. Each warp handles a token and shares
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

  for (int head = warp; head < kQueriesPerKv; head += kThreads / kWarpSize) {
    float scores[(kPageSize + 31) / 32];
    float local_max = -CUDART_INF_F;
#pragma unroll
    for (int slot = 0; slot < (kPageSize + 31) / 32; ++slot) {
      const int page_token = lane + slot * 32;
      const int64_t token = page_start + page_token;
      float score = -CUDART_INF_F;
      if (page_token < kPageSize && token < tokens) {
        float residual_score = 0.0f;
        for (int residual = 0; residual < kResidualRank; ++residual) {
          residual_score = fmaf(__bfloat162float(shared_query_code[head * kResidualRank + residual]),
              static_cast<float>(residual_code[batch * residual_stride_batch + kv_head * residual_stride_head + token * residual_stride_token + residual]), residual_score);
        }
        score = round_bfloat16(round_bfloat16(round_bfloat16(shared_scores[head * kPageSize + page_token]) + round_bfloat16(residual_score)) * scale);
      }
      scores[slot] = score;
      local_max = fmaxf(local_max, score);
    }
    const float maximum = warp_max(local_max);
    float local_sum = 0.0f;
    for (int slot = 0; slot < (kPageSize + 31) / 32; ++slot) local_sum += __expf(scores[slot] - maximum);
    const float sum = warp_sum(local_sum);
    if (lane == 0) output[batch * output_stride_batch + kv_head * output_stride_head + head * output_stride_query + page] = maximum + __logf(sum);
  }

#endif
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
    c10::BFloat16* __restrict__ mapped_key,
    int64_t mapped_key_capacity,
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

  if (mapped_key != nullptr && thread < kQueryKeyDim / 8) {
    const auto* source = reinterpret_cast<const uint4*>(
        key + key_input_base);
    auto* destination = reinterpret_cast<uint4*>(
        mapped_key +
        ((batch * kv_heads + kv_head) * mapped_key_capacity + start) *
            kQueryKeyDim);
    destination[thread] = source[thread];
  }

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
#if !defined(BASIS_DISABLE_REGISTER_ROUTER) && \
    BASIS_PAGE_SIZE == 32 && BASIS_GQA == 4 && \
    (BASIS_BASE_RANK == 8 || BASIS_BASE_RANK == 16)
  constexpr int kRegisterWarps = 8;
  constexpr int kRegisterPagesPerBlock = kRegisterWarps / 2;
  conditional_router_page_lse_register_kernel
      <<<static_cast<unsigned int>(
             base_code.size(0) * base_code.size(1) *
             ((pages + kRegisterPagesPerBlock - 1) /
              kRegisterPagesPerBlock)),
         kRegisterWarps * kWarpSize,
         0,
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
#else
  constexpr int shared_bytes = sizeof(float) * (kSharedAccumulatorElements + 16 * kPageSize)
      + sizeof(__nv_bfloat16) * (((kSharedScratchElements + 7) / 8) * 8 + kPageSize * kBasePadded + kBasePadded * kQueryKeyDim);
  assert(cudaFuncSetAttribute(conditional_router_page_lse_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes) == cudaSuccess);
  conditional_router_page_lse_kernel
      <<<static_cast<unsigned int>(
             base_code.size(0) * base_code.size(1) * pages),
         kThreads,
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
#endif
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
    bool write_rope,
    int64_t mapped_key_device_pointer,
    int64_t mapped_key_capacity) {
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
  assert(rope_cos_cache.sizes() == at::IntArrayRef(
      {write_rope ? capacity : 1, kHalfHeadDim}));
  assert(rope_sin_cache.sizes() == rope_cos_cache.sizes());
  assert(start >= 0 && start < capacity);
  assert(mapped_key_device_pointer == 0 || mapped_key_capacity == capacity);
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
          reinterpret_cast<c10::BFloat16*>(mapped_key_device_pointer),
          mapped_key_capacity,
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
