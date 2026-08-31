#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>

#include "compressed_v_decode_attention_traits.cuh"

namespace {

using basisserve::compressed_v_decode::Sm80Traits;
using basisserve::compressed_v_decode::Sm89Traits;
using basisserve::compressed_v_decode::Sm90Traits;

constexpr int kHardwareWarpSize = 32;
constexpr int kQueryKeyDim = 128;
constexpr int kVectorBytes = sizeof(uint4);
constexpr int kHalfElementsPerVector = kVectorBytes / sizeof(uint16_t);
constexpr int kGqa8QueriesPerKv = 8;
constexpr int kGqa8ValueDim = 64;
constexpr int kGqa8WarpsPerBlock = kGqa8QueriesPerKv;
constexpr int kGqa8ThreadsPerBlock = kGqa8WarpsPerBlock * kHardwareWarpSize;
constexpr int kGqa8TokenTile = 8;
constexpr int kGqa4QueriesPerKv = 4;
constexpr int kGqa4ValueDim = 96;
constexpr int kGqa4WarpsPerBlock = kGqa4QueriesPerKv;
constexpr int kGqa4ThreadsPerBlock = kGqa4WarpsPerBlock * kHardwareWarpSize;
constexpr int kGqa4TokenTile = 4;
constexpr int kSparsePageSize = 64;

template <typename scalar_t>
__device__ __forceinline__ float2 packed_pair_to_float2(uint32_t bits);

template <>
__device__ __forceinline__ float2 packed_pair_to_float2<c10::Half>(
    uint32_t bits) {
  return make_float2(
      __half2float(__ushort_as_half(static_cast<uint16_t>(bits))),
      __half2float(__ushort_as_half(static_cast<uint16_t>(bits >> 16))));
}

template <>
__device__ __forceinline__ float2 packed_pair_to_float2<c10::BFloat16>(
    uint32_t bits) {
  return make_float2(
      __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(bits))),
      __bfloat162float(
          __ushort_as_bfloat16(static_cast<uint16_t>(bits >> 16))));
}

__device__ __forceinline__ uint32_t packed_word(uint4 vector, int index) {
  switch (index) {
    case 0:
      return vector.x;
    case 1:
      return vector.y;
    case 2:
      return vector.z;
    default:
      return vector.w;
  }
}

template <typename scalar_t>
__device__ __forceinline__ float accumulate_packed_eight(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float accumulator) {
  const uint4 query_vector = *reinterpret_cast<const uint4*>(query);
  const uint4 key_vector = *reinterpret_cast<const uint4*>(key);
#pragma unroll
  for (int pair = 0; pair < kHalfElementsPerVector / 2; ++pair) {
    const float2 query_pair =
        packed_pair_to_float2<scalar_t>(packed_word(query_vector, pair));
    const float2 key_pair =
        packed_pair_to_float2<scalar_t>(packed_word(key_vector, pair));
    accumulator = fmaf(query_pair.x, key_pair.x, accumulator);
    accumulator = fmaf(query_pair.y, key_pair.y, accumulator);
  }
  return accumulator;
}

template <typename scalar_t>
__device__ __forceinline__ void accumulate_packed_value_eight(
    const scalar_t* __restrict__ value,
    float probability,
    float* __restrict__ accumulators) {
  const uint4 value_vector = *reinterpret_cast<const uint4*>(value);
#pragma unroll
  for (int pair = 0; pair < kHalfElementsPerVector / 2; ++pair) {
    const float2 value_pair =
        packed_pair_to_float2<scalar_t>(packed_word(value_vector, pair));
    accumulators[2 * pair] =
        fmaf(probability, value_pair.x, accumulators[2 * pair]);
    accumulators[2 * pair + 1] =
        fmaf(probability, value_pair.y, accumulators[2 * pair + 1]);
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kHardwareWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = kHardwareWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float subwarp8_sum(float value) {
#pragma unroll
  for (int offset = 4; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset, 8);
  }
  return value;
}

template <
    typename Architecture,
    typename scalar_t,
    int kValueDim,
    bool kDirectOutput>
__global__ void vectorized_splitk_partial_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    float* __restrict__ workspace,
    scalar_t* __restrict__ output,
    int64_t query_heads,
    int64_t queries_per_kv,
    int64_t sequence_length,
    int64_t splits,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    float scale) {
  constexpr int kWarpSize = Architecture::kWarpSize;
  constexpr int kWarpsPerBlock = Architecture::kWarpsPerBlock;
  constexpr int kTokensPerWarpStep = Architecture::kTokensPerWarpStep;
  static_assert(kWarpSize == kHardwareWarpSize);
  static_assert(kTokensPerWarpStep == 4);
  static_assert(kValueDim % kHalfElementsPerVector == 0);
  constexpr bool kVectorizedValue =
      kValueDim == 64 || kValueDim == 80 || kValueDim == 112;
  constexpr int kValueVectors = kValueDim / kHalfElementsPerVector;
  constexpr int kScalarOutputs =
      (kValueDim + kWarpSize - 1) / kWarpSize;
  constexpr int kAccumulatorCount =
      kVectorizedValue ? kHalfElementsPerVector : kScalarOutputs;
  __shared__ float warp_maxima[kWarpsPerBlock];
  __shared__ float warp_sums[kWarpsPerBlock];
  __shared__ float warp_accumulators[kWarpsPerBlock][kValueDim];
  __shared__ float merge_weights[kWarpsPerBlock];
  __shared__ float merged_sum;
  __shared__ __align__(kVectorBytes) scalar_t shared_query[kQueryKeyDim];

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int subwarp = lane / 8;
  const int sublane = lane % 8;
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int split = static_cast<int>(linear % splits);
  const int64_t row = linear / splits;
  const int query_head = static_cast<int>(row % query_heads);
  const int64_t batch = row / query_heads;
  const int kv_head = query_head / queries_per_kv;

  constexpr int kQueryVectors = kQueryKeyDim / kHalfElementsPerVector;
  if (threadIdx.x < kQueryVectors) {
    const int feature = static_cast<int>(threadIdx.x) * kHalfElementsPerVector;
    const int64_t query_base = batch * query_stride_batch +
        static_cast<int64_t>(query_head) * query_stride_head + feature;
    reinterpret_cast<uint4*>(shared_query)[threadIdx.x] =
        *reinterpret_cast<const uint4*>(query + query_base);
  }
  __syncthreads();

  const int64_t split_start = sequence_length * split / splits;
  const int64_t split_stop = sequence_length * (split + 1) / splits;
  const int64_t split_length = split_stop - split_start;
  const int64_t warp_start =
      split_start + split_length * warp / kWarpsPerBlock;
  const int64_t warp_stop =
      split_start + split_length * (warp + 1) / kWarpsPerBlock;

  float accumulator[kAccumulatorCount];
#pragma unroll
  for (int item = 0; item < kAccumulatorCount; ++item) {
    accumulator[item] = 0.0f;
  }
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  for (int64_t cursor = warp_start; cursor < warp_stop;
       cursor += kTokensPerWarpStep) {
    const int64_t token = cursor + subwarp;
    float dot = 0.0f;
    if (token < warp_stop) {
      constexpr int kFeaturesPerSubwarpLane = kQueryKeyDim / 8;
      const int feature = sublane * kFeaturesPerSubwarpLane;
      const int64_t key_base = batch * key_stride_batch +
          static_cast<int64_t>(kv_head) * key_stride_head +
          token * key_stride_token + feature;
      dot = accumulate_packed_eight(
          shared_query + feature,
          key + key_base,
          dot);
      dot = accumulate_packed_eight(
          shared_query + feature + kHalfElementsPerVector,
          key + key_base + kHalfElementsPerVector,
          dot);
    }
    dot = subwarp8_sum(dot);

    const bool token_valid = token < warp_stop;
    const float owned_score = sublane == 0 && token_valid
        ? dot * scale
        : -CUDART_INF_F;
    const float block_max = warp_max(owned_score);
    const float next_max = fmaxf(running_max, block_max);
    float previous_scale = lane == 0 ? __expf(running_max - next_max) : 0.0f;
    previous_scale = __shfl_sync(0xffffffffu, previous_scale, 0);

    const float owned_probability = sublane == 0 && token_valid
        ? __expf(owned_score - next_max)
        : 0.0f;
    const float probability0 =
        __shfl_sync(0xffffffffu, owned_probability, 0);
    const float probability1 =
        __shfl_sync(0xffffffffu, owned_probability, 8);
    const float probability2 =
        __shfl_sync(0xffffffffu, owned_probability, 16);
    const float probability3 =
        __shfl_sync(0xffffffffu, owned_probability, 24);
    const float block_sum = warp_sum(owned_probability);

    if constexpr (kVectorizedValue) {
      if (lane < kValueVectors) {
#pragma unroll
        for (int item = 0; item < kHalfElementsPerVector; ++item) {
          accumulator[item] *= previous_scale;
        }
        const int feature = lane * kHalfElementsPerVector;
        const int64_t value_base = batch * value_stride_batch +
            static_cast<int64_t>(kv_head) * value_stride_head +
            cursor * value_stride_token + feature;
        accumulate_packed_value_eight(
            value + value_base,
            probability0,
            accumulator);
        if (cursor + 1 < warp_stop) {
          accumulate_packed_value_eight(
              value + value_base + value_stride_token,
              probability1,
              accumulator);
        }
        if (cursor + 2 < warp_stop) {
          accumulate_packed_value_eight(
              value + value_base + 2 * value_stride_token,
              probability2,
              accumulator);
        }
        if (cursor + 3 < warp_stop) {
          accumulate_packed_value_eight(
              value + value_base + 3 * value_stride_token,
              probability3,
              accumulator);
        }
      }
    } else {
#pragma unroll
      for (int item = 0; item < kScalarOutputs; ++item) {
        const int feature = lane + item * kWarpSize;
        if (feature < kValueDim) {
          float partial = accumulator[item] * previous_scale;
          const int64_t value_base = batch * value_stride_batch +
              static_cast<int64_t>(kv_head) * value_stride_head +
              cursor * value_stride_token + feature;
          partial = fmaf(
              probability0,
              static_cast<float>(value[value_base]),
              partial);
          if (cursor + 1 < warp_stop) {
            partial = fmaf(
                probability1,
                static_cast<float>(value[value_base + value_stride_token]),
                partial);
          }
          if (cursor + 2 < warp_stop) {
            partial = fmaf(
                probability2,
                static_cast<float>(value[value_base + 2 * value_stride_token]),
                partial);
          }
          if (cursor + 3 < warp_stop) {
            partial = fmaf(
                probability3,
                static_cast<float>(value[value_base + 3 * value_stride_token]),
                partial);
          }
          accumulator[item] = partial;
        }
      }
    }
    running_sum = running_sum * previous_scale + block_sum;
    running_max = next_max;
  }

  if (lane == 0) {
    warp_maxima[warp] = running_max;
    warp_sums[warp] = running_sum;
  }
  if constexpr (kVectorizedValue) {
    if (lane < kValueVectors) {
#pragma unroll
      for (int item = 0; item < kHalfElementsPerVector; ++item) {
        const int feature = lane * kHalfElementsPerVector + item;
        warp_accumulators[warp][feature] = accumulator[item];
      }
    }
  } else {
#pragma unroll
    for (int item = 0; item < kScalarOutputs; ++item) {
      const int feature = lane + item * kWarpSize;
      if (feature < kValueDim) {
        warp_accumulators[warp][feature] = accumulator[item];
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    if (lane == 0) {
      float maximum = warp_maxima[0];
#pragma unroll
      for (int source = 1; source < kWarpsPerBlock; ++source) {
        maximum = fmaxf(maximum, warp_maxima[source]);
      }
      float total = 0.0f;
#pragma unroll
      for (int source = 0; source < kWarpsPerBlock; ++source) {
        const float weight = __expf(warp_maxima[source] - maximum);
        merge_weights[source] = weight;
        total += weight * warp_sums[source];
      }
      if constexpr (kDirectOutput) {
        merged_sum = total;
      } else {
        const int64_t base = (row * splits + split) * (kValueDim + 2);
        workspace[base + kValueDim] = maximum;
        workspace[base + kValueDim + 1] = total;
      }
    }
    __syncwarp();
    if constexpr (kVectorizedValue) {
#pragma unroll
      for (int vector = lane; vector < kValueVectors; vector += kWarpSize) {
#pragma unroll
        for (int item = 0; item < kHalfElementsPerVector; ++item) {
          const int feature = vector * kHalfElementsPerVector + item;
          float combined = 0.0f;
#pragma unroll
          for (int source = 0; source < kWarpsPerBlock; ++source) {
            combined = fmaf(
                merge_weights[source],
                warp_accumulators[source][feature],
                combined);
          }
          if constexpr (kDirectOutput) {
            const int64_t output_offset = batch * output_stride_batch +
                static_cast<int64_t>(query_head) * output_stride_head +
                feature * output_stride_feature;
            output[output_offset] =
                static_cast<scalar_t>(combined / merged_sum);
          } else {
            const int64_t base = (row * splits + split) * (kValueDim + 2);
            workspace[base + feature] = combined;
          }
        }
      }
    } else {
#pragma unroll
      for (int item = 0; item < kScalarOutputs; ++item) {
        const int feature = lane + item * kWarpSize;
        if (feature < kValueDim) {
          float combined = 0.0f;
#pragma unroll
          for (int source = 0; source < kWarpsPerBlock; ++source) {
            combined = fmaf(
                merge_weights[source],
                warp_accumulators[source][feature],
                combined);
          }
          if constexpr (kDirectOutput) {
            const int64_t output_offset = batch * output_stride_batch +
                static_cast<int64_t>(query_head) * output_stride_head +
                feature * output_stride_feature;
            output[output_offset] =
                static_cast<scalar_t>(combined / merged_sum);
          } else {
            const int64_t base = (row * splits + split) * (kValueDim + 2);
            workspace[base + feature] = combined;
          }
        }
      }
    }
  }
}

template <typename scalar_t, bool kDirectOutput>
__global__ void gqa8_v64_shared_kv_splitk_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    float* __restrict__ workspace,
    scalar_t* __restrict__ output,
    int64_t kv_heads,
    int64_t sequence_length,
    int64_t splits,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    float scale) {
  constexpr int kQueryVectors = kQueryKeyDim / kHalfElementsPerVector;
  constexpr int kValueVectors = kGqa8ValueDim / kHalfElementsPerVector;
  __shared__ __align__(kVectorBytes)
      scalar_t shared_query[kGqa8QueriesPerKv][kQueryKeyDim];
  __shared__ __align__(kVectorBytes)
      scalar_t shared_key[kGqa8TokenTile][kQueryKeyDim];
  __shared__ __align__(kVectorBytes)
      scalar_t shared_value[kGqa8TokenTile][kGqa8ValueDim];

  const int warp = static_cast<int>(threadIdx.x) / kHardwareWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kHardwareWarpSize;
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int split = static_cast<int>(linear % splits);
  const int64_t kv_row = linear / splits;
  const int kv_head = static_cast<int>(kv_row % kv_heads);
  const int64_t batch = kv_row / kv_heads;
  const int query_head = kv_head * kGqa8QueriesPerKv + warp;
  const int64_t query_heads = kv_heads * kGqa8QueriesPerKv;

  const int query_vector = static_cast<int>(threadIdx.x);
  if (query_vector < kGqa8QueriesPerKv * kQueryVectors) {
    const int head = query_vector / kQueryVectors;
    const int vector = query_vector % kQueryVectors;
    const int feature = vector * kHalfElementsPerVector;
    const int64_t query_base = batch * query_stride_batch +
        static_cast<int64_t>(kv_head * kGqa8QueriesPerKv + head) *
            query_stride_head +
        feature;
    reinterpret_cast<uint4*>(shared_query)[query_vector] =
        *reinterpret_cast<const uint4*>(query + query_base);
  }
  __syncthreads();

  const int64_t split_start = sequence_length * split / splits;
  const int64_t split_stop = sequence_length * (split + 1) / splits;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  for (int64_t tile_start = split_start; tile_start < split_stop;
       tile_start += kGqa8TokenTile) {
    const int key_vector = static_cast<int>(threadIdx.x);
    if (key_vector < kGqa8TokenTile * kQueryVectors) {
      const int token = key_vector / kQueryVectors;
      const int vector = key_vector % kQueryVectors;
      const int64_t source_token = tile_start + token;
      uint4 loaded = make_uint4(0, 0, 0, 0);
      if (source_token < split_stop) {
        const int64_t key_base = batch * key_stride_batch +
            static_cast<int64_t>(kv_head) * key_stride_head +
            source_token * key_stride_token +
            vector * kHalfElementsPerVector;
        loaded = *reinterpret_cast<const uint4*>(key + key_base);
      }
      reinterpret_cast<uint4*>(shared_key)[key_vector] = loaded;
    }

    const int value_thread = static_cast<int>(threadIdx.x) -
        kGqa8TokenTile * kQueryVectors;
    if (value_thread >= 0 &&
        value_thread < kGqa8TokenTile * kValueVectors) {
      const int token = value_thread / kValueVectors;
      const int vector = value_thread % kValueVectors;
      const int64_t source_token = tile_start + token;
      uint4 loaded = make_uint4(0, 0, 0, 0);
      if (source_token < split_stop) {
        const int64_t value_base = batch * value_stride_batch +
            static_cast<int64_t>(kv_head) * value_stride_head +
            source_token * value_stride_token +
            vector * kHalfElementsPerVector;
        loaded = *reinterpret_cast<const uint4*>(value + value_base);
      }
      reinterpret_cast<uint4*>(shared_value)[value_thread] = loaded;
    }
    __syncthreads();

#pragma unroll
    for (int token = 0; token < kGqa8TokenTile; ++token) {
      if (tile_start + token < split_stop) {
        float dot = 0.0f;
        if (lane < kQueryVectors) {
          const int feature = lane * kHalfElementsPerVector;
          dot = accumulate_packed_eight(
              shared_query[warp] + feature,
              shared_key[token] + feature,
              dot);
        }
        dot = warp_sum(dot);
        const float score = dot * scale;
        const float next_max = fmaxf(running_max, score);
        const float previous_scale = __expf(running_max - next_max);
        const float probability = __expf(score - next_max);
        accumulator0 = fmaf(
            probability,
            static_cast<float>(shared_value[token][lane]),
            accumulator0 * previous_scale);
        accumulator1 = fmaf(
            probability,
            static_cast<float>(
                shared_value[token][lane + kHardwareWarpSize]),
            accumulator1 * previous_scale);
        running_sum = running_sum * previous_scale + probability;
        running_max = next_max;
      }
    }
    __syncthreads();
  }

  if constexpr (kDirectOutput) {
    const float inverse_sum = 1.0f / running_sum;
    const int64_t output_base = batch * output_stride_batch +
        static_cast<int64_t>(query_head) * output_stride_head;
    output[output_base + lane * output_stride_feature] =
        static_cast<scalar_t>(accumulator0 * inverse_sum);
    output[output_base +
           (lane + kHardwareWarpSize) * output_stride_feature] =
        static_cast<scalar_t>(accumulator1 * inverse_sum);
  } else {
    const int64_t row = batch * query_heads + query_head;
    const int64_t base =
        (row * splits + split) * (kGqa8ValueDim + 2);
    workspace[base + lane] = accumulator0;
    workspace[base + lane + kHardwareWarpSize] = accumulator1;
    if (lane == 0) {
      workspace[base + kGqa8ValueDim] = running_max;
      workspace[base + kGqa8ValueDim + 1] = running_sum;
    }
  }
}

template <typename scalar_t, bool kDirectOutput>
__global__ void gqa4_v96_dense_shared_kv_splitk_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ valid_sequence_length,
    float* __restrict__ workspace,
    scalar_t* __restrict__ output,
    int64_t kv_heads,
    int64_t cache_length,
    int64_t splits,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    float scale) {
  constexpr int kQueryVectors = kQueryKeyDim / kHalfElementsPerVector;
  constexpr int kValueVectors = kGqa4ValueDim / kHalfElementsPerVector;
  __shared__ __align__(kVectorBytes)
      scalar_t shared_query[kGqa4QueriesPerKv][kQueryKeyDim];
  __shared__ __align__(kVectorBytes)
      scalar_t shared_key[kGqa4TokenTile][kQueryKeyDim];
  __shared__ __align__(kVectorBytes)
      scalar_t shared_value[kGqa4TokenTile][kGqa4ValueDim];

  const int warp = static_cast<int>(threadIdx.x) / kHardwareWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kHardwareWarpSize;
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int split = static_cast<int>(linear % splits);
  const int64_t kv_row = linear / splits;
  const int kv_head = static_cast<int>(kv_row % kv_heads);
  const int64_t batch = kv_row / kv_heads;
  const int query_head = kv_head * kGqa4QueriesPerKv + warp;
  const int64_t query_heads = kv_heads * kGqa4QueriesPerKv;

  const int query_vector = static_cast<int>(threadIdx.x);
  if (query_vector < kGqa4QueriesPerKv * kQueryVectors) {
    const int head = query_vector / kQueryVectors;
    const int vector = query_vector % kQueryVectors;
    const int feature = vector * kHalfElementsPerVector;
    const int64_t query_base = batch * query_stride_batch +
        static_cast<int64_t>(kv_head * kGqa4QueriesPerKv + head) *
            query_stride_head +
        feature;
    reinterpret_cast<uint4*>(shared_query)[query_vector] =
        *reinterpret_cast<const uint4*>(query + query_base);
  }
  __syncthreads();

  const int64_t requested_length = *valid_sequence_length;
  const int64_t sequence_length =
      requested_length < cache_length ? requested_length : cache_length;
  const int64_t split_start = sequence_length * split / splits;
  const int64_t split_stop = sequence_length * (split + 1) / splits;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  for (int64_t tile_start = split_start; tile_start < split_stop;
       tile_start += kGqa4TokenTile) {
    const int key_vector = static_cast<int>(threadIdx.x);
    if (key_vector < kGqa4TokenTile * kQueryVectors) {
      const int token = key_vector / kQueryVectors;
      const int vector = key_vector % kQueryVectors;
      const int64_t source_token = tile_start + token;
      uint4 loaded = make_uint4(0, 0, 0, 0);
      if (source_token < split_stop) {
        const int64_t key_base = batch * key_stride_batch +
            static_cast<int64_t>(kv_head) * key_stride_head +
            source_token * key_stride_token +
            vector * kHalfElementsPerVector;
        loaded = *reinterpret_cast<const uint4*>(key + key_base);
      }
      reinterpret_cast<uint4*>(shared_key)[key_vector] = loaded;
    }

    const int value_thread = static_cast<int>(threadIdx.x) -
        kGqa4TokenTile * kQueryVectors;
    if (value_thread >= 0 &&
        value_thread < kGqa4TokenTile * kValueVectors) {
      const int token = value_thread / kValueVectors;
      const int vector = value_thread % kValueVectors;
      const int64_t source_token = tile_start + token;
      uint4 loaded = make_uint4(0, 0, 0, 0);
      if (source_token < split_stop) {
        const int64_t value_base = batch * value_stride_batch +
            static_cast<int64_t>(kv_head) * value_stride_head +
            source_token * value_stride_token +
            vector * kHalfElementsPerVector;
        loaded = *reinterpret_cast<const uint4*>(value + value_base);
      }
      reinterpret_cast<uint4*>(shared_value)[value_thread] = loaded;
    }
    __syncthreads();

#pragma unroll
    for (int token = 0; token < kGqa4TokenTile; ++token) {
      if (tile_start + token < split_stop) {
        float dot = 0.0f;
        if (lane < kQueryVectors) {
          const int feature = lane * kHalfElementsPerVector;
          dot = accumulate_packed_eight(
              shared_query[warp] + feature,
              shared_key[token] + feature,
              dot);
        }
        dot = warp_sum(dot);
        const float score = dot * scale;
        const float next_max = fmaxf(running_max, score);
        const float previous_scale = __expf(running_max - next_max);
        const float probability = __expf(score - next_max);
        accumulator0 = fmaf(
            probability,
            static_cast<float>(shared_value[token][lane]),
            accumulator0 * previous_scale);
        accumulator1 = fmaf(
            probability,
            static_cast<float>(
                shared_value[token][lane + kHardwareWarpSize]),
            accumulator1 * previous_scale);
        accumulator2 = fmaf(
            probability,
            static_cast<float>(
                shared_value[token][lane + 2 * kHardwareWarpSize]),
            accumulator2 * previous_scale);
        running_sum = running_sum * previous_scale + probability;
        running_max = next_max;
      }
    }
    __syncthreads();
  }

  if constexpr (kDirectOutput) {
    const float inverse_sum = 1.0f / running_sum;
    const int64_t output_base = batch * output_stride_batch +
        static_cast<int64_t>(query_head) * output_stride_head;
    output[output_base + lane * output_stride_feature] =
        static_cast<scalar_t>(accumulator0 * inverse_sum);
    output[output_base +
           (lane + kHardwareWarpSize) * output_stride_feature] =
        static_cast<scalar_t>(accumulator1 * inverse_sum);
    output[output_base +
           (lane + 2 * kHardwareWarpSize) * output_stride_feature] =
        static_cast<scalar_t>(accumulator2 * inverse_sum);
  } else {
    const int64_t row = batch * query_heads + query_head;
    const int64_t base =
        (row * splits + split) * (kGqa4ValueDim + 2);
    workspace[base + lane] = accumulator0;
    workspace[base + lane + kHardwareWarpSize] = accumulator1;
    workspace[base + lane + 2 * kHardwareWarpSize] = accumulator2;
    if (lane == 0) {
      workspace[base + kGqa4ValueDim] = running_max;
      workspace[base + kGqa4ValueDim + 1] = running_sum;
    }
  }
}

template <typename scalar_t, bool kDirectOutput>
__global__ void gqa4_v96_paged_sparse_splitk_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ packed_key_pages,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ selected_page_ids,
    float* __restrict__ workspace,
    scalar_t* __restrict__ output,
    int64_t kv_heads,
    int64_t selected_page_slots,
    int64_t sequence_length,
    int64_t splits,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_page,
    int64_t key_stride_token,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t page_stride_batch,
    int64_t page_stride_head,
    int64_t page_stride_slot,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    float scale) {
  constexpr int kQueryVectors = kQueryKeyDim / kHalfElementsPerVector;
  constexpr int kValueVectors = kGqa4ValueDim / kHalfElementsPerVector;
  __shared__ __align__(kVectorBytes)
      scalar_t shared_query[kGqa4QueriesPerKv][kQueryKeyDim];
  __shared__ __align__(kVectorBytes)
      scalar_t shared_key[kGqa4TokenTile][kQueryKeyDim];
  __shared__ __align__(kVectorBytes)
      scalar_t shared_value[kGqa4TokenTile][kGqa4ValueDim];

  const int warp = static_cast<int>(threadIdx.x) / kHardwareWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kHardwareWarpSize;
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int split = static_cast<int>(linear % splits);
  const int64_t kv_row = linear / splits;
  const int kv_head = static_cast<int>(kv_row % kv_heads);
  const int64_t batch = kv_row / kv_heads;
  const int query_head = kv_head * kGqa4QueriesPerKv + warp;
  const int64_t query_heads = kv_heads * kGqa4QueriesPerKv;

  const int query_vector = static_cast<int>(threadIdx.x);
  if (query_vector < kGqa4QueriesPerKv * kQueryVectors) {
    const int head = query_vector / kQueryVectors;
    const int vector = query_vector % kQueryVectors;
    const int feature = vector * kHalfElementsPerVector;
    const int64_t query_base = batch * query_stride_batch +
        static_cast<int64_t>(kv_head * kGqa4QueriesPerKv + head) *
            query_stride_head +
        feature;
    reinterpret_cast<uint4*>(shared_query)[query_vector] =
        *reinterpret_cast<const uint4*>(query + query_base);
  }
  __syncthreads();

  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  for (int64_t page_slot = split; page_slot < selected_page_slots;
       page_slot += splits) {
    const int64_t page_id = selected_page_ids[
        batch * page_stride_batch +
        static_cast<int64_t>(kv_head) * page_stride_head +
        page_slot * page_stride_slot];
    if (page_id < 0) {
      continue;
    }
    const int64_t page_token_start = page_id * kSparsePageSize;
    for (int tile_start = 0; tile_start < kSparsePageSize;
         tile_start += kGqa4TokenTile) {
      const int key_vector = static_cast<int>(threadIdx.x);
      if (key_vector < kGqa4TokenTile * kQueryVectors) {
        const int token = key_vector / kQueryVectors;
        const int vector = key_vector % kQueryVectors;
        const int64_t source_token = page_token_start + tile_start + token;
        uint4 loaded = make_uint4(0, 0, 0, 0);
        if (source_token < sequence_length) {
          const int64_t key_base = batch * key_stride_batch +
              static_cast<int64_t>(kv_head) * key_stride_head +
              page_slot * key_stride_page +
              static_cast<int64_t>(tile_start + token) * key_stride_token +
              vector * kHalfElementsPerVector;
          loaded = *reinterpret_cast<const uint4*>(
              packed_key_pages + key_base);
        }
        reinterpret_cast<uint4*>(shared_key)[key_vector] = loaded;
      }

      const int value_thread = static_cast<int>(threadIdx.x) -
          kGqa4TokenTile * kQueryVectors;
      if (value_thread >= 0 &&
          value_thread < kGqa4TokenTile * kValueVectors) {
        const int token = value_thread / kValueVectors;
        const int vector = value_thread % kValueVectors;
        const int64_t source_token = page_token_start + tile_start + token;
        uint4 loaded = make_uint4(0, 0, 0, 0);
        if (source_token < sequence_length) {
          const int64_t value_base = batch * value_stride_batch +
              static_cast<int64_t>(kv_head) * value_stride_head +
              source_token * value_stride_token +
              vector * kHalfElementsPerVector;
          loaded = *reinterpret_cast<const uint4*>(value + value_base);
        }
        reinterpret_cast<uint4*>(shared_value)[value_thread] = loaded;
      }
      __syncthreads();

#pragma unroll
      for (int token = 0; token < kGqa4TokenTile; ++token) {
        if (page_token_start + tile_start + token < sequence_length) {
          float dot = 0.0f;
          if (lane < kQueryVectors) {
            const int feature = lane * kHalfElementsPerVector;
            dot = accumulate_packed_eight(
                shared_query[warp] + feature,
                shared_key[token] + feature,
                dot);
          }
          dot = warp_sum(dot);
          const float score = dot * scale;
          const float next_max = fmaxf(running_max, score);
          const float previous_scale = __expf(running_max - next_max);
          const float probability = __expf(score - next_max);
          accumulator0 = fmaf(
              probability,
              static_cast<float>(shared_value[token][lane]),
              accumulator0 * previous_scale);
          accumulator1 = fmaf(
              probability,
              static_cast<float>(
                  shared_value[token][lane + kHardwareWarpSize]),
              accumulator1 * previous_scale);
          accumulator2 = fmaf(
              probability,
              static_cast<float>(
                  shared_value[token][lane + 2 * kHardwareWarpSize]),
              accumulator2 * previous_scale);
          running_sum = running_sum * previous_scale + probability;
          running_max = next_max;
        }
      }
      __syncthreads();
    }
  }

  if constexpr (kDirectOutput) {
    const float inverse_sum = 1.0f / running_sum;
    const int64_t output_base = batch * output_stride_batch +
        static_cast<int64_t>(query_head) * output_stride_head;
    output[output_base + lane * output_stride_feature] =
        static_cast<scalar_t>(accumulator0 * inverse_sum);
    output[output_base +
           (lane + kHardwareWarpSize) * output_stride_feature] =
        static_cast<scalar_t>(accumulator1 * inverse_sum);
    output[output_base +
           (lane + 2 * kHardwareWarpSize) * output_stride_feature] =
        static_cast<scalar_t>(accumulator2 * inverse_sum);
  } else {
    const int64_t row = batch * query_heads + query_head;
    const int64_t base =
        (row * splits + split) * (kGqa4ValueDim + 2);
    workspace[base + lane] = accumulator0;
    workspace[base + lane + kHardwareWarpSize] = accumulator1;
    workspace[base + lane + 2 * kHardwareWarpSize] = accumulator2;
    if (lane == 0) {
      workspace[base + kGqa4ValueDim] = running_max;
      workspace[base + kGqa4ValueDim + 1] = running_sum;
    }
  }
}

template <typename scalar_t>
__global__ void pack_exact_key_pages_kernel(
    const scalar_t* __restrict__ exact_key,
    const int64_t* __restrict__ selected_page_ids,
    scalar_t* __restrict__ packed_key_pages,
    int64_t kv_heads,
    int64_t sequence_length,
    int64_t page_slots,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_token) {
  constexpr int kKeyVectors = kQueryKeyDim / kHalfElementsPerVector;
  constexpr int kVectorsPerPage = kSparsePageSize * kKeyVectors;
  const int64_t page_row = static_cast<int64_t>(blockIdx.x);
  const int64_t page_slot = page_row % page_slots;
  const int64_t kv_row = page_row / page_slots;
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t page_id = selected_page_ids[page_row];

  for (int vector_index = static_cast<int>(threadIdx.x);
       vector_index < kVectorsPerPage;
       vector_index += static_cast<int>(blockDim.x)) {
    const int token = vector_index / kKeyVectors;
    const int vector = vector_index % kKeyVectors;
    const int64_t source_token = page_id * kSparsePageSize + token;
    uint4 loaded = make_uint4(0, 0, 0, 0);
    if (page_id >= 0 && source_token < sequence_length) {
      const int64_t source_offset = batch * key_stride_batch +
          kv_head * key_stride_head + source_token * key_stride_token +
          vector * kHalfElementsPerVector;
      loaded = *reinterpret_cast<const uint4*>(exact_key + source_offset);
    }
    const int64_t output_vector =
        page_row * kVectorsPerPage + vector_index;
    reinterpret_cast<uint4*>(packed_key_pages)[output_vector] = loaded;
  }
}

template <typename scalar_t, int kValueDim>
__global__ void splitk_reduce_kernel(
    const float* __restrict__ workspace,
    scalar_t* __restrict__ output,
    int64_t splits,
    int64_t query_heads,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature) {
  constexpr int kOutputsPerLane =
      (kValueDim + kHardwareWarpSize - 1) / kHardwareWarpSize;
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = row / query_heads;
  const int64_t query_head = row % query_heads;
  const int64_t row_base = row * splits * (kValueDim + 2);

  float local_max = -CUDART_INF_F;
  for (int split = lane; split < splits; split += kHardwareWarpSize) {
    local_max = fmaxf(
        local_max,
        workspace[
            row_base + static_cast<int64_t>(split) * (kValueDim + 2) +
            kValueDim]);
  }
  const float maximum = warp_max(local_max);
  float local_sum = 0.0f;
  for (int split = lane; split < splits; split += kHardwareWarpSize) {
    const int64_t base =
        row_base + static_cast<int64_t>(split) * (kValueDim + 2);
    const float weight = __expf(workspace[base + kValueDim] - maximum);
    local_sum = fmaf(weight, workspace[base + kValueDim + 1], local_sum);
  }
  const float total = warp_sum(local_sum);

#pragma unroll
  for (int item = 0; item < kOutputsPerLane; ++item) {
    const int feature = lane + item * kHardwareWarpSize;
    if (feature < kValueDim) {
      float combined = 0.0f;
      for (int split = 0; split < splits; ++split) {
        const int64_t base =
            row_base + static_cast<int64_t>(split) * (kValueDim + 2);
        const float weight = __expf(workspace[base + kValueDim] - maximum);
        combined = fmaf(weight, workspace[base + feature], combined);
      }
      const int64_t output_offset = batch * output_stride_batch +
          query_head * output_stride_head +
          feature * output_stride_feature;
      output[output_offset] =
          static_cast<scalar_t>(combined / total);
    }
  }
}

template <typename scalar_t>
void launch_gqa4_v96_dense(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& valid_sequence_length,
    const at::Tensor& workspace,
    const at::Tensor& output,
    float scale,
    int64_t splits,
    cudaStream_t stream) {
  const int64_t query_heads = query.size(1);
  const int64_t blocks = query.size(0) * key.size(1) * splits;
  if (splits == 1) {
    gqa4_v96_dense_shared_kv_splitk_kernel<scalar_t, true>
        <<<static_cast<unsigned int>(blocks),
           kGqa4ThreadsPerBlock,
           0,
           stream>>>(
            query.const_data_ptr<scalar_t>(),
            key.const_data_ptr<scalar_t>(),
            value.const_data_ptr<scalar_t>(),
            valid_sequence_length.const_data_ptr<int64_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            key.size(1),
            key.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(3),
            scale);
  } else {
    gqa4_v96_dense_shared_kv_splitk_kernel<scalar_t, false>
        <<<static_cast<unsigned int>(blocks),
           kGqa4ThreadsPerBlock,
           0,
           stream>>>(
            query.const_data_ptr<scalar_t>(),
            key.const_data_ptr<scalar_t>(),
            value.const_data_ptr<scalar_t>(),
            valid_sequence_length.const_data_ptr<int64_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            key.size(1),
            key.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(3),
            scale);
    splitk_reduce_kernel<scalar_t, kGqa4ValueDim>
        <<<static_cast<unsigned int>(query.size(0) * query_heads),
           kHardwareWarpSize,
           0,
           stream>>>(
            workspace.const_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            splits,
            query_heads,
            output.stride(0),
            output.stride(1),
            output.stride(3));
  }
}

template <typename scalar_t>
void launch_gqa4_v96_paged_sparse(
    const at::Tensor& query,
    const at::Tensor& packed_key_pages,
    const at::Tensor& value,
    const at::Tensor& selected_page_ids,
    const at::Tensor& workspace,
    const at::Tensor& output,
    float scale,
    int64_t splits,
    cudaStream_t stream) {
  const int64_t query_heads = query.size(1);
  const int64_t blocks = query.size(0) * value.size(1) * splits;
  if (splits == 1) {
    gqa4_v96_paged_sparse_splitk_kernel<scalar_t, true>
        <<<static_cast<unsigned int>(blocks),
           kGqa4ThreadsPerBlock,
           0,
           stream>>>(
            query.const_data_ptr<scalar_t>(),
            packed_key_pages.const_data_ptr<scalar_t>(),
            value.const_data_ptr<scalar_t>(),
            selected_page_ids.const_data_ptr<int64_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            value.size(1),
            selected_page_ids.size(2),
            value.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            packed_key_pages.stride(0),
            packed_key_pages.stride(1),
            packed_key_pages.stride(2),
            packed_key_pages.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            selected_page_ids.stride(0),
            selected_page_ids.stride(1),
            selected_page_ids.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(3),
            scale);
  } else {
    gqa4_v96_paged_sparse_splitk_kernel<scalar_t, false>
        <<<static_cast<unsigned int>(blocks),
           kGqa4ThreadsPerBlock,
           0,
           stream>>>(
            query.const_data_ptr<scalar_t>(),
            packed_key_pages.const_data_ptr<scalar_t>(),
            value.const_data_ptr<scalar_t>(),
            selected_page_ids.const_data_ptr<int64_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            value.size(1),
            selected_page_ids.size(2),
            value.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            packed_key_pages.stride(0),
            packed_key_pages.stride(1),
            packed_key_pages.stride(2),
            packed_key_pages.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            selected_page_ids.stride(0),
            selected_page_ids.stride(1),
            selected_page_ids.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(3),
            scale);
    splitk_reduce_kernel<scalar_t, kGqa4ValueDim>
        <<<static_cast<unsigned int>(query.size(0) * query_heads),
           kHardwareWarpSize,
           0,
           stream>>>(
            workspace.const_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            splits,
            query_heads,
            output.stride(0),
            output.stride(1),
            output.stride(3));
  }
}

template <typename scalar_t>
void launch_pack_exact_key_pages(
    const at::Tensor& exact_key,
    const at::Tensor& selected_page_ids,
    const at::Tensor& packed_key_pages,
    cudaStream_t stream) {
  const int64_t blocks =
      exact_key.size(0) * exact_key.size(1) * selected_page_ids.size(2);
  pack_exact_key_pages_kernel<scalar_t>
      <<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
          exact_key.const_data_ptr<scalar_t>(),
          selected_page_ids.const_data_ptr<int64_t>(),
          packed_key_pages.mutable_data_ptr<scalar_t>(),
          exact_key.size(1),
          exact_key.size(2),
          selected_page_ids.size(2),
          exact_key.stride(0),
          exact_key.stride(1),
          exact_key.stride(2));
}

template <typename scalar_t>
void launch_gqa8_v64(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& workspace,
    const at::Tensor& output,
    float scale,
    int64_t splits,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    cudaStream_t stream) {
  const int64_t query_heads = query.size(1);
  const int64_t blocks = query.size(0) * key.size(1) * splits;
  if (splits == 1) {
    gqa8_v64_shared_kv_splitk_kernel<scalar_t, true>
        <<<static_cast<unsigned int>(blocks), kGqa8ThreadsPerBlock, 0, stream>>>(
            query.const_data_ptr<scalar_t>(),
            key.const_data_ptr<scalar_t>(),
            value.const_data_ptr<scalar_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            key.size(1),
            key.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output_stride_batch,
            output_stride_head,
            output_stride_feature,
            scale);
  } else {
    gqa8_v64_shared_kv_splitk_kernel<scalar_t, false>
        <<<static_cast<unsigned int>(blocks), kGqa8ThreadsPerBlock, 0, stream>>>(
            query.const_data_ptr<scalar_t>(),
            key.const_data_ptr<scalar_t>(),
            value.const_data_ptr<scalar_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            key.size(1),
            key.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output_stride_batch,
            output_stride_head,
            output_stride_feature,
            scale);
    splitk_reduce_kernel<scalar_t, kGqa8ValueDim>
        <<<static_cast<unsigned int>(query.size(0) * query_heads),
           kHardwareWarpSize,
           0,
           stream>>>(
            workspace.const_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            splits,
            query_heads,
            output_stride_batch,
            output_stride_head,
            output_stride_feature);
  }
}

template <typename Architecture, typename scalar_t, int kValueDim>
void launch_rank(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& workspace,
    const at::Tensor& output,
    float scale,
    int64_t splits,
    int64_t queries_per_kv,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    cudaStream_t stream) {
  constexpr int kThreadsPerBlock =
      Architecture::kWarpSize * Architecture::kWarpsPerBlock;
  const int64_t rows = query.size(0) * query.size(1);
  if (splits == 1) {
    vectorized_splitk_partial_kernel<Architecture, scalar_t, kValueDim, true>
        <<<static_cast<unsigned int>(rows), kThreadsPerBlock, 0, stream>>>(
            query.const_data_ptr<scalar_t>(),
            key.const_data_ptr<scalar_t>(),
            value.const_data_ptr<scalar_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            query.size(1),
            queries_per_kv,
            key.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output_stride_batch,
            output_stride_head,
            output_stride_feature,
            scale);
  } else {
    vectorized_splitk_partial_kernel<Architecture, scalar_t, kValueDim, false>
        <<<static_cast<unsigned int>(rows * splits), kThreadsPerBlock, 0, stream>>>(
            query.const_data_ptr<scalar_t>(),
            key.const_data_ptr<scalar_t>(),
            value.const_data_ptr<scalar_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            query.size(1),
            queries_per_kv,
            key.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output_stride_batch,
            output_stride_head,
            output_stride_feature,
            scale);
    splitk_reduce_kernel<scalar_t, kValueDim>
        <<<static_cast<unsigned int>(rows), kHardwareWarpSize, 0, stream>>>(
            workspace.const_data_ptr<float>(),
            output.mutable_data_ptr<scalar_t>(),
            splits,
            query.size(1),
            output_stride_batch,
            output_stride_head,
            output_stride_feature);
  }
}

template <typename Architecture, typename scalar_t>
void dispatch_rank(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& workspace,
    const at::Tensor& output,
    float scale,
    int64_t splits,
    int64_t queries_per_kv,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    cudaStream_t stream) {
  switch (value.size(3)) {
    case 32:
      launch_rank<Architecture, scalar_t, 32>(query, key, value, workspace, output, scale, splits, queries_per_kv, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 48:
      launch_rank<Architecture, scalar_t, 48>(query, key, value, workspace, output, scale, splits, queries_per_kv, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 64:
      launch_rank<Architecture, scalar_t, 64>(query, key, value, workspace, output, scale, splits, queries_per_kv, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 80:
      launch_rank<Architecture, scalar_t, 80>(query, key, value, workspace, output, scale, splits, queries_per_kv, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 96:
      launch_rank<Architecture, scalar_t, 96>(query, key, value, workspace, output, scale, splits, queries_per_kv, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 112:
      launch_rank<Architecture, scalar_t, 112>(query, key, value, workspace, output, scale, splits, queries_per_kv, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    default:
      TORCH_CHECK(false, "split-K decode requires rank in {32,48,64,80,96,112}");
  }
}

template <typename scalar_t>
void dispatch_architecture(
    int major,
    int minor,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& workspace,
    const at::Tensor& output,
    float scale,
    int64_t splits,
    int64_t queries_per_kv,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    cudaStream_t stream) {
  if (major == Sm80Traits::kMajor && minor == Sm80Traits::kMinor) {
    dispatch_rank<Sm80Traits, scalar_t>(
        query, key, value, workspace, output, scale, splits, queries_per_kv,
        output_stride_batch, output_stride_head, output_stride_feature, stream);
  } else if (major == Sm89Traits::kMajor && minor == Sm89Traits::kMinor) {
    dispatch_rank<Sm89Traits, scalar_t>(
        query, key, value, workspace, output, scale, splits, queries_per_kv,
        output_stride_batch, output_stride_head, output_stride_feature, stream);
  } else if (major == Sm90Traits::kMajor && minor == Sm90Traits::kMinor) {
    dispatch_rank<Sm90Traits, scalar_t>(
        query, key, value, workspace, output, scale, splits, queries_per_kv,
        output_stride_batch, output_stride_head, output_stride_feature, stream);
  } else {
    TORCH_CHECK(
        false,
        "compressed-V CUDA decode supports SM80, SM89, and SM90; got SM",
        major,
        minor);
  }
}

}  // namespace

at::Tensor compressed_v_decode_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& workspace,
    const at::Tensor& output,
    int64_t architecture,
    double scale,
    int64_t splits,
    bool feature_major_output) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(), "Q/K/V must be CUDA tensors");
  TORCH_CHECK(query.device() == key.device() && query.device() == value.device(), "Q/K/V devices differ");
  TORCH_CHECK(workspace.device() == query.device() && output.device() == query.device(), "workspace/output devices differ");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4, "Q/K/V must be rank-4");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(), "Q/K/V dtypes differ");
  TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16, "Q/K/V must be FP16 or BF16");
  TORCH_CHECK(output.scalar_type() == query.scalar_type(), "output dtype differs from Q/K/V");
  TORCH_CHECK(workspace.scalar_type() == at::kFloat, "split-K workspace must be FP32");
  TORCH_CHECK(query.size(2) == 1, "split-K decode requires one query token");
  TORCH_CHECK(query.size(3) == kQueryKeyDim && key.size(3) == kQueryKeyDim, "Q/K head dimension must be 128");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0), "Q/K/V batches differ");
  TORCH_CHECK(key.size(1) == value.size(1) && key.size(2) == value.size(2), "K/V shapes differ");
  TORCH_CHECK(key.size(1) > 0 && query.size(1) % key.size(1) == 0, "query heads must be divisible by KV heads");
  TORCH_CHECK(query.stride(3) == 1 && key.stride(3) == 1 && value.stride(3) == 1, "Q/K/V feature dimensions must be contiguous");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(query.const_data_ptr()) % kVectorBytes == 0 &&
          reinterpret_cast<uintptr_t>(key.const_data_ptr()) % kVectorBytes == 0 &&
          reinterpret_cast<uintptr_t>(value.const_data_ptr()) % kVectorBytes == 0,
      "vectorized CUDA decode requires 16-byte-aligned Q/K/V storage");
  TORCH_CHECK(
      query.stride(0) % kHalfElementsPerVector == 0 &&
          query.stride(1) % kHalfElementsPerVector == 0 &&
          key.stride(0) % kHalfElementsPerVector == 0 &&
          key.stride(1) % kHalfElementsPerVector == 0 &&
          key.stride(2) % kHalfElementsPerVector == 0 &&
          value.stride(0) % kHalfElementsPerVector == 0 &&
          value.stride(1) % kHalfElementsPerVector == 0 &&
          value.stride(2) % kHalfElementsPerVector == 0,
      "vectorized CUDA decode requires 16-byte-aligned Q/K/V rows");
  TORCH_CHECK(workspace.is_contiguous() && output.is_contiguous(), "workspace/output must be contiguous");
  TORCH_CHECK(splits == 1 || splits == 2 || splits == 4 || splits == 8 || splits == 16 || splits == 32, "splits must be in {1,2,4,8,16,32}");
  TORCH_CHECK(key.size(2) >= splits, "each split needs at least one KV token");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "scale must be finite and positive");
  const int64_t rows = query.size(0) * query.size(1);
  const int64_t value_dim = value.size(3);
  const int64_t queries_per_kv = query.size(1) / key.size(1);
  TORCH_CHECK(
      workspace.dim() == 3 && workspace.size(0) == rows &&
          workspace.size(1) == splits && workspace.size(2) == value_dim + 2,
      "invalid split-K workspace shape");
  int64_t output_stride_batch = 0;
  int64_t output_stride_head = 0;
  int64_t output_stride_feature = 0;
  if (feature_major_output) {
    TORCH_CHECK(
        output.dim() == 2 && output.size(0) == query.size(1) * value_dim &&
            output.size(1) == query.size(0),
        "feature-major split-K output must have shape [query_heads * rank, batch]");
    output_stride_batch = output.stride(1);
    output_stride_head = value_dim * output.stride(0);
    output_stride_feature = output.stride(0);
  } else {
    TORCH_CHECK(
        output.dim() == 4 && output.size(0) == query.size(0) &&
            output.size(1) == query.size(1) && output.size(2) == 1 &&
            output.size(3) == value_dim,
        "token-major split-K output must have shape [batch, query_heads, 1, rank]");
    output_stride_batch = output.stride(0);
    output_stride_head = output.stride(1);
    output_stride_feature = output.stride(3);
  }

  c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      query.scalar_type(),
      "basisserve_vectorized_splitk_compressed_v_decode",
      [&] {
        if (queries_per_kv == kGqa8QueriesPerKv &&
            value_dim == kGqa8ValueDim) {
          launch_gqa8_v64<scalar_t>(
              query,
              key,
              value,
              workspace,
              output,
              static_cast<float>(scale),
              splits,
              output_stride_batch,
              output_stride_head,
              output_stride_feature,
              stream);
        } else {
          dispatch_architecture<scalar_t>(
              static_cast<int>(architecture / 10),
              static_cast<int>(architecture % 10),
              query,
              key,
              value,
              workspace,
              output,
              static_cast<float>(scale),
              splits,
              queries_per_kv,
              output_stride_batch,
              output_stride_head,
              output_stride_feature,
              stream);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor c1_dense_gqa_v96_decode_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& valid_sequence_length,
    const at::Tensor& workspace,
    const at::Tensor& output,
    double scale,
    int64_t splits) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda() &&
          valid_sequence_length.is_cuda(),
      "dense GQA V96 Q/K/V/valid length must be CUDA tensors");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device() &&
          query.device() == valid_sequence_length.device() &&
          query.device() == workspace.device() &&
          query.device() == output.device(),
      "dense GQA V96 tensors must share one CUDA device");
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "dense GQA V96 Q/K/V must be rank-4");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type(),
      "dense GQA V96 Q/K/V dtypes differ");
  TORCH_CHECK(
      query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
      "dense GQA V96 Q/K/V must be FP16 or BF16");
  TORCH_CHECK(
      valid_sequence_length.dim() == 0 &&
          valid_sequence_length.scalar_type() == at::kLong,
      "dense GQA V96 valid sequence length must be a CUDA int64 scalar");
  TORCH_CHECK(
      workspace.scalar_type() == at::kFloat &&
          output.scalar_type() == query.scalar_type(),
      "dense GQA V96 workspace/output dtypes differ");
  TORCH_CHECK(
      query.size(2) == 1 && query.size(3) == kQueryKeyDim &&
          key.size(3) == kQueryKeyDim && value.size(3) == kGqa4ValueDim,
      "dense GQA V96 requires QK128 and Value rank 96");
  TORCH_CHECK(
      query.size(0) == key.size(0) && query.size(0) == value.size(0) &&
          key.size(1) == value.size(1) && key.size(2) == value.size(2) &&
          query.size(1) == kGqa4QueriesPerKv * key.size(1),
      "dense GQA V96 requires four Query heads per KV head");
  TORCH_CHECK(
      key.size(2) > 0 && query.stride(3) == 1 && key.stride(3) == 1 &&
          value.stride(3) == 1,
      "dense GQA V96 feature dimensions must be contiguous");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(query.const_data_ptr()) % kVectorBytes == 0 &&
          reinterpret_cast<uintptr_t>(key.const_data_ptr()) % kVectorBytes == 0 &&
          reinterpret_cast<uintptr_t>(value.const_data_ptr()) % kVectorBytes == 0,
      "dense GQA V96 Q/K/V storage must be 16-byte aligned");
  TORCH_CHECK(
      query.stride(0) % kHalfElementsPerVector == 0 &&
          query.stride(1) % kHalfElementsPerVector == 0 &&
          key.stride(0) % kHalfElementsPerVector == 0 &&
          key.stride(1) % kHalfElementsPerVector == 0 &&
          key.stride(2) % kHalfElementsPerVector == 0 &&
          value.stride(0) % kHalfElementsPerVector == 0 &&
          value.stride(1) % kHalfElementsPerVector == 0 &&
          value.stride(2) % kHalfElementsPerVector == 0,
      "dense GQA V96 Q/K/V rows must be 16-byte aligned");
  TORCH_CHECK(
      valid_sequence_length.is_contiguous() && workspace.is_contiguous() &&
          output.is_contiguous(),
      "dense GQA V96 valid length/workspace/output must be contiguous");
  TORCH_CHECK(
      splits == 1 || splits == 2 || splits == 4 || splits == 8 ||
          splits == 16 || splits == 32 || splits == 64 || splits == 128 ||
          splits == 256,
      "dense GQA V96 splits must be in {1,2,4,8,16,32,64,128,256}");
  TORCH_CHECK(
      key.size(2) >= splits && std::isfinite(scale) && scale > 0.0,
      "dense GQA V96 cache/split/scale configuration is invalid");
  const int64_t rows = query.size(0) * query.size(1);
  TORCH_CHECK(
      workspace.dim() == 3 && workspace.size(0) == rows &&
          workspace.size(1) == splits &&
          workspace.size(2) == kGqa4ValueDim + 2,
      "invalid dense GQA V96 split-K workspace shape");
  TORCH_CHECK(
      output.dim() == 4 && output.size(0) == query.size(0) &&
          output.size(1) == query.size(1) && output.size(2) == 1 &&
          output.size(3) == kGqa4ValueDim,
      "dense GQA V96 output must have shape [B,Hq,1,96]");

  c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      query.scalar_type(),
      "basisserve_gqa4_v96_dense_decode",
      [&] {
        launch_gqa4_v96_dense<scalar_t>(
            query,
            key,
            value,
            valid_sequence_length,
            workspace,
            output,
            static_cast<float>(scale),
            splits,
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor c1_pack_exact_key_pages_cuda(
    const at::Tensor& exact_key,
    const at::Tensor& selected_page_ids,
    const at::Tensor& packed_key_pages) {
  c10::cuda::CUDAGuard device_guard(exact_key.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(exact_key.get_device()).stream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      exact_key.scalar_type(),
      "basisserve_pack_exact_key_pages",
      [&] {
        launch_pack_exact_key_pages<scalar_t>(
            exact_key,
            selected_page_ids,
            packed_key_pages,
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed_key_pages;
}

at::Tensor c1_paged_sparse_decode_cuda(
    const at::Tensor& query,
    const at::Tensor& packed_key_pages,
    const at::Tensor& value,
    const at::Tensor& selected_page_ids,
    const at::Tensor& workspace,
    const at::Tensor& output,
    double scale,
    int64_t splits) {
  TORCH_CHECK(
      query.is_cuda() && packed_key_pages.is_cuda() && value.is_cuda() &&
          selected_page_ids.is_cuda(),
      "paged sparse Q/K/V/page IDs must be CUDA tensors");
  TORCH_CHECK(
      query.device() == packed_key_pages.device() &&
          query.device() == value.device() &&
          query.device() == selected_page_ids.device() &&
          query.device() == workspace.device() &&
          query.device() == output.device(),
      "paged sparse tensors must share one CUDA device");
  TORCH_CHECK(
      query.dim() == 4 && packed_key_pages.dim() == 5 && value.dim() == 4 &&
          selected_page_ids.dim() == 3,
      "paged sparse Q/K/V/page IDs have incompatible ranks");
  TORCH_CHECK(
      query.scalar_type() == packed_key_pages.scalar_type() &&
          query.scalar_type() == value.scalar_type(),
      "paged sparse Q/K/V dtypes differ");
  TORCH_CHECK(
      query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
      "paged sparse Q/K/V must be FP16 or BF16");
  TORCH_CHECK(
      selected_page_ids.scalar_type() == at::kLong,
      "selected page IDs must be int64");
  TORCH_CHECK(
      workspace.scalar_type() == at::kFloat &&
          output.scalar_type() == query.scalar_type(),
      "paged sparse workspace/output dtypes differ");
  TORCH_CHECK(
      query.size(2) == 1 && query.size(3) == kQueryKeyDim,
      "paged sparse decode requires Q=[B,H,1,128]");
  TORCH_CHECK(
      packed_key_pages.size(4) == kQueryKeyDim &&
          packed_key_pages.size(3) == kSparsePageSize,
      "packed exact-Key pages must have shape [B,Hkv,P,64,128]");
  TORCH_CHECK(
      value.size(3) == kGqa4ValueDim,
      "paged sparse C1 Value rank must be 96");
  TORCH_CHECK(
      query.size(0) == packed_key_pages.size(0) &&
          query.size(0) == value.size(0) &&
          query.size(0) == selected_page_ids.size(0),
      "paged sparse batch dimensions differ");
  TORCH_CHECK(
      packed_key_pages.size(1) == value.size(1) &&
          packed_key_pages.size(1) == selected_page_ids.size(1) &&
          query.size(1) == kGqa4QueriesPerKv * value.size(1),
      "paged sparse attention requires four Query heads per KV head");
  TORCH_CHECK(
      packed_key_pages.size(2) == selected_page_ids.size(2) &&
          selected_page_ids.size(2) > 0 && value.size(2) > 0,
      "packed exact-Key pages and selected page IDs differ");
  TORCH_CHECK(
      query.stride(3) == 1 && packed_key_pages.stride(4) == 1 &&
          value.stride(3) == 1,
      "paged sparse feature dimensions must be contiguous");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(query.const_data_ptr()) % kVectorBytes == 0 &&
          reinterpret_cast<uintptr_t>(packed_key_pages.const_data_ptr()) %
                  kVectorBytes ==
              0 &&
          reinterpret_cast<uintptr_t>(value.const_data_ptr()) % kVectorBytes ==
              0,
      "paged sparse Q/K/V storage must be 16-byte aligned");
  TORCH_CHECK(
      query.stride(0) % kHalfElementsPerVector == 0 &&
          query.stride(1) % kHalfElementsPerVector == 0 &&
          packed_key_pages.stride(0) % kHalfElementsPerVector == 0 &&
          packed_key_pages.stride(1) % kHalfElementsPerVector == 0 &&
          packed_key_pages.stride(2) % kHalfElementsPerVector == 0 &&
          packed_key_pages.stride(3) % kHalfElementsPerVector == 0 &&
          value.stride(0) % kHalfElementsPerVector == 0 &&
          value.stride(1) % kHalfElementsPerVector == 0 &&
          value.stride(2) % kHalfElementsPerVector == 0,
      "paged sparse Q/K/V rows must be 16-byte aligned");
  TORCH_CHECK(
      selected_page_ids.is_contiguous() && workspace.is_contiguous() &&
          output.is_contiguous(),
      "paged sparse page IDs/workspace/output must be contiguous");
  TORCH_CHECK(
      splits == 1 || splits == 2 || splits == 4 || splits == 8 ||
          splits == 16 || splits == 32,
      "paged sparse splits must be in {1,2,4,8,16,32}");
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "paged sparse scale must be finite and positive");
  const int64_t rows = query.size(0) * query.size(1);
  TORCH_CHECK(
      workspace.dim() == 3 && workspace.size(0) == rows &&
          workspace.size(1) == splits &&
          workspace.size(2) == kGqa4ValueDim + 2,
      "invalid paged sparse split-K workspace shape");
  TORCH_CHECK(
      output.dim() == 4 && output.size(0) == query.size(0) &&
          output.size(1) == query.size(1) && output.size(2) == 1 &&
          output.size(3) == kGqa4ValueDim,
      "paged sparse output must have shape [B,Hq,1,96]");

  c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      query.scalar_type(),
      "basisserve_gqa4_v96_paged_sparse_decode",
      [&] {
        launch_gqa4_v96_paged_sparse<scalar_t>(
            query,
            packed_key_pages,
            value,
            selected_page_ids,
            workspace,
            output,
            static_cast<float>(scale),
            splits,
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
