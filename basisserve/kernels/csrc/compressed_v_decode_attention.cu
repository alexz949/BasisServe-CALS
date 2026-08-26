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
constexpr int kQueriesPerKv = 4;
constexpr int kVectorBytes = sizeof(uint4);
constexpr int kHalfElementsPerVector = kVectorBytes / sizeof(uint16_t);

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
  constexpr bool kVectorizedValue = kValueDim == 80 || kValueDim == 112;
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
  const int kv_head = query_head / kQueriesPerKv;

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
  if (lane < splits) {
    local_max = workspace[
        row_base + static_cast<int64_t>(lane) * (kValueDim + 2) + kValueDim];
  }
  const float maximum = warp_max(local_max);
  float local_sum = 0.0f;
  if (lane < splits) {
    const int64_t base =
        row_base + static_cast<int64_t>(lane) * (kValueDim + 2);
    const float weight = __expf(workspace[base + kValueDim] - maximum);
    local_sum = weight * workspace[base + kValueDim + 1];
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

template <typename Architecture, typename scalar_t, int kValueDim>
void launch_rank(
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
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    cudaStream_t stream) {
  switch (value.size(3)) {
    case 32:
      launch_rank<Architecture, scalar_t, 32>(query, key, value, workspace, output, scale, splits, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 48:
      launch_rank<Architecture, scalar_t, 48>(query, key, value, workspace, output, scale, splits, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 64:
      launch_rank<Architecture, scalar_t, 64>(query, key, value, workspace, output, scale, splits, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 80:
      launch_rank<Architecture, scalar_t, 80>(query, key, value, workspace, output, scale, splits, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 96:
      launch_rank<Architecture, scalar_t, 96>(query, key, value, workspace, output, scale, splits, output_stride_batch, output_stride_head, output_stride_feature, stream);
      break;
    case 112:
      launch_rank<Architecture, scalar_t, 112>(query, key, value, workspace, output, scale, splits, output_stride_batch, output_stride_head, output_stride_feature, stream);
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
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    cudaStream_t stream) {
  if (major == Sm80Traits::kMajor && minor == Sm80Traits::kMinor) {
    dispatch_rank<Sm80Traits, scalar_t>(
        query, key, value, workspace, output, scale, splits,
        output_stride_batch, output_stride_head, output_stride_feature, stream);
  } else if (major == Sm89Traits::kMajor && minor == Sm89Traits::kMinor) {
    dispatch_rank<Sm89Traits, scalar_t>(
        query, key, value, workspace, output, scale, splits,
        output_stride_batch, output_stride_head, output_stride_feature, stream);
  } else if (major == Sm90Traits::kMajor && minor == Sm90Traits::kMinor) {
    dispatch_rank<Sm90Traits, scalar_t>(
        query, key, value, workspace, output, scale, splits,
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
  TORCH_CHECK(query.size(1) == kQueriesPerKv * key.size(1), "split-K decode requires GQA ratio 4");
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
            output_stride_batch,
            output_stride_head,
            output_stride_feature,
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
