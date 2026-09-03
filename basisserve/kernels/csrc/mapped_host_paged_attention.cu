#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kQueriesPerKv = 4;
constexpr int kThreads = kQueriesPerKv * kWarpSize;
constexpr int kQueryKeyDim = 128;
constexpr int kValueDim = 80;
constexpr int kPageSize = 32;
constexpr int kTokenTile = 4;
constexpr int kVectorBytes = sizeof(uint4);
constexpr int kElementsPerVector = kVectorBytes / sizeof(uint16_t);
constexpr int kKeyVectors = kQueryKeyDim / kElementsPerVector;
constexpr int kValueVectors = kValueDim / kElementsPerVector;

__device__ __forceinline__ float2 bf16_pair(uint32_t bits) {
  return make_float2(
      __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(bits))),
      __bfloat162float(
          __ushort_as_bfloat16(static_cast<uint16_t>(bits >> 16))));
}

__device__ __forceinline__ uint32_t vector_word(uint4 value, int index) {
  switch (index) {
    case 0:
      return value.x;
    case 1:
      return value.y;
    case 2:
      return value.z;
    default:
      return value.w;
  }
}

__device__ __forceinline__ float vector_dot(
    const c10::BFloat16* query,
    const c10::BFloat16* key) {
  const uint4 query_vector = *reinterpret_cast<const uint4*>(query);
  const uint4 key_vector = *reinterpret_cast<const uint4*>(key);
  float result = 0.0f;
#pragma unroll
  for (int pair = 0; pair < kElementsPerVector / 2; ++pair) {
    const float2 query_pair = bf16_pair(vector_word(query_vector, pair));
    const float2 key_pair = bf16_pair(vector_word(key_vector, pair));
    result = fmaf(query_pair.x, key_pair.x, result);
    result = fmaf(query_pair.y, key_pair.y, result);
  }
  return result;
}

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

template <bool kDirectOutput>
__global__ void mapped_host_page32_v80_kernel(
    const c10::BFloat16* __restrict__ host_key,
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ value,
    const int64_t* __restrict__ selected_page_ids,
    float* __restrict__ workspace,
    c10::BFloat16* __restrict__ output,
    int64_t kv_heads,
    int64_t capacity,
    int64_t sequence_length,
    int64_t page_slots,
    int64_t splits,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t page_stride_batch,
    int64_t page_stride_head,
    int64_t page_stride_slot,
    int64_t workspace_stride_row,
    int64_t workspace_stride_split,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature,
    float scale) {
  __shared__ __align__(kVectorBytes)
      c10::BFloat16 shared_query[kQueriesPerKv][kQueryKeyDim];
  __shared__ __align__(kVectorBytes)
      c10::BFloat16 shared_key[kTokenTile][kQueryKeyDim];
  __shared__ __align__(kVectorBytes)
      c10::BFloat16 shared_value[kTokenTile][kValueDim];

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int split = static_cast<int>(linear % splits);
  const int64_t kv_row = linear / splits;
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t query_head = kv_head * kQueriesPerKv + warp;
  const int64_t query_heads = kv_heads * kQueriesPerKv;

  const int query_vector = static_cast<int>(threadIdx.x);
  if (query_vector < kQueriesPerKv * kKeyVectors) {
    const int head = query_vector / kKeyVectors;
    const int vector = query_vector % kKeyVectors;
    const int feature = vector * kElementsPerVector;
    const int64_t source = batch * query_stride_batch +
        (kv_head * kQueriesPerKv + head) * query_stride_head + feature;
    reinterpret_cast<uint4*>(shared_query)[query_vector] =
        *reinterpret_cast<const uint4*>(query + source);
  }
  __syncthreads();

  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  for (int64_t page_slot = split; page_slot < page_slots;
       page_slot += splits) {
    const int64_t page_id = selected_page_ids[
        batch * page_stride_batch + kv_head * page_stride_head +
        page_slot * page_stride_slot];
    if (page_id < 0) {
      continue;
    }
    const int64_t page_start = page_id * kPageSize;
    for (int tile_start = 0; tile_start < kPageSize;
         tile_start += kTokenTile) {
      const int key_vector = static_cast<int>(threadIdx.x);
      if (key_vector < kTokenTile * kKeyVectors) {
        const int token = key_vector / kKeyVectors;
        const int vector = key_vector % kKeyVectors;
        const int64_t source_token = page_start + tile_start + token;
        uint4 loaded = make_uint4(0, 0, 0, 0);
        if (source_token < sequence_length) {
          const int64_t source =
              ((batch * kv_heads + kv_head) * capacity + source_token) *
                  kQueryKeyDim +
              vector * kElementsPerVector;
          loaded = *reinterpret_cast<const uint4*>(host_key + source);
        }
        reinterpret_cast<uint4*>(shared_key)[key_vector] = loaded;
      }

      const int value_thread =
          static_cast<int>(threadIdx.x) - kTokenTile * kKeyVectors;
      if (value_thread >= 0 && value_thread < kTokenTile * kValueVectors) {
        const int token = value_thread / kValueVectors;
        const int vector = value_thread % kValueVectors;
        const int64_t source_token = page_start + tile_start + token;
        uint4 loaded = make_uint4(0, 0, 0, 0);
        if (source_token < sequence_length) {
          const int64_t source = batch * value_stride_batch +
              kv_head * value_stride_head +
              source_token * value_stride_token +
              vector * kElementsPerVector;
          loaded = *reinterpret_cast<const uint4*>(value + source);
        }
        reinterpret_cast<uint4*>(shared_value)[value_thread] = loaded;
      }
      __syncthreads();

#pragma unroll
      for (int token = 0; token < kTokenTile; ++token) {
        if (page_start + tile_start + token < sequence_length) {
          float dot = 0.0f;
          if (lane < kKeyVectors) {
            dot = vector_dot(
                shared_query[warp] + lane * kElementsPerVector,
                shared_key[token] + lane * kElementsPerVector);
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
              static_cast<float>(shared_value[token][lane + kWarpSize]),
              accumulator1 * previous_scale);
          if (lane < kValueDim - 2 * kWarpSize) {
            accumulator2 = fmaf(
                probability,
                static_cast<float>(
                    shared_value[token][lane + 2 * kWarpSize]),
                accumulator2 * previous_scale);
          }
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
        query_head * output_stride_head;
    output[output_base + lane * output_stride_feature] =
        static_cast<c10::BFloat16>(accumulator0 * inverse_sum);
    output[output_base + (lane + kWarpSize) * output_stride_feature] =
        static_cast<c10::BFloat16>(accumulator1 * inverse_sum);
    if (lane < kValueDim - 2 * kWarpSize) {
      output[output_base +
             (lane + 2 * kWarpSize) * output_stride_feature] =
          static_cast<c10::BFloat16>(accumulator2 * inverse_sum);
    }
  } else {
    const int64_t row = batch * query_heads + query_head;
    const int64_t base =
        row * workspace_stride_row + split * workspace_stride_split;
    workspace[base + lane] = accumulator0;
    workspace[base + lane + kWarpSize] = accumulator1;
    if (lane < kValueDim - 2 * kWarpSize) {
      workspace[base + lane + 2 * kWarpSize] = accumulator2;
    }
    if (lane == 0) {
      workspace[base + kValueDim] = running_max;
      workspace[base + kValueDim + 1] = running_sum;
    }
  }
}

__global__ void reduce_page32_v80_kernel(
    const float* __restrict__ workspace,
    c10::BFloat16* __restrict__ output,
    int64_t splits,
    int64_t query_heads,
    int64_t workspace_stride_row,
    int64_t workspace_stride_split,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_feature) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = row / query_heads;
  const int64_t query_head = row % query_heads;
  const int64_t row_base = row * workspace_stride_row;

  float local_max = -CUDART_INF_F;
  for (int split = lane; split < splits; split += kWarpSize) {
    local_max = fmaxf(
        local_max,
        workspace[row_base + split * workspace_stride_split + kValueDim]);
  }
  const float maximum = warp_max(local_max);
  float local_sum = 0.0f;
  for (int split = lane; split < splits; split += kWarpSize) {
    const int64_t base = row_base + split * workspace_stride_split;
    local_sum = fmaf(
        __expf(workspace[base + kValueDim] - maximum),
        workspace[base + kValueDim + 1],
        local_sum);
  }
  const float total = warp_sum(local_sum);

  for (int feature = lane; feature < kValueDim; feature += kWarpSize) {
    float combined = 0.0f;
    for (int split = 0; split < splits; ++split) {
      const int64_t base = row_base + split * workspace_stride_split;
      combined = fmaf(
          __expf(workspace[base + kValueDim] - maximum),
          workspace[base + feature],
          combined);
    }
    const int64_t output_base = batch * output_stride_batch +
        query_head * output_stride_head;
    output[output_base + feature * output_stride_feature] =
        static_cast<c10::BFloat16>(combined / total);
  }
}

}  // namespace

at::Tensor mapped_host_bf16_empty_cuda(
    int64_t batch,
    int64_t kv_heads,
    int64_t capacity,
    int64_t head_dim) {
  const size_t elements = static_cast<size_t>(batch) * kv_heads * capacity *
      head_dim;
  void* pointer = nullptr;
  const cudaError_t status = cudaHostAlloc(
      &pointer,
      elements * sizeof(c10::BFloat16),
      cudaHostAllocMapped);
  assert(status == cudaSuccess);
  return at::from_blob(
      pointer,
      {batch, kv_heads, capacity, head_dim},
      [](void* allocation) { cudaFreeHost(allocation); },
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
}

at::Tensor mapped_host_append_bf16_cuda(
    const at::Tensor& host_key,
    const at::Tensor& key,
    int64_t start) {
  assert(host_key.device().is_cpu());
  assert(key.is_cuda());
  assert(host_key.scalar_type() == at::kBFloat16);
  assert(key.scalar_type() == at::kBFloat16);
  assert(host_key.dim() == 4 && key.dim() == 4);
  assert(host_key.size(0) == key.size(0));
  assert(host_key.size(1) == key.size(1));
  assert(host_key.size(3) == key.size(3));
  assert(start >= 0 && start + key.size(2) <= host_key.size(2));
  assert(key.stride(3) == 1);

  c10::cuda::CUDAGuard guard(key.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(key.get_device()).stream();
  auto* destination = host_key.mutable_data_ptr<c10::BFloat16>();
  const auto* source = key.const_data_ptr<c10::BFloat16>();
  const size_t row_bytes =
      static_cast<size_t>(key.size(3)) * sizeof(c10::BFloat16);
  const size_t destination_pitch =
      static_cast<size_t>(host_key.stride(2)) * sizeof(c10::BFloat16);
  const size_t source_pitch =
      static_cast<size_t>(key.stride(2)) * sizeof(c10::BFloat16);
  for (int64_t batch = 0; batch < key.size(0); ++batch) {
    for (int64_t head = 0; head < key.size(1); ++head) {
      const int64_t destination_offset =
          batch * host_key.stride(0) + head * host_key.stride(1) +
          start * host_key.stride(2);
      const int64_t source_offset =
          batch * key.stride(0) + head * key.stride(1);
      const cudaError_t status = cudaMemcpy2DAsync(
          destination + destination_offset,
          destination_pitch,
          source + source_offset,
          source_pitch,
          row_bytes,
          static_cast<size_t>(key.size(2)),
          cudaMemcpyDeviceToHost,
          stream);
      assert(status == cudaSuccess);
    }
  }
  return host_key;
}

int64_t mapped_host_device_pointer_cuda(const at::Tensor& host_key) {
  assert(host_key.device().is_cpu());
  void* device_pointer = nullptr;
  const cudaError_t status =
      cudaHostGetDevicePointer(&device_pointer, host_key.data_ptr(), 0);
  assert(status == cudaSuccess);
  return reinterpret_cast<int64_t>(device_pointer);
}

at::Tensor mapped_host_paged_v80_attention_cuda(
    int64_t host_key_device_pointer,
    int64_t host_key_capacity,
    const at::Tensor& query,
    const at::Tensor& value,
    const at::Tensor& selected_page_ids,
    const at::Tensor& workspace,
    const at::Tensor& output,
    int64_t sequence_length,
    double scale,
    int64_t splits) {
  assert(query.is_cuda() && value.is_cuda() && selected_page_ids.is_cuda());
  assert(query.scalar_type() == at::kBFloat16);
  assert(value.scalar_type() == at::kBFloat16);
  assert(selected_page_ids.scalar_type() == at::kLong);
  assert(workspace.scalar_type() == at::kFloat);
  assert(output.scalar_type() == at::kBFloat16);
  assert(query.size(2) == 1 && query.size(3) == kQueryKeyDim);
  assert(value.size(3) == kValueDim);
  assert(query.size(1) == kQueriesPerKv * value.size(1));
  assert(host_key_device_pointer != 0);
  assert(sequence_length > 0 && sequence_length <= host_key_capacity);
  assert(splits > 0 && splits <= selected_page_ids.size(2));

  c10::cuda::CUDAGuard guard(query.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  const int64_t kv_rows = query.size(0) * value.size(1);
  const int64_t query_heads = query.size(1);

  if (splits == 1) {
    mapped_host_page32_v80_kernel<true>
        <<<static_cast<unsigned int>(kv_rows), kThreads, 0, stream>>>(
            reinterpret_cast<const c10::BFloat16*>(host_key_device_pointer),
            query.const_data_ptr<c10::BFloat16>(),
            value.const_data_ptr<c10::BFloat16>(),
            selected_page_ids.const_data_ptr<int64_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<c10::BFloat16>(),
            value.size(1),
            host_key_capacity,
            sequence_length,
            selected_page_ids.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            selected_page_ids.stride(0),
            selected_page_ids.stride(1),
            selected_page_ids.stride(2),
            workspace.stride(0),
            workspace.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(3),
            static_cast<float>(scale));
  } else {
    mapped_host_page32_v80_kernel<false>
        <<<static_cast<unsigned int>(kv_rows * splits),
           kThreads,
           0,
           stream>>>(
            reinterpret_cast<const c10::BFloat16*>(host_key_device_pointer),
            query.const_data_ptr<c10::BFloat16>(),
            value.const_data_ptr<c10::BFloat16>(),
            selected_page_ids.const_data_ptr<int64_t>(),
            workspace.mutable_data_ptr<float>(),
            output.mutable_data_ptr<c10::BFloat16>(),
            value.size(1),
            host_key_capacity,
            sequence_length,
            selected_page_ids.size(2),
            splits,
            query.stride(0),
            query.stride(1),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            selected_page_ids.stride(0),
            selected_page_ids.stride(1),
            selected_page_ids.stride(2),
            workspace.stride(0),
            workspace.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(3),
            static_cast<float>(scale));
    reduce_page32_v80_kernel
        <<<static_cast<unsigned int>(query.size(0) * query_heads),
           kWarpSize,
           0,
           stream>>>(
            workspace.const_data_ptr<float>(),
            output.mutable_data_ptr<c10::BFloat16>(),
            splits,
            query_heads,
            workspace.stride(0),
            workspace.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(3));
  }
  assert(cudaGetLastError() == cudaSuccess);
  return output;
}
