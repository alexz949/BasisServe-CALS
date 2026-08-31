#include "ragged_allgather_common.h"

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace basisserve::ragged_ag {
namespace {

template <typename scalar_t>
__global__ void ragged_rank_major_to_token_major_tiled_kernel(
    const scalar_t* __restrict__ source,
    scalar_t* __restrict__ destination,
    int64_t batch,
    RaggedPackMetadata metadata) {
  const int source_rank = static_cast<int>(blockIdx.y);
  const int64_t width = metadata.widths[source_rank];
  const int64_t elements = batch * width;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       linear < elements;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = linear / width;
    const int64_t column = linear - token * width;
    const int64_t source_index = metadata.source_offsets[source_rank] + linear;
    const int64_t destination_index = token * metadata.total_width +
        metadata.destination_offsets[source_rank] + column;
    destination[destination_index] = source[source_index];
  }
}

__global__ void ragged_rank_major_to_token_major_uint4_kernel(
    const uint4* __restrict__ source,
    uint4* __restrict__ destination,
    int64_t batch,
    RaggedPackMetadata metadata,
    int32_t elements_per_vector) {
  const int source_rank = static_cast<int>(blockIdx.y);
  const int64_t width_vectors =
      metadata.widths[source_rank] / elements_per_vector;
  const int64_t elements = batch * width_vectors;
  const int64_t total_width_vectors =
      metadata.total_width / elements_per_vector;
  const int64_t source_offset_vectors =
      metadata.source_offsets[source_rank] / elements_per_vector;
  const int64_t destination_offset_vectors =
      metadata.destination_offsets[source_rank] / elements_per_vector;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       linear < elements;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = linear / width_vectors;
    const int64_t column = linear - token * width_vectors;
    destination[token * total_width_vectors + destination_offset_vectors + column] =
        source[source_offset_vectors + linear];
  }
}

RaggedPackMetadata make_metadata(
    const std::vector<int64_t>& widths,
    int64_t batch) {
  TORCH_CHECK(!widths.empty(), "ragged pack requires at least one source");
  TORCH_CHECK(
      widths.size() <= static_cast<size_t>(kMaximumWorldSize),
      "ragged pack supports at most ",
      kMaximumWorldSize,
      " sources");

  RaggedPackMetadata metadata{};
  metadata.world_size = static_cast<int32_t>(widths.size());
  int64_t width_prefix = 0;
  int64_t source_prefix = 0;
  for (size_t source = 0; source < widths.size(); ++source) {
    TORCH_CHECK(
        widths[source] > 0 && widths[source] <= INT32_MAX,
        "invalid ragged source width at index ",
        source,
        ": ",
        widths[source]);
    metadata.widths[source] = static_cast<int32_t>(widths[source]);
    metadata.source_offsets[source] = source_prefix;
    metadata.destination_offsets[source] = width_prefix;
    width_prefix += widths[source];
    source_prefix += batch * widths[source];
  }
  metadata.total_width = width_prefix;
  return metadata;
}

}  // namespace

void launch_ragged_pack(
    const at::Tensor& rank_major,
    at::Tensor& token_major,
    const std::vector<int64_t>& widths,
    int64_t batch,
    cudaStream_t stream) {
  TORCH_CHECK(rank_major.is_cuda(), "rank-major buffer must be CUDA");
  TORCH_CHECK(token_major.is_cuda(), "token-major buffer must be CUDA");
  TORCH_CHECK(rank_major.is_contiguous(), "rank-major buffer must be contiguous");
  TORCH_CHECK(token_major.is_contiguous(), "token-major buffer must be contiguous");
  TORCH_CHECK(
      rank_major.scalar_type() == token_major.scalar_type(),
      "ragged pack input/output dtypes must match");
  TORCH_CHECK(batch > 0, "ragged pack batch must be positive");

  const RaggedPackMetadata metadata = make_metadata(widths, batch);
  const int64_t expected_elements = batch * metadata.total_width;
  TORCH_CHECK(
      rank_major.numel() == expected_elements,
      "rank-major buffer has ",
      rank_major.numel(),
      " elements, expected ",
      expected_elements);
  TORCH_CHECK(
      token_major.numel() == expected_elements,
      "token-major buffer has ",
      token_major.numel(),
      " elements, expected ",
      expected_elements);

  constexpr int threads = 256;
  constexpr int vector_bytes = sizeof(uint4);
  const int64_t element_size = rank_major.element_size();
  const int32_t elements_per_vector =
      static_cast<int32_t>(vector_bytes / element_size);
  bool vectorized = vector_bytes % element_size == 0 &&
      reinterpret_cast<uintptr_t>(rank_major.const_data_ptr()) % vector_bytes == 0 &&
      reinterpret_cast<uintptr_t>(token_major.mutable_data_ptr()) % vector_bytes == 0 &&
      metadata.total_width % elements_per_vector == 0;
  for (size_t source = 0; source < widths.size(); ++source) {
    vectorized = vectorized && widths[source] % elements_per_vector == 0 &&
        metadata.source_offsets[source] % elements_per_vector == 0 &&
        metadata.destination_offsets[source] % elements_per_vector == 0;
  }
  int64_t maximum_work = 0;
  for (int64_t width : widths) {
    maximum_work = std::max<int64_t>(
        maximum_work,
        batch * (vectorized ? width / elements_per_vector : width));
  }
  const int64_t requested_blocks = (maximum_work + threads - 1) / threads;
  const unsigned int blocks_x = static_cast<unsigned int>(
      std::max<int64_t>(1, std::min<int64_t>(requested_blocks, 4096)));
  const dim3 blocks(blocks_x, static_cast<unsigned int>(widths.size()));
  if (vectorized) {
    ragged_rank_major_to_token_major_uint4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint4*>(rank_major.const_data_ptr()),
        static_cast<uint4*>(token_major.mutable_data_ptr()),
        batch,
        metadata,
        elements_per_vector);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        rank_major.scalar_type(),
        "basisserve_ragged_pack_tiled",
        [&] {
          ragged_rank_major_to_token_major_tiled_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  rank_major.const_data_ptr<scalar_t>(),
                  token_major.mutable_data_ptr<scalar_t>(),
                  batch,
                  metadata);
        });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace basisserve::ragged_ag
