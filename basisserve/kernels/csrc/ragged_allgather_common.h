#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace basisserve::ragged_ag {

constexpr int kMaximumWorldSize = 32;

struct RaggedPackMetadata {
  int32_t world_size;
  int64_t total_width;
  int32_t widths[kMaximumWorldSize];
  int64_t source_offsets[kMaximumWorldSize];
  int64_t destination_offsets[kMaximumWorldSize];
};

void launch_ragged_pack(
    const at::Tensor& rank_major,
    at::Tensor& token_major,
    const std::vector<int64_t>& widths,
    int64_t batch,
    cudaStream_t stream);

}  // namespace basisserve::ragged_ag
