#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace basisserve::feature_ag {

constexpr int kIpcSlotCount = 2;
constexpr int kIpcMaxPhases = 8;
constexpr int kIpcMaxChannels = 8;

enum class UniformIpcAlgorithm : int64_t {
  kAuto = 0,
  kFanout = 1,
  kFanoutWarp = 2,
  kRecursiveDoubling = 3,
  kRing = 4,
};

void launch_token_to_feature_pack(
    const at::Tensor& token_major,
    at::Tensor& feature_major,
    cudaStream_t stream);

void launch_uniform_ipc_allgather(
    const at::Tensor& peer_bases,
    void* local_base,
    int rank,
    int world_size,
    int64_t slot,
    uint64_t epoch,
    size_t slot_stride_bytes,
    size_t block_bytes,
    size_t flags_offset_bytes,
    UniformIpcAlgorithm algorithm,
    int requested_channels,
    cudaStream_t stream,
    bool check_launch);

}  // namespace basisserve::feature_ag
