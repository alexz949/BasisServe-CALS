#include "feature_ragged_allgather_common.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace basisserve::feature_ag {
namespace {

constexpr size_t kVectorBytes = sizeof(uint4);

__device__ __forceinline__ void store_release_system_u64(
    uint64_t* pointer,
    uint64_t value) {
#if __CUDA_ARCH__ >= 700
  asm volatile(
      "st.release.sys.global.u64 [%0], %1;"
      :
      : "l"(pointer), "l"(value)
      : "memory");
#else
  __threadfence_system();
  *reinterpret_cast<volatile uint64_t*>(pointer) = value;
#endif
}

__device__ __forceinline__ uint64_t load_acquire_system_u64(
    const uint64_t* pointer) {
#if __CUDA_ARCH__ >= 700
  uint64_t value;
  asm volatile(
      "ld.acquire.sys.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(pointer)
      : "memory");
  return value;
#else
  const uint64_t value = *reinterpret_cast<const volatile uint64_t*>(pointer);
  __threadfence_system();
  return value;
#endif
}

__device__ __forceinline__ uint64_t* flag_pointer(
    uint64_t base,
    size_t flags_offset_bytes,
    int64_t slot,
    int channel,
    int phase,
    int source,
    int world_size) {
  const size_t index =
      (((static_cast<size_t>(slot) * kIpcMaxChannels +
         static_cast<size_t>(channel)) *
            kIpcMaxPhases +
        static_cast<size_t>(phase)) *
           static_cast<size_t>(world_size)) +
      static_cast<size_t>(source);
  return reinterpret_cast<uint64_t*>(base + flags_offset_bytes + index * sizeof(uint64_t));
}

struct ByteRange {
  size_t offset;
  size_t bytes;
};

__device__ __forceinline__ ByteRange partition_bytes(
    size_t total_bytes,
    int channel,
    int channels) {
  // Keep channel boundaries 16-byte aligned whenever the source block itself
  // is vector-aligned. The quotient/remainder form avoids total_bytes*channel
  // overflow and distributes at most one extra unit to early channels.
  const bool vector_units = (total_bytes & (kVectorBytes - 1)) == 0;
  const size_t units = vector_units ? total_bytes / kVectorBytes : total_bytes;
  const size_t quotient = units / static_cast<size_t>(channels);
  const size_t remainder = units % static_cast<size_t>(channels);
  const size_t channel_index = static_cast<size_t>(channel);
  const size_t begin_units =
      quotient * channel_index + (channel_index < remainder ? channel_index : remainder);
  const size_t channel_units = quotient + (channel_index < remainder ? 1 : 0);
  const size_t scale = vector_units ? kVectorBytes : 1;
  return ByteRange{begin_units * scale, channel_units * scale};
}

__device__ __forceinline__ void copy_one_destination(
    const char* source,
    char* destination,
    size_t bytes) {
  const uintptr_t alignment =
      reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination) | bytes;
  if ((alignment & (kVectorBytes - 1)) == 0) {
    const auto* source_vectors = reinterpret_cast<const uint4*>(source);
    auto* destination_vectors = reinterpret_cast<uint4*>(destination);
    const size_t vectors = bytes / kVectorBytes;
    for (size_t index = threadIdx.x; index < vectors; index += blockDim.x) {
      destination_vectors[index] = source_vectors[index];
    }
    return;
  }

  for (size_t index = threadIdx.x; index < bytes; index += blockDim.x) {
    destination[index] = source[index];
  }
}

template <int kWorldSize>
__device__ __forceinline__ void copy_fanout(
    const char* source,
    const uint64_t* peer_bases,
    int rank,
    size_t destination_offset,
    size_t bytes) {
  const uintptr_t source_alignment = reinterpret_cast<uintptr_t>(source) | bytes;
  bool vectorized = (source_alignment & (kVectorBytes - 1)) == 0;
#pragma unroll
  for (int peer = 0; peer < kWorldSize; ++peer) {
    if (peer != rank) {
      const uintptr_t destination = peer_bases[peer] + destination_offset;
      vectorized = vectorized && ((destination & (kVectorBytes - 1)) == 0);
    }
  }

  if (vectorized) {
    const auto* source_vectors = reinterpret_cast<const uint4*>(source);
    const size_t vectors = bytes / kVectorBytes;
    for (size_t index = threadIdx.x; index < vectors; index += blockDim.x) {
      const uint4 value = source_vectors[index];
#pragma unroll
      for (int peer = 0; peer < kWorldSize; ++peer) {
        if (peer == rank) {
          continue;
        }
        auto* destination_vectors = reinterpret_cast<uint4*>(
            peer_bases[peer] + destination_offset);
        destination_vectors[index] = value;
      }
    }
    return;
  }

  for (size_t index = threadIdx.x; index < bytes; index += blockDim.x) {
    const char value = source[index];
#pragma unroll
    for (int peer = 0; peer < kWorldSize; ++peer) {
      if (peer == rank) {
        continue;
      }
      auto* destination = reinterpret_cast<char*>(peer_bases[peer] + destination_offset);
      destination[index] = value;
    }
  }
}

__device__ __forceinline__ void wait_for_epoch(
    const uint64_t* flag,
    uint64_t epoch) {
  unsigned int delay = 8;
  while (load_acquire_system_u64(flag) < epoch) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(delay);
    delay = delay < 256 ? delay * 2 : 256;
#endif
  }
}

template <int kWorldSize>
__global__ void uniform_ipc_fanout_warp_kernel(
    const uint64_t* __restrict__ peer_bases_global,
    char* local_base,
    int rank,
    int64_t slot,
    uint64_t epoch,
    size_t slot_stride_bytes,
    size_t block_bytes,
    size_t flags_offset_bytes) {
  __shared__ uint64_t peer_bases[kWorldSize];
  if (threadIdx.x < kWorldSize) {
    peer_bases[threadIdx.x] = peer_bases_global[threadIdx.x];
  }
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  if (warp < kWorldSize - 1) {
    const int peer = warp < rank ? warp : warp + 1;
    const size_t slot_offset = static_cast<size_t>(slot) * slot_stride_bytes;
    const size_t block_offset = static_cast<size_t>(rank) * block_bytes;
    const auto* source = local_base + slot_offset + block_offset;
    auto* destination = reinterpret_cast<char*>(
        peer_bases[peer] + slot_offset + block_offset);
    const uintptr_t alignment = reinterpret_cast<uintptr_t>(source) |
        reinterpret_cast<uintptr_t>(destination) | block_bytes;
    if ((alignment & (kVectorBytes - 1)) == 0) {
      const auto* source_vectors = reinterpret_cast<const uint4*>(source);
      auto* destination_vectors = reinterpret_cast<uint4*>(destination);
      const size_t vectors = block_bytes / kVectorBytes;
      for (size_t index = lane; index < vectors; index += warpSize) {
        destination_vectors[index] = source_vectors[index];
      }
    } else {
      for (size_t index = lane; index < block_bytes; index += warpSize) {
        destination[index] = source[index];
      }
    }
  }

  // __syncthreads() strongly orders every copy thread before thread 0. The
  // system-scope release store therefore publishes the whole CTA's remote
  // writes through transitive causality; a per-copy-thread system fence would
  // dominate the 1--2 KiB decode case.
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int peer = 0; peer < kWorldSize; ++peer) {
      store_release_system_u64(
          flag_pointer(
              peer_bases[peer], flags_offset_bytes, slot, 0, 0, rank, kWorldSize),
          epoch);
    }
#pragma unroll
    for (int source_rank = 0; source_rank < kWorldSize; ++source_rank) {
      wait_for_epoch(
          flag_pointer(
              peer_bases[rank],
              flags_offset_bytes,
              slot,
              0,
              0,
              source_rank,
              kWorldSize),
          epoch);
    }
  }
  // Propagate thread 0's system-scope acquire to the CTA and delay kernel
  // completion until every source has published this epoch.
  __syncthreads();
}

template <int kWorldSize>
__global__ void uniform_ipc_fanout_kernel(
    const uint64_t* __restrict__ peer_bases_global,
    char* local_base,
    int rank,
    int64_t slot,
    uint64_t epoch,
    size_t slot_stride_bytes,
    size_t block_bytes,
    size_t flags_offset_bytes) {
  __shared__ uint64_t peer_bases[kWorldSize];
  if (threadIdx.x < kWorldSize) {
    peer_bases[threadIdx.x] = peer_bases_global[threadIdx.x];
  }
  __syncthreads();

  const int channel = static_cast<int>(blockIdx.x);
  const int channels = static_cast<int>(gridDim.x);
  const ByteRange range = partition_bytes(block_bytes, channel, channels);
  const size_t slot_offset = static_cast<size_t>(slot) * slot_stride_bytes;
  const size_t block_offset = static_cast<size_t>(rank) * block_bytes;
  const auto* source = local_base + slot_offset + block_offset + range.offset;
  copy_fanout<kWorldSize>(
      source,
      peer_bases,
      rank,
      slot_offset + block_offset + range.offset,
      range.bytes);

  // Each ready flag has a single writer, so no peer atomics are required.
  // The block barrier orders every copy thread before thread 0's system-scope
  // release, publishing the whole CTA without a system fence per copy thread.
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int peer = 0; peer < kWorldSize; ++peer) {
      store_release_system_u64(
          flag_pointer(
              peer_bases[peer], flags_offset_bytes, slot, channel, 0, rank, kWorldSize),
          epoch);
    }
#pragma unroll
    for (int source_rank = 0; source_rank < kWorldSize; ++source_rank) {
      wait_for_epoch(
          flag_pointer(
              peer_bases[rank],
              flags_offset_bytes,
              slot,
              channel,
              0,
              source_rank,
              kWorldSize),
          epoch);
    }
  }
  __syncthreads();
}

template <int kWorldSize>
__global__ void uniform_ipc_recursive_doubling_kernel(
    const uint64_t* __restrict__ peer_bases_global,
    char* local_base,
    int rank,
    int64_t slot,
    uint64_t epoch,
    size_t slot_stride_bytes,
    size_t block_bytes,
    size_t flags_offset_bytes) {
  static_assert((kWorldSize & (kWorldSize - 1)) == 0, "world size must be a power of two");
  __shared__ uint64_t peer_bases[kWorldSize];
  if (threadIdx.x < kWorldSize) {
    peer_bases[threadIdx.x] = peer_bases_global[threadIdx.x];
  }
  __syncthreads();

  const size_t slot_offset = static_cast<size_t>(slot) * slot_stride_bytes;
  int phase = 0;
  for (int group_size = 1; group_size < kWorldSize; group_size <<= 1, ++phase) {
    const int partner = rank ^ group_size;
    const int group_start = rank & ~(group_size - 1);
    const size_t group_offset = static_cast<size_t>(group_start) * block_bytes;
    const size_t group_bytes = static_cast<size_t>(group_size) * block_bytes;
    const auto* source = local_base + slot_offset + group_offset;
    auto* destination = reinterpret_cast<char*>(
        peer_bases[partner] + slot_offset + group_offset);
    copy_one_destination(source, destination, group_bytes);

    __syncthreads();
    if (threadIdx.x == 0) {
      store_release_system_u64(
          flag_pointer(
              peer_bases[partner],
              flags_offset_bytes,
              slot,
              0,
              phase,
              rank,
              kWorldSize),
          epoch);
      wait_for_epoch(
          flag_pointer(
              peer_bases[rank],
              flags_offset_bytes,
              slot,
              0,
              phase,
              partner,
              kWorldSize),
          epoch);
    }
    __syncthreads();
  }
}

template <int kWorldSize>
__global__ void uniform_ipc_ring_kernel(
    const uint64_t* __restrict__ peer_bases_global,
    char* local_base,
    int rank,
    int64_t slot,
    uint64_t epoch,
    size_t slot_stride_bytes,
    size_t block_bytes,
    size_t flags_offset_bytes) {
  __shared__ uint64_t peer_bases[kWorldSize];
  if (threadIdx.x < kWorldSize) {
    peer_bases[threadIdx.x] = peer_bases_global[threadIdx.x];
  }
  __syncthreads();

  const int right = (rank + 1) % kWorldSize;
  const int left = (rank - 1 + kWorldSize) % kWorldSize;
  const int channel = static_cast<int>(blockIdx.x);
  const int channels = static_cast<int>(gridDim.x);
  const ByteRange range = partition_bytes(block_bytes, channel, channels);
  const size_t slot_offset = static_cast<size_t>(slot) * slot_stride_bytes;
  for (int phase = 0; phase < kWorldSize - 1; ++phase) {
    const int source_rank = (rank - phase + kWorldSize) % kWorldSize;
    const size_t block_offset = static_cast<size_t>(source_rank) * block_bytes;
    const auto* source = local_base + slot_offset + block_offset + range.offset;
    auto* destination = reinterpret_cast<char*>(
        peer_bases[right] + slot_offset + block_offset + range.offset);
    copy_one_destination(source, destination, range.bytes);

    __syncthreads();
    if (threadIdx.x == 0) {
      store_release_system_u64(
          flag_pointer(
              peer_bases[right],
              flags_offset_bytes,
              slot,
              channel,
              phase,
              rank,
              kWorldSize),
          epoch);
      wait_for_epoch(
          flag_pointer(
              peer_bases[rank],
              flags_offset_bytes,
              slot,
              channel,
              phase,
              left,
              kWorldSize),
          epoch);
    }
    __syncthreads();
  }
}

UniformIpcAlgorithm select_algorithm(
    UniformIpcAlgorithm requested,
    int world_size,
    size_t block_bytes) {
  if (requested != UniformIpcAlgorithm::kAuto) {
    return requested;
  }
  // The thresholds are intentionally conservative starting points for the
  // all-gather-only autotuner. They can be overridden by selecting an explicit
  // algorithm and should not be treated as universal hardware constants.
  const size_t fanout_limit = world_size <= 4 ? 16 * 1024 : 8 * 1024;
  if (block_bytes <= fanout_limit) {
    return UniformIpcAlgorithm::kFanout;
  }
  if (block_bytes <= 128 * 1024) {
    return UniformIpcAlgorithm::kRecursiveDoubling;
  }
  return UniformIpcAlgorithm::kRing;
}

int select_channel_count(
    UniformIpcAlgorithm algorithm,
    size_t block_bytes,
    int requested_channels) {
  TORCH_CHECK(
      requested_channels == 0 || requested_channels == 1 || requested_channels == 2 ||
          requested_channels == 4 || requested_channels == 8,
      "IPC channel count must be 0 (auto), 1, 2, 4, or 8; got ",
      requested_channels);
  if (requested_channels > 0) {
    TORCH_CHECK(
        requested_channels == 1 || algorithm == UniformIpcAlgorithm::kFanout ||
            algorithm == UniformIpcAlgorithm::kRing,
        "multi-CTA IPC channels are currently supported only by fanout and ring");
    return requested_channels;
  }
  if (algorithm != UniformIpcAlgorithm::kRing) {
    return 1;
  }
  // Large ring payloads are split into independent chunk channels. Each CTA
  // runs the full ring for one disjoint byte range and owns separate flags.
  if (block_bytes <= 128 * 1024) {
    return 1;
  }
  if (block_bytes <= 512 * 1024) {
    return 4;
  }
  return kIpcMaxChannels;
}

int select_copy_threads(size_t block_bytes) {
  if (block_bytes <= 2 * 1024) {
    return 64;
  }
  if (block_bytes <= 16 * 1024) {
    return 128;
  }
  return 256;
}

template <int kWorldSize>
void launch_for_world_size(
    const at::Tensor& peer_bases,
    void* local_base,
    int rank,
    int64_t slot,
    uint64_t epoch,
    size_t slot_stride_bytes,
    size_t block_bytes,
    size_t flags_offset_bytes,
    UniformIpcAlgorithm algorithm,
    int channels,
    cudaStream_t stream,
    bool check_launch) {
  const auto* bases = reinterpret_cast<const uint64_t*>(
      peer_bases.const_data_ptr<int64_t>());
  auto* local = static_cast<char*>(local_base);
  size_t maximum_copy_bytes = block_bytes;
  if (algorithm == UniformIpcAlgorithm::kRecursiveDoubling) {
    constexpr size_t kLargestGroup = static_cast<size_t>(kWorldSize / 2);
    TORCH_CHECK(
        block_bytes <= std::numeric_limits<size_t>::max() / kLargestGroup,
        "recursive-doubling copy span overflows size_t");
    maximum_copy_bytes *= kLargestGroup;
  }
  if ((algorithm == UniformIpcAlgorithm::kFanout ||
       algorithm == UniformIpcAlgorithm::kRing) &&
      channels > 1) {
    maximum_copy_bytes =
        (maximum_copy_bytes + static_cast<size_t>(channels) - 1) /
        static_cast<size_t>(channels);
  }
  const int copy_threads = select_copy_threads(maximum_copy_bytes);
  switch (algorithm) {
    case UniformIpcAlgorithm::kFanout:
      uniform_ipc_fanout_kernel<kWorldSize><<<channels, copy_threads, 0, stream>>>(
          bases,
          local,
          rank,
          slot,
          epoch,
          slot_stride_bytes,
          block_bytes,
          flags_offset_bytes);
      break;
    case UniformIpcAlgorithm::kFanoutWarp:
      TORCH_CHECK(channels == 1, "fanout_warp supports exactly one CTA channel");
      uniform_ipc_fanout_warp_kernel<kWorldSize>
          <<<1, (kWorldSize - 1) * 32, 0, stream>>>(
              bases,
              local,
              rank,
              slot,
              epoch,
              slot_stride_bytes,
              block_bytes,
              flags_offset_bytes);
      break;
    case UniformIpcAlgorithm::kRecursiveDoubling:
      TORCH_CHECK(channels == 1, "recursive doubling supports exactly one CTA channel");
      uniform_ipc_recursive_doubling_kernel<kWorldSize><<<1, copy_threads, 0, stream>>>(
          bases,
          local,
          rank,
          slot,
          epoch,
          slot_stride_bytes,
          block_bytes,
          flags_offset_bytes);
      break;
    case UniformIpcAlgorithm::kRing:
      uniform_ipc_ring_kernel<kWorldSize><<<channels, copy_threads, 0, stream>>>(
          bases,
          local,
          rank,
          slot,
          epoch,
          slot_stride_bytes,
          block_bytes,
          flags_offset_bytes);
      break;
    case UniformIpcAlgorithm::kAuto:
      TORCH_CHECK(false, "internal error: automatic IPC algorithm was not resolved");
      break;
  }
  if (check_launch) {
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

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
    UniformIpcAlgorithm requested_algorithm,
    int requested_channels,
    cudaStream_t stream,
    bool check_launch) {
  // The owning prepared plan validates the peer table, arena geometry, rank,
  // slot count, and stream once. This boundary is intentionally internal-only
  // so the serving fast path does not repeat tensor metadata checks.
  const UniformIpcAlgorithm algorithm =
      select_algorithm(requested_algorithm, world_size, block_bytes);
  const int channels =
      select_channel_count(algorithm, block_bytes, requested_channels);
  switch (world_size) {
    case 2:
      launch_for_world_size<2>(
          peer_bases,
          local_base,
          rank,
          slot,
          epoch,
          slot_stride_bytes,
          block_bytes,
          flags_offset_bytes,
          algorithm,
          channels,
          stream,
          check_launch);
      break;
    case 4:
      launch_for_world_size<4>(
          peer_bases,
          local_base,
          rank,
          slot,
          epoch,
          slot_stride_bytes,
          block_bytes,
          flags_offset_bytes,
          algorithm,
          channels,
          stream,
          check_launch);
      break;
    case 8:
      launch_for_world_size<8>(
          peer_bases,
          local_base,
          rank,
          slot,
          epoch,
          slot_stride_bytes,
          block_bytes,
          flags_offset_bytes,
          algorithm,
          channels,
          stream,
          check_launch);
      break;
    default:
      TORCH_CHECK(false, "uniform IPC AllGather supports TP2/TP4/TP8; got TP", world_size);
  }
}

}  // namespace basisserve::feature_ag
