#include "feature_ragged_allgather_common.h"

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace basisserve::feature_ag {
namespace {

constexpr int kTile = 32;
constexpr int kBlockRows = 8;

template <typename scalar_t>
__global__ void token_to_feature_pack_kernel(
    const scalar_t* __restrict__ token_major,
    scalar_t* __restrict__ feature_major,
    int64_t tokens,
    int64_t width) {
  __shared__ scalar_t tile[kTile][kTile + 1];

  const int64_t feature =
      static_cast<int64_t>(blockIdx.x) * kTile + threadIdx.x;
  const int64_t token_base =
      static_cast<int64_t>(blockIdx.y) * kTile + threadIdx.y;

#pragma unroll
  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    const int64_t token = token_base + offset;
    if (feature < width && token < tokens) {
      tile[threadIdx.y + offset][threadIdx.x] =
          token_major[token * width + feature];
    }
  }
  __syncthreads();

  const int64_t output_token =
      static_cast<int64_t>(blockIdx.y) * kTile + threadIdx.x;
  const int64_t feature_base =
      static_cast<int64_t>(blockIdx.x) * kTile + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    const int64_t output_feature = feature_base + offset;
    if (output_feature < width && output_token < tokens) {
      feature_major[output_feature * tokens + output_token] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

}  // namespace

void launch_token_to_feature_pack(
    const at::Tensor& token_major,
    at::Tensor& feature_major,
    cudaStream_t stream) {
  TORCH_CHECK(token_major.is_cuda(), "token-major input must be CUDA");
  TORCH_CHECK(feature_major.is_cuda(), "feature-major output must be CUDA");
  TORCH_CHECK(token_major.dim() == 2, "token-major input must be a matrix");
  TORCH_CHECK(feature_major.dim() == 2, "feature-major output must be a matrix");
  TORCH_CHECK(token_major.is_contiguous(), "token-major input must be contiguous");
  TORCH_CHECK(feature_major.is_contiguous(), "feature-major output must be contiguous");
  TORCH_CHECK(
      token_major.scalar_type() == feature_major.scalar_type(),
      "packed input and output dtypes differ");
  TORCH_CHECK(
      token_major.get_device() == feature_major.get_device(),
      "packed input and output devices differ");
  TORCH_CHECK(
      feature_major.size(0) == token_major.size(1) &&
          feature_major.size(1) == token_major.size(0),
      "feature-major output must be the transposed input shape");

  const int64_t tokens = token_major.size(0);
  const int64_t width = token_major.size(1);
  const dim3 threads(kTile, kBlockRows);
  const dim3 blocks(
      static_cast<unsigned int>((width + kTile - 1) / kTile),
      static_cast<unsigned int>((tokens + kTile - 1) / kTile));
  if (token_major.scalar_type() == at::ScalarType::Byte) {
    token_to_feature_pack_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
        token_major.const_data_ptr<uint8_t>(),
        feature_major.mutable_data_ptr<uint8_t>(),
        tokens,
        width);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        token_major.scalar_type(),
        "basisserve_token_to_feature_pack",
        [&] {
          token_to_feature_pack_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              token_major.const_data_ptr<scalar_t>(),
              feature_major.mutable_data_ptr<scalar_t>(),
              tokens,
              width);
        });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace basisserve::feature_ag
