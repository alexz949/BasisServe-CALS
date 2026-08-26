#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

namespace basisserve::feature_ag {

void launch_token_to_feature_pack(
    const at::Tensor& token_major,
    at::Tensor& feature_major,
    cudaStream_t stream);

}  // namespace basisserve::feature_ag
