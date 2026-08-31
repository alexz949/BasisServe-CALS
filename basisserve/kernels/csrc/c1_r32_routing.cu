#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kQueryKeyDim = 128;
constexpr int kRoutingRank = 32;
constexpr int kRoutingPageSize = 64;
constexpr int kQueriesPerKv = 4;
constexpr int kThreadsPerBlock = kQueriesPerKv * kWarpSize;
constexpr int kMaxTopPagesPerQuery = 32;
constexpr int kMaxRoutingPages = 2048;

template <typename scalar_t>
__global__ void project_gqa4_r32_query_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ query_projector,
    scalar_t* __restrict__ query_code,
    int64_t kv_heads,
    int64_t projector_heads,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t projector_stride_head,
    int64_t projector_stride_feature,
    int64_t projector_stride_rank) {
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int rank = static_cast<int>(threadIdx.x) % kWarpSize;
  const int64_t kv_row = static_cast<int64_t>(blockIdx.x);
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t query_head = kv_head * kQueriesPerKv + warp;
  const int64_t projector_head =
      projector_heads == kv_heads ? kv_head : query_head;

  float accumulator = 0.0f;
#pragma unroll
  for (int feature = 0; feature < kQueryKeyDim; ++feature) {
    const float query_value = static_cast<float>(
        query[batch * query_stride_batch +
              query_head * query_stride_head + feature]);
    const float projector_value = static_cast<float>(
        query_projector[projector_head * projector_stride_head +
                        static_cast<int64_t>(feature) *
                            projector_stride_feature +
                        rank * projector_stride_rank]);
    accumulator = fmaf(query_value, projector_value, accumulator);
  }
  query_code[(batch * kv_heads * kQueriesPerKv + query_head) *
                 kRoutingRank +
             rank] = static_cast<scalar_t>(accumulator);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

template <typename scalar_t>
__global__ void gqa4_r32_page_lse_kernel(
    const scalar_t* __restrict__ query_code,
    const scalar_t* __restrict__ routing_sidecar,
    scalar_t* __restrict__ page_log_mass,
    int64_t kv_heads,
    int64_t sequence_length,
    int64_t pages,
    int64_t sidecar_stride_batch,
    int64_t sidecar_stride_head,
    int64_t sidecar_stride_token,
    float scale) {
  __shared__ scalar_t shared_query[kQueriesPerKv][kRoutingRank];
  __shared__ scalar_t
      shared_sidecar[kRoutingPageSize][kRoutingRank];

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int64_t linear_page = static_cast<int64_t>(blockIdx.x);
  const int64_t page = linear_page % pages;
  const int64_t kv_row = linear_page / pages;
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t query_head = kv_head * kQueriesPerKv + warp;
  const int64_t page_start = page * kRoutingPageSize;
  const int64_t remaining_tokens = sequence_length - page_start;
  const int valid_tokens = static_cast<int>(
      remaining_tokens < kRoutingPageSize ? remaining_tokens
                                          : kRoutingPageSize);

  shared_query[warp][lane] =
      query_code[(batch * kv_heads * kQueriesPerKv + query_head) *
                     kRoutingRank +
                 lane];
  for (int index = static_cast<int>(threadIdx.x);
       index < kRoutingPageSize * kRoutingRank;
       index += kThreadsPerBlock) {
    const int token = index / kRoutingRank;
    const int rank = index % kRoutingRank;
    scalar_t value = static_cast<scalar_t>(0.0f);
    if (token < valid_tokens) {
      value = routing_sidecar[
          batch * sidecar_stride_batch + kv_head * sidecar_stride_head +
          (page_start + token) * sidecar_stride_token + rank];
    }
    shared_sidecar[token][rank] = value;
  }
  __syncthreads();

  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;
  for (int token = 0; token < valid_tokens; ++token) {
    float partial = static_cast<float>(shared_query[warp][lane]) *
        static_cast<float>(shared_sidecar[token][lane]);
    const float score = warp_sum(partial) * scale;
    if (lane == 0) {
      const float next_max = fmaxf(running_max, score);
      running_sum = running_sum * __expf(running_max - next_max) +
          __expf(score - next_max);
      running_max = next_max;
    }
  }
  if (lane == 0) {
    page_log_mass[(batch * kv_heads * kQueriesPerKv + query_head) * pages +
                  page] =
        static_cast<scalar_t>(running_max + __logf(running_sum));
  }
}

__device__ __forceinline__ bool better_page(
    float candidate_score,
    int candidate_page,
    float incumbent_score,
    int incumbent_page) {
  if (candidate_page < 0) {
    return false;
  }
  if (incumbent_page < 0) {
    return true;
  }
  return candidate_score > incumbent_score ||
      (candidate_score == incumbent_score && candidate_page < incumbent_page);
}

template <typename scalar_t>
__global__ void gqa4_topk_union_compact_kernel(
    const scalar_t* __restrict__ page_log_mass,
    int64_t* __restrict__ selected_page_ids,
    int32_t* __restrict__ selected_page_counts,
    int64_t kv_heads,
    int pages,
    int top_pages_per_query,
    int output_slots) {
  __shared__ int shared_top_pages
      [kQueriesPerKv][kMaxTopPagesPerQuery];
  __shared__ unsigned int shared_page_bits[kMaxRoutingPages / kWarpSize];

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int64_t kv_row = static_cast<int64_t>(blockIdx.x);
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t query_head = kv_head * kQueriesPerKv + warp;
  const int selected_per_query =
      top_pages_per_query < pages ? top_pages_per_query : pages;
  const int bit_words = (pages + kWarpSize - 1) / kWarpSize;

  for (int index = static_cast<int>(threadIdx.x);
       index < kQueriesPerKv * kMaxTopPagesPerQuery;
       index += kThreadsPerBlock) {
    reinterpret_cast<int*>(shared_top_pages)[index] = -1;
  }
  for (int word = static_cast<int>(threadIdx.x); word < bit_words;
       word += kThreadsPerBlock) {
    shared_page_bits[word] = 0u;
  }
  for (int slot = static_cast<int>(threadIdx.x); slot < output_slots;
       slot += kThreadsPerBlock) {
    selected_page_ids[kv_row * output_slots + slot] = -1;
  }
  __syncthreads();

  uint64_t lane_selected = 0u;
  for (int selection = 0; selection < selected_per_query; ++selection) {
    float best_score = -CUDART_INF_F;
    int best_page = -1;
    for (int page = lane; page < pages; page += kWarpSize) {
      const int local_index = page / kWarpSize;
      if ((lane_selected & (uint64_t{1} << local_index)) != 0u) {
        continue;
      }
      float score = static_cast<float>(
          page_log_mass[(batch * kv_heads * kQueriesPerKv + query_head) *
                            pages +
                        page]);
      if (isnan(score)) {
        score = -CUDART_INF_F;
      }
      if (better_page(score, page, best_score, best_page)) {
        best_score = score;
        best_page = page;
      }
    }

    int winning_lane = lane;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      const float other_score =
          __shfl_down_sync(0xffffffffu, best_score, offset);
      const int other_page =
          __shfl_down_sync(0xffffffffu, best_page, offset);
      const int other_lane =
          __shfl_down_sync(0xffffffffu, winning_lane, offset);
      if (lane < offset &&
          better_page(other_score, other_page, best_score, best_page)) {
        best_score = other_score;
        best_page = other_page;
        winning_lane = other_lane;
      }
    }
    const int winner_page =
        __shfl_sync(0xffffffffu, best_page, 0);
    const int winner_lane =
        __shfl_sync(0xffffffffu, winning_lane, 0);
    if (lane == 0) {
      shared_top_pages[warp][selection] = winner_page;
    }
    if (lane == winner_lane) {
      const int local_index = winner_page / kWarpSize;
      lane_selected |= uint64_t{1} << local_index;
    }
  }
  __syncthreads();

  for (int index = static_cast<int>(threadIdx.x);
       index < kQueriesPerKv * selected_per_query;
       index += kThreadsPerBlock) {
    const int page = reinterpret_cast<int*>(shared_top_pages)[index];
    if (page >= 0) {
      atomicOr(
          &shared_page_bits[page / kWarpSize],
          1u << (page % kWarpSize));
    }
  }
  __syncthreads();

  if (warp == 0) {
    int compact_count = 0;
    for (int base = 0; base < pages; base += kWarpSize) {
      const int page = base + lane;
      const bool present = page < pages &&
          (shared_page_bits[base / kWarpSize] & (1u << lane)) != 0u;
      const unsigned int mask =
          __ballot_sync(0xffffffffu, present);
      const int prefix = __popc(mask & ((1u << lane) - 1u));
      const int base_count =
          __shfl_sync(0xffffffffu, compact_count, 0);
      if (present) {
        selected_page_ids[kv_row * output_slots + base_count + prefix] = page;
      }
      if (lane == 0) {
        compact_count += __popc(mask);
      }
    }
    if (lane == 0) {
      selected_page_counts[kv_row] = compact_count;
    }
  }
}

template <typename scalar_t>
void launch_r32_page_lse(
    const at::Tensor& query,
    const at::Tensor& routing_sidecar,
    const at::Tensor& query_projector,
    const at::Tensor& query_code,
    const at::Tensor& page_log_mass,
    float scale,
    cudaStream_t stream) {
  const int64_t kv_heads = routing_sidecar.size(1);
  const int64_t pages = page_log_mass.size(2);
  const int64_t projection_blocks = query.size(0) * kv_heads;
  project_gqa4_r32_query_kernel<scalar_t>
      <<<static_cast<unsigned int>(projection_blocks),
         kThreadsPerBlock,
         0,
         stream>>>(
          query.const_data_ptr<scalar_t>(),
          query_projector.const_data_ptr<scalar_t>(),
          query_code.mutable_data_ptr<scalar_t>(),
          kv_heads,
          query_projector.size(0),
          query.stride(0),
          query.stride(1),
          query_projector.stride(0),
          query_projector.stride(1),
          query_projector.stride(2));
  const int64_t page_blocks = query.size(0) * kv_heads * pages;
  gqa4_r32_page_lse_kernel<scalar_t>
      <<<static_cast<unsigned int>(page_blocks),
         kThreadsPerBlock,
         0,
         stream>>>(
          query_code.const_data_ptr<scalar_t>(),
          routing_sidecar.const_data_ptr<scalar_t>(),
          page_log_mass.mutable_data_ptr<scalar_t>(),
          kv_heads,
          routing_sidecar.size(2),
          pages,
          routing_sidecar.stride(0),
          routing_sidecar.stride(1),
          routing_sidecar.stride(2),
          scale);
}

template <typename scalar_t>
void launch_r32_topk_gqa_union(
    const at::Tensor& page_log_mass,
    const at::Tensor& selected_page_ids,
    const at::Tensor& selected_page_counts,
    int64_t top_pages_per_query,
    cudaStream_t stream) {
  const int64_t kv_heads = page_log_mass.size(1) / kQueriesPerKv;
  const int64_t blocks = page_log_mass.size(0) * kv_heads;
  gqa4_topk_union_compact_kernel<scalar_t>
      <<<static_cast<unsigned int>(blocks),
         kThreadsPerBlock,
         0,
         stream>>>(
          page_log_mass.const_data_ptr<scalar_t>(),
          selected_page_ids.mutable_data_ptr<int64_t>(),
          selected_page_counts.mutable_data_ptr<int32_t>(),
          kv_heads,
          static_cast<int>(page_log_mass.size(2)),
          static_cast<int>(top_pages_per_query),
          static_cast<int>(selected_page_ids.size(2)));
}

}  // namespace

at::Tensor c1_r32_page_lse_cuda(
    const at::Tensor& query,
    const at::Tensor& routing_sidecar,
    const at::Tensor& query_projector,
    const at::Tensor& query_code,
    const at::Tensor& page_log_mass,
    double scale) {
  TORCH_CHECK(
      query.is_cuda() && routing_sidecar.is_cuda() &&
          query_projector.is_cuda() && query_code.is_cuda() &&
          page_log_mass.is_cuda(),
      "R32 page routing requires CUDA tensors");
  TORCH_CHECK(
      query.device() == routing_sidecar.device() &&
          query.device() == query_projector.device() &&
          query.device() == query_code.device() &&
          query.device() == page_log_mass.device(),
      "R32 page-routing tensors must share one CUDA device");
  TORCH_CHECK(
      query.scalar_type() == routing_sidecar.scalar_type() &&
          query.scalar_type() == query_projector.scalar_type() &&
          query.scalar_type() == query_code.scalar_type() &&
          query.scalar_type() == page_log_mass.scalar_type(),
      "R32 page-routing dtypes differ");
  TORCH_CHECK(
      query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
      "R32 page routing supports FP16 and BF16");
  TORCH_CHECK(
      query.dim() == 4 && routing_sidecar.dim() == 4 &&
          query_projector.dim() == 3 && query_code.dim() == 3 &&
          page_log_mass.dim() == 3,
      "R32 page-routing tensor ranks are incompatible");
  const int64_t batch = query.size(0);
  const int64_t query_heads = query.size(1);
  const int64_t kv_heads = routing_sidecar.size(1);
  const int64_t sequence_length = routing_sidecar.size(2);
  const int64_t pages =
      (sequence_length + kRoutingPageSize - 1) / kRoutingPageSize;
  TORCH_CHECK(
      query.size(2) == 1 && query.size(3) == kQueryKeyDim &&
          routing_sidecar.size(0) == batch &&
          routing_sidecar.size(3) == kRoutingRank &&
          query_heads == kQueriesPerKv * kv_heads,
      "R32 page routing requires QK128, R32, and GQA ratio four");
  TORCH_CHECK(
      query_projector.size(0) == kv_heads ||
          query_projector.size(0) == query_heads,
      "R32 Query projector must be per-KV-head or per-Query-head");
  TORCH_CHECK(
      query_projector.size(1) == kQueryKeyDim &&
          query_projector.size(2) == kRoutingRank,
      "R32 Query projector must have shape [heads,128,32]");
  TORCH_CHECK(
      query_code.size(0) == batch && query_code.size(1) == query_heads &&
          query_code.size(2) == kRoutingRank &&
          page_log_mass.size(0) == batch &&
          page_log_mass.size(1) == query_heads &&
          page_log_mass.size(2) == pages,
      "R32 page-routing workspace/output shapes are invalid");
  TORCH_CHECK(
      sequence_length > 0 && pages <= kMaxRoutingPages,
      "R32 page routing supports at most 128K tokens with page size 64");
  TORCH_CHECK(
      query.stride(3) == 1 && routing_sidecar.stride(3) == 1 &&
          query_projector.stride(2) == 1 && query_code.is_contiguous() &&
          page_log_mass.is_contiguous(),
      "R32 page-routing feature/output dimensions must be contiguous");
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "R32 page-routing scale must be finite and positive");

  c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      query.scalar_type(), "basisserve_r32_page_lse", [&] {
        launch_r32_page_lse<scalar_t>(
            query,
            routing_sidecar,
            query_projector,
            query_code,
            page_log_mass,
            static_cast<float>(scale),
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return page_log_mass;
}

at::Tensor c1_r32_topk_gqa_union_cuda(
    const at::Tensor& page_log_mass,
    const at::Tensor& selected_page_ids,
    const at::Tensor& selected_page_counts,
    int64_t top_pages_per_query) {
  TORCH_CHECK(
      page_log_mass.is_cuda() && selected_page_ids.is_cuda() &&
          selected_page_counts.is_cuda(),
      "R32 Top-k union requires CUDA tensors");
  TORCH_CHECK(
      page_log_mass.device() == selected_page_ids.device() &&
          page_log_mass.device() == selected_page_counts.device(),
      "R32 Top-k union tensors must share one CUDA device");
  TORCH_CHECK(
      page_log_mass.scalar_type() == at::kHalf ||
          page_log_mass.scalar_type() == at::kBFloat16,
      "R32 Top-k union page scores must be FP16 or BF16");
  TORCH_CHECK(
      selected_page_ids.scalar_type() == at::kLong &&
          selected_page_counts.scalar_type() == at::kInt,
      "R32 Top-k union outputs must be int64 IDs and int32 counts");
  TORCH_CHECK(
      page_log_mass.dim() == 3 && selected_page_ids.dim() == 3 &&
          selected_page_counts.dim() == 2,
      "R32 Top-k union tensor ranks are incompatible");
  const int64_t batch = page_log_mass.size(0);
  const int64_t query_heads = page_log_mass.size(1);
  const int64_t pages = page_log_mass.size(2);
  TORCH_CHECK(
      top_pages_per_query > 0 &&
          top_pages_per_query <= kMaxTopPagesPerQuery,
      "R32 Top-k union supports 1 through 32 pages per Query head");
  TORCH_CHECK(
      query_heads > 0 && query_heads % kQueriesPerKv == 0 && pages > 0 &&
          pages <= kMaxRoutingPages,
      "R32 Top-k union requires GQA ratio four and at most 2048 pages");
  const int64_t kv_heads = query_heads / kQueriesPerKv;
  const int64_t output_slots =
      std::min(pages, top_pages_per_query * kQueriesPerKv);
  TORCH_CHECK(
      selected_page_ids.size(0) == batch &&
          selected_page_ids.size(1) == kv_heads &&
          selected_page_ids.size(2) == output_slots &&
          selected_page_counts.size(0) == batch &&
          selected_page_counts.size(1) == kv_heads,
      "R32 Top-k union output shapes are invalid");
  TORCH_CHECK(
      page_log_mass.is_contiguous() && selected_page_ids.is_contiguous() &&
          selected_page_counts.is_contiguous(),
      "R32 Top-k union tensors must be contiguous");

  c10::cuda::CUDAGuard device_guard(page_log_mass.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
                                  page_log_mass.get_device())
                                  .stream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      page_log_mass.scalar_type(), "basisserve_r32_topk_gqa_union", [&] {
        launch_r32_topk_gqa_union<scalar_t>(
            page_log_mass,
            selected_page_ids,
            selected_page_counts,
            top_pages_per_query,
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return selected_page_ids;
}
