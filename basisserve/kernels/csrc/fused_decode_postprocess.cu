#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>

namespace {

constexpr int kThreads = 256;
constexpr int kQueriesPerKv = 4;
constexpr int kMaximumCandidates = 512;
constexpr int kItemsPerThread = 2;
constexpr int kSelectedPages = 62;
constexpr int kPageSize = 32;
constexpr int kRecentTokens = 64;
constexpr int kSupportTokens = kSelectedPages * kPageSize + kRecentTokens;
constexpr int kSelectionNetwork = 64;

__global__ void select_and_pack_kernel(
    const float* __restrict__ page_log_mass,
    const int64_t* __restrict__ candidate_ids,
    int64_t* __restrict__ selected_pages_output,
    int64_t* __restrict__ support_ids,
    int64_t kv_heads,
    int pages,
    int64_t historical,
    int64_t end,
    int64_t input_stride_batch,
    int64_t input_stride_head,
    int64_t input_stride_query) {
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  using BlockSort =
      cub::BlockRadixSort<float, kThreads, kItemsPerThread, int>;
  union TemporaryStorage {
    typename BlockReduce::TempStorage reduce;
    typename BlockSort::TempStorage sort;
  };
  __shared__ TemporaryStorage temporary;
  __shared__ float head_maximum[kQueriesPerKv];
  __shared__ float head_inverse_sum[kQueriesPerKv];
  __shared__ int selected_positions[kSelectionNetwork];
  __shared__ int64_t selected_pages[kSelectedPages];

  const int thread = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = row / kv_heads;
  const int64_t kv_head = row % kv_heads;
  const int64_t input_base =
      batch * input_stride_batch + kv_head * input_stride_head;

  if (thread == 0) {
    selected_positions[0] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int head = 0; head < kQueriesPerKv; ++head) {
    float local_maximum = -FLT_MAX;
    for (int page = thread; page < pages; page += kThreads) {
      if (page != 0) {
        local_maximum = fmaxf(
            local_maximum,
            page_log_mass[
                input_base + head * input_stride_query + page]);
      }
    }
    const float maximum =
        BlockReduce(temporary.reduce).Reduce(local_maximum, cub::Max());
    if (thread == 0) {
      head_maximum[head] = maximum;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int page = thread; page < pages; page += kThreads) {
      if (page != 0) {
        local_sum += __expf(
            page_log_mass[
                input_base + head * input_stride_query + page] -
            head_maximum[head]);
      }
    }
    const float sum = BlockReduce(temporary.reduce).Sum(local_sum);
    if (thread == 0) {
      head_inverse_sum[head] = 1.0f / sum;
    }
    __syncthreads();
  }

  float scores[kItemsPerThread];
  int page_positions[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int page = thread * kItemsPerThread + item;
    page_positions[item] = page;
    float group_score = -FLT_MAX;
    if (page > 0 && page < pages) {
      group_score = 0.0f;
#pragma unroll
      for (int head = 0; head < kQueriesPerKv; ++head) {
        const float probability = __expf(
            page_log_mass[
                input_base + head * input_stride_query + page] -
            head_maximum[head]) *
            head_inverse_sum[head];
        group_score = fmaxf(group_score, probability);
      }
    }
    scores[item] = group_score;
  }

  BlockSort(temporary.sort).SortDescending(scores, page_positions);
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int rank = thread * kItemsPerThread + item;
    if (rank < kSelectedPages - 1) {
      selected_positions[rank + 1] = page_positions[item];
    }
  }
  __syncthreads();

  if (thread >= kSelectedPages && thread < kSelectionNetwork) {
    selected_positions[thread] = INT_MAX;
  }
  __syncthreads();
  for (int width = 2; width <= kSelectionNetwork; width *= 2) {
    for (int stride = width / 2; stride > 0; stride /= 2) {
      if (thread < kSelectionNetwork) {
        const int partner = thread ^ stride;
        if (partner > thread) {
          const int left = selected_positions[thread];
          const int right = selected_positions[partner];
          const bool ascending = (thread & width) == 0;
          if ((ascending && left > right) ||
              (!ascending && left < right)) {
            selected_positions[thread] = right;
            selected_positions[partner] = left;
          }
        }
      }
      __syncthreads();
    }
  }

  if (thread < kSelectedPages) {
    const int64_t page = candidate_ids[row * pages + selected_positions[thread]];
    selected_pages[thread] = page;
    selected_pages_output[row * kSelectedPages + thread] = page;
  }
  __syncthreads();

  for (int index = thread; index < kSelectedPages * kPageSize;
       index += kThreads) {
    const int page_index = index / kPageSize;
    const int page_offset = index % kPageSize;
    const int64_t token = selected_pages[page_index] * kPageSize + page_offset;
    support_ids[row * kSupportTokens + index] =
        token < historical ? token : -1;
  }
  for (int index = thread; index < kRecentTokens; index += kThreads) {
    const int64_t token = historical + index;
    support_ids[
        row * kSupportTokens + kSelectedPages * kPageSize + index] =
        token < end ? token : -1;
  }
}

}  // namespace

void select_and_pack(
    const at::Tensor& page_log_mass,
    const at::Tensor& candidate_ids,
    const at::Tensor& selected_pages,
    const at::Tensor& support_ids,
    int64_t historical,
    int64_t end) {
  assert(page_log_mass.is_cuda() && candidate_ids.is_cuda());
  assert(selected_pages.is_cuda() && support_ids.is_cuda());
  assert(page_log_mass.scalar_type() == at::kFloat);
  assert(candidate_ids.scalar_type() == at::kLong);
  assert(selected_pages.scalar_type() == at::kLong);
  assert(support_ids.scalar_type() == at::kLong);
  assert(page_log_mass.dim() == 4 && page_log_mass.size(2) == kQueriesPerKv);
  const int64_t batch = page_log_mass.size(0);
  const int64_t kv_heads = page_log_mass.size(1);
  const int64_t pages = page_log_mass.size(3);
  assert(pages >= kSelectedPages && pages <= kMaximumCandidates);
  assert(candidate_ids.sizes() == at::IntArrayRef({batch, kv_heads, pages}));
  assert(selected_pages.sizes() ==
         at::IntArrayRef({batch, kv_heads, kSelectedPages}));
  assert(support_ids.sizes() ==
         at::IntArrayRef({batch, kv_heads, kSupportTokens}));
  assert(candidate_ids.is_contiguous() && selected_pages.is_contiguous());
  assert(support_ids.is_contiguous() && page_log_mass.stride(3) == 1);
  assert(historical >= 0 && end - historical == kRecentTokens);

  c10::cuda::CUDAGuard guard(page_log_mass.device());
  select_and_pack_kernel<<<
      static_cast<unsigned int>(batch * kv_heads),
      kThreads,
      0,
      c10::cuda::getCurrentCUDAStream(page_log_mass.get_device()).stream()>>>(
      page_log_mass.const_data_ptr<float>(),
      candidate_ids.const_data_ptr<int64_t>(),
      selected_pages.mutable_data_ptr<int64_t>(),
      support_ids.mutable_data_ptr<int64_t>(),
      kv_heads,
      static_cast<int>(pages),
      historical,
      end,
      page_log_mass.stride(0),
      page_log_mass.stride(1),
      page_log_mass.stride(2));
  assert(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("select_and_pack", &select_and_pack);
}
