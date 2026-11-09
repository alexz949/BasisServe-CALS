// Replaces select_full_group_max_pages_and_pack_cuda for any page size dividing 32.
//
// The archived selector normalizes and sorts every page of a KV head inside one block and caps
// at 8192 pages and 128 selections; Page4 at 128K has 32K pages and 496 selections, more data
// than one block can stream quickly. This selector is an exact top-k over the same full-scan page
// scores (it is not two-stage routing: no page is scored approximately or skipped):
//   1. many blocks reduce each group of 8 pages to its per-head maximum and accumulate the
//      per-head log-sum-exp;
//   2. one block per KV head picks the groups that can hold a selected page (those whose best
//      page ranks among the needed count), then selects among only their pages.
// A page score is the largest log-probability over the query heads of a KV head, which orders
// pages exactly like the archived GQA-max probability. page_log_mass is consumed: candidate
// pages have their first query head overwritten with the score.
#include <cub/block/block_scan.cuh>

namespace {

constexpr int kSelectThreads = 1024;
constexpr int kGroupPages = 8;
constexpr int kGroupThreads = 256;
constexpr int kMaxGroups = 4096;
constexpr int kMaxGroupBlocks = kMaxGroups / kGroupThreads;
constexpr int kRoutedPages = 1984 / kPageSize;
constexpr int kPinnedPages = 32 / kPageSize;
constexpr int kNeededPages = kRoutedPages - kPinnedPages;
constexpr int kRecentTokens = 64;
constexpr int kHistogramBins = 2048;
constexpr int kBinsPerThread = kHistogramBins / kSelectThreads;
// Scratch words per KV head: log-sum-exp partials, per-head group maxima, candidate groups.
constexpr int kPartialWords = kMaxGroupBlocks * kQueriesPerKv * 2;
constexpr int kGroupWords = kMaxGroups * kQueriesPerKv;
constexpr int kScratchWords = kPartialWords + kGroupWords + kMaxGroups;

struct LogSum {
  float maximum[kQueriesPerKv];
  float sum[kQueriesPerKv];
};

struct CombineLogSum {
  __device__ __forceinline__ LogSum operator()(const LogSum& left, const LogSum& right) const {
    LogSum combined;
#pragma unroll
    for (int head = 0; head < kQueriesPerKv; ++head) {
      const float maximum = fmaxf(left.maximum[head], right.maximum[head]);
      combined.maximum[head] = maximum;
      combined.sum[head] =
          (left.sum[head] > 0.0f ? left.sum[head] * __expf(left.maximum[head] - maximum) : 0.0f) +
          (right.sum[head] > 0.0f ? right.sum[head] * __expf(right.maximum[head] - maximum) : 0.0f);
    }
    return combined;
  }
};

// Unsigned key that orders like the float, with 0 reserved for "not a candidate".
__device__ __forceinline__ unsigned ordered_key(float value) {
  const unsigned bits = __float_as_uint(value);
  const unsigned key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  return key == 0u ? 1u : key;
}

__device__ __forceinline__ bool routable(int page, int pages) {
  return page >= kPinnedPages && page < pages;
}

// One thread per group of pages: per-head group maximum, and this block's share of the
// per-head log-sum-exp over routable pages.
__global__ void page_group_statistics_kernel(
    const float* __restrict__ page_log_mass,
    float* __restrict__ scratch,
    int64_t kv_heads,
    int pages,
    int blocks_per_row,
    int64_t input_stride_batch,
    int64_t input_stride_head,
    int64_t input_stride_query) {
  using BlockReduce = cub::BlockReduce<LogSum, kGroupThreads>;
  __shared__ typename BlockReduce::TempStorage temporary;
  const int thread = static_cast<int>(threadIdx.x);
  const int block = static_cast<int>(blockIdx.x) % blocks_per_row;
  const int64_t row = static_cast<int64_t>(blockIdx.x) / blocks_per_row;
  const int64_t input_base =
      (row / kv_heads) * input_stride_batch + (row % kv_heads) * input_stride_head;
  float* row_scratch = scratch + row * kScratchWords;
  const int group = block * kGroupThreads + thread;
  const int first_page = group * kGroupPages;

  LogSum local;
#pragma unroll
  for (int head = 0; head < kQueriesPerKv; ++head) {
    float values[kGroupPages];
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int item = 0; item < kGroupPages; ++item) {
      const int page = first_page + item;
      values[item] = routable(page, pages)
          ? page_log_mass[input_base + head * input_stride_query + page]
          : -CUDART_INF_F;
      maximum = fmaxf(maximum, values[item]);
    }
    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < kGroupPages; ++item) {
      if (routable(first_page + item, pages)) {
        sum += __expf(values[item] - maximum);
      }
    }
    local.maximum[head] = maximum;
    local.sum[head] = sum;
    row_scratch[kPartialWords + group * kQueriesPerKv + head] = maximum;
  }
  const LogSum total = BlockReduce(temporary).Reduce(local, CombineLogSum());
  if (thread == 0) {
#pragma unroll
    for (int head = 0; head < kQueriesPerKv; ++head) {
      row_scratch[(block * kQueriesPerKv + head) * 2] = total.maximum[head];
      row_scratch[(block * kQueriesPerKv + head) * 2 + 1] = total.sum[head];
    }
  }
}

using BinScan = cub::BlockScan<int, kSelectThreads>;

// Finds the key ranked `needed` from the top among items [first, last) of every thread, by
// histogram passes over key bits 31..21, 20..10 and 9..0. Leaves the key in *threshold and the
// number of items equal to it that are still needed in *remaining. Items with key 0 are skipped.
template <typename KeyOf>
__device__ __forceinline__ void select_threshold(
    const KeyOf& key_of,
    int first,
    int last,
    int needed,
    int* histogram,
    typename BinScan::TempStorage& scan,
    unsigned* threshold,
    int* remaining) {
  const int thread = static_cast<int>(threadIdx.x);
  if (thread == 0) {
    *threshold = 0u;
    *remaining = needed;
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < 3; ++pass) {
    const int shift = pass == 0 ? 21 : pass == 1 ? 10 : 0;
    const int bins = pass == 2 ? 1024 : kHistogramBins;
    const int fixed_shift = pass == 1 ? 21 : 10;
    for (int bin = thread; bin < kHistogramBins; bin += kSelectThreads) {
      histogram[bin] = 0;
    }
    __syncthreads();
    const unsigned fixed = *threshold;
    for (int item = first; item < last; ++item) {
      const unsigned key = key_of(item);
      if (key != 0u &&
          (pass == 0 || (key >> fixed_shift) == (fixed >> fixed_shift))) {
        atomicAdd(&histogram[(key >> shift) & (bins - 1)], 1);
      }
    }
    __syncthreads();
    // Counts from the highest bin down; each thread owns kBinsPerThread adjacent bins.
    int counts[kBinsPerThread];
    int through[kBinsPerThread];
#pragma unroll
    for (int slot = 0; slot < kBinsPerThread; ++slot) {
      counts[slot] = histogram[kHistogramBins - 1 - kBinsPerThread * thread - slot];
    }
    BinScan(scan).InclusiveSum(counts, through);
    const int wanted = *remaining;
    __syncthreads();
#pragma unroll
    for (int slot = 0; slot < kBinsPerThread; ++slot) {
      // Exactly one bin holds the wanted rank.
      if (through[slot] >= wanted && through[slot] - counts[slot] < wanted) {
        *threshold = fixed |
            (static_cast<unsigned>(kHistogramBins - 1 - kBinsPerThread * thread - slot) << shift);
        *remaining = wanted - (through[slot] - counts[slot]);
      }
    }
    __syncthreads();
  }
}

__global__ void select_pages_grouped_kernel(
    float* __restrict__ page_log_mass,
    float* __restrict__ scratch,
    int64_t* __restrict__ output,
    int64_t* __restrict__ support_ids,
    int64_t historical,
    int64_t end,
    int64_t kv_heads,
    int pages,
    int blocks_per_row,
    int64_t input_stride_batch,
    int64_t input_stride_head,
    int64_t input_stride_query) {
  __shared__ typename BinScan::TempStorage scan;
  __shared__ float log_sum[kQueriesPerKv];
  __shared__ int histogram[kHistogramBins];
  __shared__ int selected_pages[kRoutedPages];
  __shared__ unsigned threshold;
  __shared__ int remaining;
  __shared__ int candidate_groups;

  const int thread = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t input_base =
      (row / kv_heads) * input_stride_batch + (row % kv_heads) * input_stride_head;
  float* row_scratch = scratch + row * kScratchWords;
  float* group_maximum = row_scratch + kPartialWords;
  // The first word of a group's maxima is reused for the group's key.
  unsigned* group_key = reinterpret_cast<unsigned*>(group_maximum);
  int* candidates = reinterpret_cast<int*>(group_maximum + kGroupWords);
  unsigned* page_key = reinterpret_cast<unsigned*>(page_log_mass + input_base);
  const int groups = (pages + kGroupPages - 1) / kGroupPages;

  if (thread < kQueriesPerKv) {
    float maximum = -CUDART_INF_F;
    for (int block = 0; block < blocks_per_row; ++block) {
      maximum = fmaxf(maximum, row_scratch[(block * kQueriesPerKv + thread) * 2]);
    }
    float sum = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
      const float partial = row_scratch[(block * kQueriesPerKv + thread) * 2 + 1];
      if (partial > 0.0f) {
        sum += partial *
            __expf(row_scratch[(block * kQueriesPerKv + thread) * 2] - maximum);
      }
    }
    log_sum[thread] = maximum + __logf(sum);
  }
  __syncthreads();

  // A group's key is its best page's score. Groups made only of pinned pages get 0.
  const int group_chunk = (groups + kSelectThreads - 1) / kSelectThreads;
  const int first_group = min(thread * group_chunk, groups);
  const int last_group = min(first_group + group_chunk, groups);
  for (int group = first_group; group < last_group; ++group) {
    float best = -CUDART_INF_F;
#pragma unroll
    for (int head = 0; head < kQueriesPerKv; ++head) {
      best = fmaxf(best, group_maximum[group * kQueriesPerKv + head] - log_sum[head]);
    }
    group_key[group * kQueriesPerKv] =
        (group + 1) * kGroupPages > kPinnedPages ? ordered_key(best) : 0u;
  }
  __syncthreads();
  // Any selected page lies in a group whose key ranks within the needed count, because that
  // many groups each already hold a page at least as good.
  const int routable_groups = groups - kPinnedPages / kGroupPages;
  if (routable_groups > kNeededPages) {
    select_threshold(
        [&](int group) { return group_key[group * kQueriesPerKv]; },
        first_group, last_group, kNeededPages, histogram, scan, &threshold, &remaining);
  } else if (thread == 0) {
    threshold = 1u;
  }
  __syncthreads();
  const unsigned group_threshold = threshold;
  int kept = 0;
  for (int group = first_group; group < last_group; ++group) {
    kept += group_key[group * kQueriesPerKv] >= group_threshold;
  }
  int candidate_position;
  BinScan(scan).ExclusiveSum(kept, candidate_position);
  if (thread == kSelectThreads - 1) {
    candidate_groups = candidate_position + kept;
  }
  for (int group = first_group; group < last_group; ++group) {
    if (group_key[group * kQueriesPerKv] >= group_threshold) {
      candidates[candidate_position++] = group;
    }
  }
  __syncthreads();

  // Exact selection among the pages of the candidate groups, kept in page order.
  const int candidate_pages = candidate_groups * kGroupPages;
  const int page_chunk = (candidate_pages + kSelectThreads - 1) / kSelectThreads;
  const int first_item = min(thread * page_chunk, candidate_pages);
  const int last_item = min(first_item + page_chunk, candidate_pages);
  auto page_of = [&](int item) {
    return candidates[item / kGroupPages] * kGroupPages + item % kGroupPages;
  };
  for (int item = first_item; item < last_item; ++item) {
    const int page = page_of(item);
    if (routable(page, pages)) {
      float best = -CUDART_INF_F;
#pragma unroll
      for (int head = 0; head < kQueriesPerKv; ++head) {
        best = fmaxf(
            best,
            page_log_mass[input_base + head * input_stride_query + page] - log_sum[head]);
      }
      page_key[page] = ordered_key(best);
    }
  }
  __syncthreads();
  auto key_of_item = [&](int item) {
    const int page = page_of(item);
    return routable(page, pages) ? page_key[page] : 0u;
  };
  select_threshold(
      key_of_item, first_item, last_item, kNeededPages, histogram, scan, &threshold, &remaining);

  // remaining is how many pages tied at the threshold are taken, lowest page first.
  const unsigned last = threshold;
  const int ties_needed = remaining;
  int above = 0;
  int tied = 0;
  for (int item = first_item; item < last_item; ++item) {
    const unsigned key = key_of_item(item);
    above += key > last;
    tied += key == last;
  }
  int tied_before;
  BinScan(scan).ExclusiveSum(tied, tied_before);
  __syncthreads();
  const int ties_taken = min(max(ties_needed - tied_before, 0), tied);
  int position;
  BinScan(scan).ExclusiveSum(above + ties_taken, position);
  __syncthreads();
  if (thread < kPinnedPages) {
    selected_pages[thread] = thread;
  }
  int ties_left = ties_taken;
  for (int item = first_item; item < last_item; ++item) {
    const unsigned key = key_of_item(item);
    const bool tie = key == last && ties_left > 0;
    if ((key > last || tie) && position < kNeededPages) {
      selected_pages[kPinnedPages + position++] = page_of(item);
      ties_left -= tie;
    }
  }
  __syncthreads();

  for (int index = thread; index < kRoutedPages; index += kSelectThreads) {
    output[row * kRoutedPages + index] = selected_pages[index];
  }
  const int support_count = kRoutedPages * kPageSize + kRecentTokens;
  for (int index = thread; index < kRoutedPages * kPageSize;
       index += kSelectThreads) {
    const int64_t token =
        static_cast<int64_t>(selected_pages[index / kPageSize]) * kPageSize +
        index % kPageSize;
    support_ids[row * support_count + index] = token < historical ? token : -1;
  }
  for (int index = thread; index < kRecentTokens; index += kSelectThreads) {
    const int64_t token = historical + index;
    support_ids[row * support_count + kRoutedPages * kPageSize + index] =
        token < end ? token : -1;
  }
}

}  // namespace

void select_full_group_max_pages_and_pack_cuda(
    const at::Tensor& page_log_mass,
    const at::Tensor& selected_pages,
    const at::Tensor& support_ids,
    int64_t historical,
    int64_t end) {
  const int pages = static_cast<int>(page_log_mass.size(3));
  const int64_t rows = page_log_mass.size(0) * page_log_mass.size(1);
  const int blocks_per_row =
      (pages + kGroupThreads * kGroupPages - 1) / (kGroupThreads * kGroupPages);
  c10::cuda::CUDAGuard guard(page_log_mass.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(page_log_mass.get_device()).stream();
  const at::Tensor scratch = at::empty({rows, kScratchWords}, page_log_mass.options());
  page_group_statistics_kernel
      <<<static_cast<unsigned int>(rows * blocks_per_row), kGroupThreads, 0, stream>>>(
          page_log_mass.const_data_ptr<float>(),
          scratch.mutable_data_ptr<float>(),
          page_log_mass.size(1),
          pages,
          blocks_per_row,
          page_log_mass.stride(0),
          page_log_mass.stride(1),
          page_log_mass.stride(2));
  select_pages_grouped_kernel
      <<<static_cast<unsigned int>(rows), kSelectThreads, 0, stream>>>(
          page_log_mass.mutable_data_ptr<float>(),
          scratch.mutable_data_ptr<float>(),
          selected_pages.mutable_data_ptr<int64_t>(),
          support_ids.mutable_data_ptr<int64_t>(),
          historical,
          end,
          page_log_mass.size(1),
          pages,
          blocks_per_row,
          page_log_mass.stride(0),
          page_log_mass.stride(1),
          page_log_mass.stride(2));
}
