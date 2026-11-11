// Faster decode router for BasisKV (B16R16, GQA4, page sizes dividing 32): the same per-page log-sum-exp as the
// archived Table 3 register router as built by table3_page_router.py, bit for bit.
//
// Arithmetic is unchanged: every token's base code goes through the same m16n8k16 BF16 MMA against the same paired
// base_right layout, each thread owns the same (token, feature) pairs and accumulates them in the same order, the
// RoPE / bias / BF16 rounding sequence is identical, the 4-lane score reduction and the page reductions use the same
// shuffles, and the residual dot product runs over the same rank order. Only data movement differs:
//   * the archived kernel emulates BF16 arithmetic in FP32 (op, then round to BF16), and that rounding and the
//     conversions around it are its instruction bottleneck. Where both operands are already BF16 values (bias add,
//     RoPE products and sums), the same results come from packed BF16 FMA (fma.rn.bf16x2, one rounding of the exact
//     result): a product of two BF16 values is exact in FP32, and for sums FP32's 24 bits >= 2 x 8 + 2 make the
//     FP32-then-BF16 double rounding innocuous (Figueroa), so both paths give the correctly rounded BF16 value.
//     Roundings of FP32 accumulators (MMA output, dot products, scaled scores) stay FP32 -> BF16 conversions, done
//     two at a time with cvt.rn.bf16x2.f32 (each half rounded as __float2bfloat16_rn would). Subnormal results,
//     which FP32 fast-math flushes and BF16 FMA keeps, cannot occur at these magnitudes; equality is checked;
//   * the RoPE cos/sin values a thread needs are loaded into registers before the (fully unrolled) feature loop, so
//     their latency overlaps instead of sitting in a dependent loop with unroll 1;
//   * base and residual codes are read with 16-byte vector loads; each lane reads its token's residual code once
//     and scores two query heads with it (the archived kernel re-reads it per query head);
//   * 8 warps per block and 128-token tiles, which keeps the register-resident prefetch at two blocks per SM; each
//     block loops over kTiles tiles of one (request, KV head) row, so the per-row setup (base_right transpose,
//     query, bias, query code) is paid once per kTiles tiles instead of once per tile.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <cassert>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kQueriesPerKv = BASIS_GQA;
constexpr int kQueryKeyDim = 128;
constexpr int kBaseRank = BASIS_BASE_RANK;
constexpr int kResidualRank = BASIS_RESIDUAL_RANK;
constexpr int kPageSize = BASIS_PAGE_SIZE;
constexpr int kHalfHeadDim = kQueryKeyDim / 2;
constexpr int kWarps = 8;
#ifndef ROUTER_TILES
#define ROUTER_TILES 1
#endif
constexpr int kTiles = ROUTER_TILES;
constexpr int kTokensPerBlock = kWarps * 16;  // tokens per tile
constexpr int kPagesPerBlock = kTokensPerBlock / kPageSize;
constexpr int kSegments = kTokensPerBlock / kWarpSize;
constexpr int kHeadsPerItem = 2;

static_assert(kBaseRank == 16 && kResidualRank == 16 && kQueriesPerKv == 4, "B16R16 GQA4 only");
static_assert(kWarpSize % kPageSize == 0, "page size must divide 32");

__device__ __forceinline__ float round_bfloat16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

// Round two values to BF16 with one instruction; returns them as floats (same values as two round_bfloat16).
__device__ __forceinline__ float2 round_bfloat16_pair(float first, float second) {
  const __nv_bfloat162 rounded = __floats2bfloat162_rn(first, second);
  return make_float2(__low2float(rounded), __high2float(rounded));
}

__device__ __forceinline__ float page_sum(float value) {
#pragma unroll
  for (int offset = kPageSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, kPageSize);
  }
  return value;
}

__device__ __forceinline__ float page_max(float value) {
#pragma unroll
  for (int offset = kPageSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, kPageSize));
  }
  return value;
}

__global__ void __launch_bounds__(kWarps * kWarpSize) router_page_lse_fast_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ base_code,
    const __nv_bfloat16* __restrict__ residual_code,
    const __nv_bfloat16* __restrict__ base_right,
    const __nv_bfloat16* __restrict__ base_bias,
    const __nv_bfloat16* __restrict__ rope_cos,
    const __nv_bfloat16* __restrict__ rope_sin,
    const __nv_bfloat16* __restrict__ query_code,
    float* __restrict__ output,
    int64_t kv_heads,
    int64_t tokens,
    int64_t pages,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t base_stride_batch,
    int64_t base_stride_head,
    int64_t base_stride_token,
    int64_t residual_stride_batch,
    int64_t residual_stride_head,
    int64_t residual_stride_token,
    int64_t rope_stride_token,
    int64_t output_stride_batch,
    int64_t output_stride_head,
    int64_t output_stride_query,
    float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __shared__ __align__(16) __nv_bfloat16 shared_base[kTokensPerBlock * kBaseRank];
  __shared__ __align__(16) __nv_bfloat16 shared_right[kQueryKeyDim * kBaseRank];
  __shared__ float shared_query[kQueriesPerKv * kQueryKeyDim];  // BF16 values, held as FP32
  __shared__ __nv_bfloat162 shared_bias[kHalfHeadDim];            // (bias[f], bias[f + 64])
  __shared__ __nv_bfloat16 shared_residual_query[kQueriesPerKv * kResidualRank];
  __shared__ float shared_scores[kQueriesPerKv * kTokensPerBlock];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpSize;
  const int lane = thread % kWarpSize;
  const int lane_group = lane / 4;
  const int lane_in_group = lane % 4;
  const int64_t page_groups = (pages + kPagesPerBlock - 1) / kPagesPerBlock;
  const int64_t chunks = (page_groups + kTiles - 1) / kTiles;
  const int64_t chunk = static_cast<int64_t>(blockIdx.x) % chunks;
  const int64_t kv_row = static_cast<int64_t>(blockIdx.x) / chunks;
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int token_zero = warp * 16 + lane_group;

  // Per-row data, loaded once per block.
  for (int index = thread; index < kBaseRank * kQueryKeyDim; index += kWarps * kWarpSize) {
    const int rank = index / kQueryKeyDim;
    const int feature = index % kQueryKeyDim;
    const int paired_feature = (feature % kHalfHeadDim) * 2 + feature / kHalfHeadDim;
    shared_right[paired_feature * kBaseRank + rank] = base_right[kv_head * kBaseRank * kQueryKeyDim + index];
  }
  for (int index = thread; index < kQueriesPerKv * kQueryKeyDim; index += kWarps * kWarpSize) {
    const int query_head = index / kQueryKeyDim;
    const int feature = index % kQueryKeyDim;
    shared_query[index] = __bfloat162float(query[batch * query_stride_batch +
                                                 (kv_head * kQueriesPerKv + query_head) * query_stride_head + feature]);
  }
  for (int index = thread; index < kHalfHeadDim; index += kWarps * kWarpSize) {
    shared_bias[index] = __halves2bfloat162(base_bias[kv_head * kQueryKeyDim + index],
                                            base_bias[kv_head * kQueryKeyDim + index + kHalfHeadDim]);
  }
  for (int index = thread; index < kQueriesPerKv * kResidualRank; index += kWarps * kWarpSize) {
    shared_residual_query[index] =
        query_code[(batch * kv_heads + kv_head) * kQueriesPerKv * kResidualRank + index];
  }

  for (int tile = 0; tile < kTiles; ++tile) {
  const int64_t page_group = chunk * kTiles + tile;
  if (page_group >= page_groups) break;  // uniform across the block
  const int64_t token_start = page_group * kTokensPerBlock;

  // This thread's RoPE values (same tokens and features as its MMA outputs), issued first.
  // Tokens past the end are clamped to the last token: their scores are replaced by -inf before the page
  // reductions (as in the archived kernel), so their RoPE inputs never reach an output, and the clamp removes a
  // bounds check and an address computation from each of these 64 loads.
  __nv_bfloat16 cosines[2][16];
  __nv_bfloat16 sines[2][16];
#pragma unroll
  for (int token_pair = 0; token_pair < 2; ++token_pair) {
    const int64_t token = min(token_start + token_zero + token_pair * 8, tokens - 1);
    const __nv_bfloat16* cos_row = rope_cos + token * rope_stride_token + lane_in_group;
    const __nv_bfloat16* sin_row = rope_sin + token * rope_stride_token + lane_in_group;
#pragma unroll
    for (int feature_group = 0; feature_group < 16; ++feature_group) {
      cosines[token_pair][feature_group] = __ldg(cos_row + feature_group * 4);
      sines[token_pair][feature_group] = __ldg(sin_row + feature_group * 4);
    }
  }

  // Base codes: one 16-byte vector per (token, half of the 16 ranks).
  for (int index = thread; index < kTokensPerBlock * 2; index += kWarps * kWarpSize) {
    const int token_in_block = index / 2;
    const int half = index % 2;
    const int64_t token = token_start + token_in_block;
    uint4 value = make_uint4(0u, 0u, 0u, 0u);
    if (token < tokens) {
      value = *reinterpret_cast<const uint4*>(
          base_code + batch * base_stride_batch + kv_head * base_stride_head + token * base_stride_token + half * 8);
    }
    *reinterpret_cast<uint4*>(shared_base + token_in_block * kBaseRank + half * 8) = value;
  }
  __syncthreads();

  const int token_one = token_zero + 8;
  const unsigned left_zero = *reinterpret_cast<unsigned*>(shared_base + token_zero * kBaseRank + lane_in_group * 2);
  const unsigned left_one = *reinterpret_cast<unsigned*>(shared_base + token_one * kBaseRank + lane_in_group * 2);
  const unsigned left_zero_high =
      *reinterpret_cast<unsigned*>(shared_base + token_zero * kBaseRank + lane_in_group * 2 + 8);
  const unsigned left_one_high =
      *reinterpret_cast<unsigned*>(shared_base + token_one * kBaseRank + lane_in_group * 2 + 8);
  float dot_sums[2][kQueriesPerKv] = {};
#pragma unroll
  for (int feature_group = 0; feature_group < 16; ++feature_group) {
    const unsigned right_low = *reinterpret_cast<unsigned*>(
        shared_right + (feature_group * 8 + lane_group) * kBaseRank + lane_in_group * 2);
    const unsigned right_high = *reinterpret_cast<unsigned*>(
        shared_right + (feature_group * 8 + lane_group) * kBaseRank + lane_in_group * 2 + 8);
    float accumulator_zero_low;
    float accumulator_zero_high;
    float accumulator_one_low;
    float accumulator_one_high;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(accumulator_zero_low), "=f"(accumulator_zero_high),
          "=f"(accumulator_one_low), "=f"(accumulator_one_high)
        : "r"(left_zero), "r"(left_one),
          "r"(left_zero_high), "r"(left_one_high),
          "r"(right_low), "r"(right_high), "f"(0.0f));
    const int feature = feature_group * 4 + lane_in_group;
    const float reconstructed_low[2] = {accumulator_zero_low, accumulator_one_low};
    const float reconstructed_high[2] = {accumulator_zero_high, accumulator_one_high};
    const __nv_bfloat162 bias = shared_bias[feature];
    const __nv_bfloat162 signs = __floats2bfloat162_rn(-1.0f, 1.0f);
#pragma unroll
    for (int token_pair = 0; token_pair < 2; ++token_pair) {
      // Archived: biased = round(round(reconstructed) + bias), per half (low, high).
      const __nv_bfloat162 reconstructed =
          __floats2bfloat162_rn(reconstructed_low[token_pair], reconstructed_high[token_pair]);
      const __nv_bfloat162 biased = __hadd2(reconstructed, bias);
      // Archived: rotated_low = round(round(bl*c) - round(bh*s)); rotated_high = round(round(bh*c) + round(bl*s)).
      const __nv_bfloat162 cosine = __bfloat162bfloat162(cosines[token_pair][feature_group]);
      const __nv_bfloat162 sine = __bfloat162bfloat162(sines[token_pair][feature_group]);
      const __nv_bfloat162 by_cosine = __hmul2(biased, cosine);                    // (bl*c, bh*c)
      const __nv_bfloat162 by_sine = __hmul2(__lowhigh2highlow(biased), sine);     // (bh*s, bl*s)
      const __nv_bfloat162 rotated = __hfma2(by_sine, signs, by_cosine);         // (bl*c - bh*s, bh*c + bl*s)
      const float rotated_low = __low2float(rotated);
      const float rotated_high = __high2float(rotated);
#pragma unroll
      for (int query_head = 0; query_head < kQueriesPerKv; ++query_head) {
        dot_sums[token_pair][query_head] = fmaf(
            rotated_low, shared_query[query_head * kQueryKeyDim + feature], dot_sums[token_pair][query_head]);
        dot_sums[token_pair][query_head] = fmaf(
            rotated_high, shared_query[query_head * kQueryKeyDim + feature + kHalfHeadDim],
            dot_sums[token_pair][query_head]);
      }
    }
  }
#pragma unroll
  for (int token_pair = 0; token_pair < 2; ++token_pair) {
#pragma unroll
    for (int query_head = 0; query_head < kQueriesPerKv; ++query_head) {
      float score = dot_sums[token_pair][query_head];
      score += __shfl_xor_sync(0xffffffffu, score, 1, 4);
      score += __shfl_xor_sync(0xffffffffu, score, 2, 4);
      if (lane_in_group == 0) {
        shared_scores[query_head * kTokensPerBlock + token_zero + token_pair * 8] = score;
      }
    }
  }
  __syncthreads();

  // Residual scores and page log-sum-exp: item = (32-token segment, pair of query heads).
  for (int item = warp; item < kSegments * (kQueriesPerKv / kHeadsPerItem); item += kWarps) {
    const int segment = item / (kQueriesPerKv / kHeadsPerItem);
    const int head_pair = item % (kQueriesPerKv / kHeadsPerItem);
    const int64_t page = page_group * kPagesPerBlock + (segment * kWarpSize + lane) / kPageSize;
    const int64_t token = token_start + segment * kWarpSize + lane;
    __nv_bfloat16 code[kResidualRank];
    if (token < tokens) {
      const __nv_bfloat16* source = residual_code + batch * residual_stride_batch +
          kv_head * residual_stride_head + token * residual_stride_token;
      *reinterpret_cast<uint4*>(code) = *reinterpret_cast<const uint4*>(source);
      *reinterpret_cast<uint4*>(code + 8) = *reinterpret_cast<const uint4*>(source + 8);
    }
    // Archived per head: score = round(round(round(shared_score) + round(residual_score)) * scale); the two heads
    // of this item share each rounding instruction.
    float scores[kHeadsPerItem] = {-CUDART_INF_F, -CUDART_INF_F};
    if (token < tokens) {
      float residual_scores[kHeadsPerItem];
#pragma unroll
      for (int head_in_pair = 0; head_in_pair < kHeadsPerItem; ++head_in_pair) {
        const int query_head = head_pair * kHeadsPerItem + head_in_pair;
        float residual_score = 0.0f;
#pragma unroll
        for (int residual = 0; residual < kResidualRank; ++residual) {
          residual_score = fmaf(
              __bfloat162float(shared_residual_query[query_head * kResidualRank + residual]),
              __bfloat162float(code[residual]),
              residual_score);
        }
        residual_scores[head_in_pair] = residual_score;
      }
      const int head_zero = head_pair * kHeadsPerItem;
      const float2 dot = round_bfloat16_pair(
          shared_scores[head_zero * kTokensPerBlock + segment * kWarpSize + lane],
          shared_scores[(head_zero + 1) * kTokensPerBlock + segment * kWarpSize + lane]);
      const float2 residual = round_bfloat16_pair(residual_scores[0], residual_scores[1]);
      // BF16 + BF16 rounded to BF16: the packed FMA gives the archived FP32-then-round value (see above).
      const __nv_bfloat162 sum = __hadd2(__floats2bfloat162_rn(dot.x, dot.y), __floats2bfloat162_rn(residual.x, residual.y));
      const float2 total = make_float2(__low2float(sum), __high2float(sum));
      const float2 scaled = round_bfloat16_pair(total.x * scale, total.y * scale);
      scores[0] = scaled.x;
      scores[1] = scaled.y;
    }
#pragma unroll
    for (int head_in_pair = 0; head_in_pair < kHeadsPerItem; ++head_in_pair) {
      const int query_head = head_pair * kHeadsPerItem + head_in_pair;
      const float score = scores[head_in_pair];
      const float maximum = page_max(score);
      const float sum = page_sum(__expf(score - maximum));
      if (lane % kPageSize == 0 && page < pages) {
        output[batch * output_stride_batch + kv_head * output_stride_head +
               query_head * output_stride_query + page] = maximum + __logf(sum);
      }
    }
  }
  __syncthreads();  // shared_base and shared_scores are rewritten by the next tile
  }
#endif
}

at::Tensor router_page_lse_fast(
    const at::Tensor& query,
    const at::Tensor& base_code,
    const at::Tensor& residual_code,
    const at::Tensor& base_right,
    const at::Tensor& base_bias,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    const at::Tensor& query_code,
    const at::Tensor& output,
    double scale) {
  // Same contract as the archived conditional_router_page_lse with query_code_prepared = true.
  assert(query.scalar_type() == at::kBFloat16 && base_code.scalar_type() == at::kBFloat16);
  assert(residual_code.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kFloat);
  assert(query.dim() == 4 && query.size(2) == 1 && query.size(3) == kQueryKeyDim && query.stride(3) == 1);
  assert(base_code.dim() == 4 && base_code.size(3) == kBaseRank && base_code.stride(3) == 1);
  assert(residual_code.sizes() == base_code.sizes() && residual_code.stride(3) == 1);
  assert(base_code.size(1) * kQueriesPerKv == query.size(1));
  assert(base_right.is_contiguous() && base_bias.is_contiguous() && query_code.is_contiguous());
  assert(rope_cos.sizes() == at::IntArrayRef({base_code.size(2), kHalfHeadDim}));
  assert(rope_sin.sizes() == rope_cos.sizes() && rope_cos.stride(1) == 1 && rope_sin.stride(1) == 1);
  assert(rope_cos.stride(0) == rope_sin.stride(0));
  const int64_t pages = (base_code.size(2) + kPageSize - 1) / kPageSize;
  assert(output.sizes() == at::IntArrayRef({base_code.size(0), base_code.size(1), kQueriesPerKv, pages}));
  assert(output.stride(3) == 1);
  // 16-byte vector loads of base and residual codes.
  for (const at::Tensor* code : {&base_code, &residual_code}) {
    assert(reinterpret_cast<uintptr_t>(code->const_data_ptr()) % 16 == 0);
    assert(code->stride(0) % 8 == 0 && code->stride(1) % 8 == 0 && code->stride(2) % 8 == 0);
  }
  c10::cuda::CUDAGuard guard(query.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  const int64_t page_groups = (pages + kPagesPerBlock - 1) / kPagesPerBlock;
  const int64_t blocks = base_code.size(0) * base_code.size(1) * ((page_groups + kTiles - 1) / kTiles);
  router_page_lse_fast_kernel<<<static_cast<unsigned int>(blocks), kWarps * kWarpSize, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(query.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(base_code.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(residual_code.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(base_right.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(base_bias.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(rope_cos.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(rope_sin.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(query_code.const_data_ptr()),
      output.mutable_data_ptr<float>(),
      base_code.size(1), base_code.size(2), pages,
      query.stride(0), query.stride(1),
      base_code.stride(0), base_code.stride(1), base_code.stride(2),
      residual_code.stride(0), residual_code.stride(1), residual_code.stride(2),
      rope_cos.stride(0),
      output.stride(0), output.stride(1), output.stride(2),
      static_cast<float>(scale));
  assert(cudaGetLastError() == cudaSuccess);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("router_page_lse_fast", &router_page_lse_fast);
}
