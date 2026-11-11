"""Compile the archived Table 3 register router for any page size that divides 32.

The archived kernel reduces one page per warp, so it only builds for 32-token pages. Here
each warp still scores 32 tokens but reduces them in page-sized lane groups, and the page
selector is replaced by table3_page_select.cu. Page size 32 keeps the archived arithmetic.
"""

from pathlib import Path

from torch.utils.cpp_extension import load

SELECTOR = Path(__file__).with_name("table3_page_select.cu")
SOURCES = ("mapped_host_paged_attention.cpp", "mapped_host_paged_attention.cu",
           "conditional_router_page32.cu")
PAGE_REDUCE = """__device__ __forceinline__ float page_sum(float value) {
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

"""
ROUND = "__device__ __forceinline__ float round_bfloat16(float value) {"
PACK = "void select_full_group_max_pages_and_pack_cuda("
# (archived text, replacement) pairs; the first two match the archived 16-warp configuration.
EDITS = [
    ("constexpr int kRegisterWarps = 8;", "constexpr int kRegisterWarps = 16;"),
    ("    BASIS_PAGE_SIZE == 32 && BASIS_GQA == 4 && \\\n",
     "    32 % BASIS_PAGE_SIZE == 0 && BASIS_GQA == 4 && \\\n"),
    ("constexpr int kRegisterPagesPerBlock = kRegisterWarps / 2;",
     "constexpr int kRegisterPagesPerBlock = kRegisterWarps * 16 / kPageSize;"),
    ("""  for (int item = warp; item < kPagesPerBlock * kQueriesPerKv;
       item += kRegisterWarps) {
    const int page_in_block = item / kQueriesPerKv;
    const int query_head = item % kQueriesPerKv;
    const int64_t page = page_group * kPagesPerBlock + page_in_block;
    const int64_t token =
        token_start + page_in_block * kPageSize + lane;
""", """  for (int item = warp; item < kTokensPerBlock / kWarpSize * kQueriesPerKv;
       item += kRegisterWarps) {
    const int segment = item / kQueriesPerKv;
    const int query_head = item % kQueriesPerKv;
    const int64_t page = page_group * kPagesPerBlock +
        (segment * kWarpSize + lane) / kPageSize;
    const int64_t token = token_start + segment * kWarpSize + lane;
"""),
    ("""                  shared_scores[query_head * kTokensPerBlock +
                                page_in_block * kPageSize + lane]) +
""", """                  shared_scores[query_head * kTokensPerBlock +
                                segment * kWarpSize + lane]) +
"""),
    ("""    const float maximum = warp_max(score);
    const float sum = warp_sum(__expf(score - maximum));
    if (lane == 0 && page < pages) {
""", """    const float maximum = page_max(score);
    const float sum = page_sum(__expf(score - maximum));
    if (lane % kPageSize == 0 && page < pages) {
"""),
    (ROUND, PAGE_REDUCE + ROUND),
]


def build(snapshot, folder, page_size, *, value_dim, routed_tokens, sink_tokens):
    """Return the extension and how many times each archived anchor was found.

    routed_tokens counts the pinned sink; Table 3 used value_dim=128, routed_tokens=1984, sink_tokens=32."""
    folder.mkdir(parents=True, exist_ok=True)
    sources, found = [], []
    for filename in SOURCES:
        content = (snapshot / "basisserve/kernels/csrc" / filename).read_text()
        if filename == "conditional_router_page32.cu":
            for archived, replacement in EDITS:
                found.append(content.count(archived))
                content = content.replace(archived, replacement)
            found.append(content.count(PACK))
            content = content[:content.index(PACK)] + SELECTOR.read_text()
        path = folder / filename
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
        sources.append(str(path))
    module = load(
        name=f"table3_page_router_p{page_size}_v{value_dim}_t{routed_tokens}_s{sink_tokens}",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math", "--ptxas-options=-v",
            f"-DBASIS_VALUE_DIM={value_dim}", "-DBASIS_GQA=4", f"-DBASIS_PAGE_SIZE={page_size}",
            "-DBASIS_BASE_RANK=16", "-DBASIS_RESIDUAL_RANK=16",
            f"-DBASIS_ROUTED_TOKENS={routed_tokens}", f"-DBASIS_SINK_TOKENS={sink_tokens}",
        ], verbose=True)
    return module, dict(anchors_found=found, anchors_expected=[2, 2, 1, 1, 1, 1, 1, 1])
