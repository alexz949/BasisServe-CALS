"""Compile the Page32 candidate router from the Ampere WMMA kernel."""

import hashlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from benchmarks.system.local_kernel_candidates import write_source


SOURCE = Path(__file__).resolve().parents[2] / "basisserve/kernels/csrc"


def compile_fine(output):
    output = Path(output)
    original = (SOURCE / "conditional_router_page32.cu").read_text()
    start = original.index("__global__ void conditional_router_page_lse_kernel(")
    stop = original.index("__global__ void conditional_router_append_decode_kernel(", start)
    body = original[start:stop]
    assert body.count("    float scale) {") == 1
    assert body.count("const int64_t page_start = page * kPageSize;") == 1
    body = body.replace("    float scale) {", "    float scale, const int64_t* ids) {")
    body = body.replace(
        "const int64_t page_start = page * kPageSize;",
        "const int64_t page_start = ids[kv_row * pages + page] * kPageSize;",
    )
    assert body.count("= maximum + __logf(sum);") == 1
    body = body.replace(
        "= maximum + __logf(sum);",
        "= maximum == -CUDART_INF_F ? -CUDART_INF_F : maximum + __logf(sum);",
    )
    cuda = original[:start] + body + original[stop:]
    host_start = cuda.index("at::Tensor conditional_router_page_lse_cuda(")
    host_stop = cuda.index("void conditional_router_append_decode_cuda(", host_start)
    host = cuda[host_start:host_stop]
    assert host.count("double scale, bool query_code_prepared) {") == 1
    assert host.count("(base_code.size(2) + kPageSize - 1) / kPageSize;") == 1
    assert host.count("static_cast<float>(scale));") == 2
    host = host.replace(
        "double scale, bool query_code_prepared) {",
        "double scale, bool query_code_prepared, const at::Tensor& ids) {",
    )
    host = host.replace(
        "(base_code.size(2) + kPageSize - 1) / kPageSize;", "ids.size(2);"
    )
    host = host.replace(
        "static_cast<float>(scale));",
        "static_cast<float>(scale), ids.data_ptr<int64_t>());",
    )
    cuda = cuda[:host_start] + host + cuda[host_stop:]
    cpp = (SOURCE / "mapped_host_paged_attention.cpp").read_text()
    assert cpp.count("double scale, bool query_code_prepared);") == 1
    assert cpp.count('pybind11::arg("query_code_prepared"));') == 1
    cpp = cpp.replace(
        "double scale, bool query_code_prepared);",
        "double scale, bool query_code_prepared, const at::Tensor& ids);",
    )
    cpp = cpp.replace(
        'pybind11::arg("query_code_prepared"));',
        'pybind11::arg("query_code_prepared"), pybind11::arg("ids"));',
    )
    files = {
        "conditional_router_page32.cu": cuda,
        "mapped_host_paged_attention.cpp": cpp,
        "mapped_host_paged_attention.cu": (SOURCE / "mapped_host_paged_attention.cu").read_text(),
    }
    for name, contents in files.items():
        write_source(output / name, contents)
    digest = hashlib.sha256((cuda + cpp).encode()).hexdigest()[:12]
    major, minor = torch.cuda.get_device_capability()
    return load(
        name=f"basisserve_sm{major}{minor}_fine_{digest}",
        verbose=os.environ.get("BASISSERVE_VERBOSE_BUILD", "0") == "1",
        sources=[str(output / name) for name in files],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-DBASIS_VALUE_DIM=96", "-DBASIS_GQA=4", "-DBASIS_PAGE_SIZE=32",
            "-DBASIS_BASE_RANK=16", "-DBASIS_RESIDUAL_RANK=16",
            "-DBASIS_DISABLE_REGISTER_ROUTER",
        ],
    )
