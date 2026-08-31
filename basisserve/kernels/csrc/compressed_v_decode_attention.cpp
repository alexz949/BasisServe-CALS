#include <torch/extension.h>

#include <cstdint>

at::Tensor compressed_v_decode_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& workspace,
    const at::Tensor& output,
    int64_t architecture,
    double scale,
    int64_t splits,
    bool feature_major_output);

at::Tensor c1_paged_sparse_decode_cuda(
    const at::Tensor& query,
    const at::Tensor& packed_key_pages,
    const at::Tensor& value,
    const at::Tensor& selected_page_ids,
    const at::Tensor& workspace,
    const at::Tensor& output,
    double scale,
    int64_t splits);

at::Tensor c1_pack_exact_key_pages_cuda(
    const at::Tensor& exact_key,
    const at::Tensor& selected_page_ids,
    const at::Tensor& packed_key_pages);

at::Tensor c1_r32_page_lse_cuda(
    const at::Tensor& query,
    const at::Tensor& routing_sidecar,
    const at::Tensor& query_projector,
    const at::Tensor& query_code,
    const at::Tensor& page_log_mass,
    double scale);

at::Tensor c1_r32_topk_gqa_union_cuda(
    const at::Tensor& page_log_mass,
    const at::Tensor& selected_page_ids,
    const at::Tensor& selected_page_counts,
    int64_t top_pages_per_query);

at::Tensor c1_dense_gqa_v96_decode_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& valid_sequence_length,
    const at::Tensor& workspace,
    const at::Tensor& output,
    double scale,
    int64_t splits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "Exact-rank grouped-query compressed-Value CUDA decode attention";
  module.def(
      "decode",
      &compressed_v_decode_cuda,
      pybind11::arg("query"),
      pybind11::arg("key"),
      pybind11::arg("value"),
      pybind11::arg("workspace"),
      pybind11::arg("output"),
      pybind11::arg("architecture"),
      pybind11::arg("scale"),
      pybind11::arg("splits"),
      pybind11::arg("feature_major_output"));
  module.def(
      "dense_gqa_v96_decode",
      &c1_dense_gqa_v96_decode_cuda,
      pybind11::arg("query"),
      pybind11::arg("key"),
      pybind11::arg("value"),
      pybind11::arg("valid_sequence_length"),
      pybind11::arg("workspace"),
      pybind11::arg("output"),
      pybind11::arg("scale"),
      pybind11::arg("splits"));
  module.def(
      "pack_exact_key_pages",
      &c1_pack_exact_key_pages_cuda,
      pybind11::arg("exact_key"),
      pybind11::arg("selected_page_ids"),
      pybind11::arg("packed_key_pages"));
  module.def(
      "r32_page_lse",
      &c1_r32_page_lse_cuda,
      pybind11::arg("query"),
      pybind11::arg("routing_sidecar"),
      pybind11::arg("query_projector"),
      pybind11::arg("query_code"),
      pybind11::arg("page_log_mass"),
      pybind11::arg("scale"));
  module.def(
      "r32_topk_gqa_union",
      &c1_r32_topk_gqa_union_cuda,
      pybind11::arg("page_log_mass"),
      pybind11::arg("selected_page_ids"),
      pybind11::arg("selected_page_counts"),
      pybind11::arg("top_pages_per_query"));
  module.def(
      "paged_sparse_decode",
      &c1_paged_sparse_decode_cuda,
      pybind11::arg("query"),
      pybind11::arg("packed_key_pages"),
      pybind11::arg("value"),
      pybind11::arg("selected_page_ids"),
      pybind11::arg("workspace"),
      pybind11::arg("output"),
      pybind11::arg("scale"),
      pybind11::arg("splits"));
}
