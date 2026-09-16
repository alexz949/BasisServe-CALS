#include <torch/extension.h>

#include <cstdint>

at::Tensor mapped_host_bf16_empty_cuda(
    int64_t batch,
    int64_t kv_heads,
    int64_t capacity,
    int64_t head_dim);

at::Tensor mapped_host_append_bf16_cuda(
    const at::Tensor& host_key,
    const at::Tensor& key,
    int64_t start);

int64_t mapped_host_device_pointer_cuda(const at::Tensor& host_key);

at::Tensor mapped_host_paged_attention_cuda(
    int64_t host_key_device_pointer,
    int64_t host_key_capacity,
    const at::Tensor& query,
    const at::Tensor& value,
    const at::Tensor& selected_page_ids,
    const at::Tensor& workspace,
    const at::Tensor& output,
    int64_t sequence_length,
    double scale,
    int64_t splits, const at::Tensor& value_prefix, int64_t prefix_width);

at::Tensor conditional_router_query_code_cuda(const at::Tensor& query, const at::Tensor& residual_query, const at::Tensor& query_code);

at::Tensor conditional_router_page_lse_cuda(
    const at::Tensor& query,
    const at::Tensor& base_code,
    const at::Tensor& residual_code,
    const at::Tensor& base_right,
    const at::Tensor& base_bias,
    const at::Tensor& residual_query,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    const at::Tensor& query_code,
    const at::Tensor& output,
    double scale, bool query_code_prepared);

void conditional_router_append_decode_cuda(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& base_left,
    const at::Tensor& base_right,
    const at::Tensor& base_bias,
    const at::Tensor& residual_encoder,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    const at::Tensor& value_cache,
    const at::Tensor& base_cache,
    const at::Tensor& residual_cache,
    const at::Tensor& rope_cos_cache,
    const at::Tensor& rope_sin_cache,
    int64_t start,
    bool write_rope);

at::Tensor select_fixed_group_max_pages_cuda(
    const at::Tensor& page_log_mass,
    const at::Tensor& output,
    int64_t pages_per_kv_head,
    int64_t pinned_prefix_pages,
    bool force_current_page);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "Mapped-host Page32 exact-K and GPU C1-V80 decode attention";
  module.def(
      "mapped_host_bf16_empty",
      &mapped_host_bf16_empty_cuda,
      pybind11::arg("batch"),
      pybind11::arg("kv_heads"),
      pybind11::arg("capacity"),
      pybind11::arg("head_dim"));
  module.def(
      "append",
      &mapped_host_append_bf16_cuda,
      pybind11::arg("host_key"),
      pybind11::arg("key"),
      pybind11::arg("start"));
  module.def(
      "device_pointer",
      &mapped_host_device_pointer_cuda,
      pybind11::arg("host_key"));
  module.def(
      "attention",
      &mapped_host_paged_attention_cuda,
      pybind11::arg("host_key_device_pointer"),
      pybind11::arg("host_key_capacity"),
      pybind11::arg("query"),
      pybind11::arg("value"),
      pybind11::arg("selected_page_ids"),
      pybind11::arg("workspace"),
      pybind11::arg("output"),
      pybind11::arg("sequence_length"),
      pybind11::arg("scale"),
      pybind11::arg("splits"),
      pybind11::arg("value_prefix"),
      pybind11::arg("prefix_width"));
  module.def("conditional_router_query_code", &conditional_router_query_code_cuda);
  module.def(
      "conditional_router_page_lse",
      &conditional_router_page_lse_cuda,
      pybind11::arg("query"),
      pybind11::arg("base_code"),
      pybind11::arg("residual_code"),
      pybind11::arg("base_right"),
      pybind11::arg("base_bias"),
      pybind11::arg("residual_query"),
      pybind11::arg("rope_cos"),
      pybind11::arg("rope_sin"),
      pybind11::arg("query_code"),
      pybind11::arg("output"),
      pybind11::arg("scale"),
      pybind11::arg("query_code_prepared"));
  module.def(
      "conditional_router_append_decode",
      &conditional_router_append_decode_cuda,
      pybind11::arg("key"),
      pybind11::arg("value"),
      pybind11::arg("base_left"),
      pybind11::arg("base_right"),
      pybind11::arg("base_bias"),
      pybind11::arg("residual_encoder"),
      pybind11::arg("rope_cos"),
      pybind11::arg("rope_sin"),
      pybind11::arg("value_cache"),
      pybind11::arg("base_cache"),
      pybind11::arg("residual_cache"),
      pybind11::arg("rope_cos_cache"),
      pybind11::arg("rope_sin_cache"),
      pybind11::arg("start"),
      pybind11::arg("write_rope"));
  module.def(
      "select_fixed_group_max_pages",
      &select_fixed_group_max_pages_cuda,
      pybind11::arg("page_log_mass"),
      pybind11::arg("output"),
      pybind11::arg("pages_per_kv_head"),
      pybind11::arg("pinned_prefix_pages"),
      pybind11::arg("force_current_page"));
}
