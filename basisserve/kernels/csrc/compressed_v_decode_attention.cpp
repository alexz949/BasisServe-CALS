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
}
