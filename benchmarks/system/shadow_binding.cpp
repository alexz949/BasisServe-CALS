#include <torch/extension.h>
void batch_gemm_softmax(torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor, torch::Tensor,
                       int, int, int, int, float, float);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("batch_gemm_softmax", &batch_gemm_softmax);
}
