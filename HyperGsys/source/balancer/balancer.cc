#include "../../include/taskbalancer/py_balan.h"
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor BalanceSchedule(int ngs, torch::Tensor H_csrptr_tensor,
                              torch::Tensor H_colind_tensor,
                              torch::Tensor H_T_csrptr_tensor,
                              torch::Tensor H_T_colind_tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "balancer";
  m.def("balancer", &BalanceSchedule, "balancer");
}