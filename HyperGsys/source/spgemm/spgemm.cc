#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor>
hgnn_spgemm(torch::Tensor H_sp_csrptr, torch::Tensor H_sp_csrind,
            torch::Tensor H_sp_data, torch::Tensor H_T_sp_csrptr,
            torch::Tensor H_T_sp_csrind, torch::Tensor H_T_sp_data,
            torch::Tensor W);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hgnn_spgemm", &hgnn_spgemm, "SpGEMM with cuSparse");
}