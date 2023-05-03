#include "../../include/dataloader/dataloader.hpp"
#include "../../include/taskbalancer/balancer.cuh"
#include "../../include/taskbalancer/py_balan.h"

#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

// [TO DO: GPU processing]
torch::Tensor BalanceSchedule(int ngs, torch::Tensor H_csrptr_tensor,
                              torch::Tensor H_colind_tensor,
                              torch::Tensor H_T_csrptr_tensor,
                              torch::Tensor H_T_colind_tensor) {
  int H_nrow = H_csrptr_tensor.size(0) - 1;
  int H_ncol = H_T_csrptr_tensor.size(0) - 1;
  int nnz = H_colind_tensor.size(0) - 1;

  hgnn_balancer<int, float, balan_met::hgnn_ef_full> ef_balan(
      ngs, H_nrow, H_ncol, nnz, H_csrptr_tensor.data_ptr<int>(),
      H_colind_tensor.data_ptr<int>(), H_T_csrptr_tensor.data_ptr<int>(),
      H_T_colind_tensor.data_ptr<int>());

  auto optionsI = torch::TensorOptions().dtype(torch::kInt32);
  torch::Tensor key =
      torch::from_blob(ef_balan.key.data(), {ef_balan.part_keys}, optionsI);
  torch::Tensor group_row =
      torch::from_blob(ef_balan.row.data(), {ef_balan.keys}, optionsI);
  torch::Tensor group_start =
      torch::from_blob(ef_balan.group_st.data(), {ef_balan.keys}, optionsI);
  torch::Tensor group_end =
      torch::from_blob(ef_balan.group_ed.data(), {ef_balan.keys}, optionsI);
  return group_start;
}
