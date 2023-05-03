
#include "../../include/dataloader/dataloader.hpp"
#include "../../include/spgemm/spgemm.cuh"
#include "../../include/util/check.cuh"
#include "../../include/util/ramArray.cuh"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor>
hgnn_spgemm(torch::Tensor H_sp_csrptr, torch::Tensor H_sp_csrind,
            torch::Tensor H_sp_data, torch::Tensor H_T_sp_csrptr,
            torch::Tensor H_T_sp_csrind, torch::Tensor H_T_sp_data,
            torch::Tensor W) {

  int H_nrow = H_sp_csrptr.size(0) - 1;
  int H_ncol = H_T_sp_csrptr.size(0) - 1;
  int H_nnz = H_sp_csrind.size(0);
  return SpGEMM_HWHT(H_nrow, H_ncol, H_nnz, H_sp_csrptr.data_ptr<int>(),
                     H_sp_csrind.data_ptr<int>(), H_sp_data.data_ptr<float>(),
                     H_T_sp_csrptr.data_ptr<int>(),
                     H_T_sp_csrind.data_ptr<int>(),
                     H_T_sp_data.data_ptr<float>(), W.data_ptr<float>());
}