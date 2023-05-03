// #include "../../include/spmm/spmm.cuh"
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor csrspmm_rowbalance_cuda(torch::Tensor csrptr,
                                      torch::Tensor indices,
                                      torch::Tensor edge_val,
                                      torch::Tensor in_feat);

torch::Tensor csrspmm_rowbalance_degV_cuda(torch::Tensor csrptr,
                                           torch::Tensor indices,
                                           torch::Tensor edge_val,
                                           torch::Tensor in_feat,
                                           torch::Tensor degV);

torch::Tensor csrspmm_hybrid_cuda(const int M_dim, const int keys,
                                  torch::Tensor indices, torch::Tensor edge_val,
                                  torch::Tensor in_feat, torch::Tensor key_ptr,
                                  torch::Tensor group_key,
                                  torch::Tensor group_row);

torch::Tensor
csrspmm_neighborgroup_cuda(const int Mdim, torch::Tensor group_key,
                           torch::Tensor group_row, torch::Tensor indices,
                           torch::Tensor edge_val, torch::Tensor in_feat);

torch::Tensor csrspmm_rowbalance_test_cuda(const int iter, torch::Tensor csrptr,
                                           torch::Tensor indices,
                                           torch::Tensor edge_val,
                                           torch::Tensor in_feat);

torch::Tensor csrspmm_neighborgroup_test_cuda(const int iter, const int Mdim,
                                              torch::Tensor group_key,
                                              torch::Tensor group_row,
                                              torch::Tensor indices,
                                              torch::Tensor edge_val,
                                              torch::Tensor in_feat);

torch::Tensor csrspmm_edgebalance_cuda(int ncol, torch::Tensor csrptr,
                                       torch::Tensor indices,
                                       torch::Tensor edge_val,
                                       torch::Tensor in_feat);

torch::Tensor csrspmm_cusparse_cuda(int ncol, torch::Tensor sp_csrptr,
                                    torch::Tensor sp_csrind,
                                    torch::Tensor sp_data,
                                    torch::Tensor in_feat);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "spmm";
  m.def("csrspmm_rowbalance", &csrspmm_rowbalance_cuda, "csrspmm row-balanced");
  m.def("csrspmm_rowbalance_degV", &csrspmm_rowbalance_degV_cuda,
        "csrspmm row-balanced with fused degV");
  m.def("csrspmm_neighbor_group", &csrspmm_neighborgroup_cuda,
        "csrspmm neighbro group");
  m.def("csrspmm_rowbalance_test", &csrspmm_rowbalance_test_cuda,
        "csrspmm row-balanced test");
  m.def("csrspmm_neighbor_group_test", &csrspmm_neighborgroup_test_cuda,
        "csrspmm neighbor group test");
  m.def("csrspmm_hybrid", &csrspmm_hybrid_cuda, "csrspmm hybrid");
  m.def("csrspmm_edgebalance", &csrspmm_edgebalance_cuda,
        "csrspmm edge balance");
  m.def("csrspmm_cusparse", &csrspmm_cusparse_cuda, "csrspmm cusparse");
}
