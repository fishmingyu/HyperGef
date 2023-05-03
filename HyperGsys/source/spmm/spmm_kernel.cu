#include "../../include/spmm/spmm.cuh"
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

void assertTensor(torch::Tensor &T, c10::ScalarType type) {
  assert(T.is_contiguous());
  assert(T.device().type() == torch::kCUDA);
  assert(T.dtype() == type);
}

torch::Tensor csrspmm_rowbalance_cuda(torch::Tensor csrptr,
                                      torch::Tensor indices,
                                      torch::Tensor edge_val,
                                      torch::Tensor in_feat) {
  assertTensor(csrptr, torch::kInt32);
  assertTensor(indices, torch::kInt32);
  assertTensor(in_feat, torch::kFloat32);
  assertTensor(edge_val, torch::kFloat32);
  int Mdim_worker = csrptr.size(0) - 1;
  int v = Mdim_worker;
  int Ndim_worker = in_feat.size(1);
  int f = Ndim_worker;
  int e = indices.size(0);
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
  int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);

  auto out_feat = torch::zeros({v, f}, options);

  csrspmm_rowbalance_kernel<<<gridDim, blockDim>>>(
      Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(), indices.data_ptr<int>(),
      edge_val.data_ptr<float>(), in_feat.data_ptr<float>(),
      out_feat.data_ptr<float>());

  return out_feat;
}

torch::Tensor csrspmm_rowbalance_test_cuda(const int iter, torch::Tensor csrptr,
                                           torch::Tensor indices,
                                           torch::Tensor edge_val,
                                           torch::Tensor in_feat) {
  assertTensor(csrptr, torch::kInt32);
  assertTensor(indices, torch::kInt32);
  assertTensor(in_feat, torch::kFloat32);
  assertTensor(edge_val, torch::kFloat32);
  int Mdim_worker = csrptr.size(0) - 1;
  int v = Mdim_worker;
  int Ndim_worker = in_feat.size(1);
  int f = Ndim_worker;
  int e = indices.size(0);
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
  int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);

  auto out_feat = torch::zeros({v, f}, options);
  for (int i = 0; i < iter; i++)
    csrspmm_rowbalance_kernel<<<gridDim, blockDim>>>(
        Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
        indices.data_ptr<int>(), edge_val.data_ptr<float>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>());

  return out_feat;
}

torch::Tensor csrspmm_rowbalance_degV_cuda(torch::Tensor csrptr,
                                           torch::Tensor indices,
                                           torch::Tensor edge_val,
                                           torch::Tensor in_feat,
                                           torch::Tensor degV) {
  assertTensor(csrptr, torch::kInt32);
  assertTensor(indices, torch::kInt32);
  assertTensor(in_feat, torch::kFloat32);
  assertTensor(edge_val, torch::kFloat32);
  assertTensor(degV, torch::kFloat32);
  int Mdim_worker = csrptr.size(0) - 1;
  int v = Mdim_worker;
  int Ndim_worker = in_feat.size(1);
  int f = Ndim_worker;
  int e = indices.size(0);
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
  int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);

  auto out_feat = torch::zeros({v, f}, options);

  csrspmm_rowbalance_degV_kernel<<<gridDim, blockDim>>>(
      Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(), indices.data_ptr<int>(),
      edge_val.data_ptr<float>(), in_feat.data_ptr<float>(),
      out_feat.data_ptr<float>(), degV.data_ptr<float>());

  return out_feat;
}

torch::Tensor
csrspmm_neighborgroup_cuda(const int Mdim, torch::Tensor group_key,
                           torch::Tensor group_row, torch::Tensor indices,
                           torch::Tensor edge_val, torch::Tensor in_feat) {
  assertTensor(group_key, torch::kInt32);
  assertTensor(group_row, torch::kInt32);
  assertTensor(in_feat, torch::kFloat32);
  assertTensor(edge_val, torch::kFloat32);
  assertTensor(indices, torch::kInt32);
  int m = Mdim;
  int feature_size = in_feat.size(1);
  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, feature_size}, options);
  int edge_groups = group_row.size(0);
  int Mdim_worker = edge_groups;
  int Ndim_worker = feature_size;
  Index ref_block = (feature_size > 256) ? feature_size : 256;
  int Ndim_threadblock = CEIL(Ndim_worker, ref_block);
  int Ndim_thread_per_tb = min(Ndim_worker, ref_block);
  int Mdim_thread_per_tb = CEIL(ref_block, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  csrspmm_edgegroup_kernel<<<gridDim, blockDim>>>(
      edge_groups, feature_size, group_key.data_ptr<int>(),
      group_row.data_ptr<int>(), indices.data_ptr<int>(),
      edge_val.data_ptr<float>(), in_feat.data_ptr<float>(),
      out_feat.data_ptr<float>());

  return out_feat;
}

torch::Tensor csrspmm_neighborgroup_test_cuda(const int iter, const int Mdim,
                                              torch::Tensor group_key,
                                              torch::Tensor group_row,
                                              torch::Tensor indices,
                                              torch::Tensor edge_val,
                                              torch::Tensor in_feat) {
  assertTensor(group_key, torch::kInt32);
  assertTensor(group_row, torch::kInt32);
  assertTensor(in_feat, torch::kFloat32);
  assertTensor(edge_val, torch::kFloat32);
  assertTensor(indices, torch::kInt32);
  int m = Mdim;
  int feature_size = in_feat.size(1);
  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, feature_size}, options);
  int edge_groups = group_row.size(0);
  int Mdim_worker = edge_groups;
  int Ndim_worker = feature_size;
  Index ref_block = (feature_size > 256) ? feature_size : 256;
  int Ndim_threadblock = CEIL(Ndim_worker, ref_block);
  int Ndim_thread_per_tb = min(Ndim_worker, ref_block);
  int Mdim_thread_per_tb = CEIL(ref_block, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);
  for (int i = 0; i < iter; i++)
    csrspmm_edgegroup_kernel<<<gridDim, blockDim>>>(
        edge_groups, feature_size, group_key.data_ptr<int>(),
        group_row.data_ptr<int>(), indices.data_ptr<int>(),
        edge_val.data_ptr<float>(), in_feat.data_ptr<float>(),
        out_feat.data_ptr<float>());

  return out_feat;
}

torch::Tensor csrspmm_hybrid_cuda(const int M_dim, const int keys,
                                  torch::Tensor indices, torch::Tensor edge_val,
                                  torch::Tensor in_feat, torch::Tensor key_ptr,
                                  torch::Tensor group_key,
                                  torch::Tensor group_row) {
  assertTensor(indices, torch::kInt32);
  assertTensor(edge_val, torch::kFloat32);
  assertTensor(in_feat, torch::kFloat32);
  assertTensor(key_ptr, torch::kInt32);
  assertTensor(group_key, torch::kInt32);
  assertTensor(group_row, torch::kInt32);
  int v = M_dim;
  int f = in_feat.size(1);
  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({v, f}, options);
  int edge_groups = keys;
  int feature_size = f;
  int Mdim_worker = edge_groups;
  int Ndim_worker = feature_size;
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
  int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  csrspmm_hybrid_kernel<<<gridDim, blockDim>>>(
      keys, feature_size, key_ptr.data_ptr<int>(), group_key.data_ptr<int>(),
      group_row.data_ptr<int>(), indices.data_ptr<int>(),
      edge_val.data_ptr<float>(), in_feat.data_ptr<float>(),
      out_feat.data_ptr<float>());

  return out_feat;
}

torch::Tensor csrspmm_edgebalance_cuda(int ncol, torch::Tensor csrptr,
                                       torch::Tensor indices,
                                       torch::Tensor edge_val,
                                       torch::Tensor in_feat) {
  assertTensor(csrptr, torch::kInt32);
  assertTensor(indices, torch::kInt32);
  assertTensor(in_feat, torch::kFloat32);
  assertTensor(edge_val, torch::kFloat32);
  int feature_size = in_feat.size(1);
  int nrow = csrptr.size(0) - 1;
  int nnz = indices.size(0);
  int coarsen_factor =
      (feature_size >= 512) ? 4 : (feature_size >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(feature_size, (32 * coarsen_factor));

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  int ref_block = (feature_size > 256) ? feature_size : 256;
  int Nnzdim_warp_per_tb = ref_block / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      nrow,
      Nnzdim_warp_per_tb *
          thread_nz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 * thread_nz );

  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({nrow, feature_size}, options);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  // simple heuristic
  if (coarsen_factor == 4) {
    if (thread_nz == 1)
      csrspmm_edgebalance_kernel<4, 1><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    if (thread_nz == 2)
      csrspmm_edgebalance_kernel<4, 2><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    if (thread_nz == 4)
      csrspmm_edgebalance_kernel<4, 4><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      csrspmm_edgebalance_kernel<2, 1><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    if (thread_nz == 2)
      csrspmm_edgebalance_kernel<2, 2><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    if (thread_nz == 4)
      csrspmm_edgebalance_kernel<2, 4><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
  } else {
    if (thread_nz == 1)
      csrspmm_edgebalance_kernel<1, 1><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    if (thread_nz == 2)
      csrspmm_edgebalance_kernel<1, 2><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    if (thread_nz == 4)
      csrspmm_edgebalance_kernel<1, 4><<<gridDim, blockDim, smem_size>>>(
          nrow, feature_size, ncol, nnz, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
  }
  return out_feat;
}

void csrspmm_cusparse_cuda(const int ncol, torch::Tensor sp_csrptr,
                           torch::Tensor sp_csrind, torch::Tensor sp_data,
                           torch::Tensor in_feat) {
  const int nrow = sp_csrptr.size(0) - 1;
  const int nnz = sp_csrind.size(0);
  const int feature_size = in_feat.size(1);
  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({nrow, feature_size}, options);
  csrspmm_cusparse<int, float>(
      nrow, ncol, nnz, feature_size, sp_csrptr.data_ptr<int>(),
      sp_csrind.data_ptr<int>(), sp_data.data_ptr<float>(),
      in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
}