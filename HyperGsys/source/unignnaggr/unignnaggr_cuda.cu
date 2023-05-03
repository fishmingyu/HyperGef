#include "../../include/hgnnAgg.cuh"
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

void assertTensor(torch::Tensor &T, c10::ScalarType type) {
  assert(T.is_contiguous());
  assert(T.device().type() == torch::kCUDA);
  assert(T.dtype() == type);
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void
UniGNNAggrdeg_forward_kernel(int group, int feature_size, Index *group_key,
                             Index *group_row, Index *group_st, Index *group_ed,
                             Index *B_indices, DType *degE, DType *degV,
                             DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = blockIdx.y * NDIMS_PB + threadIdx.x;
  if (gid < group) {
    Index eid = __ldg(group_row + gid);
    Index rid = __ldg(group_st + gid);
    Index rd_start = __ldg(group_key + rid);
    Index rd_end = __ldg(group_key + rid + 1);

    DType B_acc = 0;
    DType degE_val = degE[eid];
    for (Index bcol = rd_start; bcol < rd_end; bcol++) {
      Index B_col_idx = B_indices[bcol];
      // shptr[bcol - start] = B_col_idx;
      B_acc += in_feature[B_col_idx * feature_size + k_idx];
    }
    B_acc *= degE_val;
    Index wid = __ldg(group_ed + gid);
    Index wr_start = __ldg(group_key + wid);
    Index wr_end = __ldg(group_key + wid + 1);

    for (Index bcol = wr_start; bcol < wr_end; bcol++) {
      Index v = B_indices[bcol];
      DType degV_val = degV[bcol];
      atomicAdd(out_feature + v * feature_size + k_idx, B_acc * degV_val);
    }
  }
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void UniGNNAggrdeg_forward_kernel_shm(
    int group, int feature_size, Index *group_key, Index *group_row,
    Index *group_st, Index *group_ed, Index *B_indices, DType *degE,
    DType *degV, DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = blockIdx.y * NDIMS_PB + threadIdx.x;
  int sh_idx = threadIdx.y * feature_size + k_idx;

  extern __shared__ DType datashm[];
  // float *datashm = (float *)&shmptr[MDIMS_PB];
  // shmptr[threadIdx.y] = __ldg(group_ed + gid);
  if (gid < group) {
    Index eid = __ldg(group_row + gid);
    Index rid = __ldg(group_st + gid);
    Index rd_start = __ldg(group_key + rid);
    Index rd_end = __ldg(group_key + rid + 1);

    DType B_acc = 0;
    DType degE_val = degE[eid];
    for (Index bcol = rd_start; bcol < rd_end; bcol++) {
      Index B_col_idx = B_indices[bcol];
      // shptr[bcol - start] = B_col_idx;
      B_acc += in_feature[B_col_idx * feature_size + k_idx];
    }
    B_acc *= degE_val;
    datashm[sh_idx] = B_acc; // k_idx
    Index wid = __ldg(group_ed + gid);
    Index wr_start = __ldg(group_key + wid);
    Index wr_end = __ldg(group_key + wid + 1);
    __syncthreads();
    DType sh_acc = datashm[k_idx] + datashm[k_idx + feature_size];
    __syncthreads();
    if ((group & 1) == 0) // even
    {
      // a, b are registers but their values are same in different warps
      // since the function is equal to shared memory, it is unnecessary to use
      // explicit shared memory(otherwise will cause performance loss)
      Index a = __ldg(group_ed + blockIdx.x * MDIMS_PB);
      Index b = __ldg(group_ed + blockIdx.x * MDIMS_PB + 1);
      if (a != b) {
        Index bcol = wr_start;
        for (; bcol < wr_end - 1; bcol += 2) {
          Index v1 = B_indices[bcol];
          Index v2 = B_indices[bcol + 1];
          DType degV_val1 = degV[v1];
          DType degV_val2 = degV[v2];
          Index B_col_adr_1 = v1 * feature_size + k_idx;
          Index B_col_adr_2 = v2 * feature_size + k_idx;
          atomicAdd(out_feature + B_col_adr_1, B_acc * degV_val1);
          atomicAdd(out_feature + B_col_adr_2, B_acc * degV_val2);
        }
        if ((wr_end - wr_start) % 2 != 0) {
          Index v = B_indices[bcol];
          DType degV_val = degV[v];
          atomicAdd(out_feature + v * feature_size + k_idx, B_acc * degV_val);
        }
      } else if (threadIdx.y == 0) {
        Index bcol = wr_start;
        for (; bcol < wr_end - 1; bcol += 2) {
          Index v1 = B_indices[bcol];
          Index v2 = B_indices[bcol + 1];
          DType degV_val1 = degV[v1];
          DType degV_val2 = degV[v2];
          Index B_col_adr_1 = v1 * feature_size + k_idx;
          Index B_col_adr_2 = v2 * feature_size + k_idx;
          atomicAdd(out_feature + B_col_adr_1, sh_acc * degV_val1);
          atomicAdd(out_feature + B_col_adr_2, sh_acc * degV_val2);
        }
        if ((wr_end - wr_start) % 2 != 0) {
          Index v = B_indices[bcol];
          DType degV_val = degV[v];
          atomicAdd(out_feature + v * feature_size + k_idx, sh_acc * degV_val);
        }
      }
    } else {
      if (gid == group - 1) {
        Index bcol = wr_start;
        for (; bcol < wr_end - 1; bcol += 2) {
          Index v1 = B_indices[bcol];
          Index v2 = B_indices[bcol + 1];
          DType degV_val1 = degV[v1];
          DType degV_val2 = degV[v2];
          Index B_col_adr_1 = v1 * feature_size + k_idx;
          Index B_col_adr_2 = v2 * feature_size + k_idx;
          atomicAdd(out_feature + B_col_adr_1, B_acc * degV_val1);
          atomicAdd(out_feature + B_col_adr_2, B_acc * degV_val2);
        }
        if ((wr_end - wr_start) % 2 != 0) {
          Index v = B_indices[bcol];
          DType degV_val = degV[v];
          atomicAdd(out_feature + v * feature_size + k_idx, B_acc * degV_val);
        }
      } else {
        Index a = __ldg(group_ed + blockIdx.x * MDIMS_PB);
        Index b = __ldg(group_ed + blockIdx.x * MDIMS_PB + 1);
        if (a != b) {
          Index bcol = wr_start;
          for (; bcol < wr_end - 1; bcol += 2) {
            Index v1 = B_indices[bcol];
            Index v2 = B_indices[bcol + 1];
            DType degV_val1 = degV[v1];
            DType degV_val2 = degV[v2];
            Index B_col_adr_1 = v1 * feature_size + k_idx;
            Index B_col_adr_2 = v2 * feature_size + k_idx;
            atomicAdd(out_feature + B_col_adr_1, B_acc * degV_val1);
            atomicAdd(out_feature + B_col_adr_2, B_acc * degV_val2);
          }
          if ((wr_end - wr_start) % 2 != 0) {
            Index v = B_indices[bcol];
            DType degV_val = degV[v];
            atomicAdd(out_feature + v * feature_size + k_idx, B_acc * degV_val);
          }
        } else if (threadIdx.y == 0) {
          Index bcol = wr_start;
          for (; bcol < wr_end - 1; bcol += 2) {
            Index v1 = B_indices[bcol];
            Index v2 = B_indices[bcol + 1];
            DType degV_val1 = degV[v1];
            DType degV_val2 = degV[v2];
            Index B_col_adr_1 = v1 * feature_size + k_idx;
            Index B_col_adr_2 = v2 * feature_size + k_idx;
            atomicAdd(out_feature + B_col_adr_1, sh_acc * degV_val1);
            atomicAdd(out_feature + B_col_adr_2, sh_acc * degV_val2);
          }
          if ((wr_end - wr_start) % 2 != 0) {
            Index v = B_indices[bcol];
            DType degV_val = degV[v];
            atomicAdd(out_feature + v * feature_size + k_idx,
                      sh_acc * degV_val);
          }
        }
      }
    }
  }
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void UniGNNAggrdeg_forward_kernel_sf(
    int group, int feature_size, Index *group_key, Index *group_row,
    Index *group_st, Index *group_ed, Index *B_indices, DType *degE,
    DType *degV, DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = threadIdx.x;
  if (gid < group) {
    Index eid = __ldg(group_row + gid);
    Index rid = __ldg(group_st + gid);
    Index rd_start = __ldg(group_key + rid);
    Index rd_end = __ldg(group_key + rid + 1);

    DType B_acc = 0;
    DType degE_val = degE[eid];
    if (k_idx < feature_size) {
      for (Index bcol = rd_start; bcol < rd_end; bcol++) {
        Index B_col_idx = B_indices[bcol];
        // shptr[bcol - start] = B_col_idx;
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      B_acc *= degE_val;
      Index wid = __ldg(group_ed + gid);
      Index wr_start = __ldg(group_key + wid);
      Index wr_end = __ldg(group_key + wid + 1);

      for (Index bcol = wr_start; bcol < wr_end; bcol++) {
        Index v = B_indices[bcol];
        DType degV_val = degV[bcol];
        atomicAdd(out_feature + v * feature_size + k_idx, B_acc * degV_val);
      }
    }
  }
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void UniGNNAggr_forward_kernel(int group, int feature_size,
                                          Index *group_key, Index *group_row,
                                          Index *group_st, Index *group_ed,
                                          Index *B_indices, DType *in_feature,
                                          DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = blockIdx.y * NDIMS_PB + threadIdx.x;
  if (gid < group) {
    Index eid = __ldg(group_row + gid);
    Index rid = __ldg(group_st + gid);
    Index rd_start = __ldg(group_key + rid);
    Index rd_end = __ldg(group_key + rid + 1);

    DType B_acc = 0;
    for (Index bcol = rd_start; bcol < rd_end; bcol++) {
      Index B_col_idx = B_indices[bcol];
      // shptr[bcol - start] = B_col_idx;
      B_acc += in_feature[B_col_idx * feature_size + k_idx];
    }
    Index wid = __ldg(group_ed + gid);
    Index wr_start = __ldg(group_key + wid);
    Index wr_end = __ldg(group_key + wid + 1);

    for (Index bcol = wr_start; bcol < wr_end; bcol++) {
      Index v = B_indices[bcol];
      atomicAdd(out_feature + v * feature_size + k_idx, B_acc);
    }
  }
}

// MDIMS_PB = 2[TODO more]
template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void
UniGNNAggr_forward_kernel(int group, int feature_size, Index *group_key,
                          Index *group_st, Index *group_ed, Index *B_indices,
                          DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = blockIdx.y * NDIMS_PB + threadIdx.x;
  int sh_idx = threadIdx.y * feature_size + k_idx;

  extern __shared__ DType datashm[];
  // float *datashm = (float *)&shmptr[MDIMS_PB];
  // shmptr[threadIdx.y] = __ldg(group_ed + gid);
  if (gid < group) {
    Index rid = __ldg(group_st + gid);
    Index rd_start = __ldg(group_key + rid);
    Index rd_end = __ldg(group_key + rid + 1);

    DType B_acc = 0;
    for (Index bcol = rd_start; bcol < rd_end; bcol++) {
      Index B_col_idx = B_indices[bcol];
      // shptr[bcol - start] = B_col_idx;
      B_acc += in_feature[B_col_idx * feature_size + k_idx];
    }
    datashm[sh_idx] = B_acc; // k_idx
    Index wid = __ldg(group_ed + gid);
    Index wr_start = __ldg(group_key + wid);
    Index wr_end = __ldg(group_key + wid + 1);
    __syncthreads();
    DType sh_acc = datashm[k_idx] + datashm[k_idx + feature_size];
    __syncthreads();
    if ((group & 1) == 0) // even
    {
      // a, b are registers but their values are same in different warps
      // since the function is equal to shared memory, it is unnecessary to use
      // explicit shared memory(otherwise will cause performance loss)
      Index a = __ldg(group_ed + blockIdx.x * MDIMS_PB);
      Index b = __ldg(group_ed + blockIdx.x * MDIMS_PB + 1);
      if (a != b) {
        Index bcol = wr_start;
        for (; bcol < wr_end - 1; bcol += 2) {
          Index B_col_adr_1 = B_indices[bcol] * feature_size + k_idx;
          Index B_col_adr_2 = B_indices[bcol + 1] * feature_size + k_idx;
          atomicAdd(out_feature + B_col_adr_1, B_acc);
          atomicAdd(out_feature + B_col_adr_2, B_acc);
        }
        if ((wr_end - wr_start) % 2 != 0) {
          Index B_col_idx_1 = B_indices[bcol];
          atomicAdd(out_feature + B_col_idx_1 * feature_size + k_idx, B_acc);
        }
      } else if (threadIdx.y == 0) {
        Index bcol = wr_start;
        for (; bcol < wr_end - 1; bcol += 2) {
          Index B_col_adr_1 = B_indices[bcol] * feature_size + k_idx;
          Index B_col_adr_2 = B_indices[bcol + 1] * feature_size + k_idx;
          atomicAdd(out_feature + B_col_adr_1, sh_acc);
          atomicAdd(out_feature + B_col_adr_2, sh_acc);
        }
        if ((wr_end - wr_start) % 2 != 0) {
          Index B_col_idx_1 = B_indices[bcol];
          atomicAdd(out_feature + B_col_idx_1 * feature_size + k_idx, sh_acc);
        }
      }
    } else {
      if (gid == group - 1) {
        Index bcol = wr_start;
        for (; bcol < wr_end - 1; bcol += 2) {
          Index B_col_adr_1 = B_indices[bcol] * feature_size + k_idx;
          Index B_col_adr_2 = B_indices[bcol + 1] * feature_size + k_idx;
          atomicAdd(out_feature + B_col_adr_1, B_acc);
          atomicAdd(out_feature + B_col_adr_2, B_acc);
        }
        if ((wr_end - wr_start) % 2 != 0) {
          Index B_col_idx_1 = B_indices[bcol];
          atomicAdd(out_feature + B_col_idx_1 * feature_size + k_idx, B_acc);
        }
      } else {
        Index a = __ldg(group_ed + blockIdx.x * MDIMS_PB);
        Index b = __ldg(group_ed + blockIdx.x * MDIMS_PB + 1);
        if (a != b) {
          Index bcol = wr_start;
          for (; bcol < wr_end - 1; bcol += 2) {
            Index B_col_adr_1 = B_indices[bcol] * feature_size + k_idx;
            Index B_col_adr_2 = B_indices[bcol + 1] * feature_size + k_idx;
            atomicAdd(out_feature + B_col_adr_1, B_acc);
            atomicAdd(out_feature + B_col_adr_2, B_acc);
          }
          if ((wr_end - wr_start) % 2 != 0) {
            Index B_col_idx_1 = B_indices[bcol];
            atomicAdd(out_feature + B_col_idx_1 * feature_size + k_idx, B_acc);
          }
        } else if (threadIdx.y == 0) {
          Index bcol = wr_start;
          for (; bcol < wr_end - 1; bcol += 2) {
            Index B_col_adr_1 = B_indices[bcol] * feature_size + k_idx;
            Index B_col_adr_2 = B_indices[bcol + 1] * feature_size + k_idx;
            atomicAdd(out_feature + B_col_adr_1, sh_acc);
            atomicAdd(out_feature + B_col_adr_2, sh_acc);
          }
          if ((wr_end - wr_start) % 2 != 0) {
            Index B_col_idx_1 = B_indices[bcol];
            atomicAdd(out_feature + B_col_idx_1 * feature_size + k_idx, sh_acc);
          }
        }
      }
    }
  }
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void
UniGNNAggr_forward_kernel_sf(int group, int feature_size, Index *group_key,
                             Index *group_row, Index *group_st, Index *group_ed,
                             Index *B_indices, DType *in_feature,
                             DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = threadIdx.x;
  if (gid < group) {
    Index eid = __ldg(group_row + gid);
    Index rid = __ldg(group_st + gid);
    Index rd_start = __ldg(group_key + rid);
    Index rd_end = __ldg(group_key + rid + 1);

    DType B_acc = 0;
    if (k_idx < feature_size) {
      for (Index bcol = rd_start; bcol < rd_end; bcol++) {
        Index B_col_idx = B_indices[bcol];
        // shptr[bcol - start] = B_col_idx;
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      Index wid = __ldg(group_ed + gid);
      Index wr_start = __ldg(group_key + wid);
      Index wr_end = __ldg(group_key + wid + 1);

      for (Index bcol = wr_start; bcol < wr_end; bcol++) {
        Index v = B_indices[bcol];
        atomicAdd(out_feature + v * feature_size + k_idx, B_acc);
      }
    }
  }
}

torch::Tensor
unignnaggrdeg_fp_cuda(torch::Tensor balan_key, torch::Tensor balan_row,
                      torch::Tensor group_start, torch::Tensor group_end,
                      torch::Tensor H_t_indices, torch::Tensor in_feat,
                      torch::Tensor degE, torch::Tensor degV) {
  assertTensor(balan_key, torch::kInt32);
  assertTensor(balan_row, torch::kInt32);
  assertTensor(group_start, torch::kInt32);
  assertTensor(group_end, torch::kInt32);
  assertTensor(H_t_indices, torch::kInt32);
  assertTensor(in_feat, torch::kFloat32);
  assertTensor(degE, torch::kFloat32);
  assertTensor(degV, torch::kFloat32);

  int nrow = degV.size(0);
  int feature_size = in_feat.size(1);
  int groups = balan_row.size(0);
  auto devid = in_feat.device().index();
  int nedge = degE.size(0);
  int nnz = H_t_indices.size(0);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({nrow, feature_size}, options);
  int MDIMS_PB = 2;
  int NDIMS_PB = 32;
  const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
  const int _MDIMS_BLOCKS = CEIL(groups, MDIMS_PB);
  dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
  dim3 blockDim(NDIMS_PB, MDIMS_PB, 1);
  if (feature_size >= 32) {
    if (nnz > 10 * nedge) {
      UniGNNAggrdeg_forward_kernel_shm<Index, DType, 2, 32>
          <<<gridDim, blockDim>>>(
              groups, feature_size, balan_key.data_ptr<int>(),
              balan_row.data_ptr<int>(), group_start.data_ptr<int>(),
              group_end.data_ptr<int>(), H_t_indices.data_ptr<int>(),
              degE.data_ptr<float>(), degV.data_ptr<float>(),
              in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    } else {
      UniGNNAggrdeg_forward_kernel<Index, DType, 2, 32><<<gridDim, blockDim>>>(
          groups, feature_size, balan_key.data_ptr<int>(),
          balan_row.data_ptr<int>(), group_start.data_ptr<int>(),
          group_end.data_ptr<int>(), H_t_indices.data_ptr<int>(),
          degE.data_ptr<float>(), degV.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    }
  } else {
    UniGNNAggrdeg_forward_kernel_sf<Index, DType, 2, 32><<<gridDim, blockDim>>>(
        groups, feature_size, balan_key.data_ptr<int>(),
        balan_row.data_ptr<int>(), group_start.data_ptr<int>(),
        group_end.data_ptr<int>(), H_t_indices.data_ptr<int>(),
        degE.data_ptr<float>(), degV.data_ptr<float>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
  }
  return out_feat;
}

torch::Tensor
unignnaggr_fp_cuda(torch::Tensor balan_key, torch::Tensor balan_row,
                   torch::Tensor group_start, torch::Tensor group_end,
                   torch::Tensor H_t_indices, torch::Tensor in_feat) {
  assertTensor(balan_key, torch::kInt32);
  assertTensor(balan_row, torch::kInt32);
  assertTensor(group_start, torch::kInt32);
  assertTensor(group_end, torch::kInt32);
  assertTensor(H_t_indices, torch::kInt32);
  assertTensor(in_feat, torch::kFloat32);

  int nrow = in_feat.size(0);
  int feature_size = in_feat.size(1);
  int groups = balan_row.size(0);
  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({nrow, feature_size}, options);
  int MDIMS_PB = 2;
  int NDIMS_PB = 32;
  const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
  const int _MDIMS_BLOCKS = CEIL(groups, MDIMS_PB);
  dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
  dim3 blockDim(NDIMS_PB, MDIMS_PB, 1);
  if (feature_size >= 32) {
    UniGNNAggr_forward_kernel<Index, DType, 2, 32><<<gridDim, blockDim>>>(
        groups, feature_size, balan_key.data_ptr<int>(),
        balan_row.data_ptr<int>(), group_start.data_ptr<int>(),
        group_end.data_ptr<int>(), H_t_indices.data_ptr<int>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>());

  } else {
    UniGNNAggr_forward_kernel_sf<Index, DType, 2, 32><<<gridDim, blockDim>>>(
        groups, feature_size, balan_key.data_ptr<int>(),
        balan_row.data_ptr<int>(), group_start.data_ptr<int>(),
        group_end.data_ptr<int>(), H_t_indices.data_ptr<int>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
  }
  return out_feat;
}
