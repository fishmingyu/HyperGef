#ifndef HGNNAGG
#define HGNNAGG

#include "./dataloader/dataloader.hpp"
#include "./taskbalancer/balancer.cuh"
#include "./util/check.cuh"
#include "./util/cuda_util.cuh"
#include "./util/gpuTimer.cuh"
#include "./util/ramArray.cuh"
#include <cuda.h>
#include <fstream>
#include <string>

#define ITER 100

enum hgnn_kernel_met {
  edge_based_fused,
  edge_based_balance,
  edge_based_parallel,
  edge_based_merge,
  edge_based_full,
  edge_based_shm,
  node_group,
  edge_group_hgnn,
  edge_group_part,
  edge_group_tune,
  sn_group,
  sn_group_tune,
  tsbalan,
  tsbalan_bl
};

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void HyperGAggr_Edgefused_Kernel(int B_row, int feature_size,
                                            Index *B_indptr, Index *B_indices,
                                            DType *in_feature,
                                            DType *out_feature) {
  int B_row_idx = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = blockIdx.y * NDIMS_PB + threadIdx.x;
  if (B_row_idx < B_row) {
    Index start = B_indptr[B_row_idx];
    Index end = B_indptr[B_row_idx + 1];
    DType B_acc = 0;
    for (Index bcol = start; bcol < end; bcol++) {
      Index B_col_idx = B_indices[bcol];
      B_acc += in_feature[B_col_idx * feature_size + k_idx];
    }
    for (Index bcol = start; bcol < end; bcol++) {
      Index B_col_idx = B_indices[bcol];
      atomicAdd(out_feature + B_col_idx * feature_size + k_idx, B_acc);
    }
  }
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void
HyperGAggr_Edgefused_Balance_Kernel(int B_row, int group, int feature_size,
                                    Index *group_key, Index *work_ind,
                                    Index *work_ts_ind, Index *B_indices,
                                    DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = blockIdx.y * NDIMS_PB + threadIdx.x;
  if (gid < group) {
    Index rid = binary_search<Index>(work_ts_ind, gid, 0, B_row);
    Index workload = work_ind[rid + 1] - work_ind[rid];
    Index offset = gid - work_ts_ind[rid];

    Index id1 = (offset / workload) + work_ind[rid];
    Index id2 = offset % workload + work_ind[rid];

    Index rd_start = __ldg(group_key + id1);
    Index rd_end = __ldg(group_key + id1 + 1);

    DType B_acc = 0;
    for (Index bcol = rd_start; bcol < rd_end; bcol++) {
      Index B_col_idx = B_indices[bcol];
      // shptr[bcol - start] = B_col_idx;
      B_acc += in_feature[B_col_idx * feature_size + k_idx];
    }

    Index wr_start = __ldg(group_key + id2);
    Index wr_end = __ldg(group_key + id2 + 1);
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
  }
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void HyperGAggr_Edgefused_Balance_Full_Kernel(
    int B_row, int group, int feature_size, Index *group_key, Index *group_st,
    Index *group_ed, Index *B_indices, DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  const int k_idx = blockIdx.y * NDIMS_PB + threadIdx.x;
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
    Index wid = __ldg(group_ed + gid);
    Index wr_start = __ldg(group_key + wid);
    Index wr_end = __ldg(group_key + wid + 1);
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
  }
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void HyperGAggr_Edgefused_Balance_Full_Kernel_sf(
    int B_row, int group, int feature_size, Index *group_key, Index *group_st,
    Index *group_ed, Index *B_indices, DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  int k_idx = threadIdx.x;
  if (gid < group) {
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
    }
  }
}

// MDIMS_PB = 2[TODO more]
template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void HyperGAggr_Edgefused_Balance_Shm_Kernel(
    int B_row, int part, int group, int feature_size, Index *group_key,
    Index *group_st, Index *group_ed, Index *B_indices, DType *in_feature,
    DType *out_feature) {
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
HyperGAggr_Edgefused_Merge_Kernel(int B_row, int feature_size, Index *B_cbptr,
                                  Index *B_cbst, Index *B_cbind,
                                  DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x; // * MDIMS_PB + threadIdx.y;
  int k_idx = threadIdx.x;
  Index rd_start = __ldg(B_cbptr + gid);
  Index rd_end = __ldg(B_cbptr + gid + 1);

  DType B_acc_1 = 0;
  DType B_acc_2 = 0;
  for (Index bcol = rd_start; bcol < rd_end; bcol++) {
    Index B_col_idx = B_cbind[bcol];
    Index indicator = B_cbst[bcol]; // 1,2,3
    // shptr[bcol - start] = B_col_idx;
    B_acc_1 +=
        in_feature[B_col_idx * feature_size + k_idx] * (indicator & 1); // e1
    B_acc_2 +=
        in_feature[B_col_idx * feature_size + k_idx] * (indicator & 2); // e2
  }

  for (Index bcol = rd_start; bcol < rd_end; bcol++) {
    Index B_col_idx = B_cbind[bcol];
    Index indicator = B_cbst[bcol]; // 1,2,3
    Index B_col_adr = B_col_idx * feature_size + k_idx;
    Index B_acc = 0;
    if (indicator == 3)
      B_acc = B_acc_1 + B_acc_2;
    else if (indicator == 2)
      B_acc = B_acc_2;
    else
      B_acc = B_acc_1;
    atomicAdd(out_feature + B_col_adr, B_acc);
  }
}

template <class Index, class DType, int MDIMS_PB, int NDIMS_PB>
__global__ void HyperGAggr_Edgefused_Parallel_Kernel(
    int B_row, int group, int feature_size, Index *work_ind, Index *group_key,
    Index *B_rowptr, Index *B_indices, DType *in_feature, DType *out_feature) {
  int gid = blockIdx.x * MDIMS_PB + threadIdx.y;
  int k_idx = threadIdx.x;
  if (gid < group) {
    Index rid = binary_search<Index>(work_ind, gid, 0, B_row);
    Index rd_start = __ldg(B_rowptr + rid);
    Index rd_end = __ldg(B_rowptr + rid + 1);

    DType B_acc = 0;
    for (Index bcol = rd_start; bcol < rd_end; bcol++) {
      Index B_col_idx = B_indices[bcol];
      // shptr[bcol - start] = B_col_idx;
      B_acc += in_feature[B_col_idx * feature_size + k_idx];
    }

    Index wr_start = __ldg(group_key + gid);
    Index wr_end = __ldg(group_key + gid + 1);
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
  }
}

template <class Index, class DType>
__global__ void HyperGAggr_sngroup_kernel(int seg_parts, int feature_size,
                                          Index *sn_key, Index *sn_row,
                                          Index *A_indices, Index *B_indptr,
                                          Index *B_indices, DType *in_feature,
                                          DType *out_feature) {
  Index group_tile = blockDim.y; // combine a set of groups together
  int k_idx = threadIdx.x;
  Index subwarp_id = threadIdx.y;
  Index group = blockIdx.x * group_tile + subwarp_id; // which node_group
  Index row = sn_row[group];
  DType A_acc = 0;
  if (group < seg_parts) {
    Index start = __ldg(sn_key + group);
    Index end = __ldg(sn_key + group + 1);
    for (Index p = start; p < end; p++) {
      Index B_row_idx = __ldg(A_indices + p);
      Index B_lb = B_indptr[B_row_idx];
      Index B_hb = B_indptr[B_row_idx + 1];
      DType B_acc = 0;
      for (Index B_col_ptr = B_lb; B_col_ptr < B_hb; B_col_ptr++) {
        Index B_col_idx = B_indices[B_col_ptr];
        // DType B_val = B_data[B_col_ptr];
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      A_acc += B_acc;
    }
    atomicAdd(out_feature + row * feature_size + k_idx, A_acc);
  }
}

template <class Index, class DType, int _MDIMS_THREADS_PER_BLOCK,
          int _NDIMS_THREADS_PER_BLOCK, int _NZ_A_PER_THREAD,
          int _NZ_B_PER_THREAD>
__global__ void HyperGAggr_sn_group_kernel_template(
    int seg_parts, int feature_size, Index *sn_key, Index *sn_row,
    Index *A_indices, Index *B_indptr, Index *B_indices, DType *in_feature,
    DType *out_feature) {
  const int threadIdx_x0 = threadIdx.x % _NDIMS_THREADS_PER_BLOCK;
  const int threadIdx_x1 = threadIdx.x / _NDIMS_THREADS_PER_BLOCK;
  const int k_idx = blockIdx.y * _NDIMS_THREADS_PER_BLOCK + threadIdx_x0;
  const Index group =
      blockIdx.x * _MDIMS_THREADS_PER_BLOCK + threadIdx_x1; // which node_group
  if (group < seg_parts) {
    const Index row = sn_row[group];
    DType A_acc = 0;
    const Index start = __ldg(sn_key + group);
    const Index end = __ldg(sn_key + group + 1);
    for (Index i = start;; i += _NZ_A_PER_THREAD) {
      int p = i + threadIdx.z;
      if (p >= end)
        break;
      Index B_row_idx = __ldg(A_indices + p);
      Index B_lb = B_indptr[B_row_idx];
      Index B_hb = B_indptr[B_row_idx + 1];
      DType B_acc = 0;
      for (Index j = B_lb;; j = j + _NZ_B_PER_THREAD) {
        int B_col_ptr = j + threadIdx.y;
        if (B_col_ptr >= B_hb)
          break;
        Index B_col_idx = B_indices[B_col_ptr];
        // DType B_val = B_data[B_col_ptr];
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      A_acc += B_acc;
    }
    atomicAdd(out_feature + row * feature_size + k_idx, A_acc);
  }
}

template <class Index, class DType, int _NZ_A_PER_THREAD, int _NZ_B_PER_THREAD>
__global__ void HyperGAggr_sn_group_kernel_template_sf(
    int seg_parts, int feature_size, Index *sn_key, Index *sn_row,
    Index *A_indices, Index *B_indptr, Index *B_indices, DType *in_feature,
    DType *out_feature) {
  const int threadIdx_x0 = threadIdx.x % feature_size;
  const int threadIdx_x1 = threadIdx.x / feature_size;
  const int k_idx = blockIdx.y * feature_size + threadIdx_x0;
  const Index group = blockIdx.x + threadIdx_x1; // which node_group
  if (group < seg_parts) {
    const Index row = sn_row[group];
    DType A_acc = 0;
    const Index start = __ldg(sn_key + group);
    const Index end = __ldg(sn_key + group + 1);
    for (Index i = start;; i += _NZ_A_PER_THREAD) {
      int p = i + threadIdx.z;
      if (p >= end)
        break;
      Index B_row_idx = __ldg(A_indices + p);
      Index B_lb = B_indptr[B_row_idx];
      Index B_hb = B_indptr[B_row_idx + 1];
      DType B_acc = 0;
      for (Index j = B_lb;; j = j + _NZ_B_PER_THREAD) {
        int B_col_ptr = j + threadIdx.y;
        if (B_col_ptr >= B_hb)
          break;
        Index B_col_idx = B_indices[B_col_ptr];
        // DType B_val = B_data[B_col_ptr];
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      A_acc += B_acc;
    }
    atomicAdd(out_feature + row * feature_size + k_idx, A_acc);
  }
}

template <class Index, class DType, int _MDIMS_THREADS_PER_BLOCK,
          int _NDIMS_THREADS_PER_BLOCK, int _NZ_PER_THREAD>
__global__ void HyperGAggr_twostepbalan_kernel_wo_schedule(
    int seg_parts, int feature_size, Index *sn_key, Index *sn_row,
    Index *neighbor_key, Index *A_indices, Index *B_indptr, Index *B_indices,
    DType *in_feature, DType *out_feature) {
  const int threadIdx_x0 = threadIdx.x % _NDIMS_THREADS_PER_BLOCK;
  const int threadIdx_x1 = threadIdx.x / _NDIMS_THREADS_PER_BLOCK;
  const int k_idx = blockIdx.y * _NDIMS_THREADS_PER_BLOCK + threadIdx_x0;
  const Index group =
      blockIdx.x * _MDIMS_THREADS_PER_BLOCK + threadIdx_x1; // which node_group
  if (group < seg_parts) {
    const Index row = sn_row[group];
    DType acc = 0;
    const Index start = __ldg(sn_key + group);
    const Index end = __ldg(sn_key + group + 1) - 1;
    if (end < start)
      return;
    const Index maxIdx = __ldg(neighbor_key + end);
    for (int i = 0; i < maxIdx; i++) {
      Index B_row_idx, itv;
      __find_row_entry(i, neighbor_key, A_indices, start, end, B_row_idx, itv);
      Index B_lb = B_indptr[B_row_idx];
      int B_col_ptr = B_lb + itv;
      Index B_col_idx = B_indices[B_col_ptr];
      acc += in_feature[B_col_idx * feature_size + k_idx];
    }
    atomicAdd(out_feature + row * feature_size + k_idx, acc); // feature
  }
}

template <class Index, class DType, int _NZ_PER_THREAD>
__global__ void HyperGAggr_twostepbalan_kernel_wo_schedul_sf(
    int seg_parts, int feature_size, Index *sn_key, Index *sn_row,
    Index *neighbor_key, Index *A_indices, Index *B_indptr, Index *B_indices,
    DType *in_feature, DType *out_feature) {
  const int threadIdx_x0 = threadIdx.x % feature_size;
  const int threadIdx_x1 = threadIdx.x / feature_size;
  const int k_idx = blockIdx.y * feature_size + threadIdx_x0;
  const Index group = blockIdx.x + threadIdx_x1; // which node_group
  if (group < seg_parts) {
    const Index row = sn_row[group];
    DType acc = 0;
    const Index start = __ldg(sn_key + group);
    const Index end = __ldg(sn_key + group + 1) - 1;
    if (end < start)
      return;
    const Index maxIdx = __ldg(neighbor_key + end);
    for (int i = 0; i < maxIdx; i++) {
      Index B_row_idx, itv;
      __find_row_entry(i, neighbor_key, A_indices, start, end, B_row_idx, itv);
      Index B_lb = B_indptr[B_row_idx];
      int B_col_ptr = B_lb + itv;
      Index B_col_idx = B_indices[B_col_ptr];
      acc += in_feature[B_col_idx * feature_size + k_idx];
    }
    atomicAdd(out_feature + row * feature_size + k_idx, acc); // feature
  }
}

template <class Index, class DType, int _MDIMS_THREADS_PER_BLOCK,
          int _NDIMS_THREADS_PER_BLOCK>
__global__ void HyperGAggr_twostepbalan_kernel_template_lf(
    int max_load, int seg_parts, int feature_size, Index *sn_key, Index *sn_row,
    Index *neighbor_key, Index *A_indices, Index *B_indptr, Index *B_indices,
    DType *in_feature, DType *out_feature) {
  const int k_idx = blockIdx.y * _NDIMS_THREADS_PER_BLOCK * 2 +
                    threadIdx.x % _NDIMS_THREADS_PER_BLOCK;
  int base = threadIdx.y * (2 * max_load - 1);
  const Index group =
      blockIdx.x * _MDIMS_THREADS_PER_BLOCK + threadIdx.y; // which node_group
  extern __shared__ Index shr_tmp[];                       // grouping
  if (group < seg_parts) {
    const Index row = sn_row[group];
    DType acc1 = 0, acc2 = 0;
    const Index start = __ldg(sn_key + group);
    const Index end = __ldg(sn_key + group + 1) - 1;
    if (end < start)
      return;
    const Index maxIdx = __ldg(neighbor_key + end);

    // cache two-step-neighbor in shared memory
    for (Index i = 0;; i += _NDIMS_THREADS_PER_BLOCK) {
      int p = i + threadIdx.x;
      if (p >= maxIdx)
        break;
      Index B_row_idx, itv;
      __find_row_entry(p, neighbor_key, A_indices, start, end, B_row_idx, itv);
      Index B_lb = B_indptr[B_row_idx];
      int B_col_ptr = B_lb + itv;
      Index B_col_idx = B_indices[B_col_ptr];
      Index sh_id = p + base;
      shr_tmp[sh_id] = B_col_idx; // B_col_idx * B_rows + B_row_idx (if HGNN)
    }
    __syncthreads();

    // ILP;hide memory lat
    if (k_idx + WARPSIZE < feature_size) {
      int st = base;
      for (; st < base + maxIdx - 1; st += 2) {
        Index buffer1 = shr_tmp[st];
        Index buffer2 = shr_tmp[st + 1];
        acc1 += in_feature[buffer1 * feature_size + k_idx];
        acc1 += in_feature[buffer2 * feature_size + k_idx];
        acc2 += in_feature[buffer1 * feature_size + k_idx + WARPSIZE];
        acc2 += in_feature[buffer2 * feature_size + k_idx + WARPSIZE];
      }

      if (maxIdx % 2 != 0) {
        acc1 += in_feature[shr_tmp[st] * feature_size + k_idx];
        acc2 += in_feature[shr_tmp[st] * feature_size + k_idx + WARPSIZE];
      }
      atomicAdd(out_feature + row * feature_size + k_idx, acc1); // feature
      atomicAdd(out_feature + row * feature_size + k_idx + WARPSIZE, acc2);
    }
  }
}

template <class Index, class DType, int _MDIMS_THREADS_PER_BLOCK,
          int _NDIMS_THREADS_PER_BLOCK>
__global__ void HyperGAggr_twostepbalan_kernel_template(
    int max_load, int seg_parts, int feature_size, Index *sn_key, Index *sn_row,
    Index *neighbor_key, Index *A_indices, Index *B_indptr, Index *B_indices,
    DType *in_feature, DType *out_feature) {
  const int k_idx = blockIdx.y * _NDIMS_THREADS_PER_BLOCK +
                    threadIdx.x % _NDIMS_THREADS_PER_BLOCK;
  int base = threadIdx.y * (2 * max_load - 1);
  const Index group =
      blockIdx.x * _MDIMS_THREADS_PER_BLOCK + threadIdx.y; // which node_group
  extern __shared__ Index shr_tmp[];                       // grouping
  if (group < seg_parts) {
    const Index row = sn_row[group];
    DType acc = 0;
    const Index start = __ldg(sn_key + group);
    const Index end = __ldg(sn_key + group + 1) - 1;
    if (end < start)
      return;
    const Index maxIdx = __ldg(neighbor_key + end);

    // cache two-step-neighbor in shared memory
    for (Index i = 0;; i += _NDIMS_THREADS_PER_BLOCK) {
      int p = i + threadIdx.x;
      if (p >= maxIdx)
        break;
      Index B_row_idx, itv;
      __find_row_entry(p, neighbor_key, A_indices, start, end, B_row_idx, itv);
      Index B_lb = B_indptr[B_row_idx];
      int B_col_ptr = B_lb + itv;
      Index B_col_idx = B_indices[B_col_ptr];
      Index sh_id = p + base;
      shr_tmp[sh_id] = B_col_idx; // B_col_idx * B_rows + B_row_idx (if HGNN)
    }
    __syncthreads();

    // ILP;hide memory lat
    if (k_idx < feature_size) {
      int st = base;
      for (; st < base + maxIdx - 1; st += 2) {
        Index buffer1 = shr_tmp[st];
        Index buffer2 = shr_tmp[st + 1];
        acc += in_feature[buffer1 * feature_size + k_idx];
        acc += in_feature[buffer2 * feature_size + k_idx];
      }

      if (maxIdx % 2 != 0)
        acc += in_feature[shr_tmp[st] * feature_size + k_idx];

      // for (int i = threadIdx.y; i < maxIdx; i += blockDim.y) {
      //   acc += in_feature[shr_tmp[i] * feature_size + k_idx];
      // }
      atomicAdd(out_feature + row * feature_size + k_idx, acc); // feature
    }
  }
}
// 64 (32) + (32) : group = 32
// a thread -> 4 feature, coarsen (128) group

template <class Index, class DType>
__global__ void HyperGAggr_twostepbalan_kernel_template_sf(
    int max_load, int seg_parts, int feature_size, Index *sn_key, Index *sn_row,
    Index *neighbor_key, Index *A_indices, Index *B_indptr, Index *B_indices,
    DType *in_feature, DType *out_feature) {
  const int k_idx = threadIdx.x;
  int base = threadIdx.y * (2 * max_load - 1);
  const Index group = blockIdx.x * blockDim.y + threadIdx.y; // which node_group
  extern __shared__ Index shr_tmp[];                         // grouping
  if (group < seg_parts) {
    const Index row = sn_row[group];
    DType acc = 0;
    const Index start = __ldg(sn_key + group);
    const Index end = __ldg(sn_key + group + 1) - 1;
    if (end < start)
      return;
    const Index maxIdx = __ldg(neighbor_key + end);

    for (Index i = 0;; i += blockDim.x) {
      int p = i + k_idx;
      if (p >= maxIdx)
        break;
      Index B_row_idx, itv;
      __find_row_entry(p, neighbor_key, A_indices, start, end, B_row_idx, itv);
      Index B_lb = B_indptr[B_row_idx];
      int B_col_ptr = B_lb + itv;
      Index B_col_idx = B_indices[B_col_ptr];
      Index sh_id = p + base;
      shr_tmp[sh_id] = B_col_idx;
    }
    __syncthreads();
    // ILP;hide memory lat
    int st = base;
    for (; st < base + maxIdx - 1; st += 2) {
      Index buffer1 = shr_tmp[st];
      Index buffer2 = shr_tmp[st + 1];
      acc += in_feature[buffer1 * feature_size + k_idx];
      acc += in_feature[buffer2 * feature_size + k_idx];
    }

    if (maxIdx % 2 != 0)
      acc += in_feature[shr_tmp[st] * feature_size + k_idx];

    // for (int i = threadIdx.y; i < maxIdx; i += blockDim.y) {
    //   acc += in_feature[shr_tmp[i] * feature_size + k_idx];
    // }

    atomicAdd(out_feature + row * feature_size + k_idx, acc); // feature
  }
}

template <class Index, class DType, int _MDIMS_THREADS_PER_BLOCK,
          int _NDIMS_THREADS_PER_BLOCK, int _NZ_A_PER_THREAD,
          int _NZ_B_PER_THREAD>
__global__ void HyperGAggr_part_edgegroup_kernel_template_v4(
    int edge_groups, int feature_size, Index *group_key, Index *group_row,
    Index *A_indices, Index *B_indptr, Index *B_indices, DType *in_feature,
    DType *out_feature) {
  const int threadIdx_x0 = threadIdx.x % _NDIMS_THREADS_PER_BLOCK;
  const int threadIdx_x1 = threadIdx.x / _NDIMS_THREADS_PER_BLOCK;
  const int k_idx = blockIdx.y * _NDIMS_THREADS_PER_BLOCK + threadIdx_x0;
  const Index group =
      blockIdx.x * _MDIMS_THREADS_PER_BLOCK + threadIdx_x1; // which node_group
  if (group < edge_groups) {
    const Index row = group_row[group];
    DType A_acc = 0;
    const Index start = __ldg(group_key + group);
    const Index end = __ldg(group_key + group + 1);
    for (Index i = start;; i += _NZ_A_PER_THREAD) {
      int p = i + threadIdx.z;
      if (p >= end)
        break;
      Index B_row_idx = __ldg(A_indices + p);
      Index B_lb = B_indptr[B_row_idx];
      Index B_hb = B_indptr[B_row_idx + 1];
      DType B_acc = 0;
      for (Index j = B_lb;; j = j + _NZ_B_PER_THREAD) {
        int B_col_ptr = j + threadIdx.y;
        if (B_col_ptr >= B_hb)
          break;
        Index B_col_idx = B_indices[B_col_ptr];
        // DType B_val = B_data[B_col_ptr];
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      A_acc += B_acc;
    }
    atomicAdd(out_feature + row * feature_size + k_idx, A_acc);
  }
}

template <class Index, class DType, int _NZ_A_PER_THREAD, int _NZ_B_PER_THREAD>
__global__ void HyperGAggr_part_edgegroup_kernel_template_v4_sf(
    int edge_groups, int feature_size, Index *group_key, Index *group_row,
    Index *A_indices, Index *B_indptr, Index *B_indices, DType *in_feature,
    DType *out_feature) {
  const int threadIdx_x0 = threadIdx.x % feature_size;
  const int threadIdx_x1 = threadIdx.x / feature_size;
  const int k_idx = blockIdx.y * feature_size + threadIdx_x0;
  const Index group = blockIdx.x + threadIdx_x1; // which node_group
  if (group < edge_groups) {
    const Index row = group_row[group];
    DType A_acc = 0;
    const Index start = __ldg(group_key + group);
    const Index end = __ldg(group_key + group + 1);
    for (Index i = start;; i += _NZ_A_PER_THREAD) {
      int p = i + threadIdx.z;
      if (p >= end)
        break;
      Index B_row_idx = __ldg(A_indices + p);
      Index B_lb = B_indptr[B_row_idx];
      Index B_hb = B_indptr[B_row_idx + 1];
      DType B_acc = 0;
      for (Index j = B_lb;; j = j + _NZ_B_PER_THREAD) {
        int B_col_ptr = j + threadIdx.y;
        if (B_col_ptr >= B_hb)
          break;
        Index B_col_idx = B_indices[B_col_ptr];
        // DType B_val = B_data[B_col_ptr];
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      A_acc += B_acc;
    }
    atomicAdd(out_feature + row * feature_size + k_idx, A_acc);
  }
}

template <class Index, class DType, int _MDIMS_THREADS_PER_BLOCK,
          int _NDIMS_THREADS_PER_BLOCK, int _NZ_A_PER_THREAD,
          int _NZ_B_PER_THREAD>
__global__ void HyperGAggr_part_edgegroup_degEVW_kernel_template_v4(
    int edge_groups, int feature_size, Index *group_key, Index *group_row,
    Index *A_indices, Index *B_indptr, Index *B_indices, DType *in_feature,
    DType *degE, DType *degV, DType *W, DType *out_feature) {
  const int threadIdx_x0 = threadIdx.x % _NDIMS_THREADS_PER_BLOCK;
  const int threadIdx_x1 = threadIdx.x / _NDIMS_THREADS_PER_BLOCK;
  const int k_idx = blockIdx.y * _NDIMS_THREADS_PER_BLOCK + threadIdx_x0;
  const Index group =
      blockIdx.x * _MDIMS_THREADS_PER_BLOCK + threadIdx_x1; // which node_group
  if (group < edge_groups) {
    const Index row = group_row[group];
    DType A_acc = 0;
    const Index start = __ldg(group_key + group);
    const Index end = __ldg(group_key + group + 1);
    for (Index i = start;; i += _NZ_A_PER_THREAD) {
      int p = i + threadIdx.z;
      if (p >= end)
        break;
      Index B_row_idx = __ldg(A_indices + p);
      Index B_lb = B_indptr[B_row_idx];
      Index B_hb = B_indptr[B_row_idx + 1];
      DType B_acc = 0;
      for (Index j = B_lb;; j = j + _NZ_B_PER_THREAD) {
        int B_col_ptr = j + threadIdx.y;
        if (B_col_ptr >= B_hb)
          break;
        Index B_col_idx = B_indices[B_col_ptr];
        // DType B_val = B_data[B_col_ptr];
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      A_acc += B_acc * degE[B_row_idx] * W[B_row_idx];
    }
    atomicAdd(out_feature + row * feature_size + k_idx, A_acc * degV[row]);
  }
}

template <class Index, class DType, int _NZ_A_PER_THREAD, int _NZ_B_PER_THREAD>
__global__ void HyperGAggr_part_edgegroup_degEVW_kernel_template_v4_sf(
    int edge_groups, int feature_size, Index *group_key, Index *group_row,
    Index *A_indices, Index *B_indptr, Index *B_indices, DType *in_feature,
    DType *degE, DType *degV, DType *W, DType *out_feature) {
  const int threadIdx_x0 = threadIdx.x % feature_size;
  const int threadIdx_x1 = threadIdx.x / feature_size;
  const int k_idx = blockIdx.y * feature_size + threadIdx_x0;
  const Index group = blockIdx.x + threadIdx_x1; // which node_group
  if (group < edge_groups) {
    const Index row = group_row[group];
    DType A_acc = 0;
    const Index start = __ldg(group_key + group);
    const Index end = __ldg(group_key + group + 1);
    for (Index i = start;; i += _NZ_A_PER_THREAD) {
      int p = i + threadIdx.z;
      if (p >= end)
        break;
      Index B_row_idx = __ldg(A_indices + p);
      Index B_lb = B_indptr[B_row_idx];
      Index B_hb = B_indptr[B_row_idx + 1];
      DType B_acc = 0;
      for (Index j = B_lb;; j = j + _NZ_B_PER_THREAD) {
        int B_col_ptr = j + threadIdx.y;
        if (B_col_ptr >= B_hb)
          break;
        Index B_col_idx = B_indices[B_col_ptr];
        // DType B_val = B_data[B_col_ptr];
        B_acc += in_feature[B_col_idx * feature_size + k_idx];
      }
      A_acc += B_acc * degE[B_row_idx] * W[B_row_idx];
    }
    atomicAdd(out_feature + row * feature_size + k_idx, A_acc * degV[row]);
  }
}

// vertex-based fused
// device hyaggr wrapper
template <class Index, class DType, hgnn_kernel_met km, balan_met bm,
          int MDIMS_PB, int NDIMS_PB, int NZ_A_THREAD, int NZ_B_THREAD>
void HyperGAggr_device(int feature_size, SpMatCsrDescr_t<Index, DType> &A,
                       SpMatCsrDescr_t<Index, DType> &B,
                       hgnn_balancer<Index, DType, bm> &balan,
                       util::RamArray<DType> &node_feature,
                       util::RamArray<DType> &out_feature) {
  if (km == hgnn_kernel_met::sn_group) {
    HyperGAggr_sngroup_kernel<Index, DType><<<balan.keys, feature_size>>>(
        balan.keys, feature_size, balan.balan_key.d_array.get(),
        balan.balan_row.d_array.get(), A.sp_csrind.d_array.get(),
        B.sp_csrptr.d_array.get(), B.sp_csrind.d_array.get(),
        node_feature.d_array.get(), out_feature.d_array.get());
  } else if (km == hgnn_kernel_met::sn_group_tune ||
             km == hgnn_kernel_met::edge_group_tune) {
    if (feature_size >= 32) {
      const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
      const int _MDIMS_BLOCKS = CEIL(balan.keys, MDIMS_PB);
      dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
      dim3 blockDim(MDIMS_PB * NDIMS_PB, NZ_B_THREAD, NZ_A_THREAD);
      HyperGAggr_sn_group_kernel_template<Index, DType, MDIMS_PB, NDIMS_PB,
                                          NZ_A_THREAD, NZ_B_THREAD>
          <<<gridDim, blockDim>>>(
              balan.keys, feature_size, balan.balan_key.d_array.get(),
              balan.balan_row.d_array.get(), A.sp_csrind.d_array.get(),
              B.sp_csrptr.d_array.get(), B.sp_csrind.d_array.get(),
              node_feature.d_array.get(), out_feature.d_array.get());
    } else {
      int _MDIMS_THREADS_PER_BLOCK = 1;
      int _NDIMS_THREADS_PER_BLOCK = feature_size;
      int _NZ_A_PER_THREAD = 4;
      int _NZ_B_PER_THREAD = 1;
      const int _NDIMS_BLOCKS = CEIL(feature_size, _NDIMS_THREADS_PER_BLOCK);
      const int _MDIMS_BLOCKS = CEIL(balan.keys, _MDIMS_THREADS_PER_BLOCK);
      dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
      dim3 blockDim(_MDIMS_THREADS_PER_BLOCK * _NDIMS_THREADS_PER_BLOCK,
                    _NZ_B_PER_THREAD, _NZ_A_PER_THREAD);
      HyperGAggr_sn_group_kernel_template_sf<Index, DType, 4, 1>
          <<<gridDim, blockDim>>>(
              balan.keys, feature_size, balan.balan_key.d_array.get(),
              balan.balan_row.d_array.get(), A.sp_csrind.d_array.get(),
              B.sp_csrptr.d_array.get(), B.sp_csrind.d_array.get(),
              node_feature.d_array.get(), out_feature.d_array.get());
    }
  } else if (km == hgnn_kernel_met::tsbalan_bl) {
    if (feature_size >= 32) {
      const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
      const int _MDIMS_BLOCKS = CEIL(balan.keys, MDIMS_PB);
      const int NZ_THREAD = NZ_A_THREAD;
      dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
      dim3 blockDim(MDIMS_PB * NDIMS_PB, NZ_THREAD, 1);
      HyperGAggr_twostepbalan_kernel_wo_schedule<Index, DType, MDIMS_PB,
                                                 NDIMS_PB, NZ_THREAD>
          <<<gridDim, blockDim>>>(
              balan.keys, feature_size, balan.balan_key.d_array.get(),
              balan.balan_row.d_array.get(), balan.neighbor_key.d_array.get(),
              A.sp_csrind.d_array.get(), B.sp_csrptr.d_array.get(),
              B.sp_csrind.d_array.get(), node_feature.d_array.get(),
              out_feature.d_array.get());
    } else {
      int _MDIMS_THREADS_PER_BLOCK = 1;
      int _NDIMS_THREADS_PER_BLOCK = feature_size;
      const int _NDIMS_BLOCKS = CEIL(feature_size, _NDIMS_THREADS_PER_BLOCK);
      const int _MDIMS_BLOCKS = CEIL(balan.keys, _MDIMS_THREADS_PER_BLOCK);
      dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
      dim3 blockDim(_MDIMS_THREADS_PER_BLOCK * _NDIMS_THREADS_PER_BLOCK, 1, 1);
      HyperGAggr_twostepbalan_kernel_wo_schedul_sf<Index, DType, 1>
          <<<gridDim, blockDim>>>(
              balan.keys, feature_size, balan.balan_key.d_array.get(),
              balan.balan_row.d_array.get(), balan.neighbor_key.d_array.get(),
              A.sp_csrind.d_array.get(), B.sp_csrptr.d_array.get(),
              B.sp_csrind.d_array.get(), node_feature.d_array.get(),
              out_feature.d_array.get());
    }
  } else if (km == hgnn_kernel_met::tsbalan) {
    if (feature_size > 64) {
      const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB * 2);
      const int _MDIMS_BLOCKS = CEIL(balan.keys, MDIMS_PB);
      dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
      dim3 blockDim(NDIMS_PB, MDIMS_PB, 1);
      HyperGAggr_twostepbalan_kernel_template_lf<Index, DType, MDIMS_PB,
                                                 NDIMS_PB>
          <<<gridDim, blockDim,
             2 * MDIMS_PB * balan.max_load_per * sizeof(Index)>>>(
              balan.max_load_per, balan.keys, feature_size,
              balan.balan_key.d_array.get(), balan.balan_row.d_array.get(),
              balan.neighbor_key.d_array.get(), A.sp_csrind.d_array.get(),
              B.sp_csrptr.d_array.get(), B.sp_csrind.d_array.get(),
              node_feature.d_array.get(), out_feature.d_array.get());
    } else if (feature_size >= 32 && feature_size <= 64) {
      const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
      const int _MDIMS_BLOCKS = CEIL(balan.keys, MDIMS_PB);
      dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
      dim3 blockDim(NDIMS_PB, MDIMS_PB, 1);
      HyperGAggr_twostepbalan_kernel_template<Index, DType, MDIMS_PB, NDIMS_PB>
          <<<gridDim, blockDim,
             2 * MDIMS_PB * balan.max_load_per * sizeof(Index)>>>(
              balan.max_load_per, balan.keys, feature_size,
              balan.balan_key.d_array.get(), balan.balan_row.d_array.get(),
              balan.neighbor_key.d_array.get(), A.sp_csrind.d_array.get(),
              B.sp_csrptr.d_array.get(), B.sp_csrind.d_array.get(),
              node_feature.d_array.get(), out_feature.d_array.get());
    } else {
      int _MDIMS_THREADS_PER_BLOCK = 1;
      int _NDIMS_THREADS_PER_BLOCK = feature_size;
      const int _MDIMS_BLOCKS = CEIL(balan.keys, _MDIMS_THREADS_PER_BLOCK);
      dim3 gridDim(_MDIMS_BLOCKS, 1, 1);
      dim3 blockDim(_NDIMS_THREADS_PER_BLOCK, MDIMS_PB, 1);
      HyperGAggr_twostepbalan_kernel_template_sf<Index, DType>
          <<<gridDim, blockDim,
             2 * MDIMS_PB * balan.max_load_per * sizeof(Index)>>>(
              balan.max_load_per, balan.keys, feature_size,
              balan.balan_key.d_array.get(), balan.balan_row.d_array.get(),
              balan.neighbor_key.d_array.get(), A.sp_csrind.d_array.get(),
              B.sp_csrptr.d_array.get(), B.sp_csrind.d_array.get(),
              node_feature.d_array.get(), out_feature.d_array.get());
    }
  } else if (km == hgnn_kernel_met::edge_based_balance) {
    const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
    const int _MDIMS_BLOCKS = CEIL(balan.keys, MDIMS_PB);
    dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
    dim3 blockDim(MIN(NDIMS_PB, feature_size), MDIMS_PB, 1);
    HyperGAggr_Edgefused_Balance_Kernel<Index, DType, MDIMS_PB, NDIMS_PB>
        <<<gridDim, blockDim>>>(
            B.nrow, balan.keys, feature_size, balan.balan_key.d_array.get(),
            balan.work_ind.d_array.get(), balan.work_ts_ind.d_array.get(),
            B.sp_csrind.d_array.get(), node_feature.d_array.get(),
            out_feature.d_array.get());
  } else if (km == hgnn_kernel_met::edge_based_parallel) {
    const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
    const int _MDIMS_BLOCKS = CEIL(balan.part_keys, MDIMS_PB);
    dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
    dim3 blockDim(MIN(NDIMS_PB, feature_size), MDIMS_PB, 1);
    HyperGAggr_Edgefused_Parallel_Kernel<Index, DType, MDIMS_PB, NDIMS_PB>
        <<<gridDim, blockDim>>>(
            B.nrow, balan.part_keys, feature_size, balan.work_ind.d_array.get(),
            balan.balan_key.d_array.get(), B.sp_csrptr.d_array.get(),
            B.sp_csrind.d_array.get(), node_feature.d_array.get(),
            out_feature.d_array.get());
  } else if (km == hgnn_kernel_met::edge_based_merge) {
    const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
    const int _MDIMS_BLOCKS = CEIL(balan.keys, MDIMS_PB);
    dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
    dim3 blockDim(MIN(NDIMS_PB, feature_size), MDIMS_PB, 1);
    HyperGAggr_Edgefused_Merge_Kernel<Index, DType, MDIMS_PB, NDIMS_PB>
        <<<gridDim, blockDim>>>(
            B.nrow, feature_size, balan.balan_key.d_array.get(),
            balan.balan_row.d_array.get(), balan.merge_col.d_array.get(),
            node_feature.d_array.get(), out_feature.d_array.get());
  } else if (km == hgnn_kernel_met::edge_based_full) {
    const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
    const int _MDIMS_BLOCKS = CEIL(balan.keys, MDIMS_PB);
    dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
    if (feature_size >= 32) {
      dim3 blockDim(MIN(NDIMS_PB, feature_size), MDIMS_PB, 1);
      HyperGAggr_Edgefused_Balance_Full_Kernel<Index, DType, MDIMS_PB, NDIMS_PB>
          <<<gridDim, blockDim>>>(
              B.nrow, balan.keys, feature_size, balan.balan_key.d_array.get(),
              balan.group_start.d_array.get(), balan.group_end.d_array.get(),
              B.sp_csrind.d_array.get(), node_feature.d_array.get(),
              out_feature.d_array.get());
    } else {
      dim3 blockDim(NDIMS_PB, MDIMS_PB, 1);
      HyperGAggr_Edgefused_Balance_Full_Kernel_sf<Index, DType, MDIMS_PB,
                                                  NDIMS_PB>
          <<<gridDim, blockDim>>>(
              B.nrow, balan.keys, feature_size, balan.balan_key.d_array.get(),
              balan.group_start.d_array.get(), balan.group_end.d_array.get(),
              B.sp_csrind.d_array.get(), node_feature.d_array.get(),
              out_feature.d_array.get());
    }
  } else if (km == hgnn_kernel_met::edge_based_shm) {
    const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
    const int _MDIMS_BLOCKS = CEIL(balan.keys, MDIMS_PB);
    dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
    dim3 blockDim(MIN(NDIMS_PB, feature_size), MDIMS_PB, 1);
    HyperGAggr_Edgefused_Balance_Shm_Kernel<Index, DType, MDIMS_PB, NDIMS_PB>
        <<<gridDim, blockDim, MDIMS_PB * feature_size * sizeof(DType)>>>(
            B.nrow, balan.max_load_per, balan.keys, feature_size,
            balan.balan_key.d_array.get(), balan.group_start.d_array.get(),
            balan.group_end.d_array.get(), B.sp_csrind.d_array.get(),
            node_feature.d_array.get(), out_feature.d_array.get());
  } else
    return;
}

// edge-based fused simple
template <class Index, class DType, hgnn_kernel_met km, int MDIMS_PB,
          int NDIMS_PB>
void HyperGAggr_device(int feature_size, SpMatCsrDescr_t<Index, DType> &A,
                       SpMatCsrDescr_t<Index, DType> &B,
                       util::RamArray<DType> &node_feature,
                       util::RamArray<DType> &out_feature) {
  const int _NDIMS_BLOCKS = CEIL(feature_size, NDIMS_PB);
  const int _MDIMS_BLOCKS = CEIL(B.nrow, MDIMS_PB);
  dim3 gridDim(_MDIMS_BLOCKS, _NDIMS_BLOCKS, 1);
  dim3 blockDim(MIN(NDIMS_PB, feature_size), MDIMS_PB, 1);
  HyperGAggr_Edgefused_Kernel<Index, DType, MDIMS_PB, NDIMS_PB>
      <<<gridDim, blockDim>>>(B.nrow, feature_size, B.sp_csrptr.d_array.get(),
                              B.sp_csrind.d_array.get(),
                              node_feature.d_array.get(),
                              out_feature.d_array.get());
}

// host ref
template <class Index, class DType>
void HyperGAggr_host(int feature_size, SpMatCsrDescr_t<Index, DType> &A,
                     SpMatCsrDescr_t<Index, DType> &B,
                     util::RamArray<DType> &node_feature,
                     util::RamArray<DType> &out_feature) {
  util::hyperaggr_reference_host(
      A.nrow, feature_size, A.sp_csrptr.h_array.get(),
      A.sp_csrind.h_array.get(), B.sp_csrptr.h_array.get(),
      B.sp_csrind.h_array.get(), node_feature.h_array.get(),
      out_feature.h_array.get());
}

// check device based on ref
template <class Index, class DType, hgnn_kernel_met km, int MDIMS_PB,
          int NDIMS_PB>
bool HyperGAggr_check(int feature_size, SpMatCsrDescr_t<Index, DType> &A,
                      SpMatCsrDescr_t<Index, DType> &B,
                      util::RamArray<DType> &node_feature,
                      util::RamArray<DType> &out_feature,
                      util::RamArray<DType> &out_ref) {
  out_feature.reset();
  out_ref.reset();
  HyperGAggr_host<Index, DType>(feature_size, A, B, node_feature, out_ref);
  HyperGAggr_device<Index, DType, km, MDIMS_PB, NDIMS_PB>(
      feature_size, A, B, node_feature, out_feature);
  out_feature.download();
  bool pass = util::check_result(
      A.nrow, feature_size, out_feature.h_array.get(), out_ref.h_array.get());
  if (pass) {
    printf("check passed!\n");
  }
  return pass;
}

// test speed
template <class Index, class DType, hgnn_kernel_met km, balan_met bm,
          int MDIMS_PB, int NDIMS_PB, int NZ_A_THREAD, int NZ_B_THREAD>
float HyperGAggr_test(int iter, int feature_size,
                      SpMatCsrDescr_t<Index, DType> &A,
                      SpMatCsrDescr_t<Index, DType> &B,
                      hgnn_balancer<Index, DType, bm> &balan,
                      util::RamArray<DType> &node_feature,
                      util::RamArray<DType> &out_feature) {
  util::gpuTimer gt;
  gt.start();
  for (int i = 0; i < iter; i++)
    HyperGAggr_device<Index, DType, km, bm, MDIMS_PB, NDIMS_PB, NZ_A_THREAD,
                      NZ_B_THREAD>(feature_size, A, B, balan, node_feature,
                                   out_feature);
  gt.end();
  float time = gt.elapsed() / iter;
  return time;
}

template <class Index, class DType, hgnn_kernel_met km, balan_met bm,
          int MDIMS_PB, int NDIMS_PB, int NZ_A_THREAD, int NZ_B_THREAD>
bool HyperGAggr_check(int feature_size, SpMatCsrDescr_t<Index, DType> &A,
                      SpMatCsrDescr_t<Index, DType> &B,
                      hgnn_balancer<Index, DType, bm> &balan,
                      util::RamArray<DType> &node_feature,
                      util::RamArray<DType> &out_feature,
                      util::RamArray<DType> &out_ref) {
  out_feature.reset();
  out_ref.reset();
  HyperGAggr_host<Index, DType>(feature_size, A, B, node_feature, out_ref);
  HyperGAggr_device<Index, DType, km, bm, MDIMS_PB, NDIMS_PB, NZ_A_THREAD,
                    NZ_B_THREAD>(feature_size, A, B, balan, node_feature,
                                 out_feature);
  out_feature.download();
  bool pass = util::check_result(
      A.nrow, feature_size, out_feature.h_array.get(), out_ref.h_array.get());
  if (!pass) {
    printf("check failed!\n");
  }
  return pass;
}

// test speed
template <class Index, class DType, hgnn_kernel_met km, int MDIMS_PB,
          int NDIMS_PB>
float HyperGAggr_test(std::fstream &fs, int iter, int feature_size,
                      SpMatCsrDescr_t<Index, DType> &A,
                      SpMatCsrDescr_t<Index, DType> &B,
                      util::RamArray<DType> &node_feature,
                      util::RamArray<DType> &out_feature) {
  util::gpuTimer gt;
  gt.start();
  for (int i = 0; i < iter; i++)
    HyperGAggr_device<Index, DType, km, MDIMS_PB, NDIMS_PB>(
        feature_size, A, B, node_feature, out_feature);
  gt.end();
  float time = gt.elapsed() / iter;
  printf("test time one baseline fused kernel: %.4f ms\n", time);
  fs << time << ",";
  return time;
}

// select kernel based on data attribute
template <class Index, class DType>
float HyperGAggr(int iter, int feature_size, int partition,
                 SpMatCsrDescr_t<Index, DType> &A,
                 SpMatCsrDescr_t<Index, DType> &B,
                 util::RamArray<DType> &node_feature,
                 util::RamArray<DType> &out_feature) {
  hgnn_balancer<Index, DType, balan_met::hgnn_ef_full> balan(partition, A, B);
  float time = 0;
  if (A.nnz > 10 * A.ncol) {
    time = HyperGAggr_test<Index, DType, hgnn_kernel_met::edge_based_shm,
                           balan_met::hgnn_ef_full, 2, 32, 1, 1>(
        iter, feature_size, A, B, balan, node_feature, out_feature);
  } else {
    time = HyperGAggr_test<Index, DType, hgnn_kernel_met::edge_based_full,
                           balan_met::hgnn_ef_full, 2, 32, 1, 1>(
        iter, feature_size, A, B, balan, node_feature, out_feature);
  }
  return time;
}

#define TRY(_KERNEL, X, Y, Z, U)                                               \
  passed = HyperGAggr_check<Index, DType, km, bm, X, Y, Z, U>(                 \
      feature_size, H, H_t, balan, node_feature, out_feature, out_ref);        \
  if (passed) {                                                                \
    tm_t = _KERNEL<Index, DType, km, bm, X, Y, Z, U>(                          \
        ITER, feature_size, H, H_t, balan, node_feature, out_feature);         \
  }                                                                            \
  if (tm_t < min) {                                                            \
    min = tm_t;                                                                \
    min_p = p;                                                                 \
  }

template <class Index, class DType, hgnn_kernel_met km, balan_met bm>
float HyperGAggr_tune(std::fstream &fs, int part[], int feature_size,
                      SpMatCsrDescr_t<Index, DType> &H,
                      SpMatCsrDescr_t<Index, DType> &H_t,
                      util::RamArray<DType> &node_feature,
                      util::RamArray<DType> &out_feature,
                      util::RamArray<DType> &out_ref) {
  int count = 0, p = part[count];
  float min = 1e9, tm_t = 0;
  int min_p = 0;
  while (p > 0) {
    hgnn_balancer<Index, DType, bm> balan(p, H, H_t);
    bool passed = 0;
    TRY(HyperGAggr_test, 2, 32, 1, 1);
    p = part[++count];
  }
  if (bm == balan_met::hgnn_eg) {
    if (km == hgnn_kernel_met::edge_group_part)
      printf("test eg time one hyperaggr: %.4f ms\n", min);
    else if (km == hgnn_kernel_met::edge_group_tune)
      printf("test eg tune time one hyperaggr: %.4f ms\n", min);
  }
  if (bm == balan_met::hgnn_sn) {
    if (km == hgnn_kernel_met::sn_group)
      printf("test sns time one hyperaggr: %.4f ms\n", min);
    else if (km == hgnn_kernel_met::sn_group_tune)
      printf("test sns tune time one hyperaggr: %.4f ms\n", min);
    else if (km == hgnn_kernel_met::tsbalan)
      printf("test ts tune time one hyperaggr: %.4f ms\n", min);
  }
  if (bm == balan_met::hgnn_ef_full) {
    if (km == hgnn_kernel_met::edge_based_full)
      printf("test ef full tune time one %.4f ms\n", min);
    else if (km == hgnn_kernel_met::edge_based_shm)
      printf("test ef shm tune time one %.4f ms\n", min);
  }
  fs << min << "," << min_p << ",";
  return min;
}
#endif