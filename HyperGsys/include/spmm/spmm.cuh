#ifndef SPMM_
#define SPMM_

#include "../dataloader/dataloader.hpp"
#include "../taskbalancer/balancer.cuh"
#include "../util/check.cuh"
#include "../util/gpuTimer.cuh"
#include "../util/ramArray.cuh"
#include <cuda.h>
#include <cusparse.h>
#include <fstream>
#include <string>

enum spmm_kernel_met {
  cusparse,
  row_balance,
  edge_balance,
  edge_group,
  hybrid
};

template <typename Index, typename DType>
void csrspmm_cusparse(const int nrow, const int ncol, const int nnz,
                      const int feature_size, int *sp_csrptr, int *sp_csrind,
                      DType *sp_data, DType *in_feature, DType *out_feature) {
  //
  // Run Cusparse-SpMM and check result
  //
  cusparseHandle_t handle;
  cusparseSpMatDescr_t csrDescr;
  cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
  float alpha = 1.0f, beta = 0.0f;

  checkCuSparseError(cusparseCreate(&handle));

  // creating sparse csr matrix
  checkCuSparseError(cusparseCreateCsr(
      &csrDescr, nrow, ncol, nnz, sp_csrptr, sp_csrind, sp_data,
      CUSPARSE_INDEX_32I, // index 32-integer for indptr
      CUSPARSE_INDEX_32I, // index 32-integer for indices
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F // datatype: 32-bit float real number
      ));

  // creating dense matrices
  checkCuSparseError(cusparseCreateDnMat(&dnMatInputDescr, ncol, feature_size,
                                         feature_size, in_feature, CUDA_R_32F,
                                         CUSPARSE_ORDER_ROW));
  checkCuSparseError(cusparseCreateDnMat(&dnMatOutputDescr, nrow, feature_size,
                                         feature_size, out_feature, CUDA_R_32F,
                                         CUSPARSE_ORDER_ROW));

  // allocate workspace buffer
  size_t workspace_size;
  checkCuSparseError(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
      &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
      &workspace_size));

  void *workspace = NULL;
  checkCudaError(cudaMalloc(&workspace, workspace_size));

  // run SpMM
  checkCuSparseError(cusparseSpMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                  &alpha, csrDescr, dnMatInputDescr, &beta,
                                  dnMatOutputDescr, CUDA_R_32F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, workspace));
  checkCuSparseError(cusparseDestroy(handle));
  checkCudaError(cudaFree(workspace));
  checkCuSparseError(cusparseDestroySpMat(csrDescr));

  checkCuSparseError(cusparseDestroyDnMat(dnMatInputDescr));
  checkCuSparseError(cusparseDestroyDnMat(dnMatOutputDescr));
}

template <typename Index, typename DType>
float csrspmm_cusparse_test(int iter, SpMatCsrDescr_t<Index, DType> &spmatA,
                            const Index feature_size, DType *in_feature,
                            DType *out_feature) {
  //
  // Run Cusparse-SpMM and check result
  //
  cusparseHandle_t handle;
  cusparseSpMatDescr_t csrDescr;
  cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
  float alpha = 1.0f, beta = 0.0f;

  checkCuSparseError(cusparseCreate(&handle));

  // creating sparse csr matrix
  checkCuSparseError(cusparseCreateCsr(
      &csrDescr, spmatA.nrow, spmatA.ncol, spmatA.nnz,
      spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
      spmatA.sp_data.d_array.get(),
      CUSPARSE_INDEX_32I, // index 32-integer for indptr
      CUSPARSE_INDEX_32I, // index 32-integer for indices
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F // datatype: 32-bit float real number
      ));

  // creating dense matrices
  checkCuSparseError(cusparseCreateDnMat(&dnMatInputDescr, spmatA.ncol,
                                         feature_size, feature_size, in_feature,
                                         CUDA_R_32F, CUSPARSE_ORDER_ROW));
  checkCuSparseError(cusparseCreateDnMat(
      &dnMatOutputDescr, spmatA.nrow, feature_size, feature_size, out_feature,
      CUDA_R_32F, CUSPARSE_ORDER_ROW));

  // allocate workspace buffer
  size_t workspace_size;
  checkCuSparseError(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
      &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
      &workspace_size));

  void *workspace = NULL;
  checkCudaError(cudaMalloc(&workspace, workspace_size));

  // run SpMM
  util::gpuTimer atimer;
  atimer.start();
  for (int i = 0; i < iter; i++)
    checkCuSparseError(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                    &alpha, csrDescr, dnMatInputDescr, &beta,
                                    dnMatOutputDescr, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, workspace));
  atimer.end();
  checkCudaError(cudaFree(workspace));

  checkCuSparseError(cusparseDestroy(handle));
  checkCuSparseError(cusparseDestroySpMat(csrDescr));

  checkCuSparseError(cusparseDestroyDnMat(dnMatInputDescr));
  checkCuSparseError(cusparseDestroyDnMat(dnMatOutputDescr));

  return atimer.elapsed();
}

template <typename Index, typename DType>
__global__ void
csrspmm_rowbalance_kernel(const Index nr, const Index feature_size,
                          const Index rowPtr[], const Index colIdx[],
                          const DType values[], const DType dnInput[],
                          DType dnOutput[]) {
  Index row_tile = blockDim.y; // 8
  Index subwarp_id = threadIdx.y;
  Index stride = row_tile * gridDim.x; // 8 * (m/8)
  Index row = blockIdx.x * row_tile + subwarp_id;
  Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  dnInput += v_id;
  dnOutput += v_id;
  DType res = 0, val;
  Index col;
  for (; row < nr; row += stride) {
    Index start = __ldg(rowPtr + row);
    Index end = __ldg(rowPtr + row + 1);
    for (Index p = start; p < end; p++) {
      col = __ldg(colIdx + p);
      val = util::__guard_load_default_one<DType>(values, p);
      res += val * __ldg(dnInput + col * feature_size);
    }
    dnOutput[row * feature_size] = res;
  }
}

template <int CoarsenFactor, int ThreadNz>
__global__ void
csrspmm_edgebalance_kernel(const int M, const int N, const int K,
                           const int nnz_, const int csr_indptr[],
                           const int csr_indices[], const float csr_data[],
                           const float B[], float C[]) {
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  extern __shared__ int shared_mem[];
  int *workspace_rowid = &shared_mem[(warp_id << 5)];
  int *workspace_colid = workspace_rowid + blockDim.x;
  float *workspace_data =
      (float *)(workspace_colid +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int global_warp_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
  int nz_start = global_warp_id * (ThreadNz * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
  float *C_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * 32;
    C_lanes[i] = C + col_offset + lane_id + i * 32;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  int stride = gridDim.x * (blockDim.x >> 5) * ThreadNz * 32;

  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  for (; nz_start < nnz; nz_start += stride) {
    // iterate over the segment of this warp
    for (int tile_base = nz_start;
         tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {

      int thread_nz_id = tile_base + lane_id;
      if (thread_nz_id < nnz) {
        workspace_colid[lane_id] = csr_indices[thread_nz_id];
        workspace_data[lane_id] =
            util::__guard_load_default_one<float>(csr_data, thread_nz_id);
      } else {
        workspace_colid[lane_id] = 0;
        workspace_data[lane_id] = 0.0f;
      }
      workspace_rowid[lane_id] =
          binary_search<int>(csr_indptr, thread_nz_id, 0, M);
      __syncwarp();

      // initialize with first value
      int k = workspace_colid[0];
      float v = workspace_data[0];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = v * B_lanes[i][k * ldB];
      }
      int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
      for (int pp = 1; pp < 32; pp++) {
        next_row = workspace_rowid[pp];
        if (next_row != row_curr) {
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
          }
          row_curr = next_row;
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = v * B_lanes[i][k * ldB];
          }
        } else {
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = c[i] + v * B_lanes[i][k * ldB];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
      }
    }
  }
  return;

Ndim_Residue:

  int valid_lane_num = CEIL(N - col_offset - lane_id, 32);

  for (; nz_start < nnz; nz_start += stride) {
    // iterate over the segment of this warp
    for (int tile_base = nz_start;
         tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {

      int thread_nz_id = tile_base + lane_id;
      if (thread_nz_id < nnz) {
        workspace_colid[lane_id] = csr_indices[thread_nz_id];
        workspace_data[lane_id] =
            util::__guard_load_default_one<float>(csr_data, thread_nz_id);
      } else {
        workspace_colid[lane_id] = 0;
        workspace_data[lane_id] = 0.0f;
      }
      workspace_rowid[lane_id] =
          binary_search<int>(csr_indptr, thread_nz_id, 0, M);
      __syncwarp();

      // initialize with first value
      int k = workspace_colid[0];
      float v = workspace_data[0];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] = v * B_lanes[i][k * ldB];
        }
      }
      int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
      for (int pp = 1; pp < 32; pp++) {
        next_row = workspace_rowid[pp];
        if (next_row != row_curr) {
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
            }
          }
          row_curr = next_row;
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              c[i] = v * B_lanes[i][k * ldB];
            }
          }
        } else {
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              c[i] = c[i] + v * B_lanes[i][k * ldB];
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
        }
      }
    }
  }
}

template <typename Index, typename DType>
void csrspmm_edgebalance(SpMatCsrDescr_t<Index, DType> &spmatA,
                         int feature_size, const float *B, float *C) {
  int coarsen_factor = (feature_size >= 512)   ? 4
                       : (feature_size >= 128) ? 2
                                               : 1;
  int Ndim_threadblock = CEIL(feature_size, (32 * coarsen_factor));

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  Index ref_block = (feature_size > 256) ? feature_size : 256;
  int Nnzdim_warp_per_tb = ref_block / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      spmatA.nrow,
      Nnzdim_warp_per_tb *
          thread_nz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 * thread_nz );

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  // simple heuristic
  if (coarsen_factor == 4) {
    if (thread_nz == 1)
      csrspmm_edgebalance_kernel<4, 1><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 2)
      csrspmm_edgebalance_kernel<4, 2><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 4)
      csrspmm_edgebalance_kernel<4, 4><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      csrspmm_edgebalance_kernel<2, 1><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 2)
      csrspmm_edgebalance_kernel<2, 2><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 4)
      csrspmm_edgebalance_kernel<2, 4><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
  } else {
    if (thread_nz == 1)
      csrspmm_edgebalance_kernel<1, 1><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 2)
      csrspmm_edgebalance_kernel<1, 2><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 4)
      csrspmm_edgebalance_kernel<1, 4><<<gridDim, blockDim, smem_size>>>(
          spmatA.nrow, feature_size, spmatA.ncol, spmatA.nnz,
          spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
          spmatA.sp_data.d_array.get(), B, C);
  }
}

template <typename Index, typename DType>
__global__ void
csrspmm_rowbalance_degV_kernel(const Index nr, const Index feature_size,
                               const Index rowPtr[], const Index colIdx[],
                               const DType values[], const DType dnInput[],
                               DType dnOutput[], const DType degV[]) {
  Index row_tile = blockDim.y; // 8
  Index subwarp_id = threadIdx.y;
  Index stride = row_tile * gridDim.x; // 8 * (m/8)
  Index row = blockIdx.x * row_tile + subwarp_id;
  Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  dnInput += v_id;
  dnOutput += v_id;
  DType res = 0, val;
  Index col;
  for (; row < nr; row += stride) {
    Index start = __ldg(rowPtr + row);
    Index end = __ldg(rowPtr + row + 1);
    for (Index p = start; p < end; p++) {
      col = __ldg(colIdx + p);
      val = util::__guard_load_default_one<DType>(values, p);
      res += val * __ldg(dnInput + col * feature_size);
    }
    dnOutput[row * feature_size] = res * degV[row];
  }
}

template <typename Index, typename DType>
__global__ void
csrspmm_edgegroup_kernel(const Index edge_groups, const Index feature_size,
                         const Index group_key[], const Index group_row[],
                         const Index colIdx[], const DType values[],
                         const DType dnInput[], DType dnOutput[]) {
  Index group_tile = blockDim.y; // combine a set of groups together
  Index subwarp_id = threadIdx.y;
  Index group = blockIdx.x * group_tile + subwarp_id; // which node_group
  Index v_id = threadIdx.x;
  if (group < edge_groups - 1) {
    Index row = group_row[group]; // get the specific row of each node group
    dnInput += v_id;
    dnOutput += v_id;
    DType res = 0, val;
    Index col;
    Index start = __ldg(group_key + group);
    Index end = __ldg(group_key + group + 1);
    for (Index p = start; p < end; p++) {
      col = __ldg(colIdx + p);
      val = util::__guard_load_default_one<DType>(values, p);
      res += val * __ldg(dnInput + col * feature_size);
    }
    atomicAdd(dnOutput + row * feature_size,
              res); // atomic, cuz different node group -> same row
  }
}

template <typename Index, typename DType>
__global__ void csrspmm_hybrid_kernel(
    const Index keys, const Index feature_size, const Index key_ptr[],
    const Index group_key[], const Index group_row[], const Index colIdx[],
    const DType values[], const DType dnInput[], DType dnOutput[]) {
  Index key_tile = blockDim.y; // combine a set of groups together
  Index subwarp_id = threadIdx.y * 2;
  Index key = blockIdx.x * key_tile * 2 + subwarp_id; // which node_group
  Index v_id = threadIdx.x;

  if (key < keys - 1) {
    Index gptr_lb = key_ptr[key];
    Index gptr_hb = key_ptr[key + 2];
    dnInput += v_id;
    dnOutput += v_id;
#pragma unroll
    for (Index group = gptr_lb; group < gptr_hb; group++) {
      Index row = group_row[group]; // get the specific row of each node group
      DType res = 0, val;
      Index col;
      Index start = __ldg(group_key + group);
      Index end = __ldg(group_key + group + 1);
      for (Index p = start; p < end; p++) {
        col = __ldg(colIdx + p);
        val = util::__guard_load_default_one<DType>(values, p);
        res += val * __ldg(dnInput + col * feature_size);
      }
      atomicAdd(dnOutput + row * feature_size,
                res); // atomic, cuz different node group -> same row
    }
  } else if (key == keys - 1) {
    Index gptr_lb = key_ptr[key];
    Index gptr_hb = key_ptr[key + 1];
    dnInput += v_id;
    dnOutput += v_id;
#pragma unroll
    for (Index group = gptr_lb; group < gptr_hb; group++) {
      Index row = group_row[group]; // get the specific row of each node group
      DType res = 0, val;
      Index col;
      Index start = __ldg(group_key + group);
      Index end = __ldg(group_key + group + 1);
      for (Index p = start; p < end; p++) {
        col = __ldg(colIdx + p);
        val = util::__guard_load_default_one<DType>(values, p);
        res += val * __ldg(dnInput + col * feature_size);
      }
      atomicAdd(dnOutput + row * feature_size,
                res); // atomic, cuz different node group -> same row
    }
  }
}

template <typename Index, typename DType>
void csrspmm_rowbalance(SpMatCsrDescr_t<Index, DType> &spmatA,
                        const Index feature_size, const DType *in_feature,
                        DType *out_feature) {
  Index Mdim_worker = spmatA.nrow;
  Index Ndim_worker = feature_size;
  Index ref_block = (feature_size > 256) ? feature_size : 256;
  Index Ndim_threadblock = CEIL(Ndim_worker, ref_block);
  Index Ndim_thread_per_tb = min(Ndim_worker, ref_block);
  Index Mdim_thread_per_tb = CEIL(ref_block, Ndim_thread_per_tb);
  Index Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  csrspmm_rowbalance_kernel<<<gridDim, blockDim>>>(
      spmatA.nrow, feature_size, spmatA.sp_csrptr.d_array.get(),
      spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), in_feature,
      out_feature);
}

template <typename Index, typename DType>
void csrspmm_nodemerge(SpMatCsrDescr_t<Index, DType> &spmatA,
                       const Index feature_size, const Index key_len,
                       const Index *key, const DType *in_feature,
                       DType *out_feature) {
  Index Mdim_worker = key_len;
  Index Ndim_worker = feature_size;
  // Index Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  Index Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
  Index Mdim_thread_per_tb = 8;

  dim3 gridDim(Mdim_worker, 1, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  csrspmm_nodemerge_kernel<<<gridDim, blockDim>>>(
      spmatA.nrow, feature_size, key, spmatA.sp_csrptr.d_array.get(),
      spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), in_feature,
      out_feature);
}

template <typename Index, typename DType>
void csrspmm_edgegroup(SpMatCsrDescr_t<Index, DType> &spmatA,
                       const Index feature_size, const Index edge_groups,
                       const Index *group_key, const Index *group_row,
                       const DType *in_feature, DType *out_feature) {
  Index Mdim_worker = edge_groups;
  Index Ndim_worker = feature_size;

  Index ref_block = (feature_size > 256) ? feature_size : 256;
  Index Ndim_threadblock = CEIL(Ndim_worker, ref_block);
  Index Ndim_thread_per_tb = min(Ndim_worker, ref_block);
  Index Mdim_thread_per_tb = CEIL(ref_block, Ndim_thread_per_tb);
  Index Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);
  // size_t shr_size = feature_size * Mdim_thread_per_tb * sizeof(DType);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  csrspmm_edgegroup_kernel<<<gridDim, blockDim>>>(
      edge_groups, feature_size, group_key, group_row,
      spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), in_feature,
      out_feature);
}

template <typename Index, typename DType>
void csrspmm_hybrid(SpMatCsrDescr_t<Index, DType> &spmatA, const int keys,
                    const Index feature_size, const Index *key_ptr,
                    const Index *group_key, const Index *group_row,
                    const DType *in_feature, DType *out_feature) {
  Index Mdim_worker = keys - 1;
  Index Ndim_worker = feature_size;
  Index ref_block = (feature_size > 256) ? feature_size : 256;
  Index Ndim_threadblock = CEIL(Ndim_worker, ref_block);
  Index Ndim_thread_per_tb = min(Ndim_worker, ref_block);
  Index Mdim_thread_per_tb = CEIL(ref_block, Ndim_thread_per_tb) * 2;
  Index Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, CEIL(Mdim_thread_per_tb, 2), 1);

  csrspmm_hybrid_kernel<<<gridDim, blockDim>>>(
      keys - 1, feature_size, key_ptr, group_key, group_row,
      spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), in_feature,
      out_feature);
}

// test speed
template <class Index, class DType, spmm_kernel_met km, balan_met bm>
void TwostepSpMM_test(std::fstream &fs, const int iter, int feature_size,
                      SpMatCsrDescr_t<Index, DType> &H,
                      SpMatCsrDescr_t<Index, DType> &H_T,
                      balancer<Index, DType, bm> &balan,
                      util::RamArray<DType> &in_feature,
                      util::RamArray<DType> &tmp_feature,
                      util::RamArray<DType> &out_feature) {
  tmp_feature.reset();
  out_feature.reset();
  util::gpuTimer atimer;
  std::string method = "";
  atimer.start();
  if (km == spmm_kernel_met::edge_group) {
    for (int i = 0; i < iter; i++) {
      csrspmm_edgegroup<Index, DType>(
          H_T, feature_size, balan.key_len_T, balan.key_T.d_array.get(),
          balan.group_row_T.d_array.get(), in_feature.d_array.get(),
          tmp_feature.d_array.get());
      csrspmm_edgegroup<Index, DType>(
          H, feature_size, balan.key_len, balan.key.d_array.get(),
          balan.group_row.d_array.get(), tmp_feature.d_array.get(),
          out_feature.d_array.get());
    }
    method += "edge group";
  } else if (km == spmm_kernel_met::hybrid) {
    for (int i = 0; i < iter; i++) {
      csrspmm_hybrid<Index, DType>(
          H_T, balan.key_len_T, feature_size, balan.key_ptr_T.d_array.get(),
          balan.key_T.d_array.get(), balan.group_row_T.d_array.get(),
          in_feature.d_array.get(), tmp_feature.d_array.get());
      csrspmm_hybrid<Index, DType>(
          H, balan.key_len, feature_size, balan.key_ptr.d_array.get(),
          balan.key.d_array.get(), balan.group_row.d_array.get(),
          tmp_feature.d_array.get(), out_feature.d_array.get());
    }
    method += "hybrid";
  }
  atimer.end();
  float time = atimer.elapsed() / iter;
  std::cout << "The time of two " << method << " spmm " << time << std::endl;
  // fs << time << "," << 4 * feature_size * H.nnz * 1.0 / time / 1e6 << ",";
  fs << time << ",";
}

// without auxiliary array
template <class Index, class DType, spmm_kernel_met km>
float TwostepSpMM_test(std::fstream &fs, const int iter, int feature_size,
                       SpMatCsrDescr_t<Index, DType> &H,
                       SpMatCsrDescr_t<Index, DType> &H_T,
                       util::RamArray<DType> &in_feature,
                       util::RamArray<DType> &tmp_feature,
                       util::RamArray<DType> &out_feature) {
  out_feature.reset();
  tmp_feature.reset();
  util::gpuTimer atimer;
  std::string method = "";
  float compute_time = 0;
  atimer.start();
  if (km == spmm_kernel_met::row_balance) {
    for (int i = 0; i < iter; i++) {
      csrspmm_rowbalance<Index, DType>(H_T, feature_size,
                                       in_feature.d_array.get(),
                                       tmp_feature.d_array.get());
      csrspmm_rowbalance<Index, DType>(H, feature_size,
                                       tmp_feature.d_array.get(),
                                       out_feature.d_array.get());
    }
    method += "row balance";
  } else if (km == spmm_kernel_met::edge_balance) {
    for (int i = 0; i < iter; i++) {
      csrspmm_edgebalance<Index, DType>(H_T, feature_size,
                                        in_feature.d_array.get(),
                                        tmp_feature.d_array.get());
      csrspmm_edgebalance<Index, DType>(H, feature_size,
                                        tmp_feature.d_array.get(),
                                        out_feature.d_array.get());
    }
    method += "edge balance";
  } else if (km == spmm_kernel_met::cusparse) {
    compute_time += csrspmm_cusparse_test<Index, DType>(
        iter, H_T, feature_size, in_feature.d_array.get(),
        tmp_feature.d_array.get());
    compute_time += csrspmm_cusparse_test<Index, DType>(
        iter, H, feature_size, tmp_feature.d_array.get(),
        out_feature.d_array.get());
    method += "cusparse";
  }
  atimer.end();
  float report_time = (km == spmm_kernel_met::cusparse)
                          ? (compute_time / iter)
                          : (atimer.elapsed() / iter);

  std::cout << "The time of two " << method << " spmm " << report_time
            << std::endl;
  // fs << report_time << "," << 4 * feature_size * H.nnz * 1.0 / report_time /
  // 1e6
  //    << ",";
  fs << report_time << ",";
  return report_time;
}

template <class Index, class DType>
void TwostepSpMM_host(int feature_size, SpMatCsrDescr_t<Index, DType> &H,
                      SpMatCsrDescr_t<Index, DType> &H_T,
                      util::RamArray<DType> &in_feature,
                      util::RamArray<DType> &tmp_feature,
                      util::RamArray<DType> &out_ref) {
  out_ref.reset();
  tmp_feature.reset();
  util::spmm_reference_host<Index, DType>(
      H_T.nrow, feature_size, H_T.sp_csrptr.h_array.get(),
      H_T.sp_csrind.h_array.get(), H_T.sp_data.h_array.get(),
      in_feature.h_array.get(), tmp_feature.h_array.get());
  util::spmm_reference_host<Index, DType>(
      H.nrow, feature_size, H.sp_csrptr.h_array.get(),
      H.sp_csrind.h_array.get(), H.sp_data.h_array.get(),
      tmp_feature.h_array.get(), out_ref.h_array.get());
}

// check device based on ref
template <class Index, class DType, spmm_kernel_met km>
bool TwostepSpMM_check(int feature_size, SpMatCsrDescr_t<Index, DType> &H,
                       SpMatCsrDescr_t<Index, DType> &H_T,
                       util::RamArray<DType> &in_feature,
                       util::RamArray<DType> &tmp_feature,
                       util::RamArray<DType> &out_feature,
                       util::RamArray<DType> &out_ref) {
  out_ref.reset();
  out_feature.reset();
  tmp_feature.reset();
  TwostepSpMM_host<Index, DType>(feature_size, H, H_T, in_feature, tmp_feature,
                                 out_ref);
  if (km == spmm_kernel_met::row_balance) {
    csrspmm_rowbalance<Index, DType>(
        H_T, feature_size, in_feature.d_array.get(), tmp_feature.d_array.get());
    csrspmm_rowbalance<Index, DType>(H, feature_size, tmp_feature.d_array.get(),
                                     out_feature.d_array.get());
  } else if (km == spmm_kernel_met::edge_balance) {
    csrspmm_edgebalance<Index, DType>(
        H_T, feature_size, in_feature.d_array.get(), tmp_feature.d_array.get());
    csrspmm_edgebalance<Index, DType>(
        H, feature_size, tmp_feature.d_array.get(), out_feature.d_array.get());
  } else if (km == spmm_kernel_met::cusparse) {
    csrspmm_cusparse<Index, DType>(
        H_T.nrow, H_T.ncol, H_T.nnz, feature_size, H_T.sp_csrptr.d_array.get(),
        H_T.sp_csrind.d_array.get(), H_T.sp_data.d_array.get(),
        in_feature.d_array.get(), tmp_feature.d_array.get());
    csrspmm_cusparse<Index, DType>(
        H.nrow, H.ncol, H.nnz, feature_size, H.sp_csrptr.d_array.get(),
        H.sp_csrind.d_array.get(), H.sp_data.d_array.get(),
        tmp_feature.d_array.get(), out_feature.d_array.get());
  }
  out_feature.download();
  bool pass = util::check_result(
      H.nrow, feature_size, out_feature.h_array.get(), out_ref.h_array.get());
  if (pass) {
    printf("check passed!\n");
  }
  return pass;
}

// check device based on ref
template <class Index, class DType, spmm_kernel_met km, balan_met bm>
bool TwostepSpMM_check(int feature_size, SpMatCsrDescr_t<Index, DType> &H,
                       SpMatCsrDescr_t<Index, DType> &H_T,
                       balancer<Index, DType, bm> &balan,
                       util::RamArray<DType> &in_feature,
                       util::RamArray<DType> &tmp_feature,
                       util::RamArray<DType> &out_feature,
                       util::RamArray<DType> &out_ref) {
  out_ref.reset();
  out_feature.reset();
  tmp_feature.reset();
  TwostepSpMM_host<Index, DType>(feature_size, H, H_T, in_feature, tmp_feature,
                                 out_ref);
  if (km == spmm_kernel_met::edge_group) {
    csrspmm_edgegroup<Index, DType>(
        H_T, feature_size, balan.key_len_T, balan.key_T.d_array.get(),
        balan.group_row_T.d_array.get(), in_feature.d_array.get(),
        tmp_feature.d_array.get());
    csrspmm_edgegroup<Index, DType>(
        H, feature_size, balan.key_len, balan.key.d_array.get(),
        balan.group_row.d_array.get(), tmp_feature.d_array.get(),
        out_feature.d_array.get());
  } else if (km == spmm_kernel_met::hybrid) {
    csrspmm_hybrid<Index, DType>(
        H_T, balan.key_len_T, feature_size, balan.key_ptr_T.d_array.get(),
        balan.key_T.d_array.get(), balan.group_row_T.d_array.get(),
        in_feature.d_array.get(), tmp_feature.d_array.get());
    csrspmm_hybrid<Index, DType>(
        H, balan.key_len, feature_size, balan.key_ptr.d_array.get(),
        balan.key.d_array.get(), balan.group_row.d_array.get(),
        tmp_feature.d_array.get(), out_feature.d_array.get());
  } else if (km == spmm_kernel_met::cusparse) {
    csrspmm_cusparse<Index, DType>(
        H_T.nrow, H_T.ncol, H_T.nnz, feature_size, H_T.sp_csrptr.d_array.get(),
        H_T.sp_csrind.d_array.get(), H_T.sp_data.d_array.get(),
        in_feature.d_array.get(), tmp_feature.d_array.get());
    csrspmm_cusparse<Index, DType>(
        H.nrow, H.ncol, H.nnz, feature_size, H.sp_csrptr.d_array.get(),
        H.sp_csrind.d_array.get(), H.sp_data.d_array.get(),
        tmp_feature.d_array.get(), out_feature.d_array.get());
  }
  out_feature.download();
  bool pass = util::check_result(
      H.nrow, feature_size, out_feature.h_array.get(), out_ref.h_array.get());
  if (pass) {
    printf("check passed!\n");
  }
  return pass;
}

#endif