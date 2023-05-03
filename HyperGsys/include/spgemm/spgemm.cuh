#ifndef SPGEMM_
#define SPGEMM_

#include "../dataloader/dataloader.hpp"
#include "../spmm/spmm.cuh"
#include "../util/ramArray.cuh"
#include "DiagMul.cuh"
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <iostream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

template <typename Index, typename DType>
cusparseSpMatDescr_t SpGEMM_HHT(SpMatCsrDescr_t<Index, DType> &H,
                                SpMatCsrDescr_t<Index, DType> &H_T,
                                util::RamArray<DType> &in_feature,
                                util::RamArray<DType> &out_feature) {
  Index *H_csrptr = H.sp_csrptr.d_array.get();
  Index *H_csrind = H.sp_csrind.d_array.get();
  DType *H_data = H.sp_data.d_array.get();
  Index *H_T_csrptr = H_T.sp_csrptr.d_array.get();
  Index *H_T_csrind = H_T.sp_csrind.d_array.get();
  DType *H_T_data = H.sp_data.d_array.get();
  int H_nrow = H.nrow;
  int H_ncol = H.ncol;
  int H_nnz = H.nnz;

  cusparseHandle_t handle = NULL;
  checkCuSparseError(cusparseCreate(&handle));
  float alpha = 1.0f;
  float beta = 0.0f;
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cudaDataType computeType = CUDA_R_32F;

  // CUSPARSE APIs

  cusparseSpMatDescr_t matA, matB, matC;
  void *dBuffer1 = NULL, *dBuffer2 = NULL;
  size_t bufferSize1 = 0, bufferSize2 = 0;

  // Create sparse matrix A in CSR format
  checkCuSparseError(cusparseCreateCsr(&matA, H_nrow, H_ncol, H_nnz, H_csrptr,
                                       H_csrind, H_data, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  checkCuSparseError(cusparseCreateCsr(&matB, H_ncol, H_nrow, H_nnz, H_T_csrptr,
                                       H_T_csrind, H_T_data, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  checkCuSparseError(cusparseCreateCsr(
      &matC, H_nrow, H_nrow, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  checkCuSparseError(cusparseSpGEMM_createDescr(&spgemmDesc));

  // ask bufferSize1 bytes for external memory
  checkCuSparseError(cusparseSpGEMM_workEstimation(
      handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));
  checkCudaError(cudaMalloc((void **)&dBuffer1, bufferSize1));
  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  checkCuSparseError(cusparseSpGEMM_workEstimation(
      handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));

  // ask bufferSize2 bytes for external memory
  checkCuSparseError(cusparseSpGEMM_compute(
      handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL));
  checkCudaError(cudaMalloc((void **)&dBuffer2, bufferSize2));

  // compute the intermediate product of A * B
  checkCuSparseError(cusparseSpGEMM_compute(
      handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));
  // get matrix C non-zero entries C_nnz
  int64_t C_num_rows, C_num_cols, C_nnz;
  checkCuSparseError(
      cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz));
  // allocate matrix C
  int *C_csrptr;
  int *C_colind;
  int *C_data;
  checkCudaError(
      cudaMalloc((void **)&C_csrptr, (C_num_rows + 1) * sizeof(int)));
  checkCudaError(cudaMalloc((void **)&C_colind, C_nnz * sizeof(int)));
  checkCudaError(cudaMalloc((void **)&C_data, C_nnz * sizeof(float)));
  // update matC with the new pointers
  checkCuSparseError(cusparseCsrSetPointers(matC, C_csrptr, C_colind, C_data));

  // copy the final products to the matrix C
  checkCuSparseError(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB,
                                         &beta, matC, computeType,
                                         CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

  // destroy matrix/vector descriptors
  checkCuSparseError(cusparseSpGEMM_destroyDescr(spgemmDesc));
  checkCuSparseError(cusparseDestroySpMat(matA));
  checkCuSparseError(cusparseDestroySpMat(matB));
  checkCuSparseError(cusparseDestroy(handle));

  // device memory deallocation
  checkCudaError(cudaFree(dBuffer1));
  checkCudaError(cudaFree(dBuffer2));

  return matC;
}

struct time_sp {
  float spgemm_time;
  float spmm_time;
};

template <typename Index, typename DType>
time_sp SpGEMM_SpMM(int iter, int feature_size,
                    SpMatCsrDescr_t<Index, DType> &H,
                    SpMatCsrDescr_t<Index, DType> &H_T,
                    util::RamArray<DType> &in_feature,
                    util::RamArray<DType> &out_feature) {
  Index *H_csrptr = H.sp_csrptr.d_array.get();
  Index *H_csrind = H.sp_csrind.d_array.get();
  DType *H_data = H.sp_data.d_array.get();
  Index *H_T_csrptr = H_T.sp_csrptr.d_array.get();
  Index *H_T_csrind = H_T.sp_csrind.d_array.get();
  DType *H_T_data = H.sp_data.d_array.get();
  int H_nrow = H.nrow;
  int H_ncol = H.ncol;
  int H_nnz = H.nnz;

  util::gpuTimer atimer;
  time_sp tsp;

  cusparseHandle_t handle = NULL;
  checkCuSparseError(cusparseCreate(&handle));
  float alpha = 1.0f;
  float beta = 0.0f;
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cudaDataType computeType = CUDA_R_32F;

  // CUSPARSE APIs

  cusparseSpMatDescr_t matA, matB, matC;
  void *dBuffer1 = NULL, *dBuffer2 = NULL;
  size_t bufferSize1 = 0, bufferSize2 = 0;

  // Create sparse matrix A in CSR format
  checkCuSparseError(cusparseCreateCsr(&matA, H_nrow, H_ncol, H_nnz, H_csrptr,
                                       H_csrind, H_data, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  checkCuSparseError(cusparseCreateCsr(&matB, H_ncol, H_nrow, H_nnz, H_T_csrptr,
                                       H_T_csrind, H_T_data, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  checkCuSparseError(cusparseCreateCsr(
      &matC, H_nrow, H_nrow, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  checkCuSparseError(cusparseSpGEMM_createDescr(&spgemmDesc));

  // Test SpGEMM compute time
  // ask bufferSize1 bytes for external memory
  checkCuSparseError(cusparseSpGEMM_workEstimation(
      handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));
  checkCudaError(cudaMalloc((void **)&dBuffer1, bufferSize1));
  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  checkCuSparseError(cusparseSpGEMM_workEstimation(
      handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));

  // ask bufferSize2 bytes for external memory
  checkCuSparseError(cusparseSpGEMM_compute(
      handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL));
  checkCudaError(cudaMalloc((void **)&dBuffer2, bufferSize2));
  atimer.start();
  for (int i = 0; i < iter; i++) {
    // compute the intermediate product of A * B
    checkCuSparseError(cusparseSpGEMM_compute(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));
  }
  atimer.end();
  tsp.spgemm_time = atimer.elapsed();

  // get matrix C non-zero entries C_nnz
  int64_t C_num_rows, C_num_cols, C_nnz;
  checkCuSparseError(
      cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz));
  // allocate matrix C
  int *C_csrptr;
  int *C_colind;
  int *C_data;
  checkCudaError(
      cudaMalloc((void **)&C_csrptr, (C_num_rows + 1) * sizeof(int)));
  checkCudaError(cudaMalloc((void **)&C_colind, C_nnz * sizeof(int)));
  checkCudaError(cudaMalloc((void **)&C_data, C_nnz * sizeof(float)));
  // update matC with the new pointers
  checkCuSparseError(cusparseCsrSetPointers(matC, C_csrptr, C_colind, C_data));

  // copy the final products to the matrix C
  checkCuSparseError(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB,
                                         &beta, matC, computeType,
                                         CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

  // SpMM
  cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;

  // creating dense matrices
  checkCuSparseError(cusparseCreateDnMat(&dnMatInputDescr, H_nrow, feature_size,
                                         feature_size, in_feature.d_array.get(),
                                         CUDA_R_32F, CUSPARSE_ORDER_ROW));
  checkCuSparseError(cusparseCreateDnMat(
      &dnMatOutputDescr, H_nrow, feature_size, feature_size,
      out_feature.d_array.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW));

  // allocate workspace buffer
  size_t workspace_size;
  checkCuSparseError(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matC, dnMatInputDescr, &beta,
      dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
      &workspace_size));

  void *workspace = NULL;
  checkCudaError(cudaMalloc(&workspace, workspace_size));

  // run SpMM
  atimer.start();
  for (int i = 0; i < iter; i++) {
    checkCuSparseError(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                    &alpha, matC, dnMatInputDescr, &beta,
                                    dnMatOutputDescr, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, workspace));
  }
  atimer.end();
  tsp.spmm_time = atimer.elapsed();
  // destroy matrix/vector descriptors
  checkCuSparseError(cusparseSpGEMM_destroyDescr(spgemmDesc));
  checkCuSparseError(cusparseDestroySpMat(matA));
  checkCuSparseError(cusparseDestroySpMat(matB));
  checkCuSparseError(cusparseDestroySpMat(matC));
  checkCuSparseError(cusparseDestroy(handle));
  checkCuSparseError(cusparseDestroyDnMat(dnMatInputDescr));
  checkCuSparseError(cusparseDestroyDnMat(dnMatOutputDescr));

  // device memory deallocation
  checkCudaError(cudaFree(dBuffer1));
  checkCudaError(cudaFree(dBuffer2));
  return tsp;
}

template <typename Index, typename DType>
bool SpGEMM_SpMM_check(int feature_size, SpMatCsrDescr_t<Index, DType> &H,
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
  SpGEMM_SpMM<Index, DType>(1, feature_size, H, H_T, in_feature, out_feature);
  out_feature.download();
  bool pass = util::check_result(
      H.nrow, feature_size, out_feature.h_array.get(), out_ref.h_array.get());
  if (pass) {
    printf("check passed!\n");
  }
  return pass;
}

template <typename Index, typename DType>
void SpGEMM_SpMM_test(std::fstream &fs, int iter, int feature_size,
                      SpMatCsrDescr_t<Index, DType> &H,
                      SpMatCsrDescr_t<Index, DType> &H_T,
                      util::RamArray<DType> &in_feature,
                      util::RamArray<DType> &out_feature) {
  time_sp tsp;
  tsp = SpGEMM_SpMM<Index, DType>(iter, feature_size, H, H_T, in_feature,
                                  out_feature);

  float spgemm_time = tsp.spgemm_time / iter;
  float spmm_time = tsp.spmm_time / iter;
  printf("spgemm time: %f spmm_time %f \n", tsp.spgemm_time / iter,
         tsp.spmm_time / iter);
  fs << spgemm_time << "," << spmm_time << ",";
}
#endif