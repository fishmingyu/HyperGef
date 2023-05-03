#ifndef CHECK_
#define CHECK_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef int Index;
typedef float DType;
const int RefThreadPerBlock = 256;

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define checkCuSparseError(a)                                                  \
  do {                                                                         \
    if (CUSPARSE_STATUS_SUCCESS != (a)) {                                      \
      fprintf(stderr, "CuSparse runTime error in line %d of file %s : %s \n",  \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define checkCudaError(a)                                                      \
  do {                                                                         \
    if (cudaSuccess != (a)) {                                                  \
      fprintf(stderr, "Cuda runTime error in line %d of file %s : %s \n",      \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace util {

template <typename DType>
bool check_result(int M, int N, DType *C, DType *C_ref) {
  bool passed = true;
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      DType c = C[i * N + j];
      DType c_ref = C_ref[i * N + j];
      if (fabs(c - c_ref) > 1e-2 * fabs(c_ref)) {
        printf(
            "Wrong result: i = %ld, j = %ld, result = %lf, reference = %lf.\n",
            i, j, c, c_ref);

        passed = false;
        break;
      }
    }
  }
  return passed;
}

// Compute spmm correct numbers. All arrays are host memory locations.
template <typename Index, typename DType>
void spmm_reference_host(int M,       // number of A-rows
                         int feature, // number of B_columns
                         Index *csr_indptr, Index *csr_indices,
                         DType *csr_values, // three arrays of A's CSR format
                         DType *B,          // assume row-major
                         DType *C_ref)      // assume row-major
{
  for (int64_t i = 0; i < M; i++) {
    Index begin = csr_indptr[i];
    Index end = csr_indptr[i + 1];
    for (Index p = begin; p < end; p++) {
      int k = csr_indices[p];
      DType val = csr_values[p];
      for (int64_t j = 0; j < feature; j++) {
        C_ref[i * feature + j] += val * B[k * feature + j];
      }
    }
  }
}

// Compute hyperaggr correct numbers. All arrays are host memory locations.
template <typename Index, typename DType>
void hyperaggr_reference_host(
    int A_row,        // number of rows of the incidence matrix
    int feature_size, // the cols of the dense matrix

    const Index *A_indptr,
    const Index *A_indices, // two arrays of the incidence matrix's CSR format
    const Index *B_indptr,
    const Index *B_indices,  // two arrays of the transpose of the incidence
                             // matrix's CSR format
    const DType *in_feature, // assume row-major
    DType *out_feature) {
  for (int A_row_idx = 0; A_row_idx < A_row; A_row_idx++) {
    Index A_lb = A_indptr[A_row_idx];
    Index A_hb = A_indptr[A_row_idx + 1];
    for (int k_idx = 0; k_idx < feature_size; k_idx++) {
      DType A_acc = 0;
      for (Index A_col_ptr = A_lb; A_col_ptr < A_hb; A_col_ptr++) {
        DType B_acc = 0;
        Index B_row_idx = A_indices[A_col_ptr];
        Index B_lb = B_indptr[B_row_idx];
        Index B_hb = B_indptr[B_row_idx + 1];
        for (Index B_col_ptr = B_lb; B_col_ptr < B_hb; B_col_ptr++) {
          Index B_col_idx = B_indices[B_col_ptr];
          // DType B_val = B_data[B_col_ptr];
          B_acc += in_feature[B_col_idx * feature_size + k_idx];
        }
        A_acc += B_acc;
      }
      out_feature[A_row_idx * feature_size + k_idx] = A_acc;
    }
  }
}

template <typename Index, typename DType>
void hgnnbp_reference_host(
    int row,          // number of rows of the incidence matrix
    int feature_size, // the cols of the dense matrix
    const Index *indptr,
    const Index *indices, // two arrays of the incidence matrix's CSR format
                          // (transposed)
    const DType *grad_output,
    const DType *node_feature, // in feature
    DType *weight_grad         // out
) {
  for (int rid = 0; rid < row; rid++) {
    Index start = indptr[rid];
    Index end = indptr[rid + 1];
    DType weight_tmp = 0;
    for (int tid = 0; tid < feature_size; tid++) {
      DType res1 = 0, res2 = 0;
      for (int ptr = start; ptr < end; ptr++) {
        Index col = indices[ptr];
        res1 += node_feature[col * feature_size + tid];
        res2 += grad_output[col * feature_size + tid];
      }
      DType res = res1 * res2;
      weight_tmp += res;
    }
    weight_grad[rid] = weight_tmp;
  }
}

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
}

} // namespace util
#endif // CHECK_