#ifndef DIAGMUL_
#define DIAGMUL_

#include <cuda.h>

#define WARP_TILE 32

// Compute in EB Tiling Method
template <typename Index, typename DType>
__global__ void DiagMul_kernel(Index A_nnz, Index *A_indice, DType *A_data,
                               DType *W) {
  Index ele_id = blockIdx.x * WARP_TILE + threadIdx.x;
  if (ele_id < A_nnz) {
    Index col_ind = A_indice[ele_id];
    A_data[ele_id] = A_data[ele_id] * sqrt(W[col_ind]);
  }
}

template <typename Index, typename DType>
void DiagMul(int A_nnz, int *A_indice, float *A_data, float *W_dev) {
  DiagMul_kernel<<<CEIL(A_nnz, WARP_TILE), WARP_TILE>>>(A_nnz, A_indice, A_data,
                                                        W_dev);
}

#endif