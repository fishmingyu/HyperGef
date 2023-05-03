#include "../util/check.cuh"
#include "../util/ramArray.cuh"
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <iostream>
#include <math.h>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

// thurst
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

template <typename DType> struct ResultTuple {
  DType upper_percent;
  DType lower_percent;
  DType gini;
  DType norm_std;
  DType norm_square;
};

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
  return old;
}

template <typename Index, typename DType>
__global__ void hist_kernel(int row, int bin_size, int up_bound, int lw_bound,
                            Index *rowptr, Index *hist_gini, DType *ul_tmp) {
  int rid = blockIdx.x * blockDim.x + threadIdx.x;
  if (rid < row) {
    int degree = rowptr[rid + 1] - rowptr[rid];
    int step = degree / bin_size; // degree/degree_max

    atomicAdd(hist_gini + step, 1);
    DType upper_sig = (degree >= up_bound) ? 1 : 0;
    DType lower_sig = (degree < lw_bound && degree != 0) ? 1 : 0;
    atomicAdd(ul_tmp, upper_sig);
    atomicAdd(ul_tmp + 1, lower_sig);
    atomicMaxFloat(ul_tmp + 2, degree);
    atomicAdd(ul_tmp + 3, degree * degree * 1.0);
  }
}

template <typename Index, typename DType>
__global__ void gini_kernel(int hist_len, int hist_sum, Index *hist_gini,
                            DType *ul_tmp) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  DType gini_tmp = 0;
  if (tid < hist_len) {
    DType percen = hist_gini[tid] * 1.0 / hist_sum; //
    gini_tmp = percen * percen;
    atomicAdd(ul_tmp + 4, gini_tmp);
  }
}

template <typename Index, typename DType>
ResultTuple<float> feature_extract_device(int row, int nnz, int bin_tu,
                                          int up_b, int lw_b, Index *rowptr,
                                          Index *colind) {
  int warpSize = 32;
  int aver = CEIL(nnz, row);
  int bin_size = aver * bin_tu; // tuning?
  int up_bound = aver * up_b;
  int lw_bound = CEIL(aver, lw_b);
  DType upper_percent, lower_percent;
  DType gini;
  DType mean_degree, degree_square_sum;
  DType norm_std, norm_square;
  int hist_len = CEIL(nnz, bin_size);
  int max_degree;

  util::RamArray<Index> hist_gini(hist_len);
  util::RamArray<DType> ul_tmp(
      5); // upper_sig, lower_sig, max_degree, square, gini
  hist_gini.fill_zero_h();
  ul_tmp.fill_zero_h();
  hist_gini.upload();
  ul_tmp.upload();
  hist_kernel<Index, DType>
      <<<dim3(CEIL(row, warpSize), 1, 1), dim3(warpSize, 1, 1)>>>(
          row, bin_size, up_bound, lw_bound, rowptr, hist_gini.d_array.get(),
          ul_tmp.d_array.get());
  // degree_max = reduce(max)
  gini_kernel<Index, DType><<<CEIL(hist_len, warpSize), warpSize>>>(
      hist_len, row, hist_gini.d_array.get(), ul_tmp.d_array.get());

  ul_tmp.download();
  hist_gini.download();

  upper_percent = ul_tmp.h_array.get()[0] / row;
  lower_percent = ul_tmp.h_array.get()[1] / row;
  max_degree = ul_tmp.h_array.get()[2];
  degree_square_sum = ul_tmp.h_array.get()[3];
  gini = 1 - ul_tmp.h_array.get()[4];
  mean_degree = nnz * 1.0 / row;
  norm_std = sqrt((degree_square_sum - mean_degree * mean_degree) * 1.0 / row) /
             max_degree;
  norm_square = sqrt((degree_square_sum)*1.0 / row) / max_degree;

  return ResultTuple<DType>{upper_percent, lower_percent, gini, norm_std,
                            norm_square};
}