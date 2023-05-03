#include "../util/check.cuh"
#include "../util/ramArray.cuh"
#include "fast_reorder_kernel.cuh"

template <typename Index, typename DType>
void jaccard_pair_node_device(int nodes, const Index *rowptr,
                              const Index *colind, Index *src, Index *dst,
                              DType *max) {
  int warpSize = 32;
  dim3 blockDim(warpSize, 1, 1);
  dim3 gridDim(nodes, 1, 1);
  jaccard_pair_node_check_kernel<Index, DType>
      <<<gridDim, blockDim>>>(rowptr, colind, src, dst, max);
}

template <typename Index, typename DType>
void jaccard_check(SpMatCsrDescr_t<Index, DType> &H, util::RamArray<Index> &src,
                   util::RamArray<Index> &dst, util::RamArray<Index> &dst_ref,
                   util::RamArray<DType> &max, util::RamArray<DType> &max_ref) {
  jaccard_pair_node_cpu<Index, DType>(
      H.nrow, H.sp_csrptr.h_array.get(), H.sp_csrind.h_array.get(),
      src.h_array.get(), dst_ref.h_array.get(), max_ref.h_array.get());
  jaccard_pair_node_device<Index, DType>(
      H.nrow, H.sp_csrptr.d_array.get(), H.sp_csrind.d_array.get(),
      src.d_array.get(), dst.d_array.get(), max.d_array.get());

  dst.download();
  max.download();
  bool pass = true;
  for (int i = 0; i < H.nrow; i++) {
    if (max.h_array.get()[i] - max_ref.h_array.get()[i] > 1e-4) {
      printf("id: %d , not equal, ref: %f, ans: %f", i,
             max_ref.h_array.get()[i], max.h_array.get()[i]);
      pass = false;
      break;
    }
  }
  if (pass)
    printf("check passed\n");
  else
    printf("check failed!\n");
}