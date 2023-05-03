#ifndef CUDA_UTIL
#define CUDA_UTIL

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define FULLMASK 0xffffffff
#define WARPSIZE 32
#define SHFL_DOWN_REDUCE(v)                                                    \
  v += __shfl_down_sync(FULLMASK, v, 16);                                      \
  v += __shfl_down_sync(FULLMASK, v, 8);                                       \
  v += __shfl_down_sync(FULLMASK, v, 4);                                       \
  v += __shfl_down_sync(FULLMASK, v, 2);                                       \
  v += __shfl_down_sync(FULLMASK, v, 1);

// end = start + how many seg parts
// itv(interval) = id in which idx of B's row
template <typename Index>
__device__ __forceinline__ void
__find_row_entry(Index id, Index *neighbor_key, Index *A_indices, Index start,
                 Index end, Index &B_row_idx, Index &itv) {
  Index lo = start, hi = end;
  // id is small, you could set the value already
  if (neighbor_key[lo] > id) {
    itv = id;
    B_row_idx = A_indices[lo];
    return;
  }
  while (lo < hi) {
    Index mid = (lo + hi) >> 1;
    if (__ldg(neighbor_key + mid) <= id) { // find the right(high)
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // case lo = hi
  while (__ldg(neighbor_key + hi) == id) {
    ++hi;
  }
  B_row_idx = A_indices[hi];
  itv = id - neighbor_key[hi - 1];
}

#endif // CUDA_UTIL