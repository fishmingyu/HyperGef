#include "../util/check.cuh"
#include <cuda.h>

template <typename Index, typename DType> struct Pair {
  Index idx;
  DType val;
};

/// Exchange a pair (max, idx).

template <typename Index, typename DType>
__device__ Pair<Index, DType> shfl_xor_sync(Pair<Index, DType> p,
                                            unsigned delta) {
  return Pair<Index, DType>{
      __shfl_xor_sync(0xFFFFFFFF, p.idx, delta),
      __shfl_xor_sync(0xFFFFFFFF, p.val, delta),
  };
}

template <typename Index, typename DType>
__device__ Pair<Index, DType> argMaxOp(Pair<Index, DType> a,
                                       Pair<Index, DType> b) {
  return a.val > b.val ? a : b;
}

template <typename Index, typename DType>
__device__ Pair<Index, DType> argMaxWarp(Pair<Index, DType> p) {
  for (int i = 16; i > 0; i >>= 1)
    p = argMaxOp(p, shfl_xor_sync(p, i));
  return p;
}

template <typename Index, typename DType>
__global__ void jaccard_pair_node_kernel(const Index *rowptr,
                                         const Index *colind, Index *src,
                                         Index *dst) {
  int nid = blockIdx.x;
  int tid = threadIdx.x;
  int lb = rowptr[nid];
  int hb = rowptr[nid + 1];
  int a_len = hb - lb;
  const Index *a_ptr = colind + lb;
  DType jaccard_max = 0;
  int max_node = 0;
  if (a_len == 0) { // none->self - loop
    if (threadIdx.x == 0) {
      src[nid] = nid;
      dst[nid] = nid;
    }
    return;
  }
  for (; tid < a_len; tid += 32) {
    Index col = colind[lb + tid];
    const Index *b_ptr = colind + rowptr[col];
    int a_cnt = 0, b_cnt = 0;
    int intersect = 0;
    int b_len = rowptr[col + 1] - rowptr[col];

    while (a_cnt < a_len && b_cnt < b_len) {
      if (a_ptr[a_cnt] > b_ptr[b_cnt]) {
        b_cnt++;
      } else if (a_ptr[a_cnt] == b_ptr[b_cnt]) {
        intersect++;
        a_cnt++;
        b_cnt++;
      } else {
        a_cnt++;
      }
    }
    int Union = a_len + b_len - intersect;
    DType jaccard = intersect * 1.0 / Union;
    if (jaccard > jaccard_max) {
      jaccard_max = jaccard;
      max_node = col;
    }
  }
  Pair<Index, DType> p{max_node, jaccard_max};
  p = argMaxWarp<Index, DType>(p);
  if (threadIdx.x == 0) {
    src[nid] = nid;
    dst[nid] = p.idx;
  }
}

template <typename Index, typename DType>
__global__ void jaccard_pair_node_check_kernel(const Index *rowptr,
                                               const Index *colind, Index *src,
                                               Index *dst, DType *max) {
  int nid = blockIdx.x;
  int tid = threadIdx.x;
  int lb = rowptr[nid];
  int hb = rowptr[nid + 1];
  int a_len = hb - lb;
  const Index *a_ptr = colind + lb;
  DType jaccard_max = 0;
  int max_node = 0;
  if (a_len == 0) { // none->self - loop
    if (threadIdx.x == 0) {
      src[nid] = nid;
      dst[nid] = nid;
      max[nid] = 0;
    }
    return;
  }
  for (; tid < a_len; tid += 32) {
    Index col = colind[lb + tid];
    const Index *b_ptr = colind + rowptr[col];
    int a_cnt = 0, b_cnt = 0;
    int intersect = 0;
    int b_len = rowptr[col + 1] - rowptr[col];

    while (a_cnt < a_len && b_cnt < b_len) {
      if (a_ptr[a_cnt] > b_ptr[b_cnt]) {
        b_cnt++;
      } else if (a_ptr[a_cnt] == b_ptr[b_cnt]) {
        intersect++;
        a_cnt++;
        b_cnt++;
      } else {
        a_cnt++;
      }
    }
    int Union = a_len + b_len - intersect;
    DType jaccard = intersect * 1.0 / Union;
    if (jaccard > jaccard_max) {
      jaccard_max = jaccard;
      max_node = col;
    }
  }
  Pair<Index, DType> p{max_node, jaccard_max};
  p = argMaxWarp<Index, DType>(p);
  if (threadIdx.x == 0) {
    src[nid] = nid;
    dst[nid] = p.idx;
    max[nid] = p.val;
  }
}

template <typename Index, typename DType>
void jaccard_pair_node_cpu(const int nodes, const Index *rowptr,
                           const Index *colind, Index *src, Index *dst,
                           DType *max) {
  for (int i = 0; i < nodes; i++) {
    int lb = rowptr[i];
    int hb = rowptr[i + 1];
    int a_len = hb - lb;
    const Index *a_ptr = colind + lb;
    DType jaccard_max = 0;
    int max_node = 0;
    if (a_len == 0) { // none -> self-loop
      src[i] = i;
      dst[i] = i;
      max[i] = 0;
      continue;
    }
    for (int j = lb; j < hb; j++) {
      Index col = colind[j];
      const Index *b_ptr = colind + rowptr[col];
      int a_cnt = 0;
      int b_cnt = 0;
      int intersect = 0;
      int b_len = rowptr[col + 1] - rowptr[col];
      while (a_cnt < a_len && b_cnt < b_len) {
        if (a_ptr[a_cnt] > b_ptr[b_cnt]) {
          b_cnt++;
        } else if (a_ptr[a_cnt] == b_ptr[b_cnt]) {
          intersect++;
          a_cnt++;
          b_cnt++;
        } else {
          a_cnt++;
        }
      }
      int Union = a_len + b_len - intersect;
      DType jaccard = intersect * 1.0 / Union;
      if (jaccard > jaccard_max) {
        jaccard_max = jaccard;
        max_node = col;
      }
    }
    src[i] = i;
    dst[i] = max_node;
    max[i] = jaccard_max;
  }
}