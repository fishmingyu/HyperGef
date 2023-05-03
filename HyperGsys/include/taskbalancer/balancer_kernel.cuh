#ifndef BALANCER_KERNEL_
#define BALANCER_KERNEL_

#include <cuda.h>
#include <map>
#include <vector>

#define FULLMASK 0xffffffff

template <typename DType>
__device__ __forceinline__ void AllReduce(DType multi, int stride,
                                          int warpSize) {
  for (; stride > 0; stride >>= 1) {
    multi += __shfl_xor_sync(FULLMASK, multi, stride, warpSize);
  }
}

template <typename Index>
__device__ __forceinline__ Index binary_search(const Index *S_csrRowPtr,
                                               Index eid, Index start,
                                               Index end) {
  Index lo = start, hi = end;
  if (lo == hi)
    return lo;
  while (lo < hi) {
    Index mid = (lo + hi) >> 1;
    if (__ldg(S_csrRowPtr + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // lo = hi
  while (__ldg(S_csrRowPtr + hi) == eid) {
    ++hi;
  }
  return hi - 1;
}

template <typename Index>
__global__ void pre_analysis_kernel_hgnn(const Index *A_indptr,
                                         const Index *A_indices,
                                         const Index *B_indptr,
                                         const Index *B_indices, Index *hist) {
  int rid = blockIdx.x;
  int pid = threadIdx.x;
  Index A_lb = A_indptr[rid];
  Index A_hb = A_indptr[rid + 1];
  Index A_acc = 0;
  Index tile_t = CEIL(A_hb - A_lb, blockDim.y);
  for (Index i = 0; i < tile_t; i++) {
    Index A_col_ptr = A_lb + pid + i * blockDim.y;
    if (A_col_ptr < A_hb) {
      Index B_row_idx = A_indices[A_col_ptr];
      Index B_lb = B_indptr[B_row_idx];
      Index B_hb = B_indptr[B_row_idx + 1];
      A_acc += B_hb - B_lb;
    }
  }
  AllReduce<Index>(A_acc, 16, 32);
  hist[rid] = A_acc;
}

template <typename Index>
__global__ void pre_analysis_kernel_spmm(int nr, const Index *A_indptr,
                                         Index *hist) {
  int rid = blockIdx.x * gridDim.x + threadIdx.x;
  if (rid == 0)
    hist[rid] = 0;
  else if (rid <= nr) {
    Index A_lb = A_indptr[rid - 1];
    Index A_hb = A_indptr[rid];
    Index A_acc = A_hb - A_lb;
    hist[rid] = A_acc;
  }
}

// [TODO]: partitioned binary search

template <typename Index>
__global__ void bin_search_simple_kernel(const int max_load, const int len,
                                         const Index tile_num,
                                         const Index *sum_hist, Index *key) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int loc = 0;
  if (id < tile_num - 1) {
    loc = binary_search(sum_hist, max_load * id, 0, len - 1);
    key[id] = loc;
  } else if (id == tile_num - 1) {
    loc = len;
    key[id] = loc;
  }
}

template <typename Index>
__global__ void spmm_edge_hist_kernel(const int max_load, const int nrow,
                                      const Index *csrptr, Index *group_hist) {
  int rid = blockIdx.x * blockDim.x + threadIdx.x;
  if (rid == 0)
    group_hist[rid] = 0;
  else if (rid <= nrow)
    group_hist[rid] = CEIL(csrptr[rid] - csrptr[rid - 1], max_load);
}

template <typename Index>
__global__ void spmm_edge_row_kernel(const int max_load, const int nrow,
                                     const int group_num, Index *group_hist_sum,
                                     Index *group_row) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < group_num) {
    Index loc = binary_search(group_hist_sum, gid, 0, nrow);
    group_row[gid] = loc;
  }
}

template <typename Index>
__global__ void spmm_edge_group_kernel(const int max_load, const int nrow,
                                       const int group_num, const Index *csrptr,
                                       Index *group_hist_sum, Index *group_row,
                                       Index *group_key) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < group_num) {
    Index row = group_row[gid];
    Index group_start = group_hist_sum[row];
    Index row_start = csrptr[row];
    Index group_key_start = (gid - group_start) * max_load + row_start;
    group_key[gid] = group_key_start;
  } else if (gid == group_num) {
    group_key[gid] = csrptr[nrow];
  }
}

template <typename Index>
__global__ void spmm_edge_row_unique(Index nrow, Index key_len,
                                     const Index *csrptr, Index *key,
                                     Index *group_row) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < key_len) {
    Index key_start = key[tid];
    Index row = binary_search(csrptr, key_start, 0, nrow);
    group_row[tid] = row;
  }
}

template <typename Index>
void hgnn_sn_balance_cpu(const int A_nrow, const int part,
                         const Index *A_indptr, const Index *A_indices,
                         const Index *B_indptr, const Index *B_indices,
                         std::vector<Index> &part_row,
                         std::vector<Index> &part_seg,
                         std::vector<Index> &neighbor_key) {
  part_seg.push_back(0);
  for (int rid = 0; rid < A_nrow; rid++) {
    int A_lb = A_indptr[rid];
    int A_hb = A_indptr[rid + 1];
    int degree_counter = 0;
    for (int col_p = A_lb; col_p < A_hb; col_p++) {
      int B_row_idx = A_indices[col_p];
      int B_lb = B_indptr[B_row_idx];
      int B_hb = B_indptr[B_row_idx + 1];
      int degree = B_hb - B_lb;
      degree_counter += degree;
      if (degree_counter > part) {
        part_row.push_back(rid);
        part_seg.push_back(col_p);
        degree_counter = degree;
      }
      neighbor_key.push_back(degree_counter);
    }
    if (part_seg.back() != A_hb) {
      part_row.push_back(rid);
      part_seg.push_back(A_hb);
    }
  }
}

// TODO cpu eg balance analysis
template <typename Index>
void hgnn_eg_balance_cpu(const int nrow, const int part, const Index *A_indptr,
                         std::vector<Index> &key,
                         std::vector<Index> &group_row) {
  key.push_back(0);
  for (int rid = 0; rid < nrow; rid++) {
    int A_lb = A_indptr[rid];
    int A_hb = A_indptr[rid + 1];
    int tmp_key = A_lb + part;
    while (tmp_key <= A_hb) {
      key.push_back(tmp_key);
      group_row.push_back(rid);
      tmp_key += part;
    }
    if (key.back() != A_hb) {
      group_row.push_back(rid);
      key.push_back(A_hb);
    }
  }
}

template <typename Index>
void hgnn_ef_balance_cpu(const int nrow, const int part, const Index *A_indptr,
                         std::vector<Index> &key, std::vector<Index> &group_row,
                         std::vector<Index> &work_prefix,
                         std::vector<Index> &work_twostep_prefix) {
  key.push_back(0);
  work_prefix.push_back(0);
  work_twostep_prefix.push_back(0);
  int work_p_sum = 0, work_ts_p_sum = 0;
  for (int rid = 0; rid < nrow; rid++) {
    int A_lb = A_indptr[rid];
    int A_hb = A_indptr[rid + 1];
    int workload = CEIL(A_hb - A_lb, part);
    int tmp_key = A_lb + part;
    while (tmp_key <= A_hb) {
      key.push_back(tmp_key);
      group_row.push_back(rid);
      tmp_key += part;
    }
    if (key.back() != A_hb) {
      group_row.push_back(rid);
      key.push_back(A_hb);
    }
    work_p_sum += workload;
    work_ts_p_sum += workload * workload;
    work_prefix.push_back(work_p_sum);
    work_twostep_prefix.push_back(work_ts_p_sum);
  }
}

template <typename Index>
void hgnn_ef_full_balance_cpu(const int nrow, const int part,
                              const Index *A_indptr, std::vector<Index> &key,
                              std::vector<Index> &row,
                              std::vector<Index> &group_st,
                              std::vector<Index> &group_ed) {
  int work_p_sum = 0;
  for (int rid = 0; rid < nrow; rid++) {
    int A_lb = A_indptr[rid];
    int A_hb = A_indptr[rid + 1];
    int workload = CEIL(A_hb - A_lb, part);
    int tmp_key = A_lb;
    while (tmp_key < A_hb) {
      key.push_back(tmp_key);
      tmp_key += part;
    }
    for (int i = 0; i < workload; i++) {
      for (int j = 0; j < workload; j++) {
        int id1 = work_p_sum + i;
        int id2 = work_p_sum + j;
        group_st.push_back(id2);
        group_ed.push_back(id1);
        row.push_back(rid);
      }
    }
    work_p_sum += workload;
  }
  if (key.back() != A_indptr[nrow]) {
    key.push_back(A_indptr[nrow]);
  }
}

template <typename Index>
void hgnn_merge_cpu(const int ncol, const int *B_indptr, const int *B_indices,
                    std::vector<Index> &key, std::vector<Index> &mgst,
                    std::vector<Index> &mgcol) {
  key.push_back(0);
  int cid = 0;
  int prefixsum = 0;
  for (; cid < ncol - 1; cid += 2) {
    std::map<int, int> tmpcol;
    int lb = B_indptr[cid];
    int hb = B_indptr[cid + 1];
    for (int i = lb; i < hb; i++) {
      int col = B_indices[i];
      tmpcol.insert(std::pair<Index, Index>(col, 1));
    }
    int cid1 = cid + 1;
    lb = B_indptr[cid1];
    hb = B_indptr[cid1 + 1];
    std::map<int, int>::iterator itr;
    for (int i = lb; i < hb; i++) {
      int col = B_indices[i];
      itr = tmpcol.find(col);
      if (itr != tmpcol.end())
        itr->second = 3;
      else
        tmpcol.insert(std::pair<Index, Index>(col, 2));
    }
    for (itr = tmpcol.begin(); itr != tmpcol.end(); ++itr) {
      mgcol.push_back(itr->first);
      mgst.push_back(itr->second);
    }
    prefixsum += tmpcol.size();
    key.push_back(prefixsum);
  }
  if (cid % 2 != 0) {
    int lb = B_indptr[cid];
    int hb = B_indptr[cid + 1];
    for (int i = lb; i < hb; i++) {
      int col = B_indices[i];
      mgcol.push_back(col);
      mgst.push_back(1);
    }
    prefixsum += hb - lb;
    key.push_back(prefixsum);
  }
}
#endif