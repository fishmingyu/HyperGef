#ifndef BALANCER_
#define BALANCER_

#include "../dataloader/dataloader.hpp"
#include "../util/ramArray.cuh"
#include "balancer_kernel.cuh"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#define WARP_TILE 32

enum balan_met {
  none,
  spmm_edge_group,
  spmm_hybrid,
  hgnn_eg,
  hgnn_sn,
  hgnn_ef,
  hgnn_ef_full,
  hgnn_merge
};

// balancer on GPU
template <typename Index, typename DType, balan_met bm> class balancer {
public:
  balancer(int _max_load_per, SpMatCsrDescr_t<Index, DType> &H,
           SpMatCsrDescr_t<Index, DType> &H_t) {
    max_load_per = _max_load_per;
    hist.create(H.nrow + 1);
    hist_T.create(H.ncol + 1);
    hist.fill_zero_h();
    hist_T.fill_zero_h();
    process(H.nrow, H.ncol, H.nnz, H.sp_csrptr.d_array.get(),
            H.sp_csrind.d_array.get(), H_t.sp_csrptr.d_array.get(),
            H_t.sp_csrind.d_array.get(), hist.d_array.get(),
            hist_T.d_array.get());
  }
  // pytorch interface
  balancer(int node_group_size, int _node_merge_size, int nrow, int ncol,
           int nnz, Index *H_csrptr, Index *H_csrind, Index *H_t_csrptr,
           Index *H_t_csrind) {
    max_load_per = node_group_size;
    node_merge_size = _node_merge_size;
    hist.create(nrow + 1);
    hist_T.create(ncol + 1);
    hist.fill_zero_h();
    hist_T.fill_zero_h();
    process(nrow, ncol, nnz, H_csrptr, H_csrind, H_t_csrptr, H_t_csrind,
            hist.d_array.get(), hist_T.d_array.get());
  }
  balancer(int node_group_size, int _node_merge_size,
           SpMatCsrDescr_t<Index, DType> &H,
           SpMatCsrDescr_t<Index, DType> &H_t) {
    max_load_per = node_group_size;
    node_merge_size = _node_merge_size;
    hist.create(H.nrow + 1);
    hist_T.create(H.ncol + 1);
    hist.fill_zero_h();
    hist_T.fill_zero_h();
    process(H.nrow, H.ncol, H.nnz, H.sp_csrptr.d_array.get(),
            H.sp_csrind.d_array.get(), H_t.sp_csrptr.d_array.get(),
            H_t.sp_csrind.d_array.get(), hist.d_array.get(),
            hist_T.d_array.get());
  }
  ~balancer(){};
  void process(int nrow, int ncol, int nnz, const Index *A_indptr,
               const Index *A_indices, const Index *B_indptr,
               const Index *B_indices, Index *hist_darray,
               Index *hist_T_darray) {
    if (bm == balan_met::spmm_edge_group | bm == balan_met::spmm_hybrid) {
      // analysis workload, generate histogram
      pre_analysis(nrow, ncol, A_indptr, A_indices, B_indptr, B_indices,
                   hist_darray, hist_T_darray);

      if (bm == balan_met::spmm_edge_group | bm == balan_met::spmm_hybrid) {
        key_len = edge_partition(nrow, A_indptr, hist_darray, key, group_row);
        key_len_T =
            edge_partition(ncol, B_indptr, hist_T_darray, key_T, group_row_T);

        if (bm == balan_met::spmm_hybrid) {
          group_row_len = key_len;
          group_row_len_T = key_len_T;
          key_len = node_partition(nnz, key_len, key.d_array.get(), key_ptr);
          key_len_T =
              node_partition(nnz, key_len_T, key_T.d_array.get(), key_ptr_T);
          key_ptr.download();
        }
      }
    }
  }

  void pre_analysis(int nrow, int ncol, const Index *A_indptr,
                    const Index *A_indices, const Index *B_indptr,
                    const Index *B_indices, Index *hist, Index *hist_T) {
    if (bm == balan_met::spmm_edge_group | bm == balan_met::spmm_hybrid) {
      spmm_edge_hist_kernel<Index><<<CEIL(nrow + 1, WARP_TILE), WARP_TILE>>>(
          max_load_per, nrow, A_indptr, hist);
      spmm_edge_hist_kernel<Index><<<CEIL(ncol + 1, WARP_TILE), WARP_TILE>>>(
          max_load_per, ncol, B_indptr, hist_T);
    }
  }

  // all_num : sum of histogram
  // hist_len : length of histogram
  int node_partition(int all_num, int hist_len, Index *hist,
                     util::RamArray<Index> &key_tmp) {
    int tile_num = CEIL(all_num, node_merge_size) + 1;
    key_tmp.create(tile_num);
    key_tmp.fill_zero_h();
    // device_bin_search<<<tile_num, 32>>>(sum_raw, key);
    // search tile brk point
    bin_search_simple_kernel<<<CEIL(tile_num, WARP_TILE), WARP_TILE>>>(
        node_merge_size, hist_len, tile_num, hist,
        key_tmp.d_array.get()); // last search is h_s[nrow - 1]
    // unique

    int *new_end = thrust::unique(thrust::device, key_tmp.d_array.get(),
                                  key_tmp.d_array.get() + tile_num);
    key_tmp.download();
    return new_end - key_tmp.d_array.get();
  }

  int edge_partition(int nrow, const Index *A_indptr, Index *hist_darray,
                     util::RamArray<Index> &gr, util::RamArray<Index> &kk) {
    thrust::device_vector<Index> d_hist(hist_darray, hist_darray + nrow + 1);
    thrust::device_vector<Index> hist_sum(d_hist.size());
    // prefix sum
    thrust::inclusive_scan(d_hist.begin(), d_hist.end(), hist_sum.begin());
    Index *sum_raw = thrust::raw_pointer_cast(hist_sum.data());
    thrust::host_vector<Index> h_s(d_hist.size());
    thrust::copy(hist_sum.begin(), hist_sum.end(), h_s.begin());

    Index hist_all_num = h_s.data()[nrow];

    int group_num = hist_all_num;
    printf("group_num %d\n", group_num);
    gr.create(group_num);
    gr.fill_zero_h();
    kk.create(group_num + 1);
    kk.fill_zero_h();
    spmm_edge_row_kernel<Index><<<CEIL(group_num, WARP_TILE), WARP_TILE>>>(
        max_load_per, nrow, group_num, sum_raw, gr.d_array.get());
    spmm_edge_group_kernel<Index>
        <<<CEIL(group_num + 1, WARP_TILE), WARP_TILE>>>(
            max_load_per, nrow, group_num, A_indptr, sum_raw, gr.d_array.get(),
            kk.d_array.get());

    kk.download();
    gr.download();

    // for (int i = 0; i < 500; i++) {
    //   printf("kk[%d] : %d\n", i, kk.h_array.get()[i]);
    //   // printf("gr[%d] : %d\n", i, gr.h_array.get()[i]);
    // }

    d_hist.clear();
    hist_sum.clear();
    h_s.clear();
    d_hist.shrink_to_fit();
    hist_sum.shrink_to_fit();
    h_s.shrink_to_fit();
    return group_num + 1;
  }

  util::RamArray<Index> key;   // output key
  util::RamArray<Index> key_T; // output key for transposed
  util::RamArray<Index> group_row;
  util::RamArray<Index> group_row_T; //
  util::RamArray<Index> key_ptr;     // ptr of group key
  util::RamArray<Index> key_ptr_T;
  int key_len;   // final output length of key
  int key_len_T; // final output length of key (transposed)
  int group_row_len;
  int group_row_len_T;
  int max_load_per; // max load per tile

private:
  int tile_num;                 // finding how many tiles
  util::RamArray<Index> hist;   // work load histogram
  util::RamArray<Index> hist_T; // work load histogram
  int node_merge_size;
};

// [TODO] balancer on GPU need to fix, here use CPU replacement
template <typename Index, typename DType, balan_met bm> class hgnn_balancer {
public:
  hgnn_balancer(int _max_load_per, int H_nrow, int H_ncol, int H_nnz,
                Index *H_csrptr, Index *H_colind, Index *H_t_csrptr,
                Index *H_t_colind) {
    max_load_per = _max_load_per;
    process(H_nrow, H_ncol, H_nnz, H_csrptr, H_colind, H_t_csrptr, H_t_colind);
  }
  hgnn_balancer(int _max_load_per, SpMatCsrDescr_t<Index, DType> &H,
                SpMatCsrDescr_t<Index, DType> &H_t) {
    max_load_per = _max_load_per;
    process(H.nrow, H.ncol, H.nnz, H.sp_csrptr.h_array.get(),
            H.sp_csrind.h_array.get(), H_t.sp_csrptr.h_array.get(),
            H_t.sp_csrind.h_array.get());
  }
  ~hgnn_balancer(){};
  void process(const int nrow, const int ncol, int nnz, const Index *A_indptr,
               const Index *A_indices, const Index *B_indptr,
               const Index *B_indices) {
    // analysis workload, generate histogram
    std::vector<int> neighbor, work_prefix, work_twostep_prefix;
    std::vector<int> mgcol, mgst; // merge

    if (bm == balan_met::hgnn_sn) {
      hgnn_sn_balance_cpu<int>(nrow, max_load_per, A_indptr, A_indices,
                               B_indptr, B_indices, row, key, neighbor);
      neighbor_key.create(neighbor.size(), neighbor);
      neighbor_key.upload();
    }
    if (bm == balan_met::hgnn_eg) {
      hgnn_eg_balance_cpu<int>(nrow, max_load_per, A_indptr, key, row);
    }
    // edge based fusion
    if (bm == balan_met::hgnn_ef) {
      hgnn_ef_balance_cpu<int>(ncol, max_load_per, B_indptr, key, row,
                               work_prefix, work_twostep_prefix);
      work_ind.create(work_prefix.size(), work_prefix);
      work_ts_ind.create(work_twostep_prefix.size(), work_twostep_prefix);
      work_ind.upload();
      work_ts_ind.upload();
    }
    if (bm == balan_met::hgnn_ef_full) {
      hgnn_ef_full_balance_cpu<int>(ncol, max_load_per, B_indptr, key, row,
                                    group_st, group_ed);
      group_start.create(group_st.size(), group_st);
      group_end.create(group_ed.size(), group_ed);
      group_start.upload();
      group_end.upload();
    }
    if (bm == balan_met::hgnn_merge) {
      hgnn_merge_cpu<int>(ncol, B_indptr, B_indices, key, row, mgcol);
      merge_col.create(mgcol.size(), mgcol);
    }

    balan_row.create(row.size(), row);
    balan_key.create(key.size(), key);

    keys = row.size();
    if (bm == balan_met::hgnn_ef) {
      keys = work_twostep_prefix.back();
      part_keys = work_prefix.back();
    }
    if (bm == balan_met::hgnn_merge) {
      keys = key.size() - 1;
    }
    if (bm == balan_met::hgnn_ef_full) {
      keys = group_st.size();
      part_keys = key.size();
    }
    balan_row.upload();
    balan_key.upload();
  }

  int max_load_per; // finding how many tiles
  std::vector<Index> row, key;
  std::vector<Index> group_st, group_ed; // full
  util::RamArray<Index> balan_row;
  util::RamArray<Index> balan_key;
  util::RamArray<Index> work_ind;
  util::RamArray<Index> work_ts_ind;
  util::RamArray<Index> neighbor_key;
  util::RamArray<Index> merge_col;
  util::RamArray<Index> group_start;
  util::RamArray<Index> group_end;
  int keys;
  int part_keys;
};

#endif