#include "../HyperGsys//include/hgnnAgg.cuh"
#include "../HyperGsys//include/spgemm/spgemm.cuh"
#include "../HyperGsys//include/spmm/spmm.cuh"
#include "../HyperGsys//include/util/ramArray.cuh"
#include "../HyperGsys/include/dataloader/dataloader.hpp"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <fstream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

__global__ void warm_up() {}

int main(int argc, char **argv) {
  // Host problem definition
  if (argc < 3) {
    printf("Input: first get the path of sparse matrix, then get the "
           "feature length of dense matrix\n");
    exit(1);
  }
  char *filename = argv[1];
  int feature_size = atoi(argv[2]);

  const int iter = 300;
  auto SpPair = DataLoader<Index, DType>(filename);

  std::fstream fs;
  fs.open("fig9.csv", std::ios::app | std::ios::in | std::ios::out);
  SpMatCsrDescr_t<Index, DType> H = std::get<0>(SpPair);
  SpMatCsrDescr_t<Index, DType> H_T = std::get<1>(SpPair);

  util::RamArray<DType> in_feature(H_T.ncol * feature_size);
  util::RamArray<DType> tmp_feature(H_T.nrow * feature_size);
  util::RamArray<DType> out_feature(H.nrow * feature_size);
  util::RamArray<DType> out_ref(H.nrow * feature_size);

  in_feature.fill_random_h();
  tmp_feature.fill_zero_h();
  out_feature.fill_zero_h();
  in_feature.upload();
  tmp_feature.upload();
  out_feature.upload();
  H.upload();
  H_T.upload();
  printf("start spmm test\n");
  // warm up
  for (int i = 0; i < 1000; i++)
    warm_up<<<1, 1>>>();

  float twospmm_time, edge_base_time, edge_partition_tune_time;
  fs << filename << "," << feature_size << ",";
  if (TwostepSpMM_check<Index, DType, spmm_kernel_met::cusparse>(
          feature_size, H, H_T, in_feature, tmp_feature, out_feature, out_ref))
    twospmm_time = TwostepSpMM_test<Index, DType, spmm_kernel_met::cusparse>(
        fs, iter, feature_size, H, H_T, in_feature, tmp_feature, out_feature);

  //   if (SpGEMM_SpMM_check<Index, DType>(feature_size, H, H_T, in_feature,
  //                                       tmp_feature, out_ref, out_feature))
  //     SpGEMM_SpMM_test<Index, DType>(fs, iter, feature_size, H, H_T,
  //     in_feature,
  //                                    out_feature);

  printf("start fused kernel test\n");

  if (HyperGAggr_check<Index, DType, hgnn_kernel_met::edge_based_fused, 2, 32>(
          feature_size, H, H_T, in_feature, out_feature, out_ref))
    edge_base_time =
        HyperGAggr_test<Index, DType, hgnn_kernel_met::edge_based_fused, 2, 32>(
            fs, ITER, feature_size, H, H_T, in_feature, out_feature);

  int tune_list[20] = {4,   6,   10,  16,  20,  30,  40,  60,  80, 100,
                       120, 150, 180, 210, 250, 300, 350, 400, -1};

  edge_partition_tune_time =
      HyperGAggr_tune<Index, DType, hgnn_kernel_met::edge_based_full,
                      balan_met::hgnn_ef_full>(fs, tune_list, feature_size, H,
                                               H_T, in_feature, out_feature,
                                               out_ref);
  // relative speedup
  fs << twospmm_time / edge_base_time << ","
     << twospmm_time / edge_partition_tune_time << "\n";
  fs.close();
  return 0;
}