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
  fs.open("fig10.csv", std::ios::app | std::ios::in | std::ios::out);
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
  // warm up
  for (int i = 0; i < 1000; i++)
    warm_up<<<1, 1>>>();

  fs << filename << "," << feature_size << ",";
  printf("start fig10 test\n");
  int tune_list[30] = {4,   6,   10,  16,  20,  30,  40,  60,
                       80,  100, 120, 150, 180, 210, 250, 300,
                       350, 400, 450, 500, 550, 600, -1};
  int count = 0;
  int p = tune_list[count];
  std::vector<float> time_list_g, time_list_ng;
  fs << "grouping,";
  while (p > 0) {
    hgnn_balancer<Index, DType, balan_met::hgnn_ef_full> balan(p, H, H_T);
    float time = 0;
    if (HyperGAggr_check<Index, DType, hgnn_kernel_met::edge_based_shm,
                         balan_met::hgnn_ef_full, 2, 32, 1, 1>(
            feature_size, H, H_T, balan, in_feature, out_feature, out_ref))
      time = HyperGAggr_test<Index, DType, hgnn_kernel_met::edge_based_shm,
                             balan_met::hgnn_ef_full, 2, 32, 1, 1>(
          iter, feature_size, H, H_T, balan, in_feature, out_feature);
    time_list_g.push_back(time);
    p = tune_list[++count];
  }
  for (auto i : time_list_g) {
    fs << i << ",";
  }
  fs << "\n, ,non-grouping,";
  count = 0;
  p = tune_list[count];
  while (p > 0) {
    hgnn_balancer<Index, DType, balan_met::hgnn_ef_full> balan(p, H, H_T);
    float time = 0;
    if (HyperGAggr_check<Index, DType, hgnn_kernel_met::edge_based_full,
                         balan_met::hgnn_ef_full, 2, 32, 1, 1>(
            feature_size, H, H_T, balan, in_feature, out_feature, out_ref))
      time = HyperGAggr_test<Index, DType, hgnn_kernel_met::edge_based_full,
                             balan_met::hgnn_ef_full, 2, 32, 1, 1>(
          iter, feature_size, H, H_T, balan, in_feature, out_feature);
    time_list_ng.push_back(time);
    p = tune_list[++count];
  }
  for (auto i : time_list_ng) {
    fs << i << ",";
  }
  fs << "\n";
  fs.close();
  return 0;
}