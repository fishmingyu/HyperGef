#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor hgnnaggr_fp_cuda(torch::Tensor balan_key, torch::Tensor balan_row,
                               torch::Tensor group_st, torch::Tensor group_ed,
                               torch::Tensor H_t_indices, torch::Tensor in_feat,
                               torch::Tensor degE, torch::Tensor degV,
                               torch::Tensor W);

torch::Tensor hgnnaggr_mean_fp_cuda(torch::Tensor H_t_csrptr,
                                    torch::Tensor H_t_indices,
                                    torch::Tensor in_feat, torch::Tensor degE,
                                    torch::Tensor degV, torch::Tensor W);

torch::Tensor hgnnaggr_mean_bp_cuda(torch::Tensor H_t_csrptr,
                                    torch::Tensor H_t_indices,
                                    torch::Tensor in_feat, torch::Tensor degE,
                                    torch::Tensor degV, torch::Tensor W);

std::vector<torch::Tensor>
hgnnaggr_max_fp_cuda(torch::Tensor H_t_csrptr, torch::Tensor H_t_indices,
                     torch::Tensor in_feat, torch::Tensor degE,
                     torch::Tensor degV, torch::Tensor W);

torch::Tensor hgnnaggr_max_bp_cuda(torch::Tensor H_t_csrptr,
                                   torch::Tensor H_t_indices,
                                   torch::Tensor in_feat, torch::Tensor degE,
                                   torch::Tensor degV, torch::Tensor W,
                                   torch::Tensor record_table);

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class HGNNAggr : public torch::autograd::Function<HGNNAggr> {
public:
  static variable_list forward(AutogradContext *ctx, Variable balan_key,
                               Variable balan_row, Variable group_st,
                               Variable group_ed, Variable csrptr_t,
                               Variable indices_t, Variable node_feat,
                               Variable degE, Variable degV, Variable W) {
    auto out = hgnnaggr_fp_cuda(balan_key, balan_row, group_st, group_ed,
                                indices_t, node_feat, degE, degV, W);
    ctx->save_for_backward({balan_key, balan_row, group_st, group_ed, csrptr_t,
                            indices_t, node_feat, degE, degV, W});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto balan_key = saved[0], balan_row = saved[1], group_st = saved[2],
         group_ed = saved[3], csrptr_t = saved[4], indices_t = saved[5],
         node_feat = saved[6], degE = saved[7], degV = saved[8], W = saved[9];

    auto grad_mat = Variable();
    grad_mat = hgnnaggr_fp_cuda(balan_key, balan_row, group_st, group_ed,
                                indices_t, grad_out, degE, degV, W);

    return {Variable(), Variable(), Variable(), Variable(), Variable(),
            Variable(), grad_mat,   Variable(), Variable(), Variable()};
  }
};

class HGNNAggr_MeanF1 : public torch::autograd::Function<HGNNAggr_MeanF1> {
public:
  static variable_list forward(AutogradContext *ctx, Variable csrptr_t,
                               Variable indices_t, Variable node_feat,
                               Variable degE, Variable degV, Variable W) {
    auto out =
        hgnnaggr_mean_fp_cuda(csrptr_t, indices_t, node_feat, degE, degV, W);
    ctx->save_for_backward({csrptr_t, indices_t, node_feat, degE, degV, W});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto csrptr_t = saved[0], indices_t = saved[1], node_feat = saved[2],
         degE = saved[3], degV = saved[4], W = saved[5];

    auto grad_mat = Variable();
    grad_mat =
        hgnnaggr_mean_bp_cuda(csrptr_t, indices_t, grad_out, degE, degV, W);

    return {Variable(), Variable(), grad_mat,
            Variable(), Variable(), Variable()};
  }
};

class HGNNAggr_MaxF1 : public torch::autograd::Function<HGNNAggr_MaxF1> {
public:
  static variable_list forward(AutogradContext *ctx, Variable csrptr_t,
                               Variable indices_t, Variable node_feat,
                               Variable degE, Variable degV, Variable W) {
    auto out =
        hgnnaggr_max_fp_cuda(csrptr_t, indices_t, node_feat, degE, degV, W);
    auto record_table = out[1];
    ctx->save_for_backward(
        {csrptr_t, indices_t, node_feat, degE, degV, W, record_table});
    return {out[0], out[1]};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto csrptr_t = saved[0], indices_t = saved[1], node_feat = saved[2],
         degE = saved[3], degV = saved[4], W = saved[5],
         record_table = saved[6];

    auto grad_mat = Variable();
    grad_mat = hgnnaggr_max_bp_cuda(csrptr_t, indices_t, grad_out, degE, degV,
                                    W, record_table);

    return {Variable(), Variable(), grad_mat,
            Variable(), Variable(), Variable()};
  }
};

torch::Tensor hgnnaggr(torch::Tensor balan_key, torch::Tensor balan_row,
                       torch::Tensor group_st, torch::Tensor group_ed,
                       torch::Tensor csrptr_t, torch::Tensor indices_t,
                       torch::Tensor node_feat, torch::Tensor degE,
                       torch::Tensor degV, torch::Tensor W) {
  return HGNNAggr::apply(balan_key, balan_row, group_st, group_ed, csrptr_t,
                         indices_t, node_feat, degE, degV, W)[0];
}

torch::Tensor hgnnaggr_mean(torch::Tensor csrptr_t, torch::Tensor indices_t,
                            torch::Tensor node_feat, torch::Tensor degE,
                            torch::Tensor degV, torch::Tensor W) {
  return HGNNAggr_MeanF1::apply(csrptr_t, indices_t, node_feat, degE, degV,
                                W)[0];
}

std::vector<torch::Tensor> hgnnaggr_max(torch::Tensor csrptr_t,
                                        torch::Tensor indices_t,
                                        torch::Tensor node_feat,
                                        torch::Tensor degE, torch::Tensor degV,
                                        torch::Tensor W) {
  return HGNNAggr_MaxF1::apply(csrptr_t, indices_t, node_feat, degE, degV, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "hgnnaggr";
  m.def("hgnnaggr", &hgnnaggr, "hgnnaggr with fused degE and degV");
  m.def("hgnnaggr_mean", &hgnnaggr_mean, "hgnnaggr with f1 mean");
  m.def("hgnnaggr_max", &hgnnaggr_max, "hgnnaggr with f1 max");
}
