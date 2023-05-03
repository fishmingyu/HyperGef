#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor
unignnaggrdeg_fp_cuda(torch::Tensor balan_key, torch::Tensor balan_row,
                      torch::Tensor group_st, torch::Tensor group_ed,
                      torch::Tensor H_t_indices, torch::Tensor in_feat,
                      torch::Tensor degE, torch::Tensor degV);

torch::Tensor unignnaggr_fp_cuda(torch::Tensor balan_key,
                                 torch::Tensor balan_row,
                                 torch::Tensor group_st, torch::Tensor group_ed,
                                 torch::Tensor H_t_indices,
                                 torch::Tensor in_feat);

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class UniGNNAggrDeg : public torch::autograd::Function<UniGNNAggrDeg> {
public:
  static variable_list forward(AutogradContext *ctx, Variable balan_key,
                               Variable balan_row, Variable group_st,
                               Variable group_ed, Variable csrptr_t,
                               Variable indices_t, Variable node_feat,
                               Variable degE, Variable degV) {
    auto out = unignnaggrdeg_fp_cuda(balan_key, balan_row, group_st, group_ed,
                                     indices_t, node_feat, degE, degV);
    ctx->save_for_backward({balan_key, balan_row, group_st, group_ed, csrptr_t,
                            indices_t, node_feat, degE, degV});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto balan_key = saved[0], balan_row = saved[1], group_st = saved[2],
         group_ed = saved[3], csrptr_t = saved[4], indices_t = saved[5],
         node_feat = saved[6], degE = saved[7], degV = saved[8];

    auto grad_mat = Variable();
    grad_mat = unignnaggrdeg_fp_cuda(balan_key, balan_row, group_st, group_ed,
                                     indices_t, grad_out, degE, degV);

    return {Variable(), Variable(), Variable(), Variable(), Variable(),
            Variable(), grad_mat,   Variable(), Variable()};
  }
};

class UniGNNAggr : public torch::autograd::Function<UniGNNAggr> {
public:
  static variable_list forward(AutogradContext *ctx, Variable balan_key,
                               Variable balan_row, Variable group_st,
                               Variable group_ed, Variable csrptr_t,
                               Variable indices_t, Variable node_feat) {
    auto out = unignnaggr_fp_cuda(balan_key, balan_row, group_st, group_ed,
                                  indices_t, node_feat);
    ctx->save_for_backward({balan_key, balan_row, group_st, group_ed, csrptr_t,
                            indices_t, node_feat});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto balan_key = saved[0], balan_row = saved[1], group_st = saved[2],
         group_ed = saved[3], csrptr_t = saved[4], indices_t = saved[5],
         node_feat = saved[6];

    auto grad_mat = Variable();
    grad_mat = unignnaggr_fp_cuda(balan_key, balan_row, group_st, group_ed,
                                  indices_t, grad_out);

    return {Variable(), Variable(), Variable(), Variable(),
            Variable(), Variable(), grad_mat};
  }
};

torch::Tensor unignnaggrdeg(torch::Tensor balan_key, torch::Tensor balan_row,
                            torch::Tensor group_st, torch::Tensor group_ed,
                            torch::Tensor csrptr_t, torch::Tensor indices_t,
                            torch::Tensor node_feat, torch::Tensor degE,
                            torch::Tensor degV) {
  return UniGNNAggrDeg::apply(balan_key, balan_row, group_st, group_ed,
                              csrptr_t, indices_t, node_feat, degE, degV)[0];
}

torch::Tensor unignnaggr(torch::Tensor balan_key, torch::Tensor balan_row,
                         torch::Tensor group_st, torch::Tensor group_ed,
                         torch::Tensor csrptr_t, torch::Tensor indices_t,
                         torch::Tensor node_feat) {
  return UniGNNAggr::apply(balan_key, balan_row, group_st, group_ed, csrptr_t,
                           indices_t, node_feat)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "unignnaggr";
  m.def("unignnaggrdeg", &unignnaggrdeg, "unignnaggr with fused degE and degV");
  m.def("unignnaggr", &unignnaggr, "unignnaggr");
}
