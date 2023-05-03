from torch.utils.cpp_extension import load
import torch
import hgnnaggr


def HGNNAggr(hyperg, in_feat, degE, degV, Wdiag, first_aggr):
    return hgnnaggr.hgnnaggr(hyperg.group_key, hyperg.group_row, hyperg.group_start, hyperg.group_end, hyperg.H_T_csrptr, hyperg.H_T_colind, in_feat, degE, degV, Wdiag)
