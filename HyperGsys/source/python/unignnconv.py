from torch.utils.cpp_extension import load
import torch
import unignnaggr


def UniGNNConvdeg(dl, in_feat, degE, degV):
    return unignnaggr.unignnconvdeg(dl.group_key, dl.group_row, dl.group_start, dl.group_end, dl.H_T_csrptr, dl.H_T_colind, in_feat, degE, degV)

def UniGNNConv(dl, in_feat):
    return unignnaggr.unignnconv(dl.group_key, dl.group_row, dl.group_start, dl.group_end, dl.H_T_csrptr, dl.H_T_colind, in_feat)
