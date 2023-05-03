from HyperGsys.source.python.hgnnaggr import HGNNAggr
import dgl
import torch.nn as nn
import torch
import time

class HyperGsysHGNN(nn.Module):
    def __init__(self, hyperg, in_channels, out_channels, first_aggr, heads=1):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        device = hyperg.device
        self.Wdiag = torch.ones(hyperg.degE.shape[0]).to(device)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hyperg = hyperg
        self.degE = hyperg.degE
        self.degV = hyperg.degV
        self.first_aggr = first_aggr

    def forward(self, X):
        X = self.W(X)
        Xv = HGNNAggr(self.hyperg, X, self.degE, self.degV, self.Wdiag, self.first_aggr)
        # Xv = HGNNAggr(self.dl, X, self.degE, self.degV, self.Wdiag)
        # print(self.Wdiag)
        # print(Xv.shape)
        return Xv
