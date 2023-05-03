from HyperGsys.source.python.unignnconv import UniGNNConv
import dgl
import torch.nn as nn
import torch
import time

class HyperGsysUinGINConv(nn.Module):
    def __init__(self, hyperg, in_channels, out_channels, first_aggr, heads=1):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hyperg = hyperg
        self.degE = self.hyperg.degE
        self.degV = self.hyperg.degV
        self.eps=nn.parameter.Parameter(torch.FloatTensor([0]))

    def forward(self, X):
        X = self.W(X)
        Xv = UniGNNConv(self.hyperg, X)
        X=(1+self.eps)*X+Xv
        # Xv = HGNNAggr(self.dl, X, self.degE, self.degV, self.Wdiag)
        # print(self.Wdiag)
        # print(Xv.shape)
        return X
