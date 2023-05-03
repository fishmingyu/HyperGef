from HyperGsys.source.python.unignnconv import UniGNNConvdeg
import dgl
import torch.nn as nn
import torch
import time

class HyperGsysUniGCNII(nn.Module):
    def __init__(self, hyperg, in_channels, out_channels, heads=1):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hyperg = hyperg
        self.degE = hyperg.degE
        self.degV = hyperg.degV

    def forward(self, X, X0, alpha, beta):
        Xv = UniGNNConvdeg(self.hyperg, X, self.degE, self.degV)
        Xi = (1-self.alpha) * Xv + alpha * X0
        X = (1-self.beta) * Xi + beta * self.W(Xi)
        
        # Xv = HGNNAggr(self.dl, X, self.degE, self.degV, self.Wdiag)
        # print(self.Wdiag)
        # print(Xv.shape)
        return X
