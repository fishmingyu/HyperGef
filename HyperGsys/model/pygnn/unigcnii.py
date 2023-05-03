from torch_scatter import scatter
import torch.nn as nn
import torch
import math

# v1: X -> XW -> AXW -> norm
class PyGHyperGCNIIConv(nn.Module):

    def __init__(self, hyperg, in_channels, out_channels, heads=8, negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=False)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.hyperg = hyperg
        self.degE = self.hyperg.degE
        self.degV = self.hyperg.degV
        self.eps=nn.parameter.Parameter(torch.FloatTensor([0]))

    def forward(self, X, vertex, edges, X0, alpha, beta):
        N = X.shape[0]
        degE = self.degE
        degV = self.degV
        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce='sum') # [E, C]
        Xe = Xe * degE 
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        Xv = Xv * degV
        X = Xv 
        Xi = (1-alpha) * X + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)
        return X