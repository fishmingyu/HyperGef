import torch.nn as nn
import torch
import dgl
import math

class DGLHyperGCNIIConv(nn.Module):
    def __init__(self, hyperg, in_channels, out_channels, heads=8, negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=False)     
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.hyperg = hyperg
        self.eps=nn.parameter.Parameter(torch.FloatTensor([0]))

    def forward(self, X, g1, g2, X0, alpha, beta):
        Xe=dgl.ops.copy_u_sum(g1,X)
        Xe = Xe * self.hyperg.degE 
        Xv=dgl.ops.copy_u_sum(g2,Xe)
        Xv = Xv * self.hyperg.degV
        Xi = (1-alpha) * Xv + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)
        return X
