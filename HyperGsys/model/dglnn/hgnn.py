# Revised from DGL documentation (https://docs.dgl.ai/notebooks/sparse/hgnn.html)
#

import torch.nn as nn
import dgl
from dgl.nn import GraphConv
import torch

# note that this new version is not accessible until dglv1.0
# X -> XW -> AXW -> norm
class DGLHGNNConv(nn.Module):
    def __init__(self, hyperg, in_channels, out_channels, first_aggr, heads=1, negative_slope=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.hyperg = hyperg
        self.first_aggr = first_aggr
        device = self.hyperg.degE.device
        self.W=torch.ones(self.hyperg.degE.shape[0],1).to(device)
        # torch.nn.init.xavier_normal_(self.W)

    def forward(self, g1, g2, X):
        X = self.linear(X)
        Xv = self.hyperg.L @ X
        return Xv
    

# we use this implementation in our experiment (2022)
# class DGLHGNNConv(nn.Module):
#     def __init__(self, hyperg, in_channels, out_channels, heads=8, negative_slope=0.2):
#         super().__init__()
#         self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)        
#         self.heads = heads
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.negative_slope = negative_slope
#         self.hyperg = hyperg
#         device = self.hyperg.degE.device
#         self.W=torch.ones(self.hyperg.degE.shape[0],1).to(device)
#         # torch.nn.init.xavier_normal_(self.W)
        

#     def forward(self, g1, g2, X):
#         X = self.linear(X)
#         Xe = dgl.ops.copy_u_sum(g1, X)
#         Xe = Xe * self.hyperg.degE
#         # print(Xe.shape,self.W.shape)
#         Xe = self.W * Xe
#         Xv = dgl.ops.copy_u_sum(g2, Xe)
#         Xv = Xv * self.hyperg.degV
#         return Xv