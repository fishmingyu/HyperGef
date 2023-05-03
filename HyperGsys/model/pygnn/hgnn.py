# Revised from PyG documentation (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html)
# 

from torch_scatter import scatter
import torch.nn as nn
import torch

# new version using pyg message passing module but it is equal to using the code below (check https://pytorch-geometric.readthedocs.io/en/latest/advanced/sparse_tensor.html#)
# X -> XW -> AXW -> norm
class PyGHGNNConv(nn.Module):
    def __init__(self, hyperg, in_channels, out_channels, first_aggr, heads=1, negative_slope=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)        
        self.heads = heads
        self.hyperg = hyperg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        device = self.hyperg.degE.device
        self.first_aggr = first_aggr
        self.W = torch.ones(self.hyperg.degE.shape[0],1, device=device)
        # torch.nn.init.xavier_normal_(self.W)
        # print(self.W)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.hyperg.degE
        degV = self.hyperg.degV
        X = self.linear(X)
        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.first_aggr) # [E, C]
        Xe = Xe * degE
        Xe = Xe * self.W
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        Xv = Xv * degV
        X = Xv 
        return X

