from torch_scatter import scatter
import torch.nn as nn
import torch

# v1: X -> XW -> AXW -> norm
class PyGHyperGINConv(nn.Module):

    def __init__(self, hyperg, in_channels, out_channels, first_aggr, heads=8, negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.eps=nn.parameter.Parameter(torch.FloatTensor([0]))

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        X = self.W(X)
        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce='sum') # [E, C]
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        X=(1+self.eps)*X+Xv
        return X