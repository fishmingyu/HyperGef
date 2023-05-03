import torch.nn as nn
import torch
import dgl

class DGLHyperGINConv(nn.Module):
    def __init__(self, args, in_channels, out_channels, first_aggr, heads=8, negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.args = args
        self.eps=nn.parameter.Parameter(torch.FloatTensor([0]))

    def forward(self, g1, g2, X):
        X=self.W(X)
        Xe=dgl.ops.copy_u_sum(g1,X)
        Xv=dgl.ops.copy_u_sum(g2,Xe)
        X=(1+self.eps)*X+Xv
        return X
