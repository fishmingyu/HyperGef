import torch
import torch.nn as nn
import torch.nn.functional as F
from . import dglnn
from . import pygnn
from . import ugsys

import math
import sys
import time

from torch_scatter import scatter
from torch_geometric.utils import softmax

__dgl_convs__ = {
    'UniGIN': dglnn.unigin.DGLHyperGINConv,
    'HGNN': dglnn.hgnn.DGLHGNNConv,
}

__pyg_convs__ = {
    'UniGIN': pygnn.unigin.PyGHyperGINConv,
    'HGNN': pygnn.hgnn.PyGHGNNConv,
}

__ugsys_convs__ = {
    'UniGIN': ugsys.unigin.HyperGsysUinGINConv,
    'HGNN': ugsys.hgnn.HyperGsysHGNN,
}


class PyGHGNN(nn.Module):
    def __init__(self, args, hyperg, nfeat, nhid, nclass, nlayer, first_aggr, nhead):
        """

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = __pyg_convs__[args.model]
        self.conv_out = Conv(hyperg, nhid * nhead, nclass, first_aggr,
                             heads=1)
        self.convs = nn.ModuleList(
            [Conv(hyperg, nfeat, nhid, first_aggr, heads=nhead)] +
            [Conv(hyperg, nhid * nhead, nhid, first_aggr, heads=nhead)
             for _ in range(nlayer-2)]
        )
        self.V = hyperg.V
        self.E = hyperg.E
        act = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        V, E = self.V, self.E
        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(X, V, E)
        return F.log_softmax(X, dim=1)


class DGLHGNN(nn.Module):
    def __init__(self, args, hyperg, nfeat, nhid, nclass, nlayer, first_aggr, nhead):
        """
        g1: the transpose of  indicence matrix (E, V)
        g2: indicence matrix (V, E)

        """
        super().__init__()
        Conv = __dgl_convs__[args.model]
        self.conv_out = Conv(hyperg, nhid * nhead, nclass, first_aggr,
                             heads=1)
        self.convs = nn.ModuleList(
            [Conv(hyperg, nfeat, nhid,first_aggr, heads=nhead)] +
            [Conv(hyperg, nhid * nhead, nhid, first_aggr, heads=nhead)
             for _ in range(nlayer-2)]
        )

        self.g1 = hyperg.g1
        self.g2 = hyperg.g2
        act = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        g1 = self.g1
        g2 = self.g2
        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(g1, g2, X)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(g1, g2, X)
        return F.log_softmax(X, dim=1)


class HGsysHGNN(nn.Module):
    def __init__(self, args, hyperg, nfeat, nhid, nclass, nlayer, first_aggr, nhead):
        super().__init__()
        Conv = __ugsys_convs__[args.model]
        self.conv_out = Conv(hyperg, nhid * nhead, nclass, first_aggr, nhead)
        self.convs = nn.ModuleList(
            [Conv(hyperg, nfeat, nhid, first_aggr, nhead)] +
            [Conv(hyperg, nhid * nhead, nhid, first_aggr, nhead)
             for _ in range(nlayer-2)]
        )

        act = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(X)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(X)
        return F.log_softmax(X, dim=1)


class UniGCNII(nn.Module):
    def __init__(self, args, hyperg, nfeat, nhid, nclass, nlayer, nhead):
        """UniGNNII
        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.hyperg = hyperg
        
        nhid = nhid * nhead
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.args = args

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        if args.backend == 'dgl':
            for _ in range(nlayer):
                self.convs.append(dglnn.unigcnii.DGLHyperGCNIIConv(hyperg, nhid, nhid)) 
        elif args.backend == 'pyg':
            for _ in range(nlayer):
                self.convs.append(pygnn.unigcnii.PyGHyperGCNIIConv(hyperg, nhid, nhid))
        elif args.backend == 'ugsys':
            for _ in range(nlayer):
                self.convs.append(ugsys.unigcnii.HyperGsysUniGCNII(hyperg, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        lamda, alpha = 0.5, 0.1 
        if self.args.backend == 'pyg':
            V, E = self.hyperg.V, self.hyperg.E 
            x = self.dropout(x)
            x = F.relu(self.convs[0](x))
            x0 = x 
            for i,con in enumerate(self.convs[1:-1]):
                x = self.dropout(x)
                beta = math.log(lamda/(i+1)+1)
                x = F.relu(con(x, V, E, x0, alpha, beta))
        elif self.args.backend == 'dgl':
            g1 = self.hyperg.g1
            g2 = self.hyperg.g2
            x = self.dropout(x)
            x = F.relu(self.convs[0](x))
            x0 = x 
            for i,con in enumerate(self.convs[1:-1]):
                x = self.dropout(x)
                beta = math.log(lamda/(i+1)+1)
                x = F.relu(con(x, g1, g2, x0, alpha, beta))

        elif self.args.backend == 'hgsys':
            x = self.dropout(x)
            x = F.relu(self.convs[0](x))
            x0 = x 
            for i,con in enumerate(self.convs[1:-1]):
                x = self.dropout(x)
                beta = math.log(lamda/(i+1)+1)
                x = F.relu(con(x, x0, alpha, beta))
        x = self.dropout(x)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)
