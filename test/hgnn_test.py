import torch
from HyperGsys.model import gnn
from HyperGsys.dataloader import Dataloader
from HyperGsys.util import *
import argparse
import dgl

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--dname', default='walmart-trips')
    p.add_argument('--model-name', type=str, default='HGNN', help='Model')
    p.add_argument('--add-self-loop', action="store_true",
                   help='add-self-loop to hypergraph')
    p.add_argument('--use-norm', action="store_true",
                   help='use norm in the final layer')
    p.add_argument('--activation', type=str, default='relu',
                   help='activation layer between UniConvs')
    p.add_argument('--nlayer', type=int, default=2,
                   help='number of hidden layers')
    p.add_argument('--nhid', type=int, default=32,
                   help='number of hidden features, note that actually it\'s #nhid x #nhead')
    p.add_argument('--nhead', type=int, default=1, help='number of conv heads')
    p.add_argument('--dropout', type=float, default=0.6,
                   help='dropout probability after UniConv layer')
    p.add_argument('--input-drop', type=float, default=0.6,
                   help='dropout probability for input layer')
    # Choose std for synthetic feature noise
    p.add_argument('--feature_noise', default='1', type=str)
    p.add_argument('--runs', default=20, type=int)
    p.add_argument('--train_prop', type=float, default=0.5)
    p.add_argument('--valid_prop', type=float, default=0.25)
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    p.add_argument('--backend', type=str, default='pyg',
                   help='backend type: pyg, dgl, hgsys')
    p.add_argument('--epochs', type=int, default=200,
                   help='number of epochs to train')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='device to use, cpu or gpu')
    p.add_argument('--seed', type=int, default=1, help='seed for randomness')
    p.add_argument('--patience', type=int, default=200,
                   help='early stop after specific epochs')
    p.add_argument('--nostdout', action="store_true",
                   help='do not output logging to terminal')
    p.add_argument('--split', type=int, default=1,
                   help='choose which train/test split to use')
    p.add_argument('--out-dir', type=str,
                   default='runs/test',  help='output dir')
    p.add_argument('--profile', type=int, default='0',  help='profiling for 1')
    p.add_argument('--reorder', type=bool, default=False,
                   help='True for reordering')
    p.add_argument('--profile_gpu', type=int, default=0)
    p.add_argument('--output', type=str, default=None, help='output file')
    return p.parse_args()

def HGNN_check(hyperg, X, W):
    Xe = dgl.ops.copy_u_sum(hyperg.g1, X)
    Xe = Xe * hyperg.degE
    Xe *= W
    # print(Xe.shape,self.W.shape)
    Xv = dgl.ops.copy_u_sum(hyperg.g2, Xe)
    Xv = Xv * hyperg.degV
    return Xv

def test():
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                    'NTU2012', 'Mushroom',
                    'coauthor_cora', 'coauthor_dblp',
                    'yelp', 'walmart-trips', 'house-committees',
                    'cora', 'citeseer', 'pubmed']

    args = parse()
    synthetic_list = ['walmart-trips', 'house-committees']

    for dname in existing_dataset:
        f_noise = 1
        if (f_noise is not None) and dname in synthetic_list:
            dl = Dataloader(name=dname, args = args, root = '../HyperGsys/data/', feature_noise=f_noise)
        else:
            dl = Dataloader(name=dname, args = args, root = '../HyperGsys/data/')
        data = dl.data
        # load data
        nfeat, nclass = dl.num_features, len(dl.data.y.unique())
        hyperg = dl.hyperg
        
        from HyperGsys.source.python.hgnnaggr import HGNNAggr
        Wdiag = torch.ones(hyperg.degE.shape[0], 1).to(args.device)
        in_feat = torch.randn(dl.X.shape[0], 2).to(args.device)
        out_feat = HGNNAggr(hyperg, in_feat, hyperg.degE, hyperg.degV, Wdiag)
        out_check = HGNN_check(hyperg, in_feat, Wdiag)
  
        assert torch.allclose(out_feat, out_check, rtol=1e-4, atol=1e-6)
