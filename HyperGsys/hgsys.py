import sys
import logging
from collections import Counter
from dataloader import Dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import os

import numpy as np
import time
import datetime
import argparse
from model import gnn
from util import *
import torch.optim as optim
import torch.nn.functional as F
import GPUtil


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--dname', default='walmart-trips')
    p.add_argument('--model', type=str, default='HGNN', help='Model')
    p.add_argument('--data-path', type=str, default='data/', help="path to data")
    p.add_argument('--add-self-loop', action="store_true",
                   help='add-self-loop to hypergraph')
    p.add_argument('--use-norm', action="store_true",
                   help='use norm in the final layer')
    p.add_argument('--activation', type=str, default='relu',
                   help='activation layer between UniConvs')
    p.add_argument('--nlayer', type=int, default=2,
                   help='number of hidden layers')
    p.add_argument('--first-aggr', type=str, default='sum', help='type of first aggregation')
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


args = parse()
# print(args.device)
# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTHONHASHSEED'] = str(args.seed)

use_norm = 'use-norm' if args.use_norm else 'no-norm'
add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'

### Load and preprocess data ###
existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                    'NTU2012', 'Mushroom',
                    'coauthor_cora', 'coauthor_dblp',
                    'yelp', 'walmart-trips', 'house-committees',
                    'cora', 'citeseer', 'pubmed']

synthetic_list = ['walmart-trips', 'house-committees']

if args.dname in existing_dataset:
    dname = args.dname
    dname = args.dname
    f_noise = args.feature_noise
    if (f_noise is not None) and dname in synthetic_list:
        dl = Dataloader(name=dname, args = args, root = args.data_path, feature_noise=f_noise)
    else:
        dl = Dataloader(name=dname, args = args, root = args.data_path)
    data = dl.data

#     Get splits
split_idx = rand_train_test_idx(
    dl.data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
train_idx = split_idx['train'].to(args.device)
test_idx = split_idx['test'].to(args.device)

# configure output directory
model_name = args.model
nlayer = args.nlayer

# load data
maxMemory = 0
trainTime = 0
inferenceTime = 0
nfeat, nclass = dl.num_features, len(dl.data.y.unique())
nlayer = args.nlayer
nhid = args.nhid
nhead = args.nhead
hyperg = dl.hyperg
first_aggr = args.first_aggr

# load model
if args.model != 'UniGCNII':
    if args.backend == 'dgl':
        model = gnn.DGLHGNN(args, hyperg, nfeat, nhid, nclass,
                            nlayer, first_aggr, nhead)
    elif args.backend == 'pyg':
        model = gnn.PyGHGNN(args, hyperg, nfeat, nhid, nclass, nlayer, first_aggr, nhead)
    elif args.backend == 'hgsys':
        model = gnn.HGsysHGNN(args, hyperg, nfeat, nhid, nclass, nlayer, first_aggr, nhead)
else:
    model = gnn.UniGCNII(args, hyperg, nfeat, nhid, nclass, nlayer, nhead)
opti = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.to(args.device)

print(f'Total Epochs: {args.epochs}')
print(model)
print(
    f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

best_test_acc, test_acc, Z = 0, 0, None

if args.profile:
    model.train()
    torch.cuda.synchronize()
    t1 = time.time()
    for epoch in range(args.epochs):
        opti.zero_grad()
        Z = model(dl.X)
        loss = F.nll_loss(Z[train_idx], dl.y[train_idx])
        loss.backward()
        opti.step()
    torch.cuda.synchronize()
    t2 = time.time()
    print(f'epoch time: {(t2-t1):.4f}')
    exit(0)

# warm up
for _ in range(10):
    model.train()
    opti.zero_grad()
    Z = model(dl.X)
    loss = F.nll_loss(Z[train_idx], dl.y[train_idx])
    loss.backward()
    opti.step()
    GPUs = GPUtil.getGPUs()
    maxMemory = max(GPUs[args.profile_gpu].memoryUsed, maxMemory)

torch.cuda.synchronize()
start = time.time()
for epoch in range(args.epochs):
    model.train()
    opti.zero_grad()
    Z = model(dl.X)
    loss = F.nll_loss(Z[train_idx], dl.y[train_idx])
    loss.backward()
    opti.step()

torch.cuda.synchronize()
end = time.time()
trainTime = (end-start)/args.epochs

model.eval()
torch.cuda.synchronize()
start = time.time()
for _ in range(args.epochs):
    Z = model(dl.X)
A = torch.cuda.memory_summary()
torch.cuda.synchronize()
end = time.time()
# print(f"model memory {A}")
inferenceTime = (end-start)/args.epochs

train_acc = accuracy(Z[train_idx], dl.y[train_idx])
test_acc = accuracy(Z[test_idx], dl.y[test_idx])

# log acc
# best_test_acc = max(best_test_acc, test_acc)
# print(f'train acc:{train_acc:.2f} | test acc:{test_acc:.2f}')
# acc = evaluate(model, dl.features, dl.labels, dl.val_mask)
# best_test_acc = max(best_test_acc, acc)
# print(f"epoch {epoch} | loss {loss.item():.4f} | test acc {acc:.4f} ")

print(f"backend {args.backend}: avg epoch time {trainTime:.4f}")

if args.output != None:
    with open(f"{args.output}", 'a') as f:
        print(f'{args.backend},{args.model},{args.dname},nlayer={args.nlayer}, nhid={args.nhid}, nhead={args.nhead},first_aggr={args.first_aggr},{trainTime},{inferenceTime}', file=f)

# print(f"final test accuracy: {np.mean(best_test_acc)} ± {np.std(best_test_acc)}")
