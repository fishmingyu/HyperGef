#!/usr/bin/env python
# coding: utf-8

import torch
import pickle
import os
import ipdb

import numpy as np
import os.path as osp
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch_sparse
from torch_scatter import scatter_add, scatter
import dgl
from HyperGsys.util import *
from HyperGsys.hypergraph import HyperGraph

class Dataloader(InMemoryDataset):
    def __init__(self, args, root = 'data/', name = None, 
                 p2raw = None,
                 train_percent = 0.01,
                 feature_noise = None,
                 transform=None, pre_transform=None):
        
        existing_dataset = ['20newsW100', 'ModelNet40', 'zoo', 
                            'NTU2012', 'Mushroom', 
                            'coauthor_cora', 'coauthor_dblp',
                            'yelp', 'walmart-trips', 'house-committees',
                            'cora', 'citeseer', 'pubmed']
        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name
        
        self.feature_noise = feature_noise
        self._train_percent = train_percent

        
        if not osp.isdir(root):
            os.makedirs(root)
            
        self.root = root
        self.myraw_dir = osp.join(root, self.name, 'raw')
        self.myprocessed_dir = osp.join(root, self.name, 'processed')
        
        super(Dataloader, self).__init__(osp.join(root, name), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.train_percent = self.data.train_percent
        self.args = args

        self.transform_data()
        self.hyperg = self.prepare_graph()
        self.X = self.data.x.to(args.device)
        self.y = self.data.y.to(args.device)
        

    def transform_data(self):
        if self.args.dname in ['ModelNet40', 'zoo', 'yelp', 'walmart-trips', 'house-committees']:
        #   Shift the y label to start with 0
            self.data.y = self.data.y - self.data.y.min()
        if not hasattr(self.data, 'n_x'):
            self.data.n_x = torch.tensor([self.data.x.shape[0]])
        if not hasattr(self.data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            self.data.num_hyperedges = torch.tensor(
                [self.data.edge_index[0].max() - self.data.n_x[0]+1])

    def prepare_graph(self):
        data_name = self.name
        data = self.data
        device = self.args.device
        hyperg = HyperGraph(data, device, data_name)
        hyperg.dgl_prepare()
        return hyperg

        # convert to sparse m
        
    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'{self.name}_noise_{self.feature_noise}']
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'data_noise_{self.feature_noise}.pt']
        else:
            file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)