import torch
import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
import dgl
from scipy.sparse import diags
from HyperGsys.balancer import balance_schedule
import dgl.sparse as dglsp

class HyperGraph:
    def __init__(self, data, device, data_name):
        self.device = device
        self.data_name = data_name
        self.num_nodes = data.x.shape[0]
        c_idx = torch.where(data.edge_index[0] == self.num_nodes)[0].min()
        V2E = data.edge_index[:, :c_idx]
        V = V2E[0]
        E = V2E[1] - self.num_nodes
        self.num_edges = len(V2E[1].unique())
        self.nnz = V2E.shape[1]
        
        # convert to sparse matrix
        data = np.ones(self.nnz)
        H = sp.coo_matrix((data, (V, E)), shape=(self.num_nodes, self.num_edges)).tocsr()
        H_T = H.transpose().tocsr()
        self.H = H
        self.H_T = H_T

        V = V.to(device)
        E = E.to(device)

        self.V, self.E = V, E
        # Compute degV and degE
        degV = H.sum(axis=1)
        degE = H.sum(axis=0)
        degV = torch.from_numpy(degV).float()
        degE = torch.from_numpy(degE).squeeze()
        degE = degE.unsqueeze(1).float()

        degV = degV.pow(-0.5)
        degE = degE.pow(-1)
        degD = degV.pow(-1)

        # when not added self-loop, some nodes might not be connected with any edge
        degV[torch.isinf(degV)] = 1

        self.degV = degV.to(device)
        self.degE = degE.to(device)
        self.degD = degD.to(device)
        
        dia_mat_V = diags(degV.squeeze().numpy(), 0)
        dia_mat_E = diags(degE.squeeze().numpy(), 0)
        L = dia_mat_V @ H @ dia_mat_E @ H_T @ dia_mat_V
        L = L.tocoo()
        L_coo = torch.stack((torch.from_numpy(L.row), torch.from_numpy(L.col)))
        self.L = dglsp.spmatrix(L_coo).to(device)

        
        H_csrptr = torch.from_numpy(self.H.indptr)
        H_colind = torch.from_numpy(self.H.indices)
        H_data = torch.from_numpy(self.H.data).float()
        H_T_csrptr = torch.from_numpy(self.H_T.indptr)
        H_T_colind = torch.from_numpy(self.H_T.indices)
        H_T_data = torch.from_numpy(self.H_T.data).float()
        self.H_csrptr = H_csrptr.to(device)
        self.H_colind = H_colind.to(device)
        self.H_data = H_data.to(device)
        self.H_T_csrptr = H_T_csrptr.to(device)
        self.H_T_colind = H_T_colind.to(device)
        self.H_T_data = H_T_data.to(device)

        self.adj_g1 = self.H_csrptr, self.H_colind, self.H_data
        self.adj_g2 = self.H_T_csrptr, self.H_T_colind, self.H_T_data
        partition_dict = {"yelp":400, "20newsW100":400, "coauthor_cora":10, "zoo":20, "NTU2012":80, "cora":210, "pubmed":40, "Mushroom":250, 
                  "coauthor_dblp":80, "house-committees":40, "walmart-trips":210, "citeseer":6, "ModelNet40":300}
        ngs = partition_dict[data_name] 
        self.balance(ngs, H_T_csrptr)

    def store_mtx(self, path):
        file_name = path + self.data_name + '.mtx'
        mmwrite(file_name, self.H)

    def store_laplacian_mtx(self, path):
        file_name = path + self.data_name + '.mtx'
        mmwrite(file_name, self.L)

    def dgl_prepare(self):
        # contruct dgl graph input
        g1 = dgl.bipartite_from_scipy(
            self.H, utype='vertex', etype='none', vtype='edge')
        g2 = dgl.bipartite_from_scipy(
            self.H_T, utype='edge', etype='none', vtype='vertex')
        self.g1 = g1.to(self.device)
        self.g2 = g2.to(self.device)

    def balance(self, ngs, H_T_csrptr):
        bs = balance_schedule(ngs, H_T_csrptr)
        self.group_start = torch.Tensor(bs.group_st).int().to(self.device)
        self.group_end = torch.Tensor(bs.group_ed).int().to(self.device)
        self.group_key = torch.Tensor(bs.balan_key).int().to(self.device)
        self.group_row = torch.Tensor(bs.balan_row).int().to(self.device)