# This code is developed based on Xiaoxiao Li, 2019/02/24
# The original version is at https://github.com/xxlya/BrainGNN_Pytorch/blob/main/imports/read_abide_stats_parall.py

import os.path as osp
from os import listdir
import os
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import scipy.io as scio
from nilearn import connectome
from tqdm import tqdm

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])
    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

def group_data_read(data_dir):
    matfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    matfiles.sort()
    Fun_dir = os.path.join(os.getcwd(), "data/Functional")
    Funfiles = [m for m in listdir(Fun_dir) if osp.isfile(osp.join(Fun_dir, m))]
    Funfiles.sort()

    batch = []
    pseudo = []
    y_list = []
    temp=[]
    subject=[]
    edge_att_list, edge_index_list,att_list = [], [], []
    for network_name, timeseries_name in tqdm(zip(matfiles, Funfiles)):
        res = individual_data_read(data_dir, network_name, Fun_dir, timeseries_name)
        temp.append(res)

    for j in range(len(temp)):
        edge_att_list.append(temp[j][0])
        edge_index_list.append(temp[j][1]+j*temp[j][4])
        att_list.append(temp[j][2])
        y_list.append(temp[j][3])
        batch.append([j]*temp[j][4])
        pseudo.append(np.diag(np.ones(temp[j][4])))
        subject.append(temp[j][5])

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch)

    data, slices = split(data, batch_torch)

    return data, slices, subject


def individual_data_read(net_dir, netfile, series_dir, seriesfile,
                         variable_Fun='ROISignals',
                         variable_bn = 'Brainnetwork'):

    SR_brain = scio.loadmat(osp.join(net_dir, netfile))[variable_bn]
    Fun_series = scio.loadmat(osp.join(series_dir, seriesfile))[variable_Fun]
    Sub_name_curr = netfile[:5]
    conn_measure = connectome.ConnectivityMeasure(kind='partial correlation')
    connectivity = conn_measure.fit_transform([Fun_series])

    Sub = os.path.join(os.getcwd(), "data", "Phenotypic_V1_0b_preprocessed1.csv")
    if not os.path.isfile(Sub):
        print(Sub + 'does not exist!')
    else:
        if Sub.endswith('.csv'):
            Sub_gro = np.genfromtxt(Sub, dtype=str,
                                      delimiter=',',
                                      skip_header=1,
                                      usecols=(2, 7))

    Sub_label = [label for idx, label in Sub_gro if idx == Sub_name_curr]
    # from adj matrix to too matrix
    num_nodes = SR_brain.shape[0]
    G = from_numpy_matrix(SR_brain)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = SR_brain[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,num_nodes)
    att = np.transpose(connectivity[0])
    label = np.array([int(x) for x in Sub_label])

    return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes,Sub_name_curr
