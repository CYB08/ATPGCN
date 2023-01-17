import torch
from torch_geometric.data import InMemoryDataset
from os import listdir
import os.path as osp
from utils.sparse_net_reader import group_data_read

class MyNetworkReader(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.subject = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        Netfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]; Netfiles.sort()
        return Netfiles

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        return

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.subject = group_data_read(self.raw_dir)

        data_list = []
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices, self.subject = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices, self.subject = self.collate(data_list)

        torch.save((self.data, self.slices, self.subject), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format('Multi_Brainnetwork', len(self))












