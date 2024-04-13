from __future__ import print_function, division
import torch
from torch_geometric.data import Data, InMemoryDataset
from typing import Optional
# from data_cova import CIFData
from data_process.Data_sub_graph import CIFData
from typing import Callable
from config.configs import args


class CIFDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform: Optional[Callable] = None):
        super(CIFDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['subgraph_cif.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for cif in cifdata:
            x = cif[0]
            y = cif[1]
            edge_index = cif[2]
            edge_weights = cif[3]
            idx = cif[4]
            peripheral_edge_attr = cif[5]
            peripheral_configuration_attr = cif[6]
            pe_attr = cif[7]
            data = Data(x=x,
                        y=y,
                        edge_index=edge_index,
                        idx=idx,
                        peripheral_edge_attr=peripheral_edge_attr,
                        peripheral_configuration_attr=peripheral_configuration_attr,
                        pe_attr=pe_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

max_distance = args.max_distance
max_edge_count = args.max_edge_count
max_distance_count = args.max_distance_count
max_edge_type = args.max_edge_type
K = args.K
kernel = args.kernel
max_edge_attr_num = args.max_edge_attr_num

cifdata = CIFData(
                  root_dir='D:\GNN_SOURCE\crystalgraph\data\pro_try',
                  max_distance=max_distance,
                  max_distance_count=max_distance_count,
                  max_edge_type=max_edge_type,
                  max_edge_count=max_edge_count,
                  K=K,
                  kernel=kernel,
                  max_edge_attr_num=max_edge_attr_num,
                  )
# dataset = CIFDataset('D:\GNN_SOURCE\crystalgraph\dataset\sub_graph_dataset')


