import torch
from torch_geometric.data import Data, InMemoryDataset
from data_process.Data import CIFData

class CIFDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CIFDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['graph_cova_cif.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for cif in cifdata:
            x = cif[0]
            y = cif[1]
            edge_index = cif[2]
            edge_weights = cif[3]
            edge_attr = cif[4]
            pos = cif[5]
            idx = cif[6]
            data = Data(x=x, y=y, edge_index=edge_index, edge_weights=edge_weights, edge_attr=edge_attr, pos=pos, idx=idx)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

cifdata = CIFData(root_dir='data/succ', max_distance=10)

