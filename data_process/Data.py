import json
import functools
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected, to_dense_adj
from pymatgen.core.structure import Structure, Molecule
from collections import defaultdict
from xenonpy.datatools import preset
from scipy.sparse.csgraph import floyd_warshall


class AtomInitializer(object):

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class CIFData(Dataset):

    def __init__(self, root_dir, max_distance=10):
        self.root_dir = root_dir
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        cif_files = [f for f in os.listdir(root_dir) if f.endswith('.cif')]
        self.cif_files = cif_files
        grouped_spacegroups = defaultdict(list)
        self.grouped_spacegroups = grouped_spacegroups
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.max_distance = max_distance

    def __len__(self):
        return len(self.cif_files)


    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id = self.cif_files[idx]
        maybe_group = []
        with open('data/space_groups/spacegroups.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                group_0 = {obj: 0 for obj in row}
        with open('data/grouped/grouped_spacegroups.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == cif_id:
                    if row[1] not in maybe_group:
                        maybe_group.append(row[1])
        for item in maybe_group:
            if item in group_0:
                group_0[item] = 1

        y = torch.Tensor(list(group_0.values()))
        # cif to graph
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id))
        # mole = Molecule.from_file(os.path.join(self.root_dir, cif_id))

        # pos
        pos = torch.from_numpy(crystal.cart_coords)
        atom_f1 = [preset.elements_completed.values[crystal[i].specie.number] for i in range(len(crystal))]
        atom_f2 = [self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]
        x = torch.tensor(np.concatenate((atom_f1, atom_f2), axis=1), dtype=torch.float32)
        source_node = []
        target_node = []
        node_num = x.shape[0]
        crystal_distance = crystal.distance_matrix
        # Fixed radius
        # all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        print(idx)
        for i, atom in enumerate(crystal):
            nbrs = crystal.get_neighbors(site=atom, r=crystal[i].specie.van_der_waals_radius, include_index=True)
            for nbr in nbrs:
                source_node.append(i)
                target_node.append(nbr.index)
        edge_index = torch.Tensor([source_node, target_node]).to(torch.long)
        original_edge_index = to_undirected(edge_index)
        adj_matrix = to_dense_adj(original_edge_index, max_num_nodes=node_num)[0]
        shortest_paths = torch.tensor(floyd_warshall(adj_matrix), dtype=torch.long)
        edge_index = torch.zeros(size=(2, node_num * node_num), dtype=torch.long)
        edge_index[0, :] = torch.arange(node_num).repeat(node_num)
        edge_index[1, :] = torch.repeat_interleave(torch.arange(node_num), node_num)
        edge_weights = shortest_paths.flatten()
        if self.max_distance:
            edge_mask_dist = edge_weights <= self.max_distance
            edge_weights = edge_weights[edge_mask_dist]
            edge_index = edge_index[:, edge_mask_dist]

        edge_fea = []
        for s in range(len(edge_index[0])):
            edge_fea.append(crystal_distance[edge_index[0][s]][edge_index[1][s]])
        edge_attr = torch.tensor(edge_fea, dtype=torch.float)
        idx = cif_id.replace(".cif.cif", "")
        print(idx)
        return (x, y, edge_index, edge_weights, edge_attr, pos, idx)
