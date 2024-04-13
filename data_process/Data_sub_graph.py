from __future__ import print_function, division
import json
import functools
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected, to_dense_adj, to_scipy_sparse_matrix, add_self_loops
from pymatgen.core.structure import Structure
from collections import defaultdict
from xenonpy.datatools import preset
import networkx as nx
from config.configs import args
from copy import deepcopy as c

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
        #初始化特征,从元素编号到原子特征的向量表示，字典格式
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        #取原子种类编号，从1到100
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        #把特征字典转换成np阵列
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class peripheral_attr(object):
    def extract_peripheral_attr_v2(
                                   adj,
                                   k_adj,
                                   max_hop_num,
                                   max_edge_type,
                                   max_edge_count,
                                   max_distance_count,
                                   ):
        num_nodes = adj.size(0)

        # component_dim=max_component_num
        # record peripheral edge information
        edge_matrix_size = [num_nodes, max_edge_type, 2]
        peripheral_edge_matrix = torch.zeros(edge_matrix_size, dtype=torch.long)
        # record node configuration
        configuration_matrix_size = [num_nodes, max_hop_num + 1]
        peripheral_configuration_matrix = torch.zeros(configuration_matrix_size, dtype=torch.long)
        for i in range(num_nodes):
            row = torch.where(k_adj[i] > 0)[0]
            # subgrapb with less than 2 nodes, no edges, thus skip
            num_sub_nodes = row.size(-1)
            if num_sub_nodes < 2:
                continue
            peripheral_subgraph = adj[row][:, row]
            s = nx.from_numpy_array(peripheral_subgraph.numpy(), create_using=nx.DiGraph)
            s_edge_list = list(nx.get_edge_attributes(s, "weight").values())
            if len(s_edge_list) == 0:
                continue
            s_edge_list = torch.tensor(s_edge_list).long()
            edge_count = torch.bincount(s_edge_list, minlength=max_edge_type + 2)  # k阶邻居的边数量，由于构造了有向图，面对无向边时采取双向两条边描述
            # remove 0 and 1
            edge_count = edge_count[2:]  # 边权重小于2的被过滤 权重为1表示直接连接的点
            sort_count, sort_type = torch.sort(edge_count,
                                               descending=True)  # count表示边数量 type表示边的索引即子图中边的权重 由于子图是基于一阶邻接矩阵的，所以需过滤掉权重为0和1的边
            sort_count = sort_count[:max_edge_type]
            sort_type = sort_type[:max_edge_type]
            sort_count[sort_count > max_edge_count] = max_edge_count
            peripheral_edge_matrix[i, :, 0] = sort_type  # 子图中边的权重
            peripheral_edge_matrix[i, :, 1] = sort_count  # 子图中边的数量
            shortest_path_matrix = peripheral_attr.nx_compute_shortest_path_length(s, max_length=max_hop_num)  # 子图中各条边之间的最短路径
            num_sub_p_edges = 0
            for j in range(num_sub_nodes):
                for h in range(1, max_hop_num + 1):
                    h_nodes = torch.where(shortest_path_matrix[j] == h)[0]  # 与j节点最短距离为h的子图节点索引
                    if h_nodes.size(-1) < 2:
                        continue
                    else:
                        pp_subgraph = peripheral_subgraph[h_nodes][:, h_nodes]  # 与j节点最小距离为h的子图中的节点组成的子图的邻接矩阵
                        num_sub_p_edges += torch.sum(pp_subgraph)  # 子图中的子图的边的数量

            configuration_feature = torch.bincount(shortest_path_matrix.view(-1),
                                                   minlength=max_hop_num + 1)  # k-hop中不同k对应的最短距离的边的数量
            # configuration_feature=configuration_feature[1:]
            configuration_feature[0] = num_sub_p_edges  # 用子图的子图中的边的数量取代子图中无连接关系的特征
            configuration_feature[
                configuration_feature > max_distance_count] = max_distance_count  # 最大子图边数量为max_distance_count
            peripheral_configuration_matrix[i, :] = configuration_feature  # [节点数， k+1] j节点的距离为k+1的子图中的边数量特征
        return peripheral_edge_matrix.unsqueeze(0), peripheral_configuration_matrix.unsqueeze(0)

    def get_peripheral_attr(self,
                            adj_list,
                            edge_attr_adj,
                            max_hop_num,
                            max_edge_type,
                            max_edge_count,
                            max_distance_count,
                            ):
        K = len(adj_list)
        if max_distance > 0:
            peripheral_edge_matrix_list = []
            peripheral_configuration_matrix_list = []
            for i in range(K):
                adj_ = c(adj_list[i])
                peripheral_edge_matrix, peripheral_configuration_matrix = peripheral_attr.extract_peripheral_attr_v2(
                                                                                                     edge_attr_adj,
                                                                                                     adj_,
                                                                                                     max_hop_num,
                                                                                                     max_edge_type,
                                                                                                     max_edge_count,
                                                                                                     max_distance_count,
                                                                                                     )
                peripheral_edge_matrix_list.append(peripheral_edge_matrix)
                peripheral_configuration_matrix_list.append(peripheral_configuration_matrix)

            peripheral_edge_attr = torch.cat(peripheral_edge_matrix_list, dim=0)
            peripheral_configuration_attr = torch.cat(peripheral_configuration_matrix_list, dim=0)
            peripheral_edge_attr = peripheral_edge_attr.transpose(0, 1)  # N * K * c * f
            peripheral_configuration_attr = peripheral_configuration_attr.transpose(0, 1)  # N * K * c * f
        else:
            peripheral_edge_attr = None
            peripheral_configuration_attr = None

        return peripheral_edge_attr, peripheral_configuration_attr

    def nx_compute_shortest_path_length(
                                        G,
                                        max_length,
                                        ):
        num_node = G.number_of_nodes()
        shortest_path_length_matrix = torch.zeros([num_node, num_node]).int()
        all_shortest_path_lengths = nx.all_pairs_shortest_path_length(G, max_length)
        for shortest_path_lengths in all_shortest_path_lengths:
            index, path_lengths = shortest_path_lengths
            for end_node, path_length in path_lengths.items():
                if end_node == index:
                    continue
                else:
                    shortest_path_length_matrix[index, end_node] = path_length
        return shortest_path_length_matrix

    def adj_K_order(self, adj, K):
        """compute the K order of adjacency given scipy matrix
        adj (coo_matrix): adjacency matrix
        K (int): number of hop
        """
        adj_list = [c(adj)]
        for i in range(K - 1):
            adj_ = adj_list[-1] @ adj
            adj_list.append(adj_) # adj的k-1次方
        for i, adj_ in enumerate(adj_list):
            adj_ = torch.from_numpy(adj_.toarray()).int()
            # prevent the precision overflow
            # adj_[adj_<0]=1e8
            adj_.fill_diagonal_(0) # 邻接矩阵对角线元素设为0 即取消自环
            adj_list[i] = adj_
        return adj_list


class CIFData(Dataset):

    def __init__(self,
                 root_dir,
                 max_distance=5,
                 max_edge_count=30,
                 max_edge_type=1,
                 max_distance_count=50,
                 K=2,
                 kernel='gd',
                 max_edge_attr_num=30,
                 ):
        self.root_dir = root_dir
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        cif_files = [f for f in os.listdir(root_dir) if f.endswith('.cif')]
        self.cif_files = cif_files
        grouped_spacegroups = defaultdict(list)
        self.grouped_spacegroups = grouped_spacegroups
        #初始化原子信息
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        #获得特征的numpy矩阵
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.peripheral = peripheral_attr()
        self.max_distance = max_distance
        self.max_edge_count = max_edge_count
        self.max_distance_count = max_distance_count
        self.max_edge_type = max_edge_type
        self.K = K
        self.kernel = kernel
        self.max_edge_attr_num = max_edge_attr_num

    def __len__(self):
        return len(self.cif_files)

    @functools.lru_cache(maxsize=None)  #构造图结构
    def __getitem__(self, idx):
        cif_id = self.cif_files[idx]
        # 打开csv文件
        maybe_group = []
        with open('D:\GNN_SOURCE\crystalgraph\data\space_groups\spacegroups.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # 逐行读取csv文件
            for row in reader:
                #group_0作为一个字典，键保存所有的空间群名称，值全为0
                group_0 = {obj: 0 for obj in row}
        with open('D:\GNN_SOURCE\crystalgraph\data\grouped\grouped_spacegroups.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == cif_id:
                    if row[1] not in maybe_group:
                        maybe_group.append(row[1])
        for item in maybe_group:
            # 检查item是否是group_0的键
            if item in group_0:
                # 如果是，将group_0中对应的值设置为1
                group_0[item] = 1

        y = torch.Tensor(list(group_0.values()))
        #cif转图结构
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id))

        #pos表示三维坐标，这里使用了笛卡尔坐标系
        # pos = torch.from_numpy(crystal.cart_coords)
        #取被选中晶体的原子特征，特征索引时原子序数
        atom_f1 = [preset.elements_completed.values[crystal[i].specie.number] for i in range(len(crystal))]
        atom_f2 = [self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]
        x = torch.tensor(np.concatenate((atom_f1, atom_f2), axis=1), dtype=torch.float32)
        # 获取邻居节点,遍历被选中节点半径在radius以内的其他节点作为邻居
        source_node = []
        target_node = []
        node_num = x.shape[0]
        crystal_distance = crystal.distance_matrix


        for i, atom_1 in enumerate(crystal):
            for j in range(i+1, len(crystal)):
                vand_distance = crystal[i].specie.van_der_waals_radius + crystal[j].specie.van_der_waals_radius
                if (crystal_distance[i][j] <= vand_distance) and (crystal_distance[i][j] != 0):
                    source_node.append(i)
                    target_node.append(j)

        edge_index = torch.Tensor([source_node, target_node]).to(torch.long)
        edge_attr = (torch.ones([edge_index.size(-1)]) * 2).long()

        # k-hop邻接矩阵
        # original_edge_index = to_undirected(edge_index)
        # adj_matrix = to_dense_adj(original_edge_index, max_num_nodes=node_num)[0]
        # shortest_paths = torch.tensor(floyd_warshall(adj_matrix), dtype=torch.long)
        # edge_index = torch.zeros(size=(2, node_num * node_num), dtype=torch.long)
        # edge_index[0, :] = torch.arange(node_num).repeat(node_num)
        # edge_index[1, :] = torch.repeat_interleave(torch.arange(node_num), node_num)
        # edge_weights = shortest_paths.flatten()
        # edge_weights = torch.clamp(edge_weights, min=0).long()
        # if self.max_distance:
        #     edge_mask_dist = edge_weights <= self.max_distance
        #     edge_weights = edge_weights[edge_mask_dist]
        #     edge_index = edge_index[:, edge_mask_dist]

        adj = to_scipy_sparse_matrix(edge_index, num_nodes=node_num)
        edge_attr_adj = torch.from_numpy(to_scipy_sparse_matrix(edge_index, edge_attr, node_num).toarray()).long()
        adj_list = self.peripheral.adj_K_order(adj, self.K) # 取消了自环的邻接矩阵的K-1次方

        if self.kernel == "gd":
            # create K-hop edge with graph diffusion kernel
            final_adj = 0
            for adj_ in adj_list:
                final_adj += adj_
            final_adj[final_adj > 1] = 1
        else:
            # process adj list to generate shortest path distance matrix with path number
            exist_adj = c(adj_list[0])
            for i in range(1, len(adj_list)):
                adj_ = c(adj_list[i])
                # mask all the edge that already exist in previous hops
                adj_[exist_adj > 0] = 0
                exist_adj = exist_adj + adj_
                exist_adj[exist_adj > 1] = 1
                adj_list[i] = adj_
            # create K-hop edge with sortest path distance kernel
            final_adj = exist_adj
        g = nx.from_numpy_array(final_adj.numpy(), create_using=nx.DiGraph)
        edge_list = g.edges
        edge_index_n = torch.from_numpy(np.array(edge_list).T).long()

        hop1_edge_attr = edge_attr_adj[edge_index_n[0, :], edge_index_n[1, :]]
        edge_attr_list = [hop1_edge_attr.unsqueeze(-1)]
        pe_attr_list = []
        for i in range(1, len(adj_list)):
            adj_ = c(adj_list[i])
            adj_[adj_ > self.max_edge_attr_num] = self.max_edge_attr_num
            # skip 1 as it is the self-loop defined in the model
            adj_[adj_ > 0] = adj_[adj_ > 0] + 1
            adj_ = adj_.long()
            hopk_edge_attr = adj_[edge_index_n[0, :], edge_index_n[1, :]].unsqueeze(-1)
            edge_attr_list.append(hopk_edge_attr)
            pe_attr_list.append(torch.diag(adj_).unsqueeze(-1))
        edge_attr = torch.cat(edge_attr_list, dim=-1)  # E * K
        if self.K > 1:
            pe_attr = torch.cat(pe_attr_list, dim=-1)  # N * K-1
        else:
            pe_attr = None

        peripheral_edge_attr, peripheral_configuration_attr = self.peripheral.get_peripheral_attr(
            adj_list,
            edge_attr_adj,
            self.max_distance,
            self.max_edge_type,
            self.max_edge_count,
            self.max_distance_count,
        )

        if max_distance > 1:
            pe_attr = torch.cat(pe_attr_list, dim=-1)

        idx = cif_id.replace(".cif.cif", "")
        print(idx)
        return (x, y, edge_index, edge_attr, idx, peripheral_edge_attr, peripheral_configuration_attr,  pe_attr)


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

data = cifdata[0]
print(data)
