import torch.nn as nn
from torch.nn import ModuleList, BatchNorm1d, Dropout
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GAT(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 dropout,
                 pool,
                 emb_sizes=None,
                 emb_input=-1,
                 device='cpu'
                 ):
        super(GAT, self).__init__()
        if emb_sizes is None:
            emb_sizes = [32, 64, 64]
        self.num_features = num_features
        self.emb_sizes = emb_sizes
        self.num_layers = len(self.emb_sizes) - 1
        self.global_pool = pool
        self.emb_input = emb_input
        self.device = device

        self.initial_mlp_modules = ModuleList(
            [
                Linear(num_features, emb_sizes[0]),
                BatchNorm1d(emb_sizes[0]),
                ReLU(),
                Linear(emb_sizes[0], emb_sizes[0]),
                BatchNorm1d(emb_sizes[0]),
                ReLU(),
                Dropout(p=dropout),
            ]
        )
        self.initial_mlp = Sequential(*self.initial_mlp_modules)

        self.initial_linear = ModuleList(
            [
                BatchNorm1d(emb_sizes[0]),
                ReLU(),
                Linear(emb_sizes[0], emb_sizes[0]),
                ReLU(),
                Linear(emb_sizes[0], num_classes),
                # ReLU(),
                Dropout(p=dropout),
            ]
        )
        self.initial_lin = Sequential(*self.initial_linear)

        cg_layers = []
        for i in range(self.num_layers):
            in_channel = emb_sizes[i]
            out_channel = emb_sizes[i + 1]
            cg_layer = GATConv(
                in_channels=in_channel, out_channels=out_channel // 2,
                heads=2,
                concat=True, dropout=0.2, bias=True
            )
            cg_layers.append(cg_layer)

        self.cg_modules = ModuleList(cg_layers)

    def pooling(self, x, batch):
        if self.global_pool == "max":
            return global_max_pool(x, batch)
        elif self.global_pool == "mean":
            return global_mean_pool(x, batch)
        elif self.global_pool == 'add' or self.global_pool == 'sum':
            return global_add_pool(x, batch)
        else:
            pass

    def forward(self, data):
        x_feat = data.x
        edge_index = data.edge_index
        edge_weights = data.edge_weights
        batch = data.batch

        x_feat = self.initial_mlp(x_feat)

        for i in range(1, self.num_layers + 1):
            edges = edge_index.T[edge_weights == 1].T
            x_feat = self.cg_modules[i - 1](x_feat, edges)

        out = self.initial_lin(self.pooling(x_feat, batch))

        return out