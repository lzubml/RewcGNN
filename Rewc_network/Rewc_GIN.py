import torch
import torch.nn as nn
from torch.nn import ModuleList, BatchNorm1d, Conv2d, Dropout, InstanceNorm1d, LayerNorm
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool



class Rewc_GIN(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 dropout,
                 pool,
                 emb_sizes=None,
                 emb_input=-1,
                 device='cpu'
                 ):
        super(Rewc_GIN, self).__init__()
        if emb_sizes is None:
            emb_sizes = [32, 64, 64]
        self.num_features = num_features
        self.emb_sizes = emb_sizes
        self.num_layers = len(self.emb_sizes) - 1
        self.global_pool = pool
        self.emb_input = emb_input
        self.device = device
        self.Relu = nn.ReLU()


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
                Dropout(p=dropout),
            ]
        )
        self.initial_lin = Sequential(*self.initial_linear)

        cg_layers = []
        for i in range(self.num_layers):
            in_channel = emb_sizes[i]
            out_channel = emb_sizes[i + 1]
            cg_layer = GINConv(
                nn=Sequential(
                BatchNorm1d(in_channel),
                Dropout(p=dropout),
                ReLU(),
                Linear(in_channel, out_channel),
                ReLU(),
                Linear(out_channel, out_channel),
                BatchNorm1d(in_channel),
            ),
                eps=0.1, train_eps=True)
            cg_layers.append(cg_layer)

        self.cg_modules = ModuleList(cg_layers)

        conv_layers = []
        for i in range(self.num_layers+1):
            conv_layer = Conv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=(i+1, 1),
                                )
            conv_layers.append(conv_layer)
        self.conv_modules = ModuleList(conv_layers)

        self.final_conv = Conv2d(in_channels=1,
                            out_channels=1,
                            kernel_size=(self.num_layers+1, 1),
                            )

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

        aggr = torch.zeros(size=(self.num_layers + 1, x_feat.shape[0], self.emb_sizes[0]),
                           dtype=torch.float).to(self.device)
        aggr[0, :, :] = x_feat

        for i in range(1, self.num_layers + 1):
            edges = edge_index.T[edge_weights == i].T
            x_feat_i = self.cg_modules[i - 1](x_feat, edges)
            aggr[i, :, :] = self.Relu(x_feat_i)

        aggr_conv = torch.zeros(size=(self.num_layers + 1, x_feat.shape[0], 1, self.emb_sizes[0]),
                                dtype=torch.float).to(self.device)

        for i in range(self.num_layers + 1):
            x_feat_conv = self.conv_modules[i](aggr[:i + 1, :, :].permute(1, 0, 2).unsqueeze(1))
            aggr_conv[i, :, :, :] = x_feat_conv.squeeze(1) + aggr[i, :, :].unsqueeze(1)

        out = self.final_conv(aggr_conv.permute(1, 2, 0, 3)).squeeze(1, 2)

        out = self.initial_lin(self.pooling(out, batch))

        return out
