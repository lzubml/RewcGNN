from network.CGCNN import CGCNN
from network.GCN import GCN
from network.GIN import GIN
from network.GAT import GAT
from Rewc_network.Rewc_GCN import Rewc_GCN
from Rewc_network.Rewc_CGCNN import Rewc_CGCNN
from Rewc_network.Rewc_GIN import Rewc_GIN
from Rewc_network.Rewc_GAT import Rewc_GAT

def get_model(args, device="cpu"):
    if (args.Rewc == False):
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        emb_input = -1
        if args.model == 'CGCNN':
            model = CGCNN(
                num_features=args.num_features,
                num_classes=args.num_classes,
                dropout=args.dropout,
                pool=args.global_pool,
                emb_sizes=emb_sizes,
                emb_input=emb_input,
                device=args.device
            ).to(device)
        elif args.model == 'GCN':
            model = GCN(
                num_features=args.num_features,
                num_classes=args.num_classes,
                dropout=args.dropout,
                pool=args.global_pool,
                emb_sizes=emb_sizes,
                emb_input=emb_input,
                device=args.device
            ).to(device)
        elif args.model == 'GIN':
            model = GIN(
                num_features=args.num_features,
                num_classes=args.num_classes,
                dropout=args.dropout,
                pool=args.global_pool,
                emb_sizes=emb_sizes,
                emb_input=emb_input,
                device=args.device
            ).to(device)
        elif args.model == 'GAT':
            model = GAT(
                num_features=args.num_features,
                num_classes=args.num_classes,
                dropout=args.dropout,
                pool=args.global_pool,
                emb_sizes=emb_sizes,
                emb_input=emb_input,
                device=args.device
            ).to(device)
        return model
    elif (args.Rewc == True):
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        emb_input = -1
        if args.model == 'GCN':
            model = Rewc_GCN(
                num_features=args.num_features,
                num_classes=args.num_classes,
                dropout=args.dropout,
                pool=args.global_pool,
                emb_sizes=emb_sizes,
                emb_input=emb_input,
                device=args.device
            ).to(device)
        elif args.model == 'CGCNN':
            model = Rewc_CGCNN(
                num_features=args.num_features,
                num_classes=args.num_classes,
                dropout=args.dropout,
                pool=args.global_pool,
                emb_sizes=emb_sizes,
                emb_input=emb_input,
                device=args.device
            ).to(device)
        elif args.model == 'GIN':
            model = Rewc_GIN(
                num_features=args.num_features,
                num_classes=args.num_classes,
                dropout=args.dropout,
                pool=args.global_pool,
                emb_sizes=emb_sizes,
                emb_input=emb_input,
                device=args.device
            ).to(device)
        elif args.model == 'GAT':
            model = Rewc_GAT(
                num_features=args.num_features,
                num_classes=args.num_classes,
                dropout=args.dropout,
                pool=args.global_pool,
                emb_sizes=emb_sizes,
                emb_input=emb_input,
                device=args.device
            ).to(device)
        return model