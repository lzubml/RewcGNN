import argparse
import torch

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", help="The root of dataset.", default="datasets/vand_fa_dataset", type=str)
parser.add_argument("--save_path", help="The path to save parm.", default='parms/cgcnn_conv_10/accuracy0.7025495750708215.pth', type=str)
parser.add_argument("--model", help="The model we will use.", default="CGCNN")
parser.add_argument("--epoches", help="Train epoches.", default=100, type=int)
parser.add_argument("--lr", help="Learning_rate.", default=8e-5, type=float)
parser.add_argument("--batch_size", help="Batch_size.", default=32, type=int)
parser.add_argument("--max_distance", help="Maximal distance in HSP model (K)", default=10, type=int)
parser.add_argument("--train_size", help="Size of train dataset.", default=0.7, type=float)
parser.add_argument("--valid_size", help="Size of valid dataset.", default=0.2, type=float)
parser.add_argument("--emb_dim", help="Size of the emb dimension.", default=256, type=int)
parser.add_argument("--dropout", help="Dropout probability.", default=0.2, type=float)
parser.add_argument("--device", help="use cpu or gpu.", default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
parser.add_argument("--num_classes", help="Number of dataset classes.", default=41, type=int)
parser.add_argument("--num_features", help="Number of dataset features numbers.", default=150, type=int)
parser.add_argument("--hiden_channels", help="The channels of hiden layer.", default=256, type=int)
parser.add_argument("--global_pool", help="What pooling method to use.", default="add", type=str)
parser.add_argument("--num_layers", help="Number of layers in the model.", default=10, type=int)
# 10  68

parser.add_argument(
    "--Rewc",
    help="Whether to use deeper GNN",
    type=str2bool,
    default=True,
)

parser.add_argument(
    "--seed",
    help="Starting seed",
    type=int,
    default=1234,
)

args = parser.parse_args()