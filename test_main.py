import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from data_process.dataset import CIFDataset
from config.configs import args
from utils.model_loader import get_model


def train():
    model.train()
    correct = 0
    total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        data.y = data.y.reshape(-1, 41).to(device)
        loss = criterion(out, data.y.to(device))
        _, predicted = torch.max(out, 1)
        total += data.y.size(0)
        real = torch.max(data.y, 1)[1]
        correct += torch.eq(predicted, real).sum().item()
        loss.backward()
        optimizer.step()
    if epoch >= 100:
        scheduler.step()
    accuracy = correct / total
    print('\n{}---------------------------------------------------------------'.format(epoch))
    print('Train Accuracy: {:.2f}%\n'.format(accuracy * 100))

def eval():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            out = model(data)
            _, predicted = torch.max(out, 1)
            data.y = data.y.reshape(-1, 41).to(device)
            total += data.y.size(0)
            real = torch.max(data.y, 1)[1]
            correct += torch.eq(predicted, real).sum().item()
    accuracy = correct / total
    print('Accuracy: {:.2f}%\n'.format(accuracy * 100))
    return accuracy

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            _, predicted = torch.max(out, 1)
            data.y = data.y.reshape(-1, 41).to(device)
            total += data.y.size(0)
            real = torch.max(data.y, 1)[1]
            correct += torch.eq(predicted, real).sum().item()
    accuracy = correct / total
    return accuracy


dataset_root = args.dataset_root
save_path = args.save_path
lr = args.lr
epochs = args.epoches
seed = args.seed
batch_size = args.batch_size
train_size = args.train_size
valid_size = args.valid_size
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
device = args.device

dataset = CIFDataset(dataset_root)
dataset = dataset.shuffle()
train_size = int(len(dataset)*train_size)
valid_size = int(len(dataset)*valid_size)
test_size = int(len(dataset) - train_size - valid_size)
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
model = get_model(args, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
criterion = torch.nn.CrossEntropyLoss()

target = {}
best_acc = 0.0
for epoch in range(epochs):
    train()
    target[epoch] = eval()
    acc = list(target.values())[epoch]
    if acc > best_acc:
        best_acc = acc
        best_params = model.state_dict()
    if epoch == (epochs-1):
        max_acc = max(target.values())
        max_acc_epoch = max(target, key=target.get)
        torch.save(best_params, save_path)
        print('Final Result----------------------------------------------------------------------------------------------------------------------------------------------\n')
        print('Max Valid epoch:{:.2f}\n'.format(epoch))
        model.load_state_dict(best_params)
        test_acc = test()
        print('Test Accuracy: {:.2f}%\n'.format(test_acc))