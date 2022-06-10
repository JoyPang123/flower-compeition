import argparse
import torch

from tqdm import tqdm

from model import ConModel
from dataset import make_loader
from utils import *


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


@torch.no_grad()        
def evaluate(model, test_loader, device):
    features, labels = [], []
    
    model.eval()
    for image, label in test_loader:
        # Load image and label to the target device
        image = image.to(device)
        label = label.to(device)
        
        # Obtain the test data features
        output = model(image)

        features.append(output)
        labels.append(label)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # Evaluate with KNN accuracy
    acc, _ = KNN(features, labels, batch_size=16)
    print("Accuracy: %.5f" % acc)  
    

def train_one_epoch(model, epoch, train_loader, criterion, optimizer, device):
    model.train()
    iters_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader, desc=f"Epochs: {epoch}")
    for idx, images in enumerate(progress_bar):
        adjust_learning_rate(optimizer, epoch + idx / iters_per_epoch, args)
        moco_m = adjust_moco_momentum(epoch + idx / iters_per_epoch, args)

        images[0] = images[0].to(device)
        images[1] = images[1].to(device)

        # compute output
        # output, target = model(images[0], images[1])
        # loss = criterion(output, target)
        loss = model(images[0], images[1], moco_m)
        loss = loss.sum()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"loss": loss.item()})
        
        
def train(model, train_loader, criterion, optimizer, args):
    max_acc = -1
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1} ", end="")
        train_one_epoch(model, epoch, train_loader, criterion, optimizer, args.device)
        # cur_acc = evaluate(model, test_loader, args.device)

    torch.save(model.state_dict(), "con.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Loader information
    parser.add_argument("--img_root", default="unlabeled_orchid", type=str)
    parser.add_argument("--label_file", default="comp_orchid/label.csv", type=str)
    
    # Model training hyper-parameters
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feature_dim", type=int, default=128)
    
    # Optmization
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument('--moco_m', type=float, default=0.999)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=10)

                        
    args = parser.parse_args()
    
    # Add device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = ConModel()
    model = nn.DataParallel(model)
    model.to(args.device)
    
    criterion = SupConLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, 
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    

    train_loader = make_loader(args.batch_size, args.img_root, "unlabeled", args.label_file)
    train(model, train_loader, criterion, optimizer, args)

