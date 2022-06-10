import argparse

import torch
from torchvision import transforms

from timm.data.auto_augment import rand_augment_transform
from tqdm import tqdm

from model import ConLinearModel
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


def load_weight(weight_file):
    weight = torch.load(weight_file)
    new_dict = {}
    
    for key, value in weight.items():
        if "module.q" in key:
            new_key = key.replace("module.", "")
            new_key = new_key.replace("q", "model", 1)
            new_dict[new_key] = value

    return new_dict


@torch.no_grad()        
def evaluate(model, test_loader, device):
    total_accuracy = 0
    
    model.eval()
    for image, label in test_loader:
        # Load image and label to the target device
        image = image.to(device)
        label = label.to(device)
        
        # Obtain the test data features
        output = model(image)

        predictions = output.argmax(dim=-1)
        total_accuracy += (predictions == label).sum() / label.shape[0]
        
    total_accuracy /= len(test_loader)
    print(f"Accuracy: {total_accuracy:.5f}")  


def train_one_epoch(model, epoch, train_loader, criterion, optimizer, args):
    model.train()
    loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}")
    for idx, (images, label) in enumerate(loader):
        # warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        
        # Load image and label to the target device
        images = images.to(args.device)
        label = label.to(args.device)
        
        # Obtain the loss with the feature embeddings
        outputs = model(images)
        loss = criterion(outputs, label)
        
        # Update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loader.set_postfix({"loss": loss.item()})

        
class FoldDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.img_root = csv_file
        self.transform = transform
    
    def __len__(self):
        return len(self.img_root)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_root.iloc[idx, 0]).convert("RGB")
        img = self.transform(img)
        return img
        

def train(model, train_loader, test_loader, criterion, optimizer, args):
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        train_one_epoch(model, epoch, train_loader, criterion, optimizer, args)
        evaluate(model, test_loader, args.device)
        
    torch.save(model.state_dict(), "li.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Loader information
    parser.add_argument("--img_root", type=str, default="comp_orchid")
    parser.add_argument("--label_file", type=str, default="comp_orchid/label.csv")
    parser.add_argument("--weight", type=str, default="con.pt")
    
    # Model training hyper-parameters
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--feature_dim", type=int, default=128)
    
    # Optmization
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument('--lr_decay_rate', type=float, default=0.2)
                        
    args = parser.parse_args()
    
    # Add device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    # Set up model (with pretrained weight), criterion, and optimizer
    model = ConLinearModel()
    new_weight = load_weight(args.weight)    
    missing_keys = model.load_state_dict(new_weight, strict=False)
    print(missing_keys)
    model = torch.nn.DataParallel(model).to(args.device)
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    train_file, test_file = split_train_test(args.label_file)
    
    # Set up train and test loader
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    rand_aug = rand_augment_transform(
        config_str="rand-m9-mstd0.5-inc1",
        hparams=dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in (0.5071, 0.4867, 0.4408)]),
        )
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=384, scale=(0.5, 1.)),
        rand_aug,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_loader = make_loader(
        args.batch_size, args.img_root, "labeled", train_file,
        transform=train_transform
    )
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        normalize
    ])
    test_loader = make_loader(
        args.batch_size, args.img_root, "labeled", test_file,
        transform=test_transform, shuffle=False
    )
    
    train(model, train_loader, test_loader, criterion, optimizer, args)

    

