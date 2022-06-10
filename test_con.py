import pandas as pd
import ttach as tta

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image
from timm.data.auto_augment import rand_augment_transform
from tqdm import tqdm

from model import ConLinearModel


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
    

def load_weight(weight_file):
    weight = torch.load(weight_file)
    new_dict = {}
    
    for key, value in weight.items():
        new_key = key.replace("module.", "")
        new_dict[new_key] = value

    return new_dict


@torch.no_grad()        
def evaluate(model, test_loader, device):
    catego = []
    
    model.eval()
    for image in tqdm(test_loader):
        # Load image and label to the target device
        image = image.to(device)
        
        # Obtain the test data features
        output = model(image)
        catego.extend(output.argmax(dim=-1).cpu().tolist())
        
    return catego


if __name__ == "__main__":
    
    # Add device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    # Set up model (with pretrained weight), criterion, and optimizer
    model = ConLinearModel().to(device)
    missing_keys = model.load_state_dict(load_weight("li.pt"))
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1, 1.1]),
    ])
    tta_model = tta.ClassificationTTAWrapper(model, tta_transform)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        normalize
    ])
    
    csv_file = pd.read_csv("target.csv")
    test_dataset = FoldDataset(csv_file, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    catego = evaluate(tta_model, test_loader, device)
    
    csv_file["category"] = catego
    csv_file.drop(columns=["path"]).to_csv("tta_384.csv", index=False)
    

