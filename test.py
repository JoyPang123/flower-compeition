import os
import glob
import PIL
from PIL import Image, ImageFile
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
import math
import numpy as np
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from functools import partial
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision.transforms import InterpolationMode
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import ttach as tta


num_classes = 219

# hyperparameter
batch_size = 64

stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

test_tfms = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


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
        

csv_file = pd.read_csv("target.csv")
test_dataset_new = FoldDataset(csv_file, test_tfms)
test_loader = DataLoader(dataset=test_dataset_new, 
                         batch_size=64,
                         shuffle=False, 
                         num_workers=2, 
                         pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_384', num_classes=219)
    def forward(self, x):
        x = self.vit(x)
        return x

model = mymodel()
model = model.to(device)
model.load_state_dict(torch.load("vit_small_patch16_384_93.61.pth"))

# tta_transform = tta.Compose([
#     tta.HorizontalFlip(),
#     tta.Multiply(factors=[0.9, 1, 1.1]),
# ])
# tta_model = tta.ClassificationTTAWrapper(model, tta_transform)
    
catego = []
model.eval()

with torch.no_grad():
    for test_image in tqdm(test_loader):
        test_image = test_image.to(device)
        
        test_output = model(test_image).argmax(-1)
        catego.extend(test_output.cpu().tolist())

csv_file["category"] = catego
csv_file.drop(columns=["path"]).to_csv("vit_s_p16_384.csv", index=False)










