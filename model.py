from copy import deepcopy

import torch
import torch.nn as nn

from torchvision import models
import timm


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name="resnet34", pretrained=True, feat_dim=256):
        super(SupConResNet, self).__init__()
        # model_fun, dim_in = model_dict[name]
        self.encoder = timm.create_model("vit_small_patch16_384", pretrained=True, num_classes=feat_dim)
        self.encoder.head = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, feat_dim)
        )
      
    def forward(self, x):
        feat = self.encoder(x)
        return F.normalize(self.head(feat), dim=-1)
        

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name="resnet34", pretrained=False, feat_dim=256, num_classes=219):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(pretrained=pretrained)
        self.encoder = timm.create_model("vit_small_patch16_384", pretrained=True, num_classes=feat_dim)
        self.encoder.head = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        return self.fc(x)
        
        
class TrainModel(nn.Module):
    def __init__(self, model_name, num_classes=219, pretrained=True):
        super(TrainModel, self).__init__()
        
        if "vit" in model_name:
            self.model = timm.create_model("vit_small_patch16_384", pretrained=True, num_classes=num_classes)
        else:
            self.model = models.__dict__[model_name](pretrained=pretrained)
            self.__reset_fc(num_classes, model_name)
    
    def __reset_fc(self, num_classes, model_name):
        if "resnet" in model_name:
            hidden_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Linear(hidden_dim, num_classes)
        elif "convnext" in model_name:
            hidden_dim = self.model.classifier[-1].weight.shape[1]
            self.model.classifier[-1] = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x) 

    
class ConModel(nn.Module):
    def __init__(self, hidden_dim=1024, dim=512, temp=0.1, pretrained=True):
        super(ConModel, self).__init__()
        
        self.temp = temp
        q_model = timm.create_model("vit_small_patch16_384", pretrained=pretrained)
        k_model = timm.create_model("vit_small_patch16_384", pretrained=pretrained)
        
        # Reset the head model
        weight_shape = q_model.head.weight.shape[1]
        q_model.head = nn.Identity()
        k_model.head = nn.Identity()
        
        # Add linear layers
        self.q = nn.Sequential(
            q_model,
            nn.Linear(weight_shape, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )
        self.k = nn.Sequential(
            k_model,
            nn.Linear(weight_shape, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
        )
        
        # Ensure the encoder k is updated through EMA
        for param_q, param_k in zip(self.q.parameters(), self.k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self, moco_m):
        for param_q, param_k in zip(self.q.parameters(), self.k.parameters()):
            param_k.data = param_k.data * moco_m + \
                           param_q.data * (1. - moco_m)

    def contrastive_loss(self, q, k):
        # Normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
       
        # Einstein sum is more intuitive
        logits = (q @ k.T) / self.temp
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)

    def forward(self, x1, x2, moco_m):
        q1 = self.predictor(self.q(x1))
        q2 = self.predictor(self.q(x2))

        # No gradients update to k
        with torch.no_grad():
            self._momentum_update_key_encoder(moco_m)
            
            k1 = self.k(x1)
            k2 = self.k(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1) 


class ConLinearModel(nn.Module):
    def __init__(self, hidden_dim=1024, num_class=219, pretrained=True):
        super(ConLinearModel, self).__init__()
        
        model = timm.create_model("vit_small_patch16_384", pretrained=pretrained)
        weight_shape = model.head.weight.shape[1]
        model.head = nn.Identity()
        
        self.model = nn.Sequential(
            model,
            nn.Linear(weight_shape, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
        )
        
        self.fc = nn.Linear(hidden_dim, num_class)
        
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
        

class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, "module")
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, "module") and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = "module." + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = "module." + k
                else:
                    j = k
                esd[k].copy_(msd[j])

