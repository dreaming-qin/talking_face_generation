import torch.nn as nn

def reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_params(model):
     for param in model.parameters():
        param.requires_grad = True
