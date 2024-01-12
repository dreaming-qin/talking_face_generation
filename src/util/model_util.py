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

def cnt_params(model):
    temp=0
    lst=[]
    for name,para in model.named_parameters():
        lst.append((name,para.nelement()))
        temp+=para.nelement()
    return temp,lst

