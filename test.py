import glob
import yaml
import random
import numpy as np
import os
import shutil
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a=nn.Linear(20,10)
        self.b=nn.Linear(10,1)
        for name, param in self.named_parameters():
            if 'b.' in name:
                param.requires_grad = False
    def forward(self,x):
        x=self.a(x)
        x=self.b(x)
        return x



if __name__=='__main__':
    model=Net()

    # 查看梯度值
    for name, parms in model.named_parameters():	
        print('-->name:', name)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("================================")

    tensor=torch.ones((5,20))
    loss=model(tensor)
    loss=loss.mean()
    # L1_func=nn.L1Loss()
    # loss=L1_func(out,1)
    loss.backward()

    # 查看梯度值
    for name, parms in model.named_parameters():	
        print('-->name:', name)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("================================")



    
