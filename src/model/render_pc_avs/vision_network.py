import os,sys
# test
if __name__=='__main__':
    sys.path.append('/workspace/code2/talking_face_generation')
    del sys.path[0]

import torch.nn as nn
import torch.nn.functional as F
from src.model.render2.base_network import BaseNetwork
from torchvision.models.resnet import ResNet, Bottleneck
import torch

model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
}


class ResNeXt50(BaseNetwork):
    def __init__(self, opt):
        super(ResNeXt50, self).__init__()
        model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
        setattr(self,f'conv1',getattr(model,f'conv1'))
        setattr(self,f'bn1',getattr(model,f'bn1'))
        setattr(self,f'relu',getattr(model,f'relu'))
        setattr(self,f'maxpool',getattr(model,f'maxpool'))
        for i in range(1,4):
            setattr(self,f'layer_{i}',getattr(model,f'layer{i}'))
        self.opt = opt
        # self.reduced_id_dim = opt.reduced_id_dim
        self.conv1x1 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        # self.fc_pre = nn.Sequential(nn.Linear(512 * Bottleneck.expansion, self.reduced_id_dim), nn.ReLU())

        '''参数量26077992
        经过更改，参数量为8959808'''
        # # test
        # cnt=0
        # lst={}
        # for name,para in self.named_parameters():
        #     name=name.split('.')[0]
        #     if name not in lst:
        #         lst[name]=0
        #     lst[name]+=para.nelement()
        #     cnt+=para.nelement()
        # print(cnt)
        # a=1


    def forward_feature(self, input):
        x = getattr(self,f'conv1')(input)
        x = getattr(self,f'bn1')(x)
        x = getattr(self,f'relu')(x)
        x = getattr(self,f'maxpool')(x)

        for i in range(1,4):
            layer=getattr(self,f'layer_{i}')
            x=layer(x)
        x = self.conv1x1(x)
        # x = self.fc_pre(x)
        return x

    def forward(self, input):
        input_batch = input.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)
        x = self.forward_feature(input_batch)
        x = F.adaptive_avg_pool2d(x, (16, 16))
        x = x.view(-1, self.opt.num_inputs, 512, 16, 16)
        x = torch.mean(x, 1)

        return  x
    

if __name__=='__main__':
    import yaml
    from argparse import Namespace

    # 指定 YAML 文件路径
    yaml_file_path = 'test.yaml'
    # 读取 YAML 文件并加载为字典
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    # 将字典转换为 Namespace 对象
    opt = Namespace(**yaml_data)
    opt.crop_size=256
    
    device=torch.device('cuda')
    model=ResNeXt50(opt).to(device)
    input_img=torch.rand((2,3,256,256)).to(device)

    out=model(input_img)
    a=1