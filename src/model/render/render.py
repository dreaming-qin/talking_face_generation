import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# test
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(3):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.model.render.base_function import LayerNorm2d, FineEncoder, FineDecoder
from src.util.model_util import cnt_params
from src.model.exp3DMM.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from src.model.exp3DMM.self_attention_pooling import SelfAttentionPooling

class Render(nn.Module):
    def __init__(
        self, 
        mapping_net, 
        warpping_net, 
        editing_net, 
        common
        ):  
        super(Render, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.warpping_net = WarpingNet(**warpping_net)
        self.editing_net = EditingNet(**editing_net, **common)


        param_num,_=cnt_params(self.mapping_net)
        print(f"mapping_net total paras number: {param_num}")
        param_num,_=cnt_params(self.warpping_net)
        print(f"warpping_net total paras number: {param_num}")
        param_num,_=cnt_params(self.editing_net)
        print(f"editing_net total paras number: {param_num}")

 
    def forward(
        self, 
        input_image, 
        driving_source, 
        stage=None
        ):
        '''input_image:[B,3,H,W]
        driving_source:[B,73(exp 3dmm+pose),2*win_size+1]
        
        返回[B,3,H,W]
        在这里，H和W为256'''
        if stage == 'warp':
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
        else:
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
            output['fake_image'] = self.editing_net(input_image, output['warp_image'], descriptor)
        return output

class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        super( MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

        self.norm=nn.LayerNorm(descriptor_nc)

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out).squeeze(dim=-1)
        out=self.norm(out).unsqueeze(dim=-1)
        return out   

class EditingNet(nn.Module):
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        layer, 
        base_nc, 
        max_nc, 
        num_res_blocks, 
        use_spect):  
        super(EditingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'norm_layer':norm_layer, 'nonlinearity':nonlinearity, 'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc

        # encoder part
        self.encoder = FineEncoder(image_nc*2, base_nc, max_nc, layer, **kwargs)
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc, max_nc, layer, num_res_blocks, **kwargs)

    def forward(self, input_image, warp_image, descriptor):
        x = torch.cat([input_image, warp_image], 1)
        x = self.encoder(x)
        gen_image = self.decoder(x, descriptor)
        return gen_image


def make_coordinate_grid_3d(spatial_size, device):
    '''
    generate 3D coordinate grid
    '''
    d, h, w = spatial_size
    x = torch.arange(w).to(device)
    y = torch.arange(h).to(device)
    z = torch.arange(d).to(device)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
    yy = y.view(1,-1, 1).repeat(d,1, w)
    xx = x.view(1,1, -1).repeat(d,h, 1)
    zz = z.view(-1,1,1).repeat(1,h,w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed,zz

class ResBlock2d(nn.Module):
    '''
    basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features,out_features,1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class UpBlock2d(nn.Module):
    '''
    basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d(nn.Module):
    '''
    basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    '''
    basic block
    '''
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    '''
    basic block
    '''
    def __init__(self, num_channels, num_down_blocks=3, block_expansion=64, max_features=512,
                 ):
        super(Encoder, self).__init__()
        self.in_conv = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.Sequential(*down_blocks)
    def forward(self, image):
        out = self.in_conv(image)
        out = self.down_blocks(out)
        return out

class Decoder(nn.Module):
    '''
    basic block
    '''
    def __init__(self,num_channels, num_down_blocks=3, block_expansion=64, max_features=512):
        super(Decoder, self).__init__()
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_conv = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.activate = nn.Tanh()
    def forward(self, feature_map):
        out = self.up_blocks(feature_map)
        out = self.out_conv(out)
        out = self.activate(out)
        return out

class AdaAT(nn.Module):
    '''
    Our proposed AdaAT operation
    '''
    def __init__(self,  para_ch,feature_ch):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.scale = nn.Sequential(
                    nn.Linear(para_ch, feature_ch),
                    nn.Sigmoid()
                )
        self.rotation = nn.Sequential(
                nn.Linear(para_ch, feature_ch),
                nn.Tanh()
            )
        self.translation = nn.Sequential(
                nn.Linear(para_ch, 2 * feature_ch),
                nn.Tanh()
            )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map,para_code):
        '''feature_map [b, 512, h/4, w/4]
        para_code [b, 256]
        '''
        batch,d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        ######## compute affine trans parameters
        para_code = self.commn_linear(para_code)
        # compute scale para
        # [b, 512, 1]
        scale = self.scale(para_code).unsqueeze(-1) * 2
        # compute rotation angle
        # [b, 512, 1]   
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159
        # transform rotation angle to ratation matrix
        # [b, 512, 4]
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        # [b, 512 ,2, 2]
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        # compute translation para
        # [b, 512, 2]
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        ########  do affine transformation
        # compute 3d coordinate grid
        # [b, h/4, w/4, 2] [b, h/4, w/4]
        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.device)
        # [b, 512, h/4, w/4, 2]
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        #  [b, 512, h/4, w/4]
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        # [b, 512, h/4, w/4, 1]
        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        # [b, 512, h/4, w/4, 2, 2]
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        # [b, 512, h/4, w/4, 2]
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        # do affine transformation on channels
        # [b, 512, h/4, w/4, 2]
        trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        # [b, 512, h/4, w/4, 3]
        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        # interpolation
        # [b, 512, h/4, w/4]
        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        return trans_feature

class AdaIN(nn.Module):
    def __init__(self,  para_ch,feature_ch,  eps=1e-5):
        super(AdaIN, self).__init__()
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.mean = nn.Linear(para_ch, feature_ch)
        self.var = nn.Linear(para_ch, feature_ch)
        self.eps = eps

    def forward(self, feature_map,para_code):
        batch, d = feature_map.size(0), feature_map.size(1)
        para_code = self.commn_linear(para_code)
        ########### compute mean and var
        mean = self.mean(para_code).unsqueeze(2).unsqueeze(3)
        var = self.var(para_code).unsqueeze(2).unsqueeze(3)
        ########## normalization
        feature_var = torch.var(feature_map.view(batch, d, -1), dim=2, keepdim=True).unsqueeze(-1) + self.eps
        feature_std = feature_var.sqrt()
        feature_mean = torch.mean(feature_map.view(batch, d, -1), dim=2, keepdim=True).unsqueeze(-1)
        feature_map = (feature_map - feature_mean) / feature_std
        norm_feature = feature_map * var + mean
        return norm_feature

class TransEncoder(nn.Module):
    def __init__(self,drving_dim):
        super(TransEncoder, self).__init__()
        self.img_feature_encoder = nn.Sequential(
            SameBlock2d(512, 256, kernel_size=3, padding=1),
            DownBlock2d(256,256,3),
            DownBlock2d(256, 256, 3),
        )

        num_heads=8
        embed_size=256

        self.drving_encoder =nn.Sequential(
            nn.Linear(drving_dim, embed_size),
            nn.LayerNorm(embed_size),
        ) 

        self.fusion = nn.MultiheadAttention(embed_size, num_heads,batch_first=True)
        self.fusion_morm=nn.LayerNorm(embed_size)
        self.self_atten=SelfAttentionPooling(input_dim=embed_size)

    def forward(self, source_image,drving):
        # source_image [b, 512, 64, 64]
        # drving [b, 256,1]

        # [b, 256, 16, 16]
        img_feature=self.img_feature_encoder(source_image)
        # [b, 256,256]
        img_feature=img_feature.reshape(img_feature.shape[0],img_feature.shape[1],-1)

        # [b,256]
        drving_feature=self.drving_encoder(drving.squeeze(dim=-1))
        # [b,256,256]
        drving_feature=drving_feature.unsqueeze(dim=1).repeat(1,256,1)

        # fusion
        # [b,256,256]
        out,_=self.fusion(drving_feature,img_feature,img_feature)
        out=self.fusion_morm(out)
        # [b,256]
        out=self.self_atten(out)

        return out

class WarpingNet(nn.Module):
    def __init__(self, img_channel,drving_dim):
        super(WarpingNet, self).__init__()
        self.appearance_encoder = nn.Sequential(
            Encoder(img_channel , num_down_blocks=2, block_expansion=64, max_features=256),
            ResBlock2d(256, 256, 3, 1),
            ResBlock2d(256, 256, 3, 1),
            ResBlock2d(256,512, 3, 1),
            ResBlock2d(512,512, 3, 1),
        )
        self.trans_encoder = TransEncoder(drving_dim)
        appearance_conv_list = []
        for i in range(3):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(512, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 512, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)

        self.adaAT1 = AdaAT(256,512)
        self.adaAT2 = AdaAT(256, 512)
        # self.adaAT1 = AdaIN(256,512)
        # self.adaAT2 = AdaIN(256, 512)
        self.adaAT3 = AdaIN(256, 512)

        self.appearance_decoder = nn.Sequential(
            ResBlock2d(512, 512, 3, 1),
            ResBlock2d(512, 256, 3, 1),
            ResBlock2d(256, 256, 3, 1),
            ResBlock2d(256, 256, 3, 1),
            Decoder(img_channel, num_down_blocks=2, block_expansion=64, max_features=512)
        )

        # torch.save(self.appearance_encoder.state_dict(),'appearance_encoder.pth')
        # torch.save(self.trans_encoder.state_dict(),'trans_encoder.pth')
        # adaAT_dict=self.appearance_conv_list.state_dict()
        # adaAT_dict.update(self.adaAT.state_dict())
        # torch.save(adaAT_dict,'adaAT.pth')
        # torch.save(self.appearance_decoder.state_dict(),'appearance_decoder.pth')
        # torch.save(self.state_dict(),'self.pth')

    def forward(self, source_image,drving):
        # source_image [b, 3, h, w]
        # source_image [b, 128]

        # concat input data
        module_in = source_image
        # compute appearance feature map
        # [b, 512, h/2, w/4]
        appearance_feature = self.appearance_encoder(module_in)
        ######################## transformation branch
        # compute paras of affine transformation
        # [b, 256]
        para_code = self.trans_encoder(appearance_feature,drving)
        ######################## feature alignment
        # [b, 512, h/2, w/4]
        appearance_feature = self.appearance_conv_list[0](appearance_feature)
        appearance_feature = self.adaAT1(appearance_feature,para_code)
        appearance_feature = self.appearance_conv_list[1](appearance_feature)
        appearance_feature = self.adaAT2(appearance_feature,para_code)
        appearance_feature = self.appearance_conv_list[2](appearance_feature)
        # [b, 512, h/2, w/4]
        appearance_feature = self.adaAT3(appearance_feature,para_code)
        # decode output image [b, 3, h, w]
        out = self.appearance_decoder(appearance_feature)
        
        return {'warp_image':out}



# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml,glob
    from src.dataset.renderDtaset import RenderDataset
    from src.model.exp3DMM.exp3DMM import Exp3DMM
    from src.util.util import get_window

    config={}
    yaml_file=['config/dataset/common.yaml',
               'config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=RenderDataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    render=Render(
        config['mapping_net'],
        config['warpping_net'],
        config['editing_net'],
        config['common'])
    # torch.save(render.state_dict(),'render.pth')

    render=render.to(device)
    with torch.no_grad():
        for data in dataloader:
            # 把数据放到device中
            for key,value in data.items():
                data[key]=value.to(device)
            
            # [B,73,win size]
            driving_source=data['driving_src']
            # [B,3,H,W]
            input_image=data['src']
            # [B,3,H,W]
            output_dict = render(input_image, driving_source)
