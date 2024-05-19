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

import src.model.render.flow_util as flow_util
from src.model.render.base_function import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder
from src.util.model_util import cnt_params

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
        self.warpping_net = WarpingNet(**warpping_net, **common)
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

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        return out   

class WarpingNet(nn.Module):
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        base_nc, 
        max_nc, 
        encoder_layer, 
        decoder_layer, 
        use_spect
        ):
        super( WarpingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'nonlinearity':nonlinearity, 'use_spect':use_spect}

        self.descriptor_nc = descriptor_nc 
        self.hourglass = ADAINHourglass(image_nc, self.descriptor_nc, base_nc,
                                       max_nc, encoder_layer, decoder_layer, **kwargs)

        self.flow_out = nn.Sequential(norm_layer(self.hourglass.output_nc), 
                                      nonlinearity,
                                      nn.Conv2d(self.hourglass.output_nc, 2, kernel_size=7, stride=1, padding=3))

        self.pool = nn.AdaptiveAvgPool2d(1)

        # test
        # self.deformation_template=None

    def forward(self, input_image, descriptor):

        final_output={}
        output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)
        deformation = flow_util.convert_flow_to_deformation(final_output['flow_field'])

        # # test 初始化deformation_template
        # if self.deformation_template is None:
        #     _, height, width, _ = deformation.shape
        #     w_deformation=torch.tensor([i*2/(width-1)-1 for i in range(width)])
        #     h_deformation=torch.tensor([i*2/(height-1)-1 for i in range(height)])
        #     zero=w_deformation.reshape(1,width).expand(height,width)
        #     one=h_deformation.reshape(height,1).expand(height,width)
        #     self.deformation_template=torch.stack((zero,one),dim=2).unsqueeze(0).to(input_image)
        # deformation+=self.deformation_template

        final_output['warp_image'] = flow_util.warp_image(input_image, deformation)
        return final_output

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
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")


    render=Render(
        config['mapping_net'],
        config['warpping_net'],
        config['editing_net'],
        config['common'])
    render=render.to(device)
    # if 'render_pre_train' in config:
    #     state_dict=torch.load(config['render_pre_train'],map_location=torch.device('cpu'))
    #     aaa={}
    #     for key,val in state_dict.items():
    #         aaa[key.replace('module.','')]=val
    #     # render.load_state_dict(aaa,strict=False)

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
