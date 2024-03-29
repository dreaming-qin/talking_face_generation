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

    def forward(self, input_image, descriptor):
        final_output={}
        output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)

        deformation = flow_util.convert_flow_to_deformation(final_output['flow_field'])
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
        batch_size=3, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    exp_3dmm=Exp3DMM(config)
    exp_3dmm=exp_3dmm.to(device)

    render=Render(
        config['mapping_net'],
        config['warpping_net'],
        config['editing_net'],
        config['common'])
    render=render.to(device)
    with torch.no_grad():
        for data in dataloader:
            for key,value in data.items():
                data[key]=value.to(device)
            aaa=data['video_input']
            transformer_video=aaa.permute(0,1,4,2,3)
            result_exp=exp_3dmm(transformer_video,data['audio_input'])

            # 数据处理
            # [B,len,73]
            driving_source=torch.cat((result_exp,data['pose']),dim=-1)
            # [B,len,win size,73]
            driving_source=get_window(driving_source,config['win_size'])
            driving_source=driving_source.permute(0,1,3,2)
            
            input_image=[]
            for img in data['img']:
                # [3,H,W]
                tmp_img=img.permute(2,0,1)
                # [len,3,H,W]
                tmp_img=tmp_img.expand(driving_source.shape[1],  -1, -1, -1)
                input_image.append(tmp_img)
            # [B,len,3,H,W]
            input_image=torch.stack(input_image)

            output_imgs=[]
            for drving,img in zip(driving_source,input_image):
                output_dict = render(img, drving)
                output_imgs.append(output_dict['fake_image'])
            output_imgs=torch.stack(output_imgs)

            a=1

