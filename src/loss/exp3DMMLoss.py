from torch import nn
import torch


# 测试代码
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))


from src.model.syncNet.sync_net import SyncNet
from src.util.model_util import freeze_params
from src.loss.renderLoss import RenderLoss
from src.util.util_3dmm import get_lm_by_3dmm, get_face
from src.loss.sync_net_loss import SyncNetLoss

class Exp3DMMLoss(nn.Module):
    r'''返回表情3DMM模块的Loss值'''
    def __init__(self, config,device):
        super(Exp3DMMLoss, self).__init__()
        self.device=device
        self.mouth_weight=config['mouth_weight']
        self.exp_weight=config['exp_weight']
        self.rec_weight=config['rec_weight']
        self.sync_weight=config['sync_weight']

        self.mse_loss=nn.MSELoss()
        self.L1_loss=nn.L1Loss()

        self.audio_exp_list=[0,1,2,3,4,5,7,10,13]



    def forward(self,audio_exp,exp,video,data):
        r'''1. 让唇形图片同步
        2. 9维3DMM
        3.GT 3DMM

        exp形状(B,win,64)
        video形状(B,3,H,W)
        '''

        # 对于唇部和表情Loss，其中一部分的Loss都需要将3DMM转为landmark，然后比较landmaark
        type_list=['']
        predict={'exp':exp,'video':video,'audio_exp':audio_exp}
        loss=0
        for type in type_list:
            # # 1.唇形图片同步
            # real_mouth=get_face(data[f'{type}id_3dmm'].reshape(-1,80),
            #     data[f'{type}gt_3dmm'].reshape(-1,64))[...,120:170,70:150]
            # fake_mouth=get_face(data[f'{type}id_3dmm'].reshape(-1,80),
            #     predict[f'{type}audio_exp'].reshape(-1,64))[...,120:170,70:150]
            # real_mouth=real_mouth.permute(0,2,3,1)
            # fake_mouth=fake_mouth.permute(0,2,3,1)
            # # 转灰度图
            # real_gray=0.299*real_mouth[...,0]+0.587*real_mouth[...,1]+0.114*real_mouth[...,2]
            # fake_gray=0.299*fake_mouth[...,0]+0.587*fake_mouth[...,1]+0.114*fake_mouth[...,2]
            # # 比较
            # mouth_loss=self.mse_loss(real_gray,fake_gray)
            # mouth_loss*=self.mouth_weight
            # loss+=mouth_loss

            # 2. 9维3DMM
            audio_choose_exp=predict[f'{type}audio_exp'][...,self.audio_exp_list]
            real_choose_exp=data[f'{type}gt_3dmm'][...,self.audio_exp_list]
            tdmm_mouth_loss=self.L1_loss(real_choose_exp,audio_choose_exp)
            # tdmm_mouth_loss=self.L1_loss(data[f'{type}gt_3dmm'],predict[f'{type}audio_exp'])
            loss+=tdmm_mouth_loss

            # 3. GT 3DMM
            tdmm_loss=self.L1_loss(predict[f'{type}exp'],data[f'{type}gt_3dmm'])
            tdmm_loss*=self.exp_weight
            loss+=tdmm_loss

            # # test，存灰度图
            # import numpy as np
            # import imageio
            # for j in range(27):
            #     real_img=real_gray[j].cpu().numpy()
            #     fake_img=fake_gray[j].cpu().numpy()
            #     img=np.concatenate((real_img,fake_img),axis=1)
            #     img=(img*255).astype(np.uint8)
            #     imageio.imsave(f'temp/{j}.png',img)


    
        return loss
    

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml,glob
    from src.dataset.exp3DMMdataset import Exp3DMMdataset

    config={}
    yaml_file=['config/dataset/common.yaml',
               'config/model/render.yaml','config/model/exp3DMM.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=Exp3DMMdataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2, 
        shuffle=True,
        drop_last=False,
        num_workers=1,
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fun=Exp3DMMLoss(config,device)
    for data in dataloader:
        for key,value in data.items():
            data[key]=value.to(device)
        ccc=loss_fun(data['gt_3dmm'],data['gt_video'],data)