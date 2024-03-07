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
from src.util.util_3dmm import get_lm_by_3dmm
from src.loss.sync_net_loss import SyncNetLoss

class Exp3DMMLoss(nn.Module):
    r'''返回表情3DMM模块的Loss值'''
    def __init__(self, config,device):
        super(Exp3DMMLoss, self).__init__()
        self.device=device
        self.landmark_weight=config['landmark_weight']
        self.exp_weight=config['exp_weight']
        self.rec_weight=config['rec_weight']
        self.sync_weight=config['sync_weight']

        self.mse_loss=nn.MSELoss()
        self.L1_loss=nn.L1Loss()
        self.logloss = nn.BCELoss()
        self.relu=nn.ReLU()
        self.render_loss=RenderLoss(config)

        # 唇形同步器
        self.sync_net=None
        self.sync_loss_function=SyncNetLoss()
        if 'sync_net_pre_train' in config:
            self.sync_net=SyncNet(**config)
            self.sync_net= torch.nn.DataParallel(self.sync_net, device_ids=config['device_id'])
            state_dict=torch.load(config['sync_net_pre_train'],map_location=torch.device('cpu'))
            self.sync_net.load_state_dict(state_dict)
            freeze_params(self.sync_net)
            self.sync_net.eval()
            self.sync_net=self.sync_net.to(self.device)


    def forward(self,exp,video,data):
        r'''exp形状(B,win,64)
        video形状(B,3,H,W)
        '''

        # 对于唇部和表情Loss，其中一部分的Loss都需要将3DMM转为landmark，然后比较landmaark
        type_list=['']
        predict={'exp':exp,'video':video}
        loss=0
        for type in type_list:
            batch_size=len(data[f'{type}id_3dmm'])
            # # 第一部分,3dmm比较
            # tdmm_loss=self.L1_loss(predict[f'{type}exp'],data[f'{type}gt_3dmm'])
            # tdmm_loss*=self.exp_weight
            # loss+=tdmm_loss

            # 第二部分，唇部landmark比较
            real_landmark=get_lm_by_3dmm(data[f'{type}id_3dmm'].reshape(-1,80),
                                         data[f'{type}gt_3dmm'].reshape(-1,64))
            fake_landmark=get_lm_by_3dmm(data[f'{type}id_3dmm'].reshape(-1,80),
                                        predict[f'{type}exp'].reshape(-1,64))
            fake_mouth_landmark=fake_landmark[:,48:]
            real_mouth_landmark=real_landmark[:,48:]
            landmark_loss=self.L1_loss(fake_mouth_landmark,real_mouth_landmark)
            landmark_loss*=self.landmark_weight
            loss+=landmark_loss

            # 第三部分唇形监督器loss
            # if self.sync_net is not None:
            #     fake_mouth_landmark=fake_lamdmark[:,48:].reshape(batch_size,-1,40)
            #     mid=fake_mouth_landmark.shape[1]//2
            #     fake_mouth_landmark=fake_mouth_landmark[:,mid-2:mid+3]
            #     # 归一化fake_mouth_landmark
            #     fake_mouth_landmark=fake_mouth_landmark.reshape(-1,20,2)
            #     # 1.按照原点对齐
            #     original = torch.sum(fake_mouth_landmark,dim=1) / fake_mouth_landmark.shape[1]
            #     fake_mouth_landmark =fake_mouth_landmark - original.reshape(-1,1,2)
            #     # 2.规整到（-1,1）中
            #     fake_mouth_landmark=fake_mouth_landmark.permute(0,2,1).reshape(-1,40)
            #     max_lmd=torch.max(torch.abs(fake_mouth_landmark),dim=1)[0]
            #     fake_mouth_landmark=(fake_mouth_landmark.T/max_lmd).T
            #     fake_mouth_landmark=fake_mouth_landmark.reshape(batch_size,-1,40)
                
            #     mid=data[f'{type}hubert'].shape[1]//2
            #     hubert=data[f'{type}hubert'][:,mid-2:mid+3].reshape(-1,10,1024)
            #     audio_e,mouth_e=self.sync_net(hubert,fake_mouth_landmark)
            #     sync_loss=self.sync_loss_function(audio_e,mouth_e,data[f'{type}label'])
            #     sync_loss*=self.sync_weight
            #     loss+=sync_loss

            # # 重建loss
            # rec_loss=self.rec_weight*self.render_loss.api_forward_for_exp_3dmm(
            #     predict[f'{type}video'],data[f'{type}gt_video'])
            # loss+=rec_loss
    
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