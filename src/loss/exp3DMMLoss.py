from torch import nn
from scipy.io import loadmat
import torch
import numpy as np
import random


# 测试代码
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))


from src.util.util_3dmm import reconstruct_idexp_lm3d
from src.model.syncNet.syncNet import LandmarkHubertSyncNet
from src.util.model_util import freeze_params
from src.loss.renderLoss import RenderLoss

class Exp3DMMLoss(nn.Module):
    r'''返回表情3DMM模块的Loss值'''
    def __init__(self, config,device):
        super(Exp3DMMLoss, self).__init__()
        self.device=device
        self.mouth_weight=config['mouth_weight']
        self.exp_weight=config['exp_weight']
        self.triple_weight=config['triple_weight']
        self.rec_weight=config['rec_weight']
        self.distance=config['triple_dis']


        self.mse_loss=nn.MSELoss()
        self.L1_loss=nn.L1Loss()
        self.logloss = nn.BCELoss()
        self.relu=nn.ReLU()
        self.render_loss=RenderLoss(config)

        model = loadmat("./BFM/BFM_model_front.mat")
        id_base = torch.from_numpy(model['idBase']).float().to(self.device) # identity basis. [3*N,80], we have 80 eigen faces for identity
        exp_base = torch.from_numpy(model['exBase']).float().to(self.device) # expression basis. [3*N,64], we have 64 eigen faces for expression
        key_points = torch.from_numpy(model['keypoints'].squeeze().astype(np.compat.long)).long().to(self.device) # vertex indices of 68 facial landmarks. starts from 1. [68,1]
        self.key_id_base = id_base.reshape([-1,3,80])[key_points, :, :].reshape([-1,80]).to(self.device)
        self.key_exp_base = exp_base.reshape([-1,3,64])[key_points, :, :].reshape([-1,64]).to(self.device)


    def forward(self,exp,pos_exp,neg_exp,
                style,pos_style,neg_style,
                video,pos_video,neg_video,
                data):
        # 对于唇部和表情Loss，其中一部分的Loss都需要将3DMM转为landmark，然后比较landmaark
        type_list=['','neg_','pos_']
        predict={'exp':exp,'neg_exp':neg_exp,'pos_exp':pos_exp,
                 'video':video,'neg_video':neg_video,'pos_video':pos_video}
        loss_mouth=0
        loss_exp=0
        rec_loss=0
        for type in type_list:
            # 先生成gt landmark
            GT_landmark=reconstruct_idexp_lm3d(data[f'{type}id_3dmm'],data[f'{type}gt_3dmm'],self.key_id_base,self.key_exp_base)
            # 再将预测的转为landmark
            predicted_landmark=reconstruct_idexp_lm3d(data[f'{type}id_3dmm'],predict[f'{type}exp'],self.key_id_base,self.key_exp_base)


            # 先计算唇部误差:
            GT_mouth = GT_landmark[:, :,48:] # [T, 20, 3]
            predicted_mouth = predicted_landmark[:, :,48:] # [T, 20, 3]
            # 第一部分
            loss_one=self.mse_loss(GT_mouth,predicted_mouth)
            loss_mouth+=self.mouth_weight[0]*loss_one

            # 再计算表情误差
            GT_other = GT_landmark[:, :,:48] # [T, 48, 3]
            predicted_other= predicted_landmark[:, :,:48] # [T, 48, 3]
            # 第三部分
            loss_three=self.mse_loss(GT_other,predicted_other)
            # 第四部分
            loss_four=self.L1_loss(predict[f'{type}exp'],data[f'{type}gt_3dmm'])
            loss_exp+=self.exp_weight[0]*loss_three+self.exp_weight[1]*loss_four

            # 重建loss
            rec_loss+=self.rec_weight*self.render_loss.api_forward(predict[f'{type}video'].squeeze(0),
                        data[f'{type}gt_video'].squeeze(0))
        # 计算三元对loss
        triple_loss=self.triple_weight*self.relu(self.mse_loss(style,pos_style)-
                               self.mse_loss(style,neg_style)+self.distance)
    
        return loss_mouth+loss_exp+triple_loss+rec_loss
    

    def get_audio_and_mouth_clip(self,mouth_lm3d,mel,batch_size=1024):
        # 为了不获得无效信息的而设计的遮罩mask
        mask=mouth_lm3d>1e-6
        mask=mask.sum(dim=-1)>1e-6
        y_len = mask.sum(dim=-1)
        mouth_lm3d=mouth_lm3d.reshape(mouth_lm3d.shape[0],-1,60)
        mel=mel.reshape(mel.shape[0],-1,1024)
        mouth_lst, mel_lst, label_lst = [], [], []
        while len(mouth_lst) < batch_size:
            for i in range(mouth_lm3d.shape[0]):
                is_pos_sample = random.choice([True, False])
                exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                mouth_clip = mouth_lm3d[i, exp_idx: exp_idx+5]
                assert mouth_clip.shape[0]==5, f"exp_idx={exp_idx},y_len={y_len[i]}"
                if is_pos_sample:
                    mel_clip = mel[i, exp_idx*2: exp_idx*2 + 10]
                    label_lst.append(1.)
                else:
                    if random.random() < 0.25:
                        wrong_spk_idx = random.randint(a=0, b=len(y_len)-1)
                        wrong_exp_idx = random.randint(a=0, b=y_len[wrong_spk_idx]-1-5)
                        while wrong_exp_idx == exp_idx:
                            wrong_exp_idx = random.randint(a=0, b=y_len[wrong_spk_idx]-1-5)
                        mel_clip = mel[wrong_spk_idx, wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10
                    elif random.random() < 0.5:
                        wrong_exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                        while wrong_exp_idx == exp_idx:
                            wrong_exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                        mel_clip = mel[i, wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10
                    else:
                        left_offset = max(-5, -exp_idx)
                        right_offset = min(5, (y_len[i]-5-exp_idx))
                        exp_offset = random.randint(a=left_offset, b=right_offset)
                        while abs(exp_offset) <= 1:
                            exp_offset = random.randint(a=left_offset, b=right_offset)
                        wrong_exp_idx = exp_offset + exp_idx
                        mel_clip = mel[i, wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10, y_len[i]-wrong_exp_idx
                    mel_clip = mel[i, wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                    label_lst.append(0.)
                mouth_lst.append(mouth_clip)
                mel_lst.append(mel_clip)
        mel_clips = torch.stack(mel_lst)
        mouth_clips = torch.stack(mouth_lst)
        labels = torch.tensor(label_lst).float().to(mel_clips.device)

        return mouth_clips,mel_clips,labels
    
# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml,glob
    from src.dataset.exp3DMMdataset import Exp3DMMdataset

    config={}
    yaml_file=['config/data_process/common.yaml','config/dataset/common.yaml',
               'config/model/render.yaml','config/model/exp3DMM.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=Exp3DMMdataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        # collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fun=Exp3DMMLoss(config,device)
    for data in dataloader:
        for key,value in data.items():
            data[key]=value.to(device)
        ccc=loss_fun(data['gt_3dmm'],data['pos_gt_3dmm'],data['neg_gt_3dmm'],
                     torch.ones(2),torch.ones(2),torch.ones(2),
                     data['gt_video'],data['pos_gt_video'],data['neg_gt_video'],
                     data)