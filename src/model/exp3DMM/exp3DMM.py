import torch
from torch import nn

# 测试代码
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(3):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))



from src.model.exp3DMM.audio_encoder import AudioEncoder
from src.model.exp3DMM.video_encoder import VideoEncoder
from src.model.exp3DMM.fusion import Fusion
from src.util.util import get_window


class Exp3DMM(nn.Module):
    '''
    输入音频MFCC和transformer后的视频，输出表情3DMM
    '''
    def __init__(self,cfg) :
        super().__init__()
        self.audio_encoder=AudioEncoder()
        self.video_encoder=VideoEncoder(**cfg['video_encoder'])
        self.fusion_module=Fusion(**cfg['fusion'])

        # test
        # temp=0
        # lst=[]
        # for name,para in self.audio_encoder.named_parameters():
        #     lst.append((name,para.nelement()))
        #     temp+=para.nelement()
        # print(f"audio_encoder total paras number: {temp}")
        # temp=0
        # lst=[]
        # for name,para in self.video_encoder.named_parameters():
        #     lst.append((name,para.nelement()))
        #     temp+=para.nelement()
        # print(f"video_encoder total paras number: {temp}")
        # temp=0
        # lst=[]
        # for name,para in self.fusion_module.named_parameters():
        #     lst.append((name,para.nelement()))
        #     temp+=para.nelement()
        # print(f"fusion_module total paras number: {temp}")
        
        self.win_size=cfg['audio_win_size']

    def forward(self, exp_3DMM, audio_MFCC, pad_mask=None):
        """exp_3DMM输入维度[b,len,64]
        audio_MFCC输入维度[B,LEN,28,mfcc dim]
        输出维度[B,len,3dmm dim]
        """
        # [B,len,audio dim]
        audio_feature=self.audio_encoder(audio_MFCC)
        video_feature=self.video_encoder(exp_3DMM, pad_mask)
        # [B,len,win_size,audio dim]
        audio_feature=get_window(audio_feature,self.win_size)
        audio_feature=audio_feature[:,self.win_size:-self.win_size]
        # [B,len,dim]
        exp3DMM=self.fusion_module(audio_feature,video_feature)
        return exp3DMM,video_feature

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml,glob
    from src.dataset.exp3DMMdataset import Exp3DMMdataset

    config={}
    yaml_file=['config/data_process/common.yaml','config/dataset/common.yaml',
               'config/model/exp3DMM.yaml']
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
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Exp3DMM(config)
    model=model.to(device)
    with torch.no_grad():
        for data in dataloader:
            for key,value in data.items():
                data[key]=value.to(device)
            
            exp=data['style_clip']
            audio=data['audio']
            pad_mask=data['mask']
            result=model(exp,audio,pad_mask)
            a=1
