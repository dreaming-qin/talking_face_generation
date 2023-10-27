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
        self.win_size=cfg['win_size']

    def forward(self, transformer_video, audio_MFCC):
        """transformer_video输入维度[b,len,3,H,W]
        audio_MFCC输入维度[B,LEN,28,mfcc dim]
        输出维度[B,len,3dmm dim]
        """
        # [B,len,audio dim]
        audio_feature=self.audio_encoder(audio_MFCC)
        video_feature=[]
        for i in range(transformer_video.shape[1]):
            # [B,video dim]
            frame_feature=self.video_encoder(transformer_video[:,i])
            video_feature.append(frame_feature)
        # [B,Len,video dim]
        video_feature=torch.stack(video_feature,dim=1)
        # [B,len,win_size,audio dim]
        audio_feature=get_window(audio_feature,self.win_size)
        # [B,len,win_size,video dim]
        video_feature=get_window(video_feature,self.win_size)
        exp3DMM=self.fusion_module(audio_feature,video_feature)
        return exp3DMM

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml,glob
    from src.dataset.exp3DMMdataset import Exp3DMMdataset

    config={}
    yaml_file=glob.glob(r'config/*/*.yaml')
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
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Exp3DMM(config)
    model=model.to(device)
    for data in dataloader:
        for key,value in data.items():
            data[key]=value.to(device)
        aaa=data['video_input']
        transformer_video=aaa.permute(0,1,4,2,3)
        result=model(transformer_video,data['audio_input'])
        a=1
