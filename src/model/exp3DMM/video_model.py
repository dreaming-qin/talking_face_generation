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



from src.model.exp3DMM.video_encoder import VideoEncoder
from src.model.exp3DMM.fusion import Fusion
from src.util.util import get_window


class VideoModel(nn.Module):
    '''
    输入音频MFCC和transformer后的视频，输出表情3DMM
    '''
    def __init__(self,cfg) :
        super().__init__()

        self.video_encoder=VideoEncoder(**cfg['video_encoder'])
        self.mapping=nn.Sequential(
            nn.Linear(cfg['audio_exp_len'],32),
            nn.Linear(32,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.Linear(256,256))
        self.fusion_module=Fusion(**cfg['fusion'])

        self.win_size=cfg['audio_win_size']

    def forward(self, transformer_video, audio_exp):
        """transformer_video输入维度[b,len,3,H,W]
        audio_exp输入维度[B,LEN(27),3dmm dim(9)]
        输出维度[B,len(27),3dmm dim(64)]
        """
        # [B,len,video dim]
        B,L,C,H,W=transformer_video.shape
        transformer_video=transformer_video.reshape(-1,C,H,W)
        video_feature=self.video_encoder(transformer_video)


        # [B,Len,video dim]
        video_feature=video_feature.reshape(B,L,-1)
        # [B,len,win_size,video dim]
        video_feature=get_window(video_feature,self.win_size)
        video_feature=video_feature[:,self.win_size:-self.win_size]

        audio_exp_mapping=self.mapping(audio_exp)
        # [B,len,win_size,video dim]
        audio_exp_mapping=get_window(audio_exp_mapping,self.win_size)
        audio_exp_mapping=audio_exp_mapping[:,self.win_size:-self.win_size]
        # test，测试音频是否能同步唇形
        exp3DMM=self.fusion_module(video_feature,audio_exp_mapping)

        return exp3DMM
    
# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml,glob
    from src.dataset.exp3DMMdataset import Exp3DMMdataset

    config={}
    yaml_file=['config/dataset/common.yaml',
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
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Exp3DMM(config)
    model= torch.nn.DataParallel(model, device_ids=config['device_id'])
    state_dict=torch.load('checkpoint/exp3DMM/epoch_18_metrices_0.6989557775132689.pth',
                map_location=torch.device('cpu'))
    model.load_state_dict(state_dict,strict=False)
    with torch.no_grad():
        for data in dataloader:
            for key,value in data.items():
                data[key]=value.to(device)
            
            exp=data['video'].permute(0,1,4,2,3)
            audio=data['audio']
            result=model(exp,audio)
            a=1
