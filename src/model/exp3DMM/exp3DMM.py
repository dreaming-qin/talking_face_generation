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
from src.util.model_util import cnt_params


class Exp3DMM(nn.Module):
    '''
    输入音频MFCC和transformer后的视频，输出表情3DMM
    '''
    def __init__(self,cfg) :
        super().__init__()
        self.audio_encoder=AudioEncoder(**cfg['audio_encoder'])
        self.video_encoder=VideoEncoder(**cfg['video_encoder'])
        self.fusion_module=Fusion(**cfg['fusion'])

        audio_cnt,_=cnt_params(self.audio_encoder)
        print(f"audio_encoder total paras number: {audio_cnt}")
        video_cnt,_=cnt_params(self.video_encoder)
        print(f"video_encoder total paras number: {video_cnt}")
        fusion_cnt,_=cnt_params(self.fusion_module)
        print(f"fusion_module total paras number: {fusion_cnt}")
        
        self.win_size=cfg['audio_win_size']

    def forward(self, transformer_video, audio_MFCC):
        """transformer_video输入维度[b,len(27),3,H,W]
        audio_MFCC输入维度[B,LEN(27),len(28),mfcc dim(13)]
        输出维度[B,len(27),3dmm dim(64)]
        """
        # [B,len,audio dim]
        audio_feature=self.audio_encoder(audio_MFCC)
        B,L,C,H,W=transformer_video.shape
        transformer_video=transformer_video.reshape(-1,C,H,W)
        video_feature=self.video_encoder(transformer_video)
        # [B,Len,video dim]
        video_feature=video_feature.reshape(B,L,-1)


        # # test，测试audio feature与video feature大小
        # import imageio
        # import numpy as np
        # # 1.获得原来的audio feature
        # state_dict=torch.load('checkpoint/exp3DMM/epoch_17_metrices_0.9269133167494708.pth',
        #             map_location=torch.device('cpu'))
        # temp_dice={}
        # for key,value in state_dict.items():
        #     temp_dice[key[7:]]=value
        # self.load_state_dict(temp_dice)
        # o_audio_feature=self.audio_encoder(audio_MFCC).squeeze()
        # # 2. 获得新的audio feture
        # state_dict=torch.load('checkpoint/exp3DMM/epoch_18_metrices_0.6989557775132689.pth',
        #             map_location=torch.device('cpu'))
        # temp_dice={}
        # for key,value in state_dict.items():
        #     temp_dice[key[7:]]=value
        # self.load_state_dict(temp_dice)
        # n_audio_feature=self.audio_encoder(audio_MFCC).squeeze()
        # # 3. 合并音频特征
        # feature=torch.cat((o_audio_feature,n_audio_feature))
        # # feature=torch.cat((audio_feature.squeeze(),video_feature))
        # feature=feature.cpu().numpy()
        # min_,max_=feature.min(),feature.max()
        # feature=(255*(feature-min_)/(max_-min_)).astype(np.uint8)
        # imageio.imsave('temp_oa2na.png',feature)


        # test，测试音频是否能同步唇形
        # exp3DMM=self.fusion_module(audio_feature,audio_feature)

        exp3DMM=self.fusion_module(audio_feature,video_feature)
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
    with torch.no_grad():
        for data in dataloader:
            for key,value in data.items():
                data[key]=value.to(device)
            
            exp=data['video'].permute(0,1,4,2,3)
            audio=data['audio']
            result=model(exp,audio)
            a=1
