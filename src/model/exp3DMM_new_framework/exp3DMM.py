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



from src.model.exp3DMM.audio_model import AudioModel
from src.model.exp3DMM.video_model import VideoModel
from src.util.model_util import cnt_params


class Exp3DMM(nn.Module):
    '''
    输入音频MFCC和transformer后的视频，输出表情3DMM
    '''
    def __init__(self,cfg) :
        super().__init__()

        self.audio_model=AudioModel(cfg)
        self.audio_exp_list=[0,1,2,3,4,5,7,10,13]
        cfg['audio_exp_len']=len(self.audio_exp_list)
        self.video_model=VideoModel(cfg)
        
        self.win_size=cfg['audio_win_size']


        audio_cnt,_=cnt_params(self.audio_model)
        print(f"audio_model total paras number: {audio_cnt}")
        video_cnt,_=cnt_params(self.video_model)
        print(f"video_model total paras number: {video_cnt}")
        

    def forward(self, transformer_video, audio_MFCC):
        """transformer_video输入维度[b,len,3,H,W]
        audio_MFCC输入维度[B,LEN,28,mfcc dim]
        输出维度都是[B,len(27),3dmm dim]
        """
        # [B,len(37),3dmm dim(64)]
        audio_exp=self.audio_model(audio_MFCC)
        # 抽出影响最大的n维
        audio_choose_exp_detach=audio_exp[...,self.audio_exp_list].detach()

        # [B,len(27),3dmm dim(64)]
        video_exp=self.video_model(transformer_video,audio_choose_exp_detach)

        fake_exp=video_exp.clone()
        fake_exp[...,self.audio_exp_list]+=audio_choose_exp_detach[:,self.win_size:-self.win_size]
        
        return audio_exp[:,self.win_size:-self.win_size],fake_exp
    
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
