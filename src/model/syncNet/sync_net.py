import torch
from torch import nn
from torch.nn import functional as F

# test
if __name__=='__main__':
    import os,sys
    path=sys.path[0]
    sys.path.pop(0)
    for _ in range(3):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.model_util import cnt_params
from src.model.exp3DMM.audio_encoder import AudioEncoder
from src.model.exp3DMM.video_encoder import VideoEncoder
from src.util.util_3dmm import get_lm_by_3dmm, get_face


class SyncNet(nn.Module):
    def __init__(self, cfg):
        super(SyncNet, self).__init__()

        self.audio_encoder=AudioEncoder(**cfg['audio_encoder'])
        self.video_encoder=VideoEncoder(**cfg['video_encoder'])

        cnt,_=cnt_params(self.audio_encoder)
        print(f"SyncNet audio_encoder total paras number: {cnt}")
        cnt,_=cnt_params(self.video_encoder)
        print(f"SyncNet video_encoder total paras number: {cnt}")

    def forward(self, audio_mfcc, mouth_img): 
        """transformer_video输入维度[b,3,H,W]
        audio_MFCC输入维度[B,28,mfcc dim]"""

        # [B, audio feature dim]
        audio_feature=self.audio_encoder(audio_mfcc.unsqueeze(1))
        audio_feature=audio_feature.squeeze(1)
        # [B, video feature dim]
        video_feature=self.video_encoder(mouth_img)

        audio_feature = F.normalize(audio_feature, p=2, dim=1)
        video_feature = F.normalize(video_feature, p=2, dim=1)

        return audio_feature,video_feature



if __name__ == '__main__':
    import os,sys
    import yaml
    from src.dataset.sync_net_dataset import SyncNetDataset
    from src.loss.sync_net_loss import SyncNetLoss

    config={}
    yaml_file=['config/dataset/common.yaml',
               'config/model/sync_net.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=SyncNetDataset(config,sample_per_video=20)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    syncnet = SyncNet(config)
    syncnet=syncnet.to(device)

    loss_fun=SyncNetLoss()
    with torch.no_grad():
        for data in dataloader:
            for key,value in data.items():
                data[key]=value.to(device)

            # 将exp和id转为灰度图[B,3,H,W]
            mouth_img=get_face(data[f'id'].reshape(-1,80),
                data[f'exp'].reshape(-1,64))[...,120:170,70:150]
            # 转成256*256大小的唇部图形
            mouth_img = F.interpolate(mouth_img, mode='bilinear', size=(256,256),align_corners=False)

            a_e,v_e=syncnet(data['mfcc'],mouth_img)
            ccc=loss_fun(a_e,v_e,data['label'])
            print(ccc)

