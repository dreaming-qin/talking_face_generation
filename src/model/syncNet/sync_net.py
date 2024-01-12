import torch
from torch import nn
from torch.nn import functional as F

if __name__=='__main__':
    import os,sys
    path=sys.path[0]
    sys.path.pop(0)
    for _ in range(3):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class SyncNet(nn.Module):
    def __init__(self, landmark_dim,audio_dim,**_):
        super(SyncNet, self).__init__()

        # hubert = torch.rand(B, 1, , t=10)
        self.hubert_encoder = nn.Sequential(
            Conv1d(audio_dim, 128, kernel_size=3, stride=1, padding=1),

            Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0),)


        # mouth = torch.rand(B, 20*3, t=5)
        self.mouth_encoder = nn.Sequential(
            Conv1d(landmark_dim, 96, kernel_size=3, stride=1, padding=1),

            Conv1d(96, 128, kernel_size=3, stride=1, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0),)
        self.logloss = nn.BCELoss()

    def forward(self, hubert, mouth_lm): 
        # hubert := (B, T=10, C=1024)
        # mouth_lm3d := (B, T=5, C=60)
        hubert = hubert.transpose(1,2)
        mouth_lm = mouth_lm.transpose(1,2)
        mouth_embedding = self.mouth_encoder(mouth_lm)
        audio_embedding = self.hubert_encoder(hubert)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        mouth_embedding = mouth_embedding.view(mouth_embedding.size(0), -1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        mouth_embedding = F.normalize(mouth_embedding, p=2, dim=1)
        return audio_embedding, mouth_embedding



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


    syncnet = SyncNet(**config)
    syncnet=syncnet.to(device)

    loss_fun=SyncNetLoss()

    for data in dataloader:
        for key,value in data.items():
            data[key]=value.to(device)
        a_e,m_e=syncnet(data['hubert'],data['mouth_landmark'])
        ccc=loss_fun(a_e,m_e,data['label'])
        print(ccc)

