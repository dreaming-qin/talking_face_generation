from torch import nn


# 测试代码
if __name__=='__main__':
    import os,sys
    path=sys.path[0]
    sys.path.pop(0)
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))



class SyncNetLoss(nn.Module):
    r'''返回render模块的Loss值'''
    def __init__(self):
        super(SyncNetLoss, self).__init__()
        self.logloss = nn.BCELoss()

    def forward(self, audio_embedding, mouth_embedding, label):
        r'''
        输入：
            audio_embedding (B, dim), tensor
            mouth_embedding (B, dim), tensor
            label (B), tensor
        '''
        gt_d = label.float().view(-1,1).to(audio_embedding.device)
        d = nn.functional.cosine_similarity(audio_embedding, mouth_embedding)
        loss = self.logloss(d.unsqueeze(1), gt_d)
        return loss
    
    


# 测试代码
if __name__=='__main__':
    import os,sys
    import torch
    import yaml,glob
    from src.dataset.sync_net_dataset import SyncNetDataset

    config={}
    yaml_file=['config/dataset/common.yaml',
               'config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=SyncNetDataset(config,type='train',sample_per_video=20)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fun=SyncNetLoss()
    for data in dataloader:
        for key,value in data.items():
            data[key]=value.to(device)
        a_e=torch.rand((len(data['label']),1024))
        m_e=torch.rand((len(data['label']),1024))
        ccc=loss_fun(a_e,m_e,data['label'])
        print(ccc)
