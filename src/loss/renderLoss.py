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


from src.util.model_util import freeze_params
from src.model.vgg19 import Vgg19

class RenderLoss(nn.Module):
    r'''返回render模块的Loss值'''
    def __init__(self, config,device):
        super(RenderLoss, self).__init__()
        self.device=device
        self.weight=config['weight']
        self.vgg_weight=config['vgg_weight']

        self.L1_loss=nn.L1Loss()

        self.vgg=Vgg19().to(device)
        freeze_params(self.vgg)

    def forward(self,predicted_video, data):
        r'''predicted_video[B,len,H,W,3]
        
        返回loss'''
        # 首先，比较L1 Loss
        gt_video=data['raw_video']
        loss_one=self.L1_loss(predicted_video,gt_video)

        # 其次，比较VGG Loss
        loss_two = 0
        x_video=predicted_video.permute(0,1,4,2,3)
        y_video=gt_video.permute(0,1,4,2,3)
        for x,y in zip(x_video,y_video):
            x_vgg = self.vgg(x)
            y_vgg = self.vgg(y)
            for i, weight in enumerate(self.vgg_weight):
                value = torch.abs(x_vgg[i] - y_vgg[i]).mean()
                loss_two += weight * value
        
        # 序列的GAN先留空，因为输入不是序列性的

        loss=loss_one*self.weight[0]+loss_two*self.weight[1]
        return loss
    

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml,glob
    from src.dataset.renderDtaset import RenderDataset

    config={}
    yaml_file=['config/data_process/common.yaml','config/dataset/common.yaml',
               'config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=RenderDataset(config,type='train',max_len=30)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fun=RenderLoss(config,device)
    for data in dataloader:
        for key,value in data.items():
            data[key]=value.to(device)
        ccc=loss_fun(data['raw_video'],data)