from torch import nn

class Exp3DMMLoss(nn.Module):
    r'''返回'''
    def __init__(self, mouth_weight,exp_weight,**_):
        super(Exp3DMMLoss, self).__init__()
        self.mouth_weight=mouth_weight
        self.exp_weight=exp_weight

    def forward(self,predicted_exp_3DMM, GT_exp_3DMM):
        # 计算关于唇部的Loss

        # 计算关于表情的Loss

        return torch.sum(1 - cosine_d) / cosine_d.shape[0]     