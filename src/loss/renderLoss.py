from torch import nn
import torch
import torch.nn.functional as F
from math import exp


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
    def __init__(self, config):
        super(RenderLoss, self).__init__()
        self.vgg=Vgg19()
        freeze_params(self.vgg)
        self.vgg.eval()

        self.L1_loss=nn.L1Loss()
        self.ssim_loss=SSIM()

        self.num_scales=config['num_scales']
        self.resize_mode=config['resize_mode']
        self.vgg_weight=config['vgg_weight']
        # 重构权重
        self.rec_weight=config['rec_low_weight']

    def forward(self,predicted_video, data,stage=None):
        r'''predicted_video[B,3,H,W]
        
        返回loss'''
        target=data['target']
        self.vgg=self.vgg.to(predicted_video.device)
        predicted_video, target = \
            apply_imagenet_normalization(predicted_video), \
            apply_imagenet_normalization(target)
        
        
        loss=0
        if stage == 'exp':
            # 比较L1 Loss
            loss_one=self.rec_weight[0]*self.L1_loss(predicted_video,target)
            # 获得ssim loss
            loss_two=(1-self.ssim_loss(predicted_video,target))
            loss_two=self.rec_weight[1]*loss_two
            # 序列的GAN先留空，因为输入不是序列性的
            loss+=loss_one+loss_two
            return loss
        
        for scale in range(self.num_scales):
            # 比较VGG Loss
            loss_three = 0
            # [B,3,H,W]
            x_vgg = self.vgg(predicted_video)
            y_vgg = self.vgg(target)
            for i, weight in enumerate(self.vgg_weight):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                loss_three += weight * value
            loss+=loss_three

            if stage != 'warp':
                # 比较L1 Loss
                loss_one=self.rec_weight[0]*self.L1_loss(predicted_video,target)
                # 获得ssim loss
                loss_two=(1-self.ssim_loss(predicted_video,target))
                loss_two=self.rec_weight[1]*loss_two
                # 序列的GAN先留空，因为输入不是序列性的
                loss+=loss_one+loss_two

            # Downsample the input and target.
            if scale != self.num_scales - 1:
                predicted_video = F.interpolate(
                    predicted_video, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        return loss

    def api_forward_for_exp_3dmm(self,predicted_video, gt,):
        data={'target':gt}
        return self(predicted_video,data,stage='exp')


def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
 
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
 
# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
 
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret
 
 
# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        # [B,3,H,W]
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml,glob
    from src.dataset.renderDtaset import RenderDataset

    config={}
    yaml_file=['config/dataset/common.yaml',
               'config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=RenderDataset(config,type='train',frame_num=5)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fun=RenderLoss(config)
    for data in dataloader:
        for key,value in data.items():
            data[key]=value.to(device)
        ccc=loss_fun(data['target'],data)
        ccc=loss_fun(data['target'],data,stage='warp')
