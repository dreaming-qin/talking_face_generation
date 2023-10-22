import torch
from torch import nn

from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .fusion import Fusion


class Exp3DMM(nn.Module):
    '''
    输入音频MFCC和transformer后的视频，输出表情3DMM
    '''
    def __init__(self,cfg) :
        super().__init__()
        self.audio_encoder=AudioEncoder()
        self.video_encoder=VideoEncoder(**cfg['video_encoder'])
        self.fusion_module=Fusion(**cfg['fusion'])

    def forward(self, transformer_video, audio_MFCC):
        """
        Args:
            content (_type_): (B, num_frames, window, C_dmodel)
            style_code (_type_): (B, C_dmodel)

        Returns:
            face3d: (B, L_clip, C_3dmm)
        """
        audio_feature=self.audio_encoder(audio_MFCC)
        video_feature=self.video_encoder(transformer_video)
        exp3DMM=self.fusion_module(audio_feature,video_feature)
        return exp3DMM


