from torch import nn
import torch

class AudioEncoder(nn.Module):
    r'''音频编码器，输入一串音频的MFCC，输出音频特征，长度为256'''
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
        )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
        )
        # self.audio_eocder_fc = nn.Sequential(
        #     nn.Linear(1024 *12,128),
        #     nn.ReLU(True),
        #     nn.Linear(128,256),
        #     nn.ReLU(True),
        # )


    def forward(self,audio):
        '''audio输入维度[B,len,28,mfcc dim]
        输出维度[B,len,1024]'''
        audio_feature = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            audio_feature.append(current_feature)
        audio_feature = torch.stack(audio_feature, dim = 1)

        return audio_feature
    

def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        layer.append(activation())
    return layer
