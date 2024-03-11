
import os
from moviepy.editor import VideoFileClip
import torch
import pickle
import zlib
import numpy as np
import glob
from tqdm import tqdm
import imageio
import torchvision
import shutil
from PIL import Image
from random import sample
from torch import nn



# test
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.data_process.video_process_util import video_to_3DMM_and_pose
from scipy.io import loadmat
from src.model.syncNet.sync_net import SyncNet
from src.util.util_3dmm import get_lm_by_3dmm,get_face
from Deep3DFaceRecon_pytorch.util.util import draw_landmarks
from src.util.model_util import cnt_params
from src.model.exp3DMM.self_attention_pooling import SelfAttentionPooling
from src.model.exp3DMM.fusion import PositionalEncoding





'''从pkl文件中生成视频'''
# with open('data/001.pkl','rb') as f:
#     info=f.read()
# info=zlib.decompress(info)
# info= pickle.loads(info)
# process_video=info['face_video']
# video_array=process_video
# imageio.mimsave('001.mp4',video_array,fps=30)

'''对齐视频用的在mateittalk的main_end2end.py的crop_image_tem中，其它预处理方法在其getdata()中'''

'''将音频无损的添加到视频中'''
# cmd ='ffmpeg -i {} -i {} -loglevel error -c copy -map 0:0 -map 1:1 -y -shortest {}'.format(
#     video,audio,out) 

'''横向合并视频'''
# command ='ffmpeg -i {} -i {} -loglevel error -y -shortest -filter_complex hstack=inputs=2 {}'.format(
#     emo_file,f'{result_path}/emotion.mp4','temp.mp4')

'''降采样为25帧'''
# f'ffmpeg -y -i {video_path} -loglevel error -r 25 temp.mp4'
'''转为图片序列'''
# cmd=f'ffmpeg -i {fake_video} -loglevel error -r 25 -y temp/%05d.png'


'''获得16k的音频'''
# audio_command = 'ffmpeg -i {} -loglevel error -y -f wav -acodec pcm_s16le -ar 16000 {}'.format(
#     video_path, '{}/{}'.format(audio_save_dir,os.path.basename('file.mov').replace('.pkl','.mov')))

'''获得第一帧作为源图片'''
# cmd = 'ffmpeg -i {} -loglevel error -y -vframes 1 {}'




if __name__=='__main__':
    # # 定义编码器，词典大小为10，要把token编码成128维的向量
    # embedding = nn.Embedding(10, 128)
    # # 定义transformer，模型维度为128（也就是词向量的维度）
    # transformer = nn.Transformer(d_model=128, batch_first=True) # batch_first一定不要忘记
    # # 定义源句子，可以想想成是 <bos> 我 爱 吃 肉 和 菜 <eos> <pad> <pad>
    # src = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2, 2]])
    # # 定义目标句子，可以想想是 <bos> I like eat meat and vegetables <eos> <pad>
    # tgt = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2]])
    # # 将token编码后送给transformer（这里暂时不加Positional Encoding）
    # outputs = transformer(embedding(src), embedding(tgt))
    # outputs.size()


    input=torch.ones((1,37,28,12))

    # 初始化模型
    num_encoder_layers=7
    feature_dim=256

    mapping_net=nn.Sequential(
        nn.Linear(12,64),
        nn.Linear(64,128),
        nn.ReLU(True),
        nn.Linear(128,256),
    )

    
    encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8,
                    dim_feedforward=512,batch_first=True)
    encoder_norm = nn.LayerNorm(feature_dim)
    encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    self_atten=SelfAttentionPooling(input_dim=feature_dim)

    # 音频长度28
    pos=PositionalEncoding(feature_dim, n_position=28)

    # test
    temp=0
    temp+=cnt_params(encoder)[0]
    temp+=cnt_params(mapping_net)[0]
    temp+=cnt_params(self_atten)[0]
    temp+=cnt_params(pos)[0]
    print(temp)

    # 前向传播
    out=mapping_net(input)

    B,L,_,dim=out.shape
    out=out.reshape(B*L,-1,dim)
    pos_embd=pos(out.shape[1])
    # pytorch中的transformer未进行pos操作，在这进行操作
    out=out+pos_embd
    out=encoder(out)

    out=self_atten(out)
    out=out.reshape(B,L,dim)
    print(out.shape)

    







