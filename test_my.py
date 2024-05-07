
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
import random
import librosa
import soundfile as sf
import python_speech_features
from torch.multiprocessing import Pool, set_start_method
import scipy.io.wavfile as wave
from moviepy.editor import AudioFileClip
import torch.nn.functional as F
from scipy.io import loadmat
from PIL import Image
try:
    from PIL.Image import Resampling
    RESAMPLING_METHOD = Resampling.BICUBIC
except ImportError:
    from PIL.Image import BICUBIC
    RESAMPLING_METHOD = BICUBIC
import cv2




# test
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))


from Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from Deep3DFaceRecon_pytorch.util.nvdiffrast import MeshRenderer

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


'''无损将图片序列压缩为视频'''
# cmd='ffmpeg -i temp2/%4d.png -c:v libx265 -pix_fmt yuv420p -preset ultrafast -x265-params lossless=1 -y temp.mp4'


'''往data中加入path'''
if __name__=='__main__':
    set_start_method('spawn')

    device=torch.device('cuda')
    model=nn.Conv2d(3,3,(3,3)).to(device)
    x=torch.rand((1,3,6,6)).to(device)

    out=model(x)
    # out=F.interpolate(out, scale_factor=2)
    loss=out.mean()
    loss.backward()

    a=1