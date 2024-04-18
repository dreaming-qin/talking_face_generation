
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



# test
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.data_process.video_process_util import video_to_3DMM_and_pose,get_face_image
from scipy.io import loadmat
from src.model.syncNet.sync_net import SyncNet
from src.util.util_3dmm import get_lm_by_3dmm,get_face
from Deep3DFaceRecon_pytorch.util.util import draw_landmarks
from src.util.model_util import cnt_params
from src.model.exp3DMM.self_attention_pooling import SelfAttentionPooling
from src.model.exp3DMM.fusion import PositionalEncoding

from src.util.data_process.audio_config_pc_avs import AudioConfig




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


audio_process=AudioConfig(num_frames_per_clip=5,hop_size=160)

def format_data(file):
    with open(file,'rb') as f:
        byte_file=f.read()
    byte_file=zlib.decompress(byte_file)
    data= pickle.loads(byte_file)
    video_file=data['path']
    
    # fps指定帧数，保证音频片段和视频片段同步
    audio_file=video_file.replace('.mp4','.wav')
    my_audio_clip = AudioFileClip(video_file)
    my_audio_clip.write_audiofile(audio_file,fps=16000)

    
    '''将音频转为MFCC'''
    speech, samplerate = sf.read(audio_file)
    # 16kHz对应25帧的换算方式
    fps=samplerate*25/16000
    if len(speech.shape)==2:
        speech=speech[:,0]
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,samplerate,winstep=1/(fps*4))

    ind = 3
    input_mfcc = []
    while ind <= int(mfcc.shape[0]/4) - 4:
        t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4]
        input_mfcc.append(t_mfcc)
        ind += 1
    input_mfcc=np.array(input_mfcc)

    
    print('进程{}处理{}，裁剪了{}'.format(os.getpid(),file,len(data['face_video'])-len(input_mfcc)))
    if len(input_mfcc)>=len(data['face_video']):
        input_mfcc=input_mfcc[:len(data['face_video'])]
    else:
        temp_last=torch.tensor(input_mfcc[-1]).expand(len(data['face_video'])-len(input_mfcc),-1,-1)
        input_mfcc=np.concatenate((input_mfcc,temp_last.numpy()))

    assert len(data['face_video'])==len(input_mfcc)

    data['audio_mfcc']=np.array(input_mfcc)
    # 存入新文件
    new_file=file.replace('data_mead2','data_mead3')
    os.makedirs(os.path.dirname(new_file),exist_ok=True)
    data = pickle.dumps(data)
    data=zlib.compress(data)
    with open(new_file,'wb') as f:
        f.write(data)
    
    # 删除音频文件
    os.remove(audio_file)


'''往data中加入path'''
if __name__=='__main__':
    set_start_method('spawn')
    
    with open('data_mead/format_data/eval/0/M003_front_angry_level_3_015.pkl','rb') as f:
        byte_file=f.read()
    byte_file=zlib.decompress(byte_file)
    data= pickle.loads(byte_file)

    # 重新处理音频，生成data_vox2
    file_list=glob.glob('data_mead/format_data/test/*/*.pkl')
    random.shuffle(file_list)
    file_list=file_list[:50]
    temp=[]
    for file in file_list:
        temp.append(os.path.abspath(file))
    np.save('test_dataset.npy',temp)

    








