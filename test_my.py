
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


# test
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.data_process.video_process_util import video_to_3DMM_and_pose
from scipy.io import loadmat





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
# cmd ='ffmpeg -i {} -i {} -loglevel error -c copy -map 0:0 -map 1:1 -y {}'.format(
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




'''往data中加入path'''
if __name__=='__main__':
    # 创建视频
    file_list=sorted(glob.glob(f'data2/format_data/*/*/*.pkl'))
    for file in tqdm(file_list):
        # 解压pkl文件
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)

        face_video=data['face_video']

        save_file=file.replace('data2','video').replace('.pkl','.mp4')
        os.makedirs(os.path.dirname(save_file),exist_ok=True)

        # 存结果为视频，需要加上音频
        reader = imageio.get_reader(data['path'])
        # 获得fps
        fps = reader.get_meta_data()['fps']
        reader.close()
        # 先把视频存好
        imageio.mimsave(save_file,face_video,fps=fps)