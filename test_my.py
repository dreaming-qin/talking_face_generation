
import os
from moviepy.editor import VideoFileClip
import torch
import pickle
import zlib
import numpy as np
import glob
from tqdm import tqdm


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

'''将视频中的音频与另一个视频文件结合'''
# def add_mp3(video_src1, video_src2, video_dst):
#     ' 将video_src1的音频嵌入video_src2视频中'
#     video_src1 = VideoFileClip(video_src1)
#     video_src2 = VideoFileClip(video_src2)
#     audio = video_src1.audio
#     videoclip2 = video_src2.set_audio(audio)
#     videoclip2.write_videofile(video_dst, codec='libx264')

# video_src1 = '001.mp4'
# video_src2 = 'temp.mp4'
# video_dst = '002.mp4'
# add_mp3(video_src1, video_src2, video_dst)


'''往data中加入path'''
if __name__=='__main__':
    
    file_list = glob.glob(f'data2/*/*/*/*.pkl')
    for file in tqdm(file_list):
        # 解压pkl文件
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        # test
        str_clip=os.path.basename(file).replace('.pkl','.mp4').split('_')
        data['path']=f'/workspace/dataset/MEAD/{str_clip[0]}/video/{str_clip[1]}/{str_clip[2]}'+\
            f'/{str_clip[3]}_{str_clip[4]}/{str_clip[5]}'
        data.pop('align_video')
        
        # 获得最终数据，使用压缩操作
        save_file=file.replace('data2','data')
        os.makedirs(os.path.dirname(save_file),exist_ok=True)
        info = pickle.dumps(data)
        info=zlib.compress(info)
        with open(save_file,'wb') as f:
            f.write(info)