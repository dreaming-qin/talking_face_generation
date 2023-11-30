
import os
from moviepy.editor import VideoFileClip
import torch
import pickle
import zlib
import numpy as np


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

'''查看改了尺寸后的pkl文件有什么不同不'''


def process_pose(data):
    """Get pose parameters from mat file

    Returns:
        pose_params (numpy.ndarray): shape (L_video, 9), angle, translation, crop paramters
    """
    mat_dict = data

    np_3dmm = mat_dict["coeff"]
    angles = np_3dmm[:, 224:227]
    translations = np_3dmm[:, 254:257]

    np_trans_params = mat_dict["transform_params"]
    crop = np_trans_params[:, -3:]

    # [len,9]
    pose_params = np.concatenate((angles, translations, crop), axis=1)

    
    return np.array(pose_params).tolist()

def process_exp_3DMM(data):
    data_3DMM=data['coeff']

    # [len,64]
    face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range


    return np.array(face3d_exp).tolist()


if __name__=='__main__':
    # video_to_3DMM_and_pose('data')
    # 加载mat
    data_crop=loadmat('data/0.mat')
    data_crop_pose,data_crop_exp=process_pose(data_crop),process_exp_3DMM(data_crop)
    data_no_crop=loadmat('data/1.mat')
    data_no_crop_pose,data_no_crop_exp=process_pose(data_no_crop),process_exp_3DMM(data_no_crop)

    # 解压pkl文件
    with open('data/angry_001.pkl','rb') as f:
        byte_file=f.read()
    byte_file=zlib.decompress(byte_file)
    data_yuan= pickle.loads(byte_file)
    data_yuan=data_yuan['face_coeff']
    data_yuan_pose,data_yuan_exp=process_pose(data_yuan),process_exp_3DMM(data_yuan)
    data_yuan=1

    aaa=1