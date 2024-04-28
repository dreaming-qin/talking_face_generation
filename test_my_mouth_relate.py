
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




'''往data中加入path'''
if __name__=='__main__':
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    L1_function=torch.nn.L1Loss()
    error_dict={}
    cnt=0

    face_landmark_index=[i for i in range(48,68)]+[7,8,9]
    no_face_landmark_index=[i for i in range(68) if i not in face_landmark_index]

    # 测试3dmm哪个参数和唇部最有关
    file_list=sorted(glob.glob('data_mead/format_data/test/0/*.pkl'))
    for file in tqdm(file_list):
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        data_3DMM=data['face_coeff']['coeff']
        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
        face3d_exp=torch.tensor(face3d_exp).to(device)
        face3d_id=data_3DMM[:, 0:80]
        face3d_id=torch.tensor(face3d_id).to(device)

        # 1.随机选择10个3DMM组
        index=sample(range(len(face3d_id)), 10)
        face3d_id=face3d_id[index]
        face3d_exp=face3d_exp[index]
        gt_landmark=get_face(face3d_id,face3d_exp)

        # 2. 对3DMM每个加1，观察唇部和非唇部关键点的变换幅值
        # 次数加1
        cnt+=1
        for i in range(1,65):
            if i not in error_dict:
                error_dict[i]={}
                error_dict[i]['face']=0
                error_dict[i]['no_face']=0
            face3d_exp_copy=face3d_exp.clone().detach()
            face3d_exp_copy[:,i-1]+=1
            landmark=get_face(face3d_id,face3d_exp_copy)


            # # face彩图版本的
            # # mouth_img=landmark[...,120:170,70:150]
            # error=torch.abs(gt_landmark-landmark)
            # face_error=error[...,120:210,70:150].sum().item()
            # error_dict[i]['face']+=face_error
            # error_dict[i]['no_face']+=error.sum().item()-face_error


            # face灰图版本的
            gt_gray=0.299*gt_landmark[:,0]+0.587*gt_landmark[:,1]+0.114*gt_landmark[:,2]
            gray=0.299*landmark[:,0]+0.587*landmark[:,1]+0.114*landmark[:,2]
            error=torch.abs(gt_gray-gray)
            face_error=error[...,120:210,70:150].sum().item()
            error_dict[i]['face']+=face_error
            error_dict[i]['no_face']+=error.sum().item()-face_error

            # # test，存图片，左边是原图，右边是3dmm加1后的图片
            # for j in range(1):
            #     real_img=gt_gray[j].cpu().numpy()
            #     fake_img=gray[j].cpu().numpy()
            #     img=np.concatenate((real_img,fake_img),axis=1)
            #     img=(img*255).astype(np.uint8)
            #     imageio.imsave(f'temp/{i}.png',img)


            # # landmark版本的
            # error=torch.abs(gt_landmark-landmark)
            # face_error=error[:,face_landmark_index].sum()
            # no_face_error=error[:,no_face_landmark_index].sum()
            # error_dict[i]['face']+=face_error.item()
            # error_dict[i]['no_face']+=no_face_error.item()

            # # test，存图片，左边是原图，右边是3dmm加1后的图片
            # if i==1:
            #     img=np.zeros((1,224,224,3))
            #     lmark=gt_landmark[0].unsqueeze(0).cpu().numpy()
            #     real_img=draw_landmarks(img,lmark)[0]
            # img=np.zeros((1,224,224,3))
            # lmark=landmark[0].unsqueeze(0).cpu().numpy()
            # fake_img=draw_landmarks(img,lmark)[0]
            # img=np.concatenate((real_img,fake_img),axis=1)
            # os.makedirs(f'temp/{os.path.basename(file)}',exist_ok=True)
            # imageio.imsave(f'temp/{os.path.basename(file)}/{i}.png',img.astype(np.uint8))

            

    # 3. 求均值，打印结果
    face_sum=0
    no_face_sum=0
    for key in range(1,65):
        face_sum+=error_dict[key]['face']
        no_face_sum+=error_dict[key]['no_face']

    # # 纯数值版本的
    # face_sum=1
    # no_face_sum=1

    for key in range(1,65):
        face_mean=error_dict[key]['face']
        no_face_mean=error_dict[key]['no_face']
        print('表情系数%s：对唇部的影响是%.3f%%，对非唇部的影响是%.3f%%' % (key,(face_mean/face_sum)*100,
            (no_face_mean/no_face_sum)*100))

            




