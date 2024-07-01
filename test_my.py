
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
import matplotlib.pyplot as plt
from torch.multiprocessing import  set_start_method
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
from src.dataset.renderDtaset import RenderDataset

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


facemodel=ParametricFaceModel(
            bfm_folder='BFM', camera_distance=10.0, focal=1015.0, center=112.0,
            is_train=False, default_name='BFM_model_front.mat'
        )

render=MeshRenderer(
            rasterize_fov=12.59363743796881, znear=5.0, zfar=15.0, 
            rasterize_size=224, use_opengl=False
        )


class MaskFormat():
    def format(self,pred_mask,data):
        out_images=[]
        for i in range(len(pred_mask)):
            t=np.zeros((2,1))
            t[0,0]=data['face_coeff']['transform_params'][i][3]
            t[1,0]=data['face_coeff']['transform_params'][i][4]
            s=data['face_coeff']['transform_params'][i][2]
            out_img= self.image_transform(pred_mask[i][0],s,t)
            out_images.append(out_img[None])
        return torch.stack(out_images, 0)

    def image_transform(self, images,s,t):
        img= self.align_img(images,s,t)        
        return img    


    # utils for face reconstruction
    def align_img(self,img, s,t,target_size=224.):
        """
        Return:
            transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
            img_new            --PIL.Image  (target_size, target_size, 3)
            lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
            mask_new           --PIL.Image  (target_size, target_size)
        
        Parameters:
            img                --PIL.Image  (raw_H, raw_W, 3)
            lm                 --numpy.array  (68, 2), y direction is opposite to v direction
            lm3D               --numpy.array  (5, 3)
            mask               --PIL.Image  (raw_H, raw_W, 3)
        """

        # processing the image
        img_new = self.resize_n_crop_img(img, t, s, target_size=target_size)

        return img_new

    # resize and crop images for face reconstruction
    def resize_n_crop_img(self,img, t, s, target_size=224.):
        w0, h0 = 256,256
        w = int(w0*s)
        h = int(h0*s)
        left = int(w/2 - target_size/2 + float((t[0][0] - w0/2)*s))
        right_left=max(0,left)
        right = int(left + target_size)
        right_right=min(w,right)
        up = int(h/2 - target_size/2 + float((h0/2 - t[1])*s))
        right_up=max(0,up)
        below = int(up + target_size)
        right_below=min(h,below)

        new_mask=torch.zeros((h,w)).to(img)
        new_mask[right_up:right_below,right_left:right_right]=\
            img[right_up-up:224-below+right_below,right_left-left:224-right+right_right]
        new_mask = F.interpolate(new_mask.reshape(1,1,h,w), size = (h0, w0), mode='bilinear')
        new_mask=new_mask.reshape(h0,w0)

        return new_mask


class ImgFormat():
    def format(self,pred_face,data):
        out_images=[]
        for i in range(len(pred_face)):
            t=np.zeros((2,1))
            t[0,0]=data['face_coeff']['transform_params'][i][3]
            t[1,0]=data['face_coeff']['transform_params'][i][4]
            s=data['face_coeff']['transform_params'][i][2]
            out_img= self.image_transform(pred_face[i],s,t)
            out_images.append(out_img[None])
        return torch.cat(out_images, 0)

    def image_transform(self, images,s,t):
        img= self.align_img(images,s,t)        
        return img    


    # utils for face reconstruction
    def align_img(self,img, s,t,target_size=224.):
        """
        Return:
            transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
            img_new            --PIL.Image  (target_size, target_size, 3)
            lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
            mask_new           --PIL.Image  (target_size, target_size)
        
        Parameters:
            img                --PIL.Image  (raw_H, raw_W, 3)
            lm                 --numpy.array  (68, 2), y direction is opposite to v direction
            lm3D               --numpy.array  (5, 3)
            mask               --PIL.Image  (raw_H, raw_W, 3)
        """

        # processing the image
        img_new = self.resize_n_crop_img(img, t, s, target_size=target_size)

        return img_new

    # resize and crop images for face reconstruction
    def resize_n_crop_img(self,img, t, s, target_size=224.):
        w0, h0 = 256,256
        w = int(w0*s)
        h = int(h0*s)
        left = int(w/2 - target_size/2 + float((t[0][0] - w0/2)*s))
        right_left=max(0,left)
        right = int(left + target_size)
        right_right=min(w,right)
        up = int(h/2 - target_size/2 + float((h0/2 - t[1])*s))
        right_up=max(0,up)
        below = int(up + target_size)
        right_below=min(h,below)

        new_mask=torch.zeros((3,h,w)).to(img)
        new_mask[:,right_up:right_below,right_left:right_right]=\
            img[:,right_up-up:224-below+right_below,right_left-left:224-right+right_right]
        new_mask = F.interpolate(new_mask.reshape(1,3,h,w), size = (h0, w0), mode='bilinear')
        new_mask=new_mask.reshape(3,h0,w0)

        return new_mask



def compute_visuals(input_img,pred_face,pred_mask):
    '''input_img (B,3,224,224)'''
    with torch.no_grad():
        output_vis = pred_face * pred_mask + (1 - pred_mask) * input_img
        output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
        
        output_vis_numpy = output_vis_numpy_raw

        output_vis = torch.tensor(
                output_vis_numpy / 255., dtype=torch.float32
            ).permute(0, 3, 1, 2).to(pred_face)
    return output_vis


'''往data中加入path'''
if __name__=='__main__':
    set_start_method('spawn')
    r'''
    生成fake face video
    对mead，使用3DMM生成人脸作为fake face video
    对vox，使用face video作为fake face video
    '''

    device=torch.device('cuda:0')
    facemodel.to(device)
    render.to(device)
    img_format=ImgFormat()
    mask_format=MaskFormat()

    pkl_list=sorted(glob.glob(f'data_mix/format_data/*/*/*.pkl'))
    for pkl_file in tqdm(pkl_list):
        # 解压pkl文件
        with open(pkl_file,'rb') as f:
            # data= pickle.load(f)
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)

        # 把3DMM拿出来，生成人脸
        output_coeff=torch.tensor(data['face_coeff']['coeff']).to(device)
        pred_vertex, pred_tex, pred_color, pred_lm = facemodel.compute_for_render(output_coeff)
        _, _, pred_face = render(pred_vertex, facemodel.face_buf, feat=pred_color)

        # img=img_format.format(data).to(device)
        pred_face=img_format.format(pred_face,data)
        real=data['face_video']
        img=(torch.tensor(real).to(device)/255).permute(0,3,1,2)
        fake=compute_visuals(img,pred_face,pred_mask)
        fake=(fake.permute(0,2,3,1).detach().cpu().numpy()*255).astype(np.uint8)

        # 存在data_mead_fake_face中
        assert fake.shape==data['face_video'].shape
        data['fake_face_video']=fake
        info = pickle.dumps(data)
        info=zlib.compress(info)
        os.makedirs(os.path.dirname(pkl_file.replace('data_mix','data_mix_fake_face')),exist_ok=True)
        with open(pkl_file.replace('data_mix','data_mix_fake_face'),'wb') as f:
            f.write(info)

        
        # 解压测试
        with open(pkl_file.replace('data_mix','data_mix_fake_face'),'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        video=data['path']

        # out=np.concatenate((real,fake),axis=2)
        # imageio.mimsave('aaa.mp4',out,fps=25)






