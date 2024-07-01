import torch
import numpy as np
import imageio
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip


# test 
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.pop(0)
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from Deep3DFaceRecon_pytorch.util.nvdiffrast import MeshRenderer

facemodel=ParametricFaceModel(
            bfm_folder='BFM', camera_distance=10.0, focal=1015.0, center=112.0,
            is_train=False, default_name='BFM_model_front.mat'
        )

render=MeshRenderer(
            rasterize_fov=12.59363743796881, znear=5.0, zfar=15.0, 
            rasterize_size=224, use_opengl=False
        )

# 标准面部上色模板
pred_color=np.load('color.npy')
pred_color=torch.tensor(pred_color).float().contiguous()

def get_lm_by_3dmm(id_3dmm,exp_3dmm):
    r'''通过3dmm的身份和表情参数，获得2d唇部landmark
    参数：
        id_3dmm：(B，80)，tensor
        exp_3dmm：(B, 64)，tensor
    返回：
        landmark：(B,68,2)，tensor
    '''
    device=id_3dmm.device
    facemodel.to(device)
    face_shape = facemodel.compute_shape(id_3dmm, exp_3dmm)

    # 旋转和平移设置为0
    rotation = facemodel.compute_rotation(torch.zeros(len(id_3dmm),3).to(device))
    face_shape_transformed = facemodel.transform(face_shape, rotation,torch.zeros(len(id_3dmm),3).to(device))

    face_vertex = facemodel.to_camera(face_shape_transformed)
    
    face_proj = facemodel.to_image(face_vertex)
    landmark = facemodel.get_landmarks(face_proj)

    return landmark

def get_face(id_3dmm,exp_3dmm,angle=None,translation=None):
    r'''通过3dmm的身份和表情参数，获得旋转和平移为0的人脸
    参数：
        id_3dmm：(B，80)，tensor
        exp_3dmm：(B, 64)，tensor
        angle(3)，tensor ，为None时将设置为0
        translation(B, 3)，为None时将设置为0
    返回：
        face：(B,3,224,224)，tensor
    '''
    device=id_3dmm.device
    facemodel.to(device)
    render.to(device)

    face_shape = facemodel.compute_shape(id_3dmm, exp_3dmm)
    # 旋转和平移设置为0
    if angle is None:
        angle=torch.zeros(len(id_3dmm),3).to(device)
    if translation is None:
        translation = torch.zeros(len(id_3dmm),3).to(device)

    rotation = facemodel.compute_rotation(angle)
    face_shape_transformed = facemodel.transform(face_shape, rotation,translation)
    pred_vertex = facemodel.to_camera(face_shape_transformed)

    pred_color_copy=pred_color.expand(len(exp_3dmm),-1,-1).to(device)
    _, _, pred_face = render(pred_vertex, facemodel.face_buf, feat=pred_color_copy)
    
    return pred_face

def save_face_video(id_3dmm,exp_3dmm,save_file,audio_file=None):
    '''通过3dmm的身份和表情参数，渲染人脸并保存
    参数：
        id_3dmm：(B，80)，tensor
        exp_3dmm：(B, 64)，tensor
    '''
    face=get_face(id_3dmm,exp_3dmm)
    face=(face*255).permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)
    imageio.mimsave('dhaj.mp4',face,fps=25)
    if audio_file is None:
        shutil.copy('dhaj.mp4',save_file)
    else:
        # 读取视频和音频文件
        video_clip = VideoFileClip("dhaj.mp4")
        audio_clip = AudioFileClip(audio_file)
        # 将音频与视频合成为新的视频
        video_with_audio = video_clip.set_audio(audio_clip)
        # 保存新的视频文件
        video_with_audio.write_videofile(save_file, codec="libx264", fps=25)
    os.remove('dhaj.mp4')


if __name__=='__main__':
    import zlib,pickle
    from PIL import Image
    import glob

    device=torch.device('cuda:1')

    file='data_mead/format_data/test/0/M005_front_surprised_level_3_003.pkl'
    file='data_mead/format_data/test/0/M003_front_angry_level_3_025.pkl'
    with open(file,'rb') as f:
        byte_file=f.read()
    byte_file=zlib.decompress(byte_file)
    data= pickle.loads(byte_file)
    data_3DMM=data['face_coeff']['coeff']
    face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
    face3d_exp=torch.tensor(face3d_exp).to(device)

    for i in range(64):
        os.makedirs('temp/one',exist_ok=True)
        face3d_exp=torch.zeros_like(face3d_exp)
        face3d_exp[...,i]=1
        face3d_id=data_3DMM[:, 0:80]
        face3d_id=torch.tensor(face3d_id).to(device)
        save_face_video(face3d_id,face3d_exp,f'temp/one/{i}.mp4',data['path'])

        os.makedirs('temp/zero',exist_ok=True)
        face3d_exp=torch.zeros_like(face3d_exp)
        face3d_exp[...,i]=0
        face3d_id=data_3DMM[:, 0:80]
        face3d_id=torch.tensor(face3d_id).to(device)
        save_face_video(face3d_id,face3d_exp,f'temp/zero/{i}.mp4',data['path'])

        os.makedirs('temp/fu_one',exist_ok=True)
        face3d_exp=torch.zeros_like(face3d_exp)
        face3d_exp[...,i]=-1
        face3d_id=data_3DMM[:, 0:80]
        face3d_id=torch.tensor(face3d_id).to(device)
        save_face_video(face3d_id,face3d_exp,f'temp/fu_one/{i}.mp4',data['path'])
