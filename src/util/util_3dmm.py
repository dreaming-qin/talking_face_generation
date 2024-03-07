import torch
import numpy as np

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

def get_face(id_3dmm,exp_3dmm):
    r'''通过3dmm的身份和表情参数，获得人脸
    参数：
        id_3dmm：(B，80)，tensor
        exp_3dmm：(B, 64)，tensor
    返回：
        face：(B,3,224,224)，tensor
    '''
    device=id_3dmm.device
    facemodel.to(device)
    render.to(device)

    face_shape = facemodel.compute_shape(id_3dmm, exp_3dmm)
    # 旋转和平移设置为0
    rotation = facemodel.compute_rotation(torch.zeros(len(id_3dmm),3).to(device))
    face_shape_transformed = facemodel.transform(face_shape, rotation,torch.zeros(len(id_3dmm),3).to(device))
    pred_vertex = facemodel.to_camera(face_shape_transformed)

    pred_color_copy=pred_color.expand(len(exp_3dmm),-1,-1).to(device)
    _, _, pred_face = render(pred_vertex, facemodel.face_buf, feat=pred_color_copy)
    
    return pred_face

if __name__=='__main__':
    import zlib,pickle
    from PIL import Image
    import glob

    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    file_list=sorted(glob.glob('data_mead/format_data/test/0/*.pkl'))
    for file in file_list:
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        data_3DMM=data['face_coeff']['coeff']
        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
        face3d_exp=torch.tensor(face3d_exp).to(device)
        face3d_id=data_3DMM[:, 0:80]
        face3d_id=torch.tensor(face3d_id).to(device)
        face=get_face(face3d_id,face3d_exp)

        pic=face[0].cpu().numpy().transpose(1,2,0)
        pic=(pic*255).astype(np.uint8)
        image_pil = Image.fromarray(pic)

        # test
        gray=0.299*image_pil[...,0]+0.587*image_pil[...,1]+0.114*image_pil[...,2]
        gray=gray.astype(np.uint8)
        image_pil = Image.fromarray(gray)

        # image_pil=image_pil.convert('L')
        image_pil.save('temp.png')

        face2=get_face(face3d_id,face3d_exp)

        pic=face2[0].cpu().numpy().transpose(1,2,0)
        image_pil = Image.fromarray((pic*255).astype(np.uint8))
        image_pil=image_pil.convert('L')
        image_pil.save('temp2.png')

