import torch

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

facemodel=ParametricFaceModel(
            bfm_folder='BFM', camera_distance=10.0, focal=1015.0, center=112.0,
            is_train=False, default_name='BFM_model_front.mat'
        )


@torch.no_grad()
def get_lm_by_3dmm(id_3dmm,exp_3dmm):
    r'''通过3dmm的身份和表情参数，获得2d唇部landmark
    参数：
        id_3dmm：(B，80)，tensor
        exp_3dmm：(B, 64)，tensor
    返回：
        landmark：(B,20,2)，tensor
    注意：
        （1）该方法主要为syncNet中需要将3dmm转为landmark做准备
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

    mouth_lamdmark=landmark[:,48:]

    return mouth_lamdmark



if __name__=='__main__':
    import zlib,pickle
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open('data/format_data/eval/10/angry_001.pkl','rb') as f:
        byte_file=f.read()
    byte_file=zlib.decompress(byte_file)
    data= pickle.loads(byte_file)
    data_3DMM=data['face_coeff']['coeff']
    face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
    face3d_exp=torch.tensor(face3d_exp).to(device)
    face3d_id=data_3DMM[:, 0:80]
    face3d_id=torch.tensor(face3d_id).to(device)
    lmd=get_lm_by_3dmm(face3d_id,face3d_exp)


    # test
    import numpy as np
    from Deep3DFaceRecon_pytorch.util.util import tensor2im, save_image,draw_landmarks
    for i in range(len(lmd)):
        image_numpy = np.zeros((1,224,224,3)).astype(np.uint8)
        image_numpy = draw_landmarks(image_numpy,lmd[i].cpu().numpy().reshape(1,-1,2))
        save_image(image_numpy.reshape(224,224,3), f'temp/{i}.jpg')
    cmd='ffmpeg -f image2 -i temp/%d.jpg -r 30 -y -loglevel error temp.mp4'
    os.system(cmd)
    cmd ='ffmpeg -i {} -i {} -loglevel error -c copy -map 0:0 -map 1:1 -y -shortest {}'.format(
        'temp.mp4','data/001.mp4','temp2.mp4') 
    os.system(cmd)
