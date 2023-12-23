import numpy as np
import glob
import os

# test 
if __name__=='__main__':
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.data_process.video_process_util import video_to_3DMM_and_pose
from scipy.io import loadmat
from src.util.util import remove_file



'''AED取值范围[0,正无穷]，越小越好'''


def AED_by_dir(predict_video_dir,gt_video_dir,is_get_3dmm=False,is_delete_3dmm_file=False):
    '''由于每执行一次模型都要初始化，因此按照dir的形式抓取更有效率
    获得所有文件夹的结果'''
    if is_get_3dmm:
        video_to_3DMM_and_pose(predict_video_dir)
        video_to_3DMM_and_pose(gt_video_dir)

    # 获得信息后，开始比对
    fake_mats=sorted(glob.glob(f'{predict_video_dir}/*.mat'))
    real_mats=sorted(glob.glob(f'{gt_video_dir}/*.mat'))
    distance=[]
    for fake_mat,real_mat in zip(fake_mats,real_mats):
        assert os.path.basename(fake_mat)==os.path.basename(real_mat)
        fake_exp,_=get_exp_and_pose(fake_mat)
        real_exp,_=get_exp_and_pose(real_mat)

        # 计算L1距离，获得一个视频的平均APD
        min_len=min(len(fake_exp),len(real_exp))
        distance.append(np.abs(fake_exp[:min_len]-real_exp[:min_len]).mean())

        # 删除无用文件
        remove_file(fake_mat.replace('.mat','.txt'))
        remove_file(real_mat.replace('.mat','.txt'))
        if is_delete_3dmm_file:
            remove_file(fake_mat)
            remove_file(real_mat)

    return sum(distance)/len(distance)


def get_exp_and_pose(mat_file):
    fake_data=loadmat(mat_file)

    data_3DMM=fake_data['coeff']
    # [len,64]
    face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range

    mat_dict = fake_data
    np_3dmm = mat_dict["coeff"]
    angles = np_3dmm[:, 224:227]
    translations = np_3dmm[:, 254:257]
    np_trans_params = mat_dict["transform_params"]
    crop = np_trans_params[:, -3:]
    # [len,9]
    pose_params = np.concatenate((angles, translations), axis=1)

    return face3d_exp,pose_params

if __name__=='__main__':

    method=['atvg','eamm','wav2lip_no_pose','make','pc_avs','exp3DMM']

    # test, 以一个视频为单位
    for a in method:
        with open(f'result/{a}/result/AED.txt','w',encoding='utf8') as f:
            
            video_to_3DMM_and_pose(f'result/{a}/result/fake')
            video_to_3DMM_and_pose(f'result/{a}/result/real')

            # 获得信息后，开始比对
            fake_mats=sorted(glob.glob(f'result/{a}/result/fake/*.mat'))
            real_mats=sorted(glob.glob(f'result/{a}/result/real/*.mat'))
            distance=[]
            for fake_mat,real_mat in zip(fake_mats,real_mats):
                assert os.path.basename(fake_mat)==os.path.basename(real_mat)
                fake_exp,_=get_exp_and_pose(fake_mat)
                real_exp,_=get_exp_and_pose(real_mat)

                # 计算L1距离，获得一个视频的平均APD
                temp=np.abs(fake_exp-real_exp).mean()
                f.write(f'{os.path.basename(fake_mat)}\t{temp}\n')