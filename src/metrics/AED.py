import numpy as np
import glob

# test 
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.data_process.video_process_util import video_to_3DMM_and_pose
from scipy.io import loadmat
from src.util.util import remove_file
from src.metrics.APD import get_exp_and_pose



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
        distance.append(np.abs(fake_exp-real_exp).mean())

        # 删除无用文件
        remove_file(fake_mat.replace('.mat','.txt'))
        remove_file(real_mat.replace('.mat','.txt'))
        if is_delete_3dmm_file:
            remove_file(fake_mat)
            remove_file(real_mat)

    return sum(distance)/len(distance)


if __name__=='__main__':
    print(AED_by_dir('temp','temp',is_get_3dmm=False))