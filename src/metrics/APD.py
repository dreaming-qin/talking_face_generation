import numpy as np
import glob
import os
import cv2
import imageio
import matplotlib.pyplot as plt

# test 
if __name__=='__main__':
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'TDDFA_V2_master'))

from TDDFA_V2_master.demo import get_pose



'''APD取值范围[0,正无穷]，越小越好'''


def APD(predict_video,gt_video):
    '''输入维度(len,H,W,3)'''
    fake_pose=[]
    real_pose=[]
    for predict,gt in zip(predict_video,gt_video):
        predict = cv2.cvtColor(predict,cv2.COLOR_RGB2BGR)
        gt = cv2.cvtColor(gt,cv2.COLOR_RGB2BGR) 
        fake_result=get_pose(predict)
        real_result=get_pose(gt)
        if fake_result is None or real_result is None:
            continue
        # predict是(H,W,3)
        fake_pose.append(fake_result)
        real_pose.append(real_result)
    fake_pose,real_pose=np.array(fake_pose),np.array(real_pose)
    # 删除缩放因子s
    fake_pose=np.delete(fake_pose, 3, axis=1)
    real_pose=np.delete(real_pose, 3, axis=1)
    distance=np.abs(fake_pose-real_pose).mean()
    return distance

def APD_by_path(predict_video_file,gt_video_file):
    pre_reader = imageio.get_reader(predict_video_file)
    predict_video=[im for im in pre_reader]
    pre_reader.close()

    gt_reader = imageio.get_reader(gt_video_file)
    gt_video=[im for im in gt_reader]
    gt_reader.close()

    return APD(predict_video,gt_video)


def APD_by_dir(predict_video_dir,gt_video_dir):
    ans=[]
    predict_video_dir=sorted(glob.glob(f'{predict_video_dir}/*.mp4'))
    gt_video_dir=sorted(glob.glob(f'{gt_video_dir}/*.mp4'))
    for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
        assert os.path.basename(predict_video_file)==os.path.basename(gt_video_file)
        temp=APD_by_path(predict_video_file,gt_video_file)
        if temp!=-1:
            ans.append(temp)
    if len(ans)==0:
        # -1代表不成功
        return -1
    average_distance = sum(ans) / len(ans)
    return average_distance



if __name__=='__main__':

    method=['atvg','eamm','wav2lip_no_pose','make','pc_avs','exp3DMM']
    # method=['atvg','pc_avs']

    # # test, 以一个视频为单位
    # for a in method:
    #     with open(f'result/{a}/result/APD.txt','w',encoding='utf8') as f:
    #         ans=[]
    #         predict_video_dir=sorted(glob.glob(f'result/{a}/result/fake/*.mp4'))
    #         gt_video_dir=sorted(glob.glob(f'result/{a}/result/real/*.mp4'))
    #         for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
    #             temp= APD_by_path(predict_video_file, gt_video_file)
    #             f.write(f'{os.path.basename(predict_video_file)}\t{temp}\n')


    # test，测试整个视频
    result=[]
    for a in method:
        temp= APD_by_dir(f'result/LRW/{a}/result/fake', f'result/LRW/{a}/result/real')
        result.append(temp)
    for a,b in zip(result,method):
        print(f'{b}:{a}')


    # test,以一个帧为单位
    # for a in method:
    #     with open(f'result/{a}/result/4_APD.txt','w',encoding='utf8') as f:
    #         ans=[]
    #         predict_video_file=f'result/{a}/result/fake/4.mp4'
    #         gt_video_file=f'result/{a}/result/real/4.mp4'
            
    #         pre_reader = imageio.get_reader(predict_video_file)
    #         predict_video=[im for im in pre_reader]
    #         pre_reader.close()

    #         gt_reader = imageio.get_reader(gt_video_file)
    #         gt_video=[im for im in gt_reader]
    #         gt_reader.close()

    #         index=0
    #         for predict,gt in zip(predict_video,gt_video):
    #             index+=1
    #             predict = cv2.cvtColor(predict,cv2.COLOR_RGB2BGR)
    #             gt = cv2.cvtColor(gt,cv2.COLOR_RGB2BGR)
    #             if index==16:
    #                 aaa=1
    #             fake_pose=get_pose(predict)
    #             real_pose=get_pose(gt)
    #             if fake_pose is None or real_pose is None:
    #                 continue
    #             fake_pose=np.delete(fake_pose, 3, axis=0)
    #             real_pose=np.delete(real_pose, 3, axis=0)
    #             distance=np.abs(fake_pose-real_pose).mean()
                
    #             f.write(f'{index}\t{distance}\n')

