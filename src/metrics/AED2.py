import numpy as np
import glob
import os
import cv2
import imageio
import matplotlib.pyplot as plt
from  torchvision.models import resnet18
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor
import shutil


# test 
if __name__=='__main__':
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'TDDFA_V2_master'))

from src.metrics.lib.face_align_cuda import main as get_face




'''APD取值范围[0,正无穷]，越小越好'''
device=torch.device('cuda:0')
model=resnet18(num_classes=7)
# 加载模型
checkpoint = torch.load('checkpoint/AED/Resnet18_FER+_pytorch.pth.tar')
pretrained_state_dict = checkpoint['state_dict']
model_state_dict = model.state_dict()
for key in pretrained_state_dict:
    if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
        pass
    else:
        model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]
model.load_state_dict(model_state_dict)
# 抓最后一层特征
stage_indices = {'avgpool':'result'}
model = create_feature_extractor(model, stage_indices).to(device)
model.eval()

transform=transforms.Compose([transforms.Resize(224), 
        transforms.ToTensor()])


@torch.no_grad()
def AED_by_path(predict_video_file,gt_video_file):
    # 预处理视频数据
    fake_frame_dir='temp/fake/frame'
    real_frame_dir='temp/real/frame'
    os.makedirs(fake_frame_dir,exist_ok=True)
    os.makedirs(real_frame_dir,exist_ok=True)
    cmd = 'ffmpeg -i {:} -loglevel error -f image2 {:}/%07d.png'.format(predict_video_file, fake_frame_dir)
    os.system(cmd)
    cmd = 'ffmpeg -i {:} -loglevel error -f image2 {:}/%07d.png'.format(gt_video_file, real_frame_dir)
    os.system(cmd)

    fake_img_dir='temp/fake/face'
    real_img_dir='temp/real/face'
    os.makedirs(fake_img_dir,exist_ok=True)
    os.makedirs(real_img_dir,exist_ok=True)
    get_face(fake_frame_dir,fake_img_dir)
    get_face(real_frame_dir,real_img_dir)

    fake_exp=[]
    real_exp=[]
    fake_img_list=sorted(glob.glob(f'{fake_img_dir}/*.png'))
    real_img_list=sorted(glob.glob(f'{real_img_dir}/*.png'))
    for fake_img,real_img in zip(fake_img_list,real_img_list):
        fake_img = Image.open(fake_img).convert("RGB")
        real_img = Image.open(real_img).convert("RGB")
        fake_img = transform(fake_img).reshape(1,3,224,224).to(device)
        real_img = transform(real_img).reshape(1,3,224,224).to(device)

        fake_result=model(fake_img)['result'].squeeze()
        real_result=model(real_img)['result'].squeeze()

        fake_exp.append(fake_result)
        real_exp.append(real_result)
    fake_exp=torch.stack(fake_exp)
    real_exp=torch.stack(real_exp)
    mean_by_frame=torch.abs(fake_exp-real_exp).mean(dim=1)
    shutil.rmtree('temp')

    return mean_by_frame.mean().item()


def AED_by_dir(predict_video_dir,gt_video_dir):
    ans=[]
    predict_video_dir=sorted(glob.glob(f'{predict_video_dir}/*.mp4'))
    gt_video_dir=sorted(glob.glob(f'{gt_video_dir}/*.mp4'))
    for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
        assert os.path.basename(predict_video_file)==os.path.basename(gt_video_file)
        temp=AED_by_path(predict_video_file,gt_video_file)
        if temp!=-1:
            ans.append(temp)
    if len(ans)==0:
        # -1代表不成功
        return -1
    average_distance = sum(ans) / len(ans)
    return average_distance



if __name__=='__main__':

    method=['atvg','eamm','wav2lip_no_pose','make','pc_avs','exp3DMM']
    # method=['wav2lip_no_pose','pc_avs']

    # # test, 以一个视频为单位
    for a in method:
        with open(f'result/LRW/{a}/result/AED.txt','w',encoding='utf8') as f:
            ans=[]
            predict_video_dir=sorted(glob.glob(f'result/LRW/{a}/result/fake/*.mp4'))
            gt_video_dir=sorted(glob.glob(f'result/LRW/{a}/result/real/*.mp4'))
            for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
                temp= AED_by_path(predict_video_file, gt_video_file)
                f.write(f'{os.path.basename(predict_video_file)}\t{temp}\n')


    # test，测试整个视频
    result=[]
    for a in method:
        temp= AED_by_dir(f'result/LRW/{a}/result/fake', f'result/LRW/{a}/result/real')
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

