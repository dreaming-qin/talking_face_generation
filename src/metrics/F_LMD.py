import dlib
import cv2
import numpy as np
import imageio
from imutils import face_utils
import os
import glob
import skimage.transform as trans
import matplotlib.pyplot as plt


'''F-LMD取值范围[0,正无穷]，越小越好'''


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./checkpoint/shape_predictor_68_face_landmarks.dat')


def scale(mouth_land):
    # lmd会随着人脸大小会改变，因此进行规范化
    gt=(120,110)
    x_scale=gt[0]/mouth_land[16][0]
    y_scale=gt[1]/mouth_land[8][1]
    scale=(x_scale+y_scale)/2
    mouth_land[:,0]=mouth_land[:,0]*scale
    mouth_land[:,1]=mouth_land[:,1]*scale
    return mouth_land


def F_LMD(predict_video,gt_video):
    '''输入维度(len,H,W,3)'''
    distance=[]
    for predict,gt in zip(predict_video,gt_video):
        # predict是(H,W,3)
        predict = cv2.cvtColor(predict,cv2.COLOR_RGB2GRAY) #  将图像转换为灰度图
        gt = cv2.cvtColor(gt,cv2.COLOR_RGB2GRAY) #  将图像转换为灰度图
        
        real_rects = detector(gt, 1)
        fake_rects = detector(predict, 1)
        # 未检测到人脸，不要
        if len(real_rects) ==0 or len(fake_rects)==0:
            continue

        # 先获得预测的landmark
        fake_rect = fake_rects[0]
        fake_mouth_land = get_landmark(predict, fake_rect)
        fake_mouth_land=scale(fake_mouth_land)
        # 然后获得gt的landmark
        real_rect = real_rects[0]
        real_mouth_land = get_landmark(gt, real_rect)
        real_mouth_land=scale(real_mouth_land)


        # test，输出图片
        # plt.plot(fake_mouth_land.T[0], fake_mouth_land.T[1],'x',color='k',alpha=0.5)
        # plt.plot(real_mouth_land.T[0], real_mouth_land.T[1],'o',color='k',alpha=0.5)
        # plt.savefig('temp.jpg')
        # plt.close()
        # imageio.imsave('gt.jpg',gt)
        # imageio.imsave('predict.jpg',predict)



        dis = (fake_mouth_land-real_mouth_land)**2
        dis = np.sum(dis,axis=1)
        dis = np.sqrt(dis)
        dis = np.sum(dis,axis=0)
        

        distance.append(dis/len(fake_mouth_land))

    if len(distance)==0:
        # -1代表不成功
        return -1
    average_distance = sum(distance) / len(distance)
    return average_distance

def F_LMD_by_path(predict_video_file,gt_video_file):
    pre_reader = imageio.get_reader(predict_video_file)
    predict_video=[im for im in pre_reader]
    pre_reader.close()

    gt_reader = imageio.get_reader(gt_video_file)
    gt_video=[im for im in gt_reader]
    gt_reader.close()
    return F_LMD(predict_video,gt_video)


def F_LMD_by_dir(predict_video_dir,gt_video_dir):
    ans=[]
    predict_video_dir=sorted(glob.glob(f'{predict_video_dir}/*.mp4'))
    gt_video_dir=sorted(glob.glob(f'{gt_video_dir}/*.mp4'))
    for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
        assert os.path.basename(predict_video_file)==os.path.basename(gt_video_file)
        temp=F_LMD_by_path(predict_video_file,gt_video_file)
        if temp!=-1:
            ans.append(temp)
    if len(ans)==0:
        # -1代表不成功
        return -1
    average_distance = sum(ans) / len(ans)
    return average_distance



def get_landmark(images, rect):
    shape = predictor(images, rect)
    shape = face_utils.shape_to_np(shape)
    original = np.sum(shape,axis=0) / len(shape)
    shape = shape - original
    return shape


if __name__=='__main__':

    method=['atvg','eamm','wav2lip_no_pose','make','pc_avs','exp3DMM']
    method=['atvg']

    # test, 以一个视频为单位
    for a in method:
        with open(f'result/{a}/result/m_lmd.txt','w',encoding='utf8') as f:
            ans=[]
            predict_video_dir=sorted(glob.glob(f'result/{a}/result/fake/*.mp4'))
            gt_video_dir=sorted(glob.glob(f'result/{a}/result/real/*.mp4'))
            for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
                temp= F_LMD_by_path(predict_video_file, gt_video_file)
                f.write(f'{os.path.basename(predict_video_file)}\t{temp}\n')


    # test，测试整个视频
    # result=[]
    # for a in method:
    #     temp= F_LMD_by_path(f'result/{a}/result/fake/3.mp4', f'result/{a}/result/real/3.mp4')
    #     result.append(temp)
    # for a,b in zip(result,method):
    #     print(f'{b}:{a}')


    # test,以一个帧为单位
    for a in method:
        with open(f'result/{a}/result/3_m_lmd.txt','w',encoding='utf8') as f:
            ans=[]
            predict_video_file=f'result/{a}/result/fake/3.mp4'
            gt_video_file=f'result/{a}/result/real/3.mp4'
            
            pre_reader = imageio.get_reader(predict_video_file)
            predict_video=[im for im in pre_reader]
            pre_reader.close()

            gt_reader = imageio.get_reader(gt_video_file)
            gt_video=[im for im in gt_reader]
            gt_reader.close()

            index=0
            for predict,gt in zip(predict_video,gt_video):
                # predict是(H,W,3)
                predict = cv2.cvtColor(predict,cv2.COLOR_RGB2GRAY) #  将图像转换为灰度图
                gt = cv2.cvtColor(gt,cv2.COLOR_RGB2GRAY) #  将图像转换为灰度图
                
                real_rects = detector(gt, 1)
                fake_rects = detector(predict, 1)
                # 未检测到人脸，不要
                if len(real_rects) ==0 or len(fake_rects)==0:
                    continue

                # 先获得预测的landmark
                fake_rect = fake_rects[0]
                fake_mouth_land = get_landmark(predict, fake_rect)
                fake_mouth_land=scale(fake_mouth_land)
                # 然后获得gt的landmark
                real_rect = real_rects[0]
                real_mouth_land = get_landmark(gt, real_rect)
                real_mouth_land=scale(real_mouth_land)


                # test，输出图片
                plt.plot(fake_mouth_land.T[0], fake_mouth_land.T[1],'x',color='k',alpha=0.5)
                plt.plot(real_mouth_land.T[0], real_mouth_land.T[1],'o',color='k',alpha=0.5)
                plt.savefig('temp.jpg')
                plt.close()
                imageio.imsave('gt.jpg',gt)
                imageio.imsave('predict.jpg',predict)



                dis = (fake_mouth_land-real_mouth_land)**2
                dis = np.sum(dis,axis=1)
                dis = np.sqrt(dis)
                dis = np.sum(dis,axis=0)
                
                index+=1
                f.write(f'{index}\t{dis/len(fake_mouth_land)}\n')
