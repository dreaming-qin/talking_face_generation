import cv2 as cv
from skimage.metrics import structural_similarity
import imageio
import os
import glob

'''ssim取值范围[0,1]，越大越好'''

def ssim(predict_video,gt_video):
    r''''返回视频的ssim均值
    predict_video(len,H,W,3)
    gt_video(len,H,W,3)'''
    

    ans=0
    for predict,gt in zip(predict_video,gt_video):
        # predict是(H,W,3)
        image1 = cv.cvtColor(predict,cv.COLOR_BGR2GRAY) #  将图像转换为灰度图

        image2 = cv.cvtColor(gt,cv.COLOR_BGR2GRAY) #  将图像转换为灰度图

        ans += structural_similarity(image1, image2)
    ans/=min(len(predict_video),len(gt_video))
    return ans


def ssim_by_path(predict_video_file,gt_video_file):
    pre_reader = imageio.get_reader(predict_video_file)
    predict_driving_video=[im for im in pre_reader]
    pre_reader.close()

    gt_reader = imageio.get_reader(gt_video_file)
    gt_driving_video=[im for im in gt_reader]
    gt_reader.close()

    return ssim(predict_driving_video,gt_driving_video)


def ssim_by_dir(predict_video_dir,gt_video_dir):
    ans=[]
    predict_video_dir=sorted(glob.glob(f'{predict_video_dir}/*.mp4'))
    gt_video_dir=sorted(glob.glob(f'{gt_video_dir}/*.mp4'))
    for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
        assert os.path.basename(predict_video_file)==os.path.basename(gt_video_file)
        temp=ssim_by_path(predict_video_file,gt_video_file)
        if temp!=-1:
            ans.append(temp)
    if len(ans)==0:
        # -1代表不成功
        return -1
    average_distance = sum(ans) / len(ans)
    return average_distance


if __name__=='__main__':
    ssim_by_path('0.mp4','1.mp4')

