import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
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
        # imageio.imsave('gt.jpg',gt)
        # imageio.imsave('predict.jpg',predict)
        # image1 = cv2.cvtColor(predict,cv2.COLOR_RGB2BGR) 
        # image2 = cv2.cvtColor(gt,cv2.COLOR_RGB2BGR)
        ans += structural_similarity(predict, gt,channel_axis=2)
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
    # ssim_by_path('0.mp4','1.mp4')

    method=['atvg','eamm','wav2lip_no_pose','make','pc_avs','exp3DMM']
    # method=['pc_avs','atvg']

    result=[]
    for a in method:
        with open(f'result/{a}/result/0_ssim.txt','w',encoding='utf8') as f:
            ans=[]
            predict_video_dir=sorted(glob.glob(f'result/{a}/result/fake/0.mp4'))
            gt_video_dir=sorted(glob.glob(f'result/{a}/result/real/0.mp4'))
            for predict_video_file,gt_video_file in zip(predict_video_dir,gt_video_dir):
                pre_reader = imageio.get_reader(predict_video_file)
                predict_video=[im for im in pre_reader]
                pre_reader.close()

                gt_reader = imageio.get_reader(gt_video_file)
                gt_video=[im for im in gt_reader]
                gt_reader.close()

                for i,(predict,gt) in enumerate(zip(predict_video,gt_video)):
                    temp= structural_similarity(predict, gt,channel_axis=2)
                    # imageio.imsave('gt.jpg',gt)
                    # imageio.imsave('predict.jpg',predict)
                    if i==32:
                        ccc=1
                    f.write(f'{i}\t{temp}\n')


    for a,b in zip(result,method):
        print(f'{b}:{a}')

