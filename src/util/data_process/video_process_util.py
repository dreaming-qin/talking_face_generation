# 测试 
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(3):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from skimage import transform as tf
import cv2
import dlib
import numpy as np
from src.util.augmentation import AllAugmentationTransform
import imageio
from skimage.transform import resize
from Deep3DFaceRecon_pytorch.extract_kp_videos import keypoints
from Deep3DFaceRecon_pytorch.face_recon_videos import get3DMM
import pickle
import os
import glob
from scipy.io import loadmat


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./checkpoint/shape_predictor_68_face_landmarks.dat')

def process_video(video_dir,config):
    r'''输入视频目录，处理该目录下所有mp4的视频，生成的信与视频目录放在一块'''

    # 获得transformer video
    filenames = glob.glob(f'{video_dir}/*.mp4')
    for video_path in filenames:
        info={}
        import time
        start=time.time()
        process_video=transformer_video(video_path,config)
        print(time.time()-start)
        info['transformer_video']=np.array(process_video,dtype=np.uint8)
        with open(video_path.replace('.mp4','_temp1.pkl'),'wb') as f:
            pickle.dump(info,f)

    # 获得3DMM信息
    video_to_3DMM_and_pose(video_dir)

    # 获得所有信息后，开始合并所需要的信息
    filenames = glob.glob(f'{video_dir}/*.mp4')
    for video_path in filenames:
        info={}
        info['path']=video_path
        with open(video_path.replace('.mp4','_temp1.pkl'),'rb') as f:
            process_video= pickle.load(f)['transformer_video']
            info['process_video']=process_video
        info['face_coeff']=loadmat(video_path.replace('.mp4','.mat'))
        # 获得最终数据
        with open(video_path.replace('.mp4','_video.pkl'),'wb') as f:
            info =  pickle.dumps(info)
            f.write(info)
        # 完了之后，删除没有必要的数据
        os.remove(video_path.replace('.mp4','.mat'))
        os.remove(video_path.replace('.mp4','_temp1.pkl'))
        os.remove(video_path.replace('.mp4','.txt'))
    return


def transformer_video(video_path,config):
    '''对视频进行预处理，返回transformer后的numpy数组
    只能对一个视频处理的方法
    '''
    reader = imageio.get_reader(video_path)
    driving_video = []
    try:
        driving_video=[im for im in reader]
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    driving_video = get_aligned_image(driving_video,config)
    # transformed_video = get_transformed_image(driving_video,config)
    transformed_video=driving_video
    transformed_video=np.array(np.array(transformed_video)*255, dtype=np.uint8)

    return transformed_video


def video_to_3DMM_and_pose(video_dir):
    '''video转为3DMM和pose信息
    对一个文件夹内的所有视频进行操作
    '''
    # keypoints对一个文件夹内的所有视频，以批次方法进行操作
    keypoints(video_dir,video_dir)
    # get3DMM对一个文件夹内的所有视频，以批次方法进行操作
    get3DMM(video_dir,video_dir)


def get_aligned_image(driving_video,config):
    aligned_array = []
    video_array = np.array(driving_video)

    detect_face=[]
    first_index=1e10
    for i in range(len(video_array)):
        image=np.array(video_array[i] * 255, dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)  #detect human face
        detect_face.append(rects)
        if len(rects)!=0:
            template = predictor(gray, rects[-1]) #detect 68 points
            template = shape_to_np(template)
            first_index=min(first_index,i)


    # 获得参照物图片信息
    template = predictor(gray, detect_face[first_index][-1]) #detect 68 points
    template = shape_to_np(template)
    # 在这里，将会更新config里的面部遮罩坐标，精心调制版本
    top=min(template[50][1],template[52][1])
    bottom=max(template[56][1],template[57][1])
    left=template[48][0]
    right=template[54][0]
    config['augmentation_params']['crop_mouth_param']['center_x']=(left+right)//2
    config['augmentation_params']['crop_mouth_param']['center_y']=(top+bottom)//2+10
    config['augmentation_params']['crop_mouth_param']['mask_width']=right-left+10
    config['augmentation_params']['crop_mouth_param']['mask_height']=bottom-top+25
    
    
    # 根据参照物图片，将进行align
    def align(rects):
        dst=np.zeros((256,256,3))
        if len(rects)!=0:
            shape = predictor(gray, rects[-1]) #detect 68 points
            shape = shape_to_np(shape)
            pts2 = np.float32(template[:35,:])
            pts1 = np.float32(shape[:35,:]) #eye and nose

            tform = tf.SimilarityTransform()
            tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
            dst = tf.warp(image, tform, output_shape=(256, 256))

            dst = np.array(dst, dtype=np.float32)
            aligned_array.append(dst)
        return dst
    aligned_array=[align(rects) for rects in detect_face ]

    return aligned_array


def get_transformed_image(driving_video, config):
    video_array = np.array(driving_video)
    transformations = AllAugmentationTransform(**config['augmentation_params'])
    transformed_array = transformations(video_array)
    return transformed_array


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

# 测试
if __name__=='__main__':
    # 希望各个方法能够做到：输入视频路径，输出pickl的pkl文件
    import yaml
    with open(r'config\data_process\common.yaml',encoding='utf8') as f:
        config=yaml.safe_load(f)
    with open(r'config\dataset\common.yaml',encoding='utf8') as f:
        config.update(yaml.safe_load(f))
    process_video(r'data',config)
