# 测试 
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(3):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

import cv2
import dlib
import numpy as np
from src.util.augmentation import AllAugmentationTransform
from Deep3DFaceRecon_pytorch.extract_kp_videos import keypoints
from Deep3DFaceRecon_pytorch.face_recon_videos import get3DMM
import pickle
import os
import glob


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./checkpoint/shape_predictor_68_face_landmarks.dat')

def process_video(video_dir):
    r'''输入视频目录，处理该目录下所有mp4的视频，生成的信与视频目录放在一块'''

    # 获得transformer video
    filenames = glob.glob(f'{video_dir}/*.mp4')
    for video_path in filenames:
        info={}
        process_video,mouth_mask=__process_video__(video_path)
        info['face_video']=np.array(process_video,dtype=np.uint8)
        info['mouth_mask']=np.array(mouth_mask)
        with open(video_path.replace('.mp4','_temp1.pkl'),'wb') as f:
            pickle.dump(info,f)

    return

def __process_video__(video_path):
    '''对视频进行预处理，返回transformer后的numpy数组
    只能对一个视频处理的方法
    '''
    cap = cv2.VideoCapture(video_path)  
    driving_video=[]
    while(cap.isOpened()):  
        ret, frame = cap.read()
        # 如果读到了结尾
        if not ret:
            break
        driving_video.append(frame) 
    cap.release()

    process_video,mouth_mask = get_face_image(driving_video)

    # # test，测试遮罩效果
    # temp=[]
    # for i in range(mouth_mask.shape[0]):
    #     mask = np.random.rand(mouth_mask[i][1]-mouth_mask[i][0],mouth_mask[i][3]-mouth_mask[i][2], 3)
    #     mask=(mask*255).astype(np.uint8)
    #     img = process_video[i].copy()
    #     img[mouth_mask[i][0]:mouth_mask[i][1], mouth_mask[i][2]:mouth_mask[i][3], :] = mask
    #     temp.append(img)
    # video_array=np.array(temp)
    # video_height = video_array.shape[1]
    # video_width = video_array.shape[2]
    # out_video_size = (video_width,video_height)
    # output_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    # video_write_capture = cv2.VideoWriter('mouth.mp4', output_video_fourcc, 30, out_video_size)
    # for frame in video_array:
    #     video_write_capture.write(frame)
    # video_write_capture.release()
    

    return process_video,mouth_mask


def video_to_3DMM_and_pose(video_dir):
    '''video转为3DMM和pose信息
    对一个文件夹内的所有视频进行操作
    '''
    # keypoints对一个文件夹内的所有视频，以批次方法进行操作
    keypoints(video_dir,video_dir)
    # get3DMM对一个文件夹内的所有视频，以批次方法进行操作
    get3DMM(video_dir,video_dir)


def get_face_image(driving_video):
    video_array = np.array(driving_video, dtype=np.uint8)
    top=100000
    bottom=-1
    left=100000
    right=-1
    # config['augmentation_para
    # 初始化gray list
    for i in range(len(video_array)):
        gray = cv2.cvtColor(video_array[i], cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)  #detect human face
        # # test
        # aaa=rects[-1]
        # pic=video_array[i][aaa.top()-200:aaa.bottom()+100,aaa.left()-150:aaa.right()+150]
        # cv2.imwrite("temp.jpg", pic)
        if len(rects)!=0:
            top=max(min(top,rects[-1].top()-200),0)
            bottom=min(max(bottom,rects[-1].bottom()+100),video_array.shape[1])
            left=max(min(left,rects[-1].left()-150),0)
            right=min(max(right,rects[-1].right()+150),video_array.shape[2])
        # test,测试只检测3张，加快速度
        # if i==3:
        #     break
    
    if top==100000:
        # 返回一个全零数组，到时丢弃数据好判断
        return np.zeros((len(video_array),256,256,3)),np.zeros((len(video_array),4))
    
    # 裁剪图片
    temp_video_array=[]
    for pic in video_array:
        temp_video_array.append(pic[top:bottom,left:right])
    video_array=np.array(temp_video_array)
    # resize图片
    video_array = [cv2.resize(frame, (256, 256)) for frame in video_array]
    video_array=np.array(video_array)

    # test，存视频
    # video_height = video_array.shape[1]
    # video_width = video_array.shape[2]
    # out_video_size = (video_width,video_height)
    # output_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    # video_write_capture = cv2.VideoWriter('temp.mp4', output_video_fourcc, 30, out_video_size)
    # for frame in video_array:
    #     video_write_capture.write(frame)
    # video_write_capture.release()

    # 获得面部遮罩
    mouth_mask=[]
    flag=False
    for i in range(len(video_array)):
        gray = cv2.cvtColor(video_array[i], cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)  #detect human face
        if len(rects)!=0:
            template = predictor(gray,rects[-1]) #detect 68 points
            template = shape_to_np(template)
            top=min(template[50][1],template[52][1])
            bottom=max(template[56][1],template[57][1])
            left=template[48][0]
            right=template[54][0]
            mouth_mask.append([top,bottom,left,right])
            flag=True
        else:
            mouth_mask.append([])
    if  not flag:
        # 返回一个全零数组，到时丢弃数据好判断
        return np.zeros((len(video_array),256,256,3)),np.zeros((len(video_array),4))
    
    # test
    # mouth_mask[0]=[]
    # mouth_mask[1]=[]
    # mouth_mask[97]=[]
    # mouth_mask[96]=[]
    # mouth_mask[6]=[]
    # mouth_mask[7]=[]
    # mouth_mask[8]=[]
    # mouth_mask[9]=[]
    # mouth_mask[11]=[]
    # mouth_mask[13]=[]

    mouth_mask=deal_mouth_mask(mouth_mask)

    return video_array,mouth_mask

def deal_mouth_mask(mouth_mask):
    '''mouth mask中有些无法探测，在这里补齐'''
    # 先补齐前面的
    # 找到第一个不是空的
    for i in range(len(mouth_mask)):
        if len(mouth_mask[i]) !=0:
            start=i
            break
    # 补齐
    for i in range(start):
        mouth_mask[i]+=mouth_mask[start]

    # 再补齐后面的
    # 找到倒数第一个不是空的
    for i in range(len(mouth_mask)-1,0,-1):
        if len(mouth_mask[i]) !=0:
            end=i
            break
    # 补齐
    for i in range(len(mouth_mask)-1,end,-1):
        mouth_mask[i]+=mouth_mask[end]

    # 最后补齐中间的
    for i in range(len(mouth_mask)):
        # 遇到空时
        if len(mouth_mask[i])==0:
            start=i-1
            end=i
            while True:
                if len(mouth_mask[end])!=0:
                    break
                end+=1
            # 用线性叠加方式赋值
            left=np.array(mouth_mask[start])
            right=np.array(mouth_mask[end])
            step=((right-left)/(end-start))
            for j in range(start+1,end):
                temp=np.array(mouth_mask[j-1]).astype(np.float64)
                temp+=step
                temp=temp.tolist()
                mouth_mask[j]+=temp
    return np.around(mouth_mask).astype(np.int16)

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
    process_video(r'data')
