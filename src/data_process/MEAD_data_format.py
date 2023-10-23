# 测试 
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.data_process.audio_process_util import process_audio
from src.util.data_process.video_process_util import process_video,video_to_3DMM_and_pose
import glob
import pickle
import numpy as np
import zlib
from torch.multiprocessing import Pool, set_start_method
import os

import logging


def logger_config(log_path,logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

logger = logger_config(log_path='data_process.log', logging_name='data process log')

def format_data_by_cuda(dir_name):
    # 先调用两个方法，获得video和audio的数据
    global logger
    logger.info('进程{}处理文件夹{}的内容'.format(os.getpid(),dir_name))

    video_to_3DMM_and_pose(dir_name)
    process_audio(dir_name)
    
    logger.info('已结束：进程{}任务处理文件夹{}的内容'.format(os.getpid(),dir_name))


def format_data_no_use_cuda(dir_name):
    # 先调用两个方法，获得video和audio的数据
    global logger
    logger.info('进程{}处理文件夹{}的内容'.format(os.getpid(),dir_name))
    process_video(dir_name,process_3DMM=False)
    # process_audio(dir_name)

    # 整理数据
    filenames = glob.glob(f'{dir_name}/*.mp4')
    for video_path in filenames:
        info={}
        # 先处理视频的数据
        with open(video_path.replace('.mp4','_video.pkl'),'rb') as f:
            temp = pickle.load(f)
        frame_index=[]
        align_video=[]
        for i,frame in enumerate(temp['process_video']):
            # 当全零像素小于总像素的3/4时，认为是可以作为数据输入的有效帧
            if np.sum(frame==0)<3*256*256/4:
                frame_index.append(i)
                align_video.append(frame)
        info['align_video']=np.array(align_video)
        info['face_coeff']=temp['face_coeff']
        info['frame_index']=np.array(frame_index)
        info['path']=temp['path']

        # 再处理音频的数据
        with open(video_path.replace('.mp4','_audio.pkl'),'rb') as f:
            temp = pickle.load(f)
        info['audio_mfcc']=temp['input_audio_mfcc']
        info['audio_hugebert']=temp['syncNet_audio']

        # 获得最终数据，使用压缩操作
        info = pickle.dumps(info)
        info=zlib.compress(info)
        with open(video_path.replace('.mp4','.pkl'),'wb') as f:
            f.write(info)
        # # 解压测试
        # with open(video_path.replace('.mp4','_temp.pkl'),'rb') as f:
        #     a=f.read()
        # a=zlib.decompress(a)
        # a= pickle.loads(a)

        # 完了之后，删除没有必要的数据
        os.remove(video_path.replace('.mp4','_video.pkl'))
        os.remove(video_path.replace('.mp4','_audio.pkl'))
    
    logger.info('已结束：进程{}任务处理文件夹{}的内容'.format(os.getpid(),dir_name))

if __name__=='__main__':
    set_start_method('spawn')
    # 希望各个方法能够做到：输入视频路径，输出pickl的pkl文件
    import yaml
    with open(r'config/data_process/common.yaml',encoding='utf8') as f:
        config=yaml.safe_load(f)
    with open(r'config/dataset/common.yaml',encoding='utf8') as f:
        config.update(yaml.safe_load(f))

    dataset_root=config['mead_root_path']
    dir_list=glob.glob(f'{dataset_root}/*/video/front/*/*')

    # dir_list=['data']
    # for file_list in dir_list:
    #     format_data(file_list)
    
    workers=5
    pool = Pool(workers)
    for _ in pool.imap_unordered(format_data_by_cuda,dir_list):
        None
    pool.close()

    print(logger.info('\n使用cuda获得的数据已处理完毕，现在处理不使用cuda的\n'))

    pool = Pool(workers)
    for _ in pool.imap_unordered(format_data_no_use_cuda,dir_list):
        None
    pool.close()
    