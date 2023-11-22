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
from src.util.logger import logger_config
import glob
import pickle
import numpy as np
import zlib
from torch.multiprocessing import Pool, set_start_method
import os
import random
import shutil
from scipy.io import loadmat





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
    process_video(dir_name)

    logger.info('已结束：进程{}任务处理文件夹{}的内容'.format(os.getpid(),dir_name))


def merge_data(dir_name):
    # 获得所有信息后，开始合并所需要的信息

    # 对视频的整理
    filenames = glob.glob(f'{dir_name}/*.mp4')
    for video_path in filenames:
        info={}
        info['path']=video_path
        with open(video_path.replace('.mp4','_temp1.pkl'),'rb') as f:
            process_video= pickle.load(f)
        info['face_video']=process_video['face_video']
        info['mouth_mask']=process_video['mouth_mask']
        info['face_coeff']=loadmat(video_path.replace('.mp4','.mat'))
        # 获得最终数据
        with open(video_path.replace('.mp4','_video.pkl'),'wb') as f:
            info =  pickle.dumps(info)
            f.write(info)
        # 完了之后，删除没有必要的数据
        os.remove(video_path.replace('.mp4','.mat'))
        os.remove(video_path.replace('.mp4','_temp1.pkl'))
        os.remove(video_path.replace('.mp4','.txt'))
    

    # 对音频的整理很简单，在处理方法中直接生成，不需要在这整理
    # 整理数据
    filenames = glob.glob(f'{dir_name}/*.mp4')
    for video_path in filenames:
        info={}
        # 先获得视频的数据
        with open(video_path.replace('.mp4','_video.pkl'),'rb') as f:
            temp = pickle.load(f)
        info.update(temp)


        # 再获得音频的数据
        with open(video_path.replace('.mp4','_audio.pkl'),'rb') as f:
            temp = pickle.load(f)
        info['audio_mfcc']=temp['input_audio_mfcc']
        info['audio_hugebert']=temp['syncNet_audio']

        # 获得最终数据，使用压缩操作
        info = pickle.dumps(info)
        info=zlib.compress(info)
        with open(video_path.replace('.mp4','.pkl'),'wb') as f:
            f.write(info)
        # # test,解压测试
        # with open(video_path.replace('.mp4','_temp.pkl'),'rb') as f:
        #     a=f.read()
        # a=zlib.decompress(a)
        # a= pickle.loads(a)

        # 完了之后，删除没有必要的数据
        os.remove(video_path.replace('.mp4','_video.pkl'))
        os.remove(video_path.replace('.mp4','_audio.pkl'))

        # 有些没有视频信息的，需要删除
        info=zlib.decompress(info)
        info =  pickle.loads(info)
        frame=info['face_video'][0]
        if np.sum(frame==0)>3*256*256/4:
            os.remove(video_path.replace('.mp4','.pkl'))
        
        # test，测试遮罩效果
        # import imageio
        # mouth_mask=info['mouth_mask']
        # process_video=info['face_video']
        # temp=[]
        # for i in range(mouth_mask.shape[0]):
        #     mask = np.random.rand(mouth_mask[i][1]-mouth_mask[i][0],mouth_mask[i][3]-mouth_mask[i][2], 3)
        #     mask=(mask*255).astype(np.uint8)
        #     img = process_video[i].copy()
        #     img[mouth_mask[i][0]:mouth_mask[i][1], mouth_mask[i][2]:mouth_mask[i][3], :] = mask
        #     temp.append(img)
        # video_array=np.array(temp)
        # imageio.mimsave('mask.mp4',video_array)



def merge_data2(dir_name):
    # 获得所有信息后，开始合并所需要的信息
    # 从dormat_data入手
    filenames = glob.glob(f'{dir_name}/*.pkl')

    logger.info('进程{}合并文件夹{}的内容'.format(os.getpid(),dir_name))

    # 存到这
    for file in filenames:
        info={}
        with open(file,'rb') as f:
            data=f.read()
        data=zlib.decompress(data)
        data= pickle.loads(data)
        info.update(data)
        info.pop('frame_index')
        info.pop('path')

        # 再从视频中拿东西
        video_path=data['path']
        if not os.path.exists(video_path.replace('.mp4','_temp1.pkl')):
            continue
        with open(video_path.replace('.mp4','_temp1.pkl'),'rb') as f:
            process_video= pickle.load(f)
        info['face_video']=process_video['face_video']
        info['mouth_mask']=process_video['mouth_mask']
        # 完了之后，删除没有必要的数据
        os.remove(video_path.replace('.mp4','_temp1.pkl'))

        
        # 没有视频信息的，直接弃用
        frame=info['face_video'][0]
        if np.sum(frame==0)>3*256*256/4:
            continue

        # 获得最终数据，使用压缩操作
        save_file=file.replace('data2','data')
        os.makedirs(os.path.dirname(save_file),exist_ok=True)
        info = pickle.dumps(info)
        info=zlib.compress(info)
        with open(save_file,'wb') as f:
            f.write(info)

    logger.info('已结束：进程{}任务合并文件夹{}的内容'.format(os.getpid(),dir_name))




if __name__=='__main__':
    set_start_method('spawn')
    # 希望各个方法能够做到：输入视频路径，输出pickl的pkl文件
    import yaml
    config={}
    with open(r'config/dataset/common.yaml',encoding='utf8') as f:
        config.update(yaml.safe_load(f))
    dataset_root=config['mead_root_path']
    

    # 对于一些不符合规范的视频，删除掉，不然处理会产生问题
    # unvalid_file=[]
    # filenames=sorted(glob.glob(f'{dataset_root}/*/video/*/*/*/*.mp4'))
    # for file in filenames:
    #     if os.path.getsize(file)==0:
    #         unvalid_file.append(file)
    # for file in unvalid_file:
    #     os.remove(file)

    dir_list=sorted(glob.glob(f'{dataset_root}/*/video/front/*/*'))

    # index=0
    # for i,dir in enumerate(dir_list):
    #     if '/workspace/dataset/MEAD/W026/video/front/happy/level_2' in dir:
    #         index=i
    #         break
    # dir_list=dir_list[index:]

    # test
    # dir_list=['data']
    # for file_list in dir_list:
    #     # format_data_by_cuda(file_list)
    #     format_data_no_use_cuda(file_list)
    # dir_list=['data2/format_data/train/0']
    # merge_data2(file_list)

    
    # workers=4
    # pool = Pool(workers)
    # for _ in pool.imap_unordered(format_data_by_cuda,dir_list):
    #     None
    # pool.close()
    # print(logger.info('\n使用cuda获得的数据已处理完毕，现在处理不使用cuda的\n'))
    # workers=5
    # pool = Pool(workers)
    # for _ in pool.imap_unordered(format_data_no_use_cuda,dir_list):
    #     None
    # pool.close()
    print(logger.info('\n开始合并数据\n'))
    workers=5
    dir_list=sorted(glob.glob('data2/format_data/*/*'))
    pool = Pool(workers)
    for _ in pool.imap_unordered(merge_data2,dir_list):
        None
    pool.close()




    # # 接下类是转移文件
    # dataset_root=config['mead_root_path']
    # filenames=sorted(glob.glob(f'{dataset_root}/*/video/*/*/*/*.pkl'))
    # out_path=config['format_output_path']
    # # 拿500作为验证集，500作为测试集，其余作为训练集
    # temp_index=sorted(random.sample(range(len(filenames)),1000))
    # eval_file=filenames[temp_index[:500]]
    # test_file=filenames[temp_index[500:]]
    # train_file=np.delete(np.array(filenames),temp_index)
    # cnt=0
    # for file in eval_file.tolist():
    #     if cnt%500==0:
    #         path=os.path.join(out_path,f'eval/{cnt//500}')
    #         os.makedirs(path,exist_ok=True)
    #     file_list=file.split('/')
    #     out_name=f'{file_list[-6]}_{file_list[-4]}_{file_list[-3]}_{file_list[-2]}_{file_list[-1]}'
    #     out_name=os.path.join(path,out_name)
    #     shutil.copyfile(file,out_name)
    #     cnt+=1
    # cnt=0
    # for file in test_file.tolist():
    #     if cnt%500==0:
    #         path=os.path.join(out_path,f'test/{cnt//500}')
    #         os.makedirs(path,exist_ok=True)
    #     file_list=file.split('/')
    #     out_name=f'{file_list[-6]}_{file_list[-4]}_{file_list[-3]}_{file_list[-2]}_{file_list[-1]}'
    #     out_name=os.path.join(path,out_name)
    #     shutil.copyfile(file,out_name)
    #     cnt+=1
    # cnt=0
    # for file in train_file.tolist():
    #     if cnt%500==0:
    #         path=os.path.join(out_path,f'train/{cnt//500}')
    #         os.makedirs(path,exist_ok=True)
    #     file_list=file.split('/')
    #     out_name=f'{file_list[-6]}_{file_list[-4]}_{file_list[-3]}_{file_list[-2]}_{file_list[-1]}'
    #     out_name=os.path.join(path,out_name)
    #     shutil.copyfile(file,out_name)
    #     cnt+=1
    # for file in filenames:
    #     os.remove(file)

    