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
    

    # # 对于一些不符合规范的视频，删除掉，不然处理会产生问题
    # unvalid_file=[]
    # filenames=sorted(glob.glob(f'{dataset_root}/*/video/*/*/*/*.mp4'))
    # for file in filenames:
    #     if os.path.getsize(file)==0:
    #         unvalid_file.append(file)
    # for file in unvalid_file:
    #     os.remove(file)


    # dir_list=sorted(glob.glob(f'{dataset_root}/*/video/front/*/*'))
    # # index=0
    # # for i,dir in enumerate(dir_list):
    # #     if '/workspace/dataset/MEAD/W011/video/front/surprised/level_1' in dir:
    # #         index=i
    # #         break
    # # dir_list=dir_list[index:]

    # # test
    # # dir_list=['data','data2','data3','data4','data5']
    # # for file_list in dir_list:
    # #     format_data_by_cuda(file_list)
    # #     format_data_no_use_cuda(file_list)
    
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

    # # 对于没有video的文件，删除
    # out_path=config['format_output_path']
    # filenames=glob.glob(f'{out_path}/*/*/*.pkl')
    # a=[]
    # for i in range(len(filenames)):
    #     file=filenames[i]
    #     print(i)
    #     # 解压pkl文件
    #     with open(file,'rb') as f:
    #         byte_file=f.read()
    #     byte_file=zlib.decompress(byte_file)
    #     data= pickle.loads(byte_file)
    #     if data['align_video'].shape[0]==0:
    #         a.append(file)
    #         # os.remove(file)
    # print(len(a))
    # for aaa in a:
    #     os.remove(aaa)


    