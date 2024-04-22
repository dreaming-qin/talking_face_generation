
import os
import glob
import pickle
import numpy as np
import zlib
from torch.multiprocessing import Pool, set_start_method
import random
import shutil
from scipy.io import loadmat
import torch
import imageio
from tqdm import tqdm
import imageio
import cv2
import dlib

# test 
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



logger = logger_config(log_path='data_process.log', logging_name='data process log')
dataset_name='vox'

def check_data(dir_name):
    r'''选择需要的150000个数据，并进行检验'''
    # 在对数据进行预处理之前，先检查数据是否有问题
    global logger
    logger.info('进程{}检验文件夹{}的内容'.format(os.getpid(),dir_name))

    detector = dlib.get_frontal_face_detector()
    filenames = sorted(glob.glob(f'{dir_name}/*.mp4'))
    delete_num=0
    for file in filenames:
        # 检验：是否能够正常读取视频
        try:
            reader = imageio.get_reader(file)
            # 信息表示为整型
            driving_video=[im for im in reader]
            reader.close()
        except:
            reader.close()
            os.remove(file)
            delete_num+=1
            continue
        
        # # 检验：第一帧是否有人脸
        # gray = cv2.cvtColor(driving_video[0], cv2.COLOR_BGR2GRAY)
        # rects = detector(gray, 1) 
        # if len(rects)==0:
        #     os.remove(file)
        #     delete_num+=1
        #     continue
    
    logger.info(f'删除了{delete_num}个视频')
    
    logger.info('已结束：进程{}检验文件夹{}的内容'.format(os.getpid(),dir_name))


def format_data(dir_name):
    # 先调用两个方法，获得video和audio的数据
    global logger,dataset_name
    logger.info('进程{}处理文件夹{}的内容'.format(os.getpid(),dir_name))
    
    process_video(dir_name,dataset_name)

    os.makedirs(f'{dir_name}/temp',exist_ok=True)
    # 从process_video方法中获得的裁剪面部结果生成视频，从这个视频中获得3dmm
    video_list = sorted(glob.glob(f'{dir_name}/*.mp4'))
    for video_path in video_list:
        with open(video_path.replace('.mp4','_temp1.pkl'),'rb') as f:
            info= pickle.load(f)
        face_video=info['face_video']
        imageio.mimsave(f'{dir_name}/temp/{os.path.basename(video_path)}',face_video)
    # 获得3dmm
    video_to_3DMM_and_pose(f'{dir_name}/temp')
    # 将结果搬回原文件夹中
    video_list = sorted(glob.glob(f'{dir_name}/*.mp4'))
    for video_path in video_list:
        shutil.move('{}/temp/{}'.format(dir_name,os.path.basename(video_path).replace('.mp4','.txt')),
                    video_path.replace('.mp4','.txt'))
        shutil.move('{}/temp/{}'.format(dir_name,os.path.basename(video_path).replace('.mp4','.mat')),
                    video_path.replace('.mp4','.mat'))
    # 删除temp文件夹
    shutil.rmtree(f'{dir_name}/temp')

    process_audio(dir_name)
    
    logger.info('已结束：进程{}任务处理文件夹{}的内容'.format(os.getpid(),dir_name))


def merge_data(dir_name):
    # 获得所有信息后，开始合并所需要的信息
    global logger
    logger.info('进程{}合并文件夹{}的内容'.format(os.getpid(),dir_name))

    # 对视频的整理
    filenames = sorted(glob.glob(f'{dir_name}/*.mp4'))
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
        os.remove(video_path.replace('.mp4','_temp1.pkl'))
        os.remove(video_path.replace('.mp4','.mat'))
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

        # 获得音频的数据
        with open(video_path.replace('.mp4','_audio.pkl'),'rb') as f:
            temp = pickle.load(f)
        info['audio_mfcc']=temp['audio_mfcc']

        # 完了之后，删除pkl文件
        os.remove(video_path.replace('.mp4','_video.pkl'))
        os.remove(video_path.replace('.mp4','_audio.pkl'))
        
        # 没有视频信息的，直接弃用
        frame=info['face_video'][0]
        if np.sum(frame==0)>3*256*256/4:
            continue

        # # 没有音频信息的，直接弃用
        # word=info['audio_word']
        # if len(word)==0:
        #     continue

        # 对齐音频数据
        print('进程{}处理{}，裁剪了{}'.format(os.getpid(),video_path,len(info['face_video'])-len(info['audio_mfcc'])))
        if len(info['audio_mfcc'])>=len(info['face_video']):
            info['audio_mfcc']=info['audio_mfcc'][:len(info['face_video'])]
        else:
            temp_last=torch.tensor(info['audio_mfcc'][-1]).expand(len(info['face_video'])-len(info['audio_mfcc']),-1,-1)
            info['audio_mfcc']=np.concatenate((info['audio_mfcc'],temp_last.numpy()))
        # 检验长度对齐
        assert len(info['face_coeff']['coeff'])==len(info['face_video'])
        assert len(info['audio_mfcc'])==len(info['face_video'])

        # 获得最终数据，使用压缩操作
        info = pickle.dumps(info)
        info=zlib.compress(info)
        with open(video_path.replace('.mp4','.pkl'),'wb') as f:
            f.write(info)
    
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

    logger.info('已结束：进程{}任务合并文件夹{}的内容'.format(os.getpid(),dir_name))


def move_data(config):
    global logger
    logger.info(f'转移数据中...')
    # 接下是转移文件
    dataset_root=config['voxceleb2_root_path']
    filenames=sorted(glob.glob(f'{dataset_root}/train_part/*/*.pkl'))
    out_path=config['format_output_path']
    # # 拿300作为验证集，300作为测试集，其余作为训练集
    random.shuffle(filenames)
    eval_file=filenames[:300]
    test_file=filenames[300:600]
    train_file=filenames[600:]
    # 测试验证集，测试集是否有在训练集中
    eval_set,test_set,train_set=set(eval_file),set(test_file),set(train_file)
    for eval in eval_set:
        if eval in train_set:
            logger.info(f'验证集的数据{eval}在训练集中')
    for test in test_set:
        if test in train_set:
            logger.info(f'测试集的数据{test}在训练集中')

    # 开始转移文件
    cnt=0
    for file in eval_file:
        if cnt%500==0:
            path=os.path.join(out_path,f'eval/{cnt//500}')
            os.makedirs(path,exist_ok=True)
        out_name=os.path.basename(file)
        out_name=os.path.join(path,out_name)
        shutil.move(file,out_name)
        cnt+=1
    cnt=0
    for file in test_file:
        if cnt%500==0:
            path=os.path.join(out_path,f'test/{cnt//500}')
            os.makedirs(path,exist_ok=True)
        out_name=os.path.basename(file)
        out_name=os.path.join(path,out_name)
        shutil.move(file,out_name)
        cnt+=1
    cnt=0
    for file in train_file:
        if cnt%500==0:
            path=os.path.join(out_path,f'train/{cnt//500}')
            os.makedirs(path,exist_ok=True)
        out_name=os.path.basename(file)
        out_name=os.path.join(path,out_name)
        shutil.move(file,out_name)
        cnt+=1
    logger.info(f'转移数据完毕')


    # 所有文件的解压测试
    out_path=config['format_output_path']
    pkl_files=sorted(glob.glob(f'{out_path}/*/*/*.pkl'))
    delete_file=[]
    logger.info(f'对pkl文件进行解压测试中...')
    for file in tqdm(pkl_files):
        try:
            with open(file,'rb') as f:
                byte_file=f.read()
            byte_file=zlib.decompress(byte_file)
            data= pickle.loads(byte_file)
        except:
            delete_file.append(file)    
    print(delete_file)
    logger.info(f'需要删除的文件有：')
    for file in delete_file:
        os.remove(file)
        logger.info(f'{file}')
    logger.info(f'删除完成')


if __name__=='__main__':
    set_start_method('spawn')

    # 希望各个方法能够做到：输入视频路径，输出pickl的pkl文件
    import yaml
    config={}
    with open(r'config/dataset/common.yaml',encoding='utf8') as f:
        config.update(yaml.safe_load(f))
    dataset_root=config['voxceleb2_root_path']
    
    # 保证输出路径
    assert 'vox' in config['format_output_path'], 'format output path {} is not voxceleb2'.format(
        config['format_output_path'])


    # 抽取数据
    # filenames=glob.glob(f'{dataset_root}/mp4/*/*/*.mp4')
    # random.shuffle(filenames)
    # # 需要删除的视频比例是10%，因此先选择160000个数据
    # filenames=filenames[:160000]
    # np.save('vox_list.npy',np.array(filenames))
    # 由于vox的数据来源于随机抽检，而我们的数据预处理方法基于文件夹，需要创建临时文件夹进行数据存放
    # 视频命名格式是id_标识_编号.mp4
    # 存储路径是'{dataset_root}/train_part/{index}/*.mp4'
    # filenames=np.load('vox_list.npy').tolist()
    # save_path=f'{dataset_root}/train_part'
    # index=-1
    # for i,file in tqdm(enumerate(filenames)):
    #     if i%500==0:
    #         index+=1
    #         os.makedirs(f'{save_path}/{index}',exist_ok=True)
    #     file_copy=file
    #     dir_list=[]
    #     for _ in range(3):
    #         dir_list.append(os.path.basename(file_copy))
    #         file_copy=os.path.dirname(file_copy)
    #     file_name=f'{dir_list[2]}_{dir_list[1]}_{dir_list[0]}'
    #     file_name=f'{save_path}/{index}/{file_name}'
    #     shutil.copyfile(file,file_name)

    # dir_list=sorted(glob.glob(f'{dataset_root}/train_part/*'))
    # 只用36个就行
    index_list=[i for i in range(36)]
    dir_list=[]
    for index in index_list:
        dir_list.append(f'{dataset_root}/train_part/{index}')

    # index=0
    # for i,dir in enumerate(dir_list):
    #     if '/workspace/dataset/voxceleb/train/train_part/23' in dir:
    #         index=i
    #         break
    # dir_list=dir_list[index:]

    # test
    # dir_list=['data_vox']
    # for file_list in dir_list:
    #     check_data(file_list)
    #     format_data(file_list)
    #     merge_data(file_list)
        # move_data(config)

    
    # workers=5
    # pool = Pool(workers)
    # for _ in pool.imap_unordered(check_data,dir_list):
    #     None

    # print(logger.info('\n检验数据完毕，现在开始处理数据\n'))
    # workers=3
    # pool = Pool(workers)
    # for _ in pool.imap_unordered(format_data,dir_list):
    #     None
    # pool.close()

    # print(logger.info('\n获得的数据已处理完毕，现在合并文件\n'))
    # workers=5
    # pool = Pool(workers)
    # for _ in pool.imap_unordered(merge_data,dir_list):
    #     None
    # pool.close()

    move_data(config)


