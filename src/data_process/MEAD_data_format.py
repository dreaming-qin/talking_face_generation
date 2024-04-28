
import glob
import pickle
import numpy as np
import zlib
from torch.multiprocessing import Pool, set_start_method
import os
import random
import shutil
from scipy.io import loadmat
import torch
import imageio
from tqdm import tqdm

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
dataset_name='mead'

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
        info['audio_word']=temp['audio_word']

        # 完了之后，删除pkl文件
        os.remove(video_path.replace('.mp4','_video.pkl'))
        os.remove(video_path.replace('.mp4','_audio.pkl'))
        
        # 没有视频信息的，直接弃用
        frame=info['face_video'][0]
        if np.sum(frame==0)>3*256*256/4:
            continue

        # 没有音频信息的，直接弃用
        word=info['audio_word']
        if len(word)==0:
            continue

        # 对齐音频数据
        if len(info['audio_word'])>=len(info['face_video']):
            info['audio_word']=info['audio_word'][:len(info['face_video'])]
        else:
            temp_last=torch.tensor(31).expand(len(info['face_video'])-len(info['audio_word']))
            info['audio_word']=np.concatenate((info['audio_word'],temp_last.numpy()))
        # 检验长度对齐
        assert len(info['face_coeff']['coeff'])==len(info['face_video'])
        assert len(info['audio_word'])==len(info['face_video'])

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
    dataset_root=config['mead_root_path']
    filenames=sorted(glob.glob(f'{dataset_root}/*/video/*/*/*/*.pkl'))
    out_path=config['format_output_path']
    # 拿300作为验证集，300作为测试集，其余作为训练集
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
        file_list=file.split('/')
        out_name=f'{file_list[-6]}_{file_list[-4]}_{file_list[-3]}_{file_list[-2]}_{file_list[-1]}'
        out_name=os.path.join(path,out_name)
        shutil.move(file,out_name)
        cnt+=1
    cnt=0
    for file in test_file:
        if cnt%500==0:
            path=os.path.join(out_path,f'test/{cnt//500}')
            os.makedirs(path,exist_ok=True)
        file_list=file.split('/')
        out_name=f'{file_list[-6]}_{file_list[-4]}_{file_list[-3]}_{file_list[-2]}_{file_list[-1]}'
        out_name=os.path.join(path,out_name)
        shutil.move(file,out_name)
        cnt+=1
    cnt=0
    for file in train_file:
        if cnt%500==0:
            path=os.path.join(out_path,f'train/{cnt//500}')
            os.makedirs(path,exist_ok=True)
        file_list=file.split('/')
        out_name=f'{file_list[-6]}_{file_list[-4]}_{file_list[-3]}_{file_list[-2]}_{file_list[-1]}'
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
    dataset_root=config['mead_root_path']

    # 保证输出路径
    assert 'mead' in config['format_output_path'], 'format output path {} is not mead'.format(
        config['format_output_path'])


    # 对于一些不符合规范的视频，删除掉，不然处理会产生问题
    # unvalid_file=[]
    # filenames=sorted(glob.glob(f'{dataset_root}/*/video/*/*/*/*.mp4'))
    # for file in filenames:
    #     if os.path.getsize(file)==0:
    #         unvalid_file.append(file)
    # for file in unvalid_file:
    #     os.remove(file)

    dir_list=sorted(glob.glob(f'{dataset_root}/*/video/front/*/*'))

    index=0
    for i,dir in enumerate(dir_list):
        if '/workspace/dataset/MEAD/M012/video/front/happy/level_1' in dir:
            index=i
            break
    dir_list=dir_list[index:]

    # test
    dir_list=['data_mead']
    for file_list in dir_list:
        format_data(file_list)
        merge_data(file_list)

    
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

    # move_data(config)


