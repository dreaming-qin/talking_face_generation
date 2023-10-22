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
from src.util.data_process.video_process_util import process_video
import glob
import pickle
import numpy as np
import zlib

def format_data(dir_name):
    filenames = glob.glob(f'{dir_name}/*.mp4')
    info={}
    for video_path in filenames:
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

if __name__=='__main__':
    # 希望各个方法能够做到：输入视频路径，输出pickl的pkl文件
    import yaml
    with open(r'config\data_process\common.yaml',encoding='utf8') as f:
        config=yaml.safe_load(f)
    with open(r'config\dataset\common.yaml',encoding='utf8') as f:
        config.update(yaml.safe_load(f))

    dataset_root=config['mead_root_path']
    dir_list=glob.glob(f'{dataset_root}/*/video/*/*/*')
    
    for dir_name in dir_list:
        process_video(dir_name,config)
        process_audio(dir_name)
        format_data(dir_name)