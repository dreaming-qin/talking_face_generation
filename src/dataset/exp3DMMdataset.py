import glob
import zlib
import pickle
import torch
import numpy as np
import dlib
import cv2
from torch.nn.utils.rnn import pad_sequence
import random


# 测试代码
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))

from src.util.augmentation import AllAugmentationTransform
from src.util.data_process.video_process_util import shape_to_np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./checkpoint/shape_predictor_68_face_landmarks.dat')

class Exp3DMMdataset(torch.utils.data.Dataset):
    r'''用于训练EXP 3DMM提取模块的数据集，需要获得输入，以及LOSS函数需要的输入'''

    def __init__(self,config,type='train',max_len=None):
        format_path=config['format_output_path']
        self.filenames=sorted(glob.glob(f'{format_path}/{type}/*/*.pkl'))
        self.transformed_video_args=config['augmentation_params']
        self.max_len=max_len

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        r'''由于gpu大小问题，只能将帧长度设置为70，在这里进行更改'''
        file=self.filenames[idx]
        # 解压pkl文件
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        out={}
        # 下列的self.process方法中，需要根据frame_index拿出合法数据
        video_data,data['frame_index']=self.process_video(data,self.max_len)
        out['video_input']=torch.tensor(video_data).float()
        audio_data=self.process_audio(data)
        out['audio_input']=torch.tensor(audio_data[0]).float()
        out['audio_hubert']=torch.tensor(audio_data[1]).float()
        data_3DMM=self.process_3DMM(data)
        out['exp_3DMM']=torch.tensor(data_3DMM[0]).float()
        out['id_3DMM']=torch.tensor(data_3DMM[1]).float()

        return out

    def process_video(self,data,max_len=None):
        video_data=data['align_video']
        frame_index=data['frame_index']
        # 获得参照物图片的信息，更新transformer视频时面部遮罩坐标
        for i in range(video_data.shape[0]):
            template_video_array=video_data[i]
            template_gray = cv2.cvtColor(template_video_array, cv2.COLOR_BGR2GRAY)
            template_rect = detector(template_gray, 1)
            if len(template_rect)!=0:
                template = predictor(template_gray, template_rect[-1]) #detect 68 points
                template = shape_to_np(template)
                # 在这里，将会更新transformer视频时面部遮罩坐标
                top=min(template[50][1],template[52][1])
                bottom=max(template[56][1],template[57][1])
                left=template[48][0]
                right=template[54][0]
                self.transformed_video_args['crop_mouth_param']['center_x']=(left+right)//2
                self.transformed_video_args['crop_mouth_param']['center_y']=(top+bottom)//2
                self.transformed_video_args['crop_mouth_param']['mask_width']=right-left+10
                self.transformed_video_args['crop_mouth_param']['mask_height']=bottom-top+10
                break

        # 当超出长度限制时，需要截断
        if  (max_len is not None) and (video_data.shape[0]>max_len):
            # 由于要生成序列性的视频，需要取一段连续的序列
            temp_index=random.choice(range(0,frame_index.shape[0]-max_len))
            video_data=video_data[temp_index:temp_index+max_len]
            frame_index=frame_index[temp_index:temp_index+max_len]

        
        # 更新transformer参数后，开始transformer video
        input_video=self.transformed_video(video_data/255)
        return input_video,frame_index

    def transformed_video(self,driving_video):
        video_array = np.array(driving_video)
        transformations = AllAugmentationTransform(**self.transformed_video_args)
        transformed_array = transformations(video_array)
        return np.array(transformed_array)

    def process_audio(self,data):
        frame_index_list=data['frame_index']
        input_audio_data=data['audio_mfcc']
        input_audio_hubert=data['audio_hugebert']
        audio_input=[]
        audio_hubert=[]

        # 获取有效数据
        for frame_index in frame_index_list:
            # 对于音频，越界的情况只有在末尾发生，因此直接取最后一个
            if frame_index>=len(input_audio_data):
                audio_input.append(input_audio_data[-1])
            else:
                audio_input.append(input_audio_data[frame_index])
            if (2*frame_index+1)>=len(input_audio_hubert):
                audio_hubert.append(input_audio_hubert[-2:])
            else:
                audio_hubert.append(input_audio_hubert[2*frame_index:2*(frame_index+1)])

        return np.array(audio_input),np.array(audio_hubert)

    def process_3DMM(self,data):
        frame_index_list=data['frame_index']
        data_3DMM=data['face_coeff']['coeff']

        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
        face3d_id=data_3DMM[:, 0:80]

        # 获取有效数据，3DMM不像音频，不存在越界情况，因此可以直接通过frame_index获得
        exp_3DMM=[face3d_exp[frame_index] for frame_index in frame_index_list]
        id_3DMM=[face3d_id[frame_index] for frame_index in frame_index_list]

        return np.array(exp_3DMM),np.array(id_3DMM)

    def collater(self, samples):
        # 对齐数据
        audio_input = [s['audio_input'] for s in samples]
        audio_hubert = [s['audio_hubert'] for s in samples]
        video_input = [s['video_input'] for s in samples]
        exp_3DMM = [s['exp_3DMM'] for s in samples]
        id_3DMM = [s['id_3DMM'] for s in samples]

        audio_input=pad_sequence(audio_input,batch_first =True)
        audio_hubert=pad_sequence(audio_hubert,batch_first =True)
        video_input=pad_sequence(video_input,batch_first =True)
        exp_3DMM=pad_sequence(exp_3DMM,batch_first =True)
        id_3DMM=pad_sequence(id_3DMM,batch_first =True)

        data={}
        data.update({
            'audio_input': audio_input,
            'audio_hubert': audio_hubert,
            'video_input': video_input,
            'exp_3DMM': exp_3DMM,
            'id_3DMM': id_3DMM
        })
        return data

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml

    config={}
    yaml_file=glob.glob(r'config/*/*.yaml')
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=Exp3DMMdataset(config,type='eval',max_len=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )     
    for data in dataloader:
        for key,value in data.items():
            print('{}形状为{}'.format(key,value.shape))
