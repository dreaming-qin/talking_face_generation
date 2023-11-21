import glob
import zlib
import pickle
import torch
import numpy as np
import dlib
from torch.nn.utils.rnn import pad_sequence
import random
import os



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



def get_video_style_clip(face3d_exp, style_max_len, start_idx="random", dtype=torch.float32):
    

    face3d_exp = torch.tensor(face3d_exp, dtype=dtype)
    length = face3d_exp.shape[0]
    if length >= style_max_len:
        clip_num_frames = style_max_len
        if start_idx == "random":
            clip_start_idx = np.random.randint(low=0, high=length - clip_num_frames + 1)
        elif start_idx == "middle":
            clip_start_idx = (length - clip_num_frames + 1) // 2
        elif isinstance(start_idx, int):
            clip_start_idx = start_idx
        else:
            raise ValueError(f"Invalid start_idx {start_idx}")

        face3d_clip = face3d_exp[clip_start_idx : clip_start_idx + clip_num_frames]
        pad_mask = torch.tensor([False] * style_max_len)
    else:
        padding = torch.zeros(style_max_len - length, face3d_exp.shape[1])
        face3d_clip = torch.cat((face3d_exp, padding), dim=0)
        pad_mask = torch.tensor([False] * length + [True] * (style_max_len - length))

    return face3d_clip, pad_mask



class Exp3DMMdataset(torch.utils.data.Dataset):
    r'''用于训练EXP 3DMM提取模块的数据集，需要获得输入，以及LOSS函数需要的输入'''

    def __init__(self,config,type='train',frame_num=1):
        format_path=config['format_output_path']
        self.filenames=sorted(glob.glob(f'{format_path}/{type}/*/*.pkl'))
        self.emo_dict={}
        for file in self.filenames:
            name=os.path.basename(file)
            emo=name.split('_')[0]
            if not emo in self.emo_dict:
                self.emo_dict[emo]=[]
            self.emo_dict[emo].append(file)
        self.frame_num=frame_num
        self.audio_win_size=config['audio_win_size']
        self.exp_3dmm_win_size=config['render_win_size']

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        r'''由于gpu大小问题，只能将帧长度设置为70，在这里进行更改'''
        file=self.filenames[idx]
        out={}
        out.update(self.get_data(file,type=''))
        emo=os.path.basename(file)[:os.path.basename(file).rfind('_')]
        pos_index=random.choice(range(0,len(self.emo_dict[emo])))
        pos_file=self.emo_dict[emo][pos_index]
        out.update(self.get_data(pos_file,type='pos_'))
        while True:
            temp = random.choice(list(self.emo_dict))
            if temp != emo:
                emo=temp
                break
        neg_index=random.choice(range(0,len(self.emo_dict[emo])))
        neg_file=self.emo_dict[emo][neg_index]
        out.update(self.get_data(neg_file,type='neg_'))
        return out

    def get_data(self,file,type):
        # 解压pkl文件
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        out={}
        # 下列的self.process方法中，需要根据frame_index拿出合法数据
        video_data,img,data['frame_index']=self.process_video(data)
        video_data=(video_data/255*2)-1
        #[frame num,3,H,W]
        out[f'{type}gt_video']=torch.tensor(video_data).float().permute(0,3,1,2)
        img=(img/255*2)-1
        #[3,H,W]
        out[f'{type}img']=torch.tensor(img).float().permute(2,0,1)

        data_3DMM=self.process_3DMM(data)
        out[f'{type}gt_3dmm']=torch.tensor(data_3DMM[0]).float()
        out[f'{type}id_3dmm']=torch.tensor(data_3DMM[1]).float()

        audio_data=self.process_audio(data)
        out[f'{type}audio']=torch.tensor(audio_data[0]).float()
        # out['audio_hubert']=torch.tensor(audio_data[1]).float()

        # 获得style clip，使用同一个人，相同情感的情况下的作为style clip
        emo=os.path.basename(file)[:os.path.basename(file).rfind('_')]
        pos_index=random.choice(range(0,len(self.emo_dict[emo])))
        pos_file=self.emo_dict[emo][pos_index]
        # 解压pkl文件
        with open(pos_file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        data_3DMM=data['face_coeff']['coeff']
        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
        style_clip, pad_mask = get_video_style_clip(face3d_exp, style_max_len=256, start_idx=0)
        out[f'{type}style_clip']=style_clip.float()
        out[f'{type}mask']=pad_mask
        return out

    def process_video(self,data):
        video_data=data['face_video']
        # 从video中随机选择一张当源图片
        temp_index=random.choice(range(0,len(video_data)))
        img=video_data[temp_index]

        # 当超出长度限制时，需要截断
        if   (video_data.shape[0]>self.frame_num):
            # 由于要生成序列性的视频，需要取一段连续的序列
            temp_index=random.sample(range(len(video_data)),self.frame_num)
            video_data=video_data[temp_index]
            frame_index=temp_index

        return video_data,img,frame_index

    def process_audio(self,data):
        frame_index_list=data['frame_index'].copy()
        for _ in range(self.audio_win_size):
            frame_index_list.append(frame_index_list[-1]+1)
            frame_index_list.insert(0,max(frame_index_list[0]-1,0))
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
        frame_index_list=data['frame_index'].copy()
        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
        face3d_id=data_3DMM[:, 0:80]
        for _ in range(self.exp_3dmm_win_size):
            # 有问题
            frame_index_list.append(min(frame_index_list[-1]+1,len(face3d_exp)))
            frame_index_list.insert(0,max(frame_index_list[0]-1,0))
        data_3DMM=data['face_coeff']['coeff']


        # 获取有效数据，3DMM不像音频，不存在越界情况，因此可以直接通过frame_index获得
        exp_3DMM=[face3d_exp[frame_index] for frame_index in frame_index_list]
        id_3DMM=[face3d_id[frame_index] for frame_index in frame_index_list]

        return np.array(exp_3DMM),np.array(id_3DMM)

    # def collater(self, samples):
    #     # 对齐数据
    #     emo_list=['','pos_','neg_']
    #     video_input=[]
    #     tgt_input=[]
    #     audio_input=[]
    #     for emo in emo_list:
    #         pass
    #     audio_input = [s['audio_input'] for s in samples]
    #     audio_hubert = [s['audio_hubert'] for s in samples]
    #     video_input = [s['video_input'] for s in samples]
    #     exp_3DMM = [s['exp_3DMM'] for s in samples]
    #     id_3DMM = [s['id_3DMM'] for s in samples]

    #     audio_input=pad_sequence(audio_input,batch_first =True)
    #     audio_hubert=pad_sequence(audio_hubert,batch_first =True)
    #     video_input=pad_sequence(video_input,batch_first =True)
    #     exp_3DMM=pad_sequence(exp_3DMM,batch_first =True)
    #     id_3DMM=pad_sequence(id_3DMM,batch_first =True)

    #     data={}
    #     data.update({
    #         'audio_input': audio_input,
    #         'audio_hubert': audio_hubert,
    #         'video_input': video_input,
    #         'exp_3DMM': exp_3DMM,
    #         'id_3DMM': id_3DMM
    #     })
    #     return data

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml

    config={}
    yaml_file=['config/data_process/common.yaml','config/dataset/common.yaml',
               'config/model/exp3DMM.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=Exp3DMMdataset(config,type='train',audio_len=10)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        # collate_fn=dataset.collater
    )     
    for data in dataloader:
        for key,value in data.items():
            print('{}形状为{}'.format(key,value.shape))
