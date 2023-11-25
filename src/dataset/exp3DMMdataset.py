import glob
import zlib
import pickle
import torch
import numpy as np
import dlib
from torch.nn.utils.rnn import pad_sequence
import random
import os
import copy



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
        sample=self.filenames[idx]
        emo=os.path.basename(sample)[:os.path.basename(sample).rfind('_')]
        # 随机选择正样本
        pos_index=random.choice(range(0,len(self.emo_dict[emo])))
        pos_file=self.emo_dict[emo][pos_index]
        # 随机选择负样本
        while True:
            temp = random.choice(list(self.emo_dict))
            if temp != emo:
                emo=temp
                break
        neg_index=random.choice(range(0,len(self.emo_dict[emo])))
        neg_file=self.emo_dict[emo][neg_index]
        out={}
        out.update(self.get_data(sample,type=''))
        out.update(self.get_data(neg_file,type='neg_'))
        out.update(self.get_data(pos_file,type='pos_'))
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
        #[frame ,3,H,W]
        out[f'{type}gt_video']=torch.tensor(video_data).float().permute(0,3,1,2)
        img=(img/255*2)-1
        #[frame,3,H,W]
        out[f'{type}img']=torch.tensor(img).float().permute(0,3,1,2)

        data_3DMM=self.process_3DMM(data)
        # (frame,win,64)
        out[f'{type}gt_3dmm']=torch.tensor(data_3DMM[0]).float()
        # (frame,win,80)
        out[f'{type}id_3dmm']=torch.tensor(data_3DMM[1]).float()

        pose_data=self.process_pose(data)
        # (frame,win,9)
        out[f'{type}pose']=torch.tensor(pose_data).float()


        audio_data=self.process_audio(data)
        # (frame,win,28,dim)
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
        # (frame,max_len,64)
        out[f'{type}style_clip']=style_clip.float().expand(self.frame_num,-1,-1)
        # (frame,max_len)
        out[f'{type}mask']=pad_mask.expand(self.frame_num,-1)
        return out

    def process_video(self,data):
        video_data=data['face_video']

        temp_index=random.sample(range(len(video_data)),self.frame_num)
        
        src=[]
        target=[]
        for i in range(self.frame_num):
            target.append(video_data[temp_index[i]])
            src.append(video_data[temp_index[(i+1)%self.frame_num]])
        if self.frame_num==1:
            src=[video_data[random.choice(range(len(video_data)))]]

        frame_index=[]
        for i in temp_index:
            frame_index.append([i])

        return np.array(target),np.array(src),frame_index

    def process_audio(self,data):
        frame_index_list=copy.deepcopy(data['frame_index'])
        for temp in frame_index_list:
            for _ in range(self.audio_win_size+self.exp_3dmm_win_size):
                temp.append(temp[-1]+1)
                temp.insert(0,max(temp[0]-1,0))
        input_audio_data=data['audio_mfcc']
        input_audio_hubert=data['audio_hugebert']
        audio_input=[]
        audio_hubert=[]

        # 获取有效数据
        for temp in frame_index_list:
            input=[]
            hubert=[]
            for frame_index in temp:
                # 对于音频，越界的情况只有在末尾发生，因此直接取最后一个
                if frame_index>=len(input_audio_data):
                    input.append(input_audio_data[-1])
                else:
                    input.append(input_audio_data[frame_index])
                if (2*frame_index+1)>=len(input_audio_hubert):
                    hubert.append(input_audio_hubert[-2:])
                else:
                    hubert.append(input_audio_hubert[2*frame_index:2*(frame_index+1)])
            audio_hubert.append(hubert)
            audio_input.append(input)

        return np.array(audio_input),np.array(audio_hubert)

    def process_3DMM(self,data):
        frame_index_list=copy.deepcopy(data['frame_index'])
        data_3DMM=data['face_coeff']['coeff']
        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
        face3d_id=data_3DMM[:, 0:80]
        for temp in frame_index_list:
            for _ in range(self.exp_3dmm_win_size):
                temp.append(min(temp[-1]+1,len(face3d_exp)-1))
                temp.insert(0,max(temp[0]-1,0))

        # [frame num,win size,9]
        exp_3DMM=[]
        id_3DMM=[]
        for temp in frame_index_list:
            exp_3DMM.append([face3d_exp[frame_index] for frame_index in temp])
            id_3DMM.append([face3d_id[frame_index] for frame_index in temp])

        return np.array(exp_3DMM),np.array(id_3DMM)

    def process_pose(self,data):
        frame_index_list=copy.deepcopy(data['frame_index'])
        mat_dict = data['face_coeff']
        np_3dmm = mat_dict["coeff"]
        for temp in frame_index_list:
            for _ in range(self.exp_3dmm_win_size):
                temp.append(min(temp[-1]+1,len(np_3dmm)-1))
                temp.insert(0,max(temp[0]-1,0))


        angles = np_3dmm[:, 224:227]
        translations = np_3dmm[:, 254:257]
        np_trans_params = mat_dict["transform_params"]
        crop = np_trans_params[:, -3:]
        # [len,9]
        pose_params = np.concatenate((angles, translations, crop), axis=1)

        # [frame num,win size,9]
        num_ans=[]
        for temp in frame_index_list:
            num_ans.append([pose_params[frame_index] for frame_index in temp])
        
        return np.array(num_ans)

    def collater(self, samples):
        # 对齐数据
        triple_list=['','pos_','neg_']
        data_key=['gt_video','img','gt_3dmm','id_3dmm','audio','style_clip','mask','pose']
        data={}
        for key in data_key:
            for triple in triple_list:
                temp=[s[f'{triple}{key}'] for s in samples]
                data[f'{triple}{key}']=torch.cat(temp)

        return data

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml

    config={}
    yaml_file=['config/dataset/common.yaml','config/model/render.yaml',
               'config/model/exp3DMM.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=Exp3DMMdataset(config,type='train',frame_num=1)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2, 
        shuffle=True,
        drop_last=False,
        num_workers=1,
        collate_fn=dataset.collater
    )     
    for data in dataloader:
        for key,value in data.items():
            print('{}形状为{}'.format(key,value.shape))
        break
