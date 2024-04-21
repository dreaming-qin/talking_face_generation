import glob
import zlib
import pickle
import torch
import numpy as np
import dlib
import random
import os
import copy
import time

import imageio



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

    def __init__(self,config,type='train',frame_num=1):
        format_path=config['format_output_path']
        self.filenames=sorted(glob.glob(f'{format_path}/{type}/*/*.pkl'))
        self.frame_num=frame_num
        self.exp_3dmm_win_size=config['render_win_size']
        self.mask_width=config['mask_width']
        self.mask_height=config['mask_height']


        self.format_path=format_path
        self.img_dict={}
        if 'mead' in format_path:
            for file in self.filenames:
                name,_,emo,_,_,video_num=os.path.basename(file)[:-4].split('_')
                if 'neutral' == emo:
                    if not f'{name}_{emo}' in self.img_dict:
                        self.img_dict[f'{name}_{emo}' ]={}
                    if not f'{video_num}' in self.img_dict[f'{name}_{emo}']:
                        self.img_dict[f'{name}_{emo}'][f'{video_num}' ]=[]
                    self.img_dict[f'{name}_{emo}'][f'{video_num}' ].append(file)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample=self.filenames[idx]
        out={}
        out.update(self.get_data(sample,type=''))
        return out

    def get_data(self,file,type):
        # 解压pkl文件
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        out={}

        # 对参考图片，使用中性视频中的图片表示
        if 'mead' in self.format_path:
            name,_,_,_,_,video_num=os.path.basename(file)[:-4].split('_')
            if f'{video_num}' not in self.img_dict[f'{name}_neutral']:
                src_key=random.sample(self.img_dict[f'{name}_neutral'].keys(),1)[0]
                src_file=self.img_dict[f'{name}_neutral'][src_key][0]
            else:
                src_file=self.img_dict[f'{name}_neutral'][f'{video_num}'][0]
            with open(src_file,'rb') as f:
                byte_file=f.read()
            byte_file=zlib.decompress(byte_file)
            src_data= pickle.loads(byte_file)

        # 下列的self.process方法中，需要根据frame_index拿出合法数据
        video_data,img,data['frame_index']=self.process_video(data,src_data)
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
        out[f'{type}audio']=torch.tensor(audio_data).float()

        # 获得输入视频
        mask_video=self.process_mask_video(data)
        out[f'{type}video']=torch.tensor((mask_video/255*2)-1).float()
        return out

    def process_mask_video(self,data):
        frame_index_list=random.sample(range(len(data['face_video'])),self.frame_num)
        frame_index_list=[[i] for i in frame_index_list]
        for temp in frame_index_list:
            for _ in range(self.exp_3dmm_win_size):
                temp.append(min(temp[-1]+1,len(data['face_video'])-1))
                temp.insert(0,max(temp[0]-1,0))
                # temp+=[temp[-1],temp[-1]]

        mask_video=[]
        for index_list in frame_index_list:
            temp=[]
            for index in index_list:
                img=data['face_video'][index].copy()
                # mask=data['mouth_mask'][index]
                # if mask[1]-mask[0]>0 and mask[3]-mask[2]>0:
                #     top,bottom,left,right=mask
                #     top_temp=max(0,min(top,top-((self.mask_height-bottom+top)//2)))
                #     bottom_temp=min(img.shape[0],max(bottom,bottom+((self.mask_height-bottom+top)//2)))
                #     left_temp=max(0,min(left,left-((self.mask_width-right+left)//2)))
                #     right_temp=min(img.shape[1],max(right,right+((self.mask_width-right+left)//2)))
                #     noise=np.random.randint(0,256,(bottom_temp-top_temp,right_temp-left_temp,3))
                #     img[top_temp:bottom_temp,left_temp:right_temp]=noise
                temp.append(img)
                # # test
                # imageio.imsave('temp1.jpg',data['face_video'][index])
                # imageio.imsave('temp2.jpg',img)
            mask_video.append(temp)
        
        return np.array(mask_video)


    def process_video(self,data,src_data):
        video_data=data['face_video']
        src_video_data=src_data['face_video']

        temp_index=random.sample(range(len(video_data)),self.frame_num)
        src_temp_index=random.sample(range(len(src_video_data)),self.frame_num)
        
        target=[]
        src=[]
        for i in range(self.frame_num):
            target.append(video_data[temp_index[i]])
            src.append(src_video_data[src_temp_index[i]])
            
        # if self.frame_num==1:
        #     src=[src_video_data[random.choice(range(len(src_video_data)))]]

        frame_index=[]
        for i in temp_index:
            frame_index.append([i])

        return np.array(target),np.array(src),frame_index

    def process_audio(self,data):
        frame_index_list=copy.deepcopy(data['frame_index'])
        for temp in frame_index_list:
            for _ in range(self.exp_3dmm_win_size):
                temp.append(temp[-1]+1)
                temp.insert(0,max(temp[0]-1,0))
        input_audio_data=data['audio_mfcc']
        audio_input=[]

        # 获取有效数据
        for temp in frame_index_list:
            input=[]
            for frame_index in temp:
                # 对于音频，越界的情况只有在末尾发生，因此直接取最后一个
                if frame_index>=len(input_audio_data):
                    input.append(input_audio_data[-1])
                else:
                    input.append(input_audio_data[frame_index])
            audio_input.append(input)

        return np.array(audio_input)

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
        triple_list=['']
        data_key=['gt_video','img','gt_3dmm','id_3dmm','audio','video','pose']
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
       
    dataset=Exp3DMMdataset(config,type='train',frame_num=2)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )     
    for data in dataloader:
        for key,val in data.items():
            print(f'{key}的形状是{val.shape}')
