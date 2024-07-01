import glob
import zlib
import pickle
import torch
import numpy as np
import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader


# 测试代码
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))




class RenderDataset(torch.utils.data.Dataset):
    r'''用于训练render模块的数据集，需要获得输入，以及LOSS函数需要的输入
    使用PTRender方式进行改进'''

    def __init__(self,config,type='train',frame_num=2):
        format_path=config['format_output_path']
        self.filenames=sorted(glob.glob(f'{format_path}/{type}/*/*.pkl'))
        self.win_size=config['render_win_size']
        self.frame_num=frame_num
        self.type=type

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file=self.filenames[idx]
        # 解压pkl文件
        with open(file,'rb') as f:
            # data= pickle.load(f)
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        out={}

        self.update_index_list(data)

        # 下列的self.process方法中，需要根据frame_index拿出合法数据
        video_data=self.process_video(data)
        # [frame num,H,W,3]
        src=(video_data[0]/255*2)-1
        # [frame num,H,W,3]
        tgt=(video_data[1]/255*2)-1
        #[frame num,3,H,W]
        out['src']=torch.Tensor(src).float().permute(0,3,1,2)
        # [frame num,3,H,W]
        out['target']=torch.Tensor(tgt).float().permute(0,3,1,2)

        # [frame num,win size,9]
        src_pose,tgt_pose=self.process_pose(data)
        # [frame num,9]
        src_pose=torch.tensor(src_pose).float()
        out['src_pose']=src_pose
        tgt_pose=torch.tensor(tgt_pose).float().permute(0,2,1)
        # [frame num,9,win size]
        out['tgt_pose']=tgt_pose
        
        # [frame num,win size,64]
        tgt_exp=self.process_exp_3DMM(data)
        tgt_exp=torch.tensor(tgt_exp).float().permute(0,2,1)
        # [frame num,64,win size]
        out['tgt_exp']=tgt_exp

        
        # [frame num,win size,64]
        src_id=self.process_src_inf(data)
        src_id=torch.tensor(src_id).float()
        # [frame num,64]
        out['src_inf']=src_id

        return out
    
    def process_src_inf(self,data):
        data_3DMM=data['face_coeff']['coeff']

        src_index=data['src_index']

        # [frame num,win size,64]
        src_exp_3DMM=data_3DMM[src_index]

        return np.array(src_exp_3DMM)

    def update_index_list(self,data):
        video_data=data['face_video']
        tgt_temp_index=random.sample(range(len(video_data)),self.frame_num)
        src_temp_index=random.sample(range(len(video_data)),self.frame_num)
        data['tgt_index']=tgt_temp_index
        data['src_index']=src_temp_index
        return data

    def process_video(self,data):
        r'''返回(src,target),frame_index这样的数据'''
        video_data=data['face_video']
        if 'fake_face_video' in data:
            src_video_data=data['fake_face_video']
        else:
            src_video_data=data['face_video']


        # 由于要生成序列性的视频，需要取一段连续的序列
        # 从video中随机选择frame_num张帧
        temp_index=data['tgt_index']
        src_temp_index=data['src_index']

        src=[]
        target=[]
        for i in range(self.frame_num):
            target.append(video_data[temp_index[i]])
            src.append(src_video_data[src_temp_index[i]])

        # if self.frame_num==1:
        #     src=[video_data[random.choice(range(len(video_data)))]]

        src=np.array(src)
        target=np.array(target)

        return (src,target)

    def process_pose(self,data):
        """Get pose parameters from mat file

        Returns:
            pose_params (numpy.ndarray): shape (L_video, 9), angle, translation, crop paramters
        """
        mat_dict = data['face_coeff']

        np_3dmm = mat_dict["coeff"]
        angles = np_3dmm[:, 224:227]
        translations = np_3dmm[:, 254:257]

        np_trans_params = mat_dict["transform_params"]
        crop = np_trans_params[:, -3:]

        # [len,9]
        pose_params = np.concatenate((angles, translations, crop), axis=1)

        src_index=data['src_index']
        tgt_index=data['tgt_index']

        # [frame num,win size,9]
        src_ans=pose_params[src_index]
        tgt_ans=self.get_window(tgt_index,pose_params)
        
        return np.array(src_ans),np.array(tgt_ans)

    def process_exp_3DMM(self,data):
        data_3DMM=data['face_coeff']['coeff']

        # [len,64]
        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range

        tgt_index=data['tgt_index']

        # [frame num,win size,64]
        tgt_exp_3DMM=self.get_window(tgt_index,face3d_exp)

        return np.array(tgt_exp_3DMM)

    def collater(self, samples):
        # 对齐数据
        key_dict={'src','target','tgt_pose','tgt_exp','src_inf','src_pose'}
        data={}
        for key in key_dict:
            value = [data for s in samples for data in s[key]]
            value=torch.stack(value)
            data[key]=value

        return data
        

    def get_window(self,frame_index,array):
        '''array[len,dim]
        frame_index: 选中帧
        
        return : [len(frame_index), win size,dim]'''
        num_ans=[]
        for index in frame_index: 
            batch_ans=[]
            for i in range(index - self.win_size, index + self.win_size + 1):
                if i < 0:
                    batch_ans.append(array[0])
                elif i >= array.shape[0]:
                    batch_ans.append(array[-1])
                else:
                    batch_ans.append(array[i])
            num_ans.append(np.array(batch_ans))
        return np.array(num_ans)

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml
    import torchvision

    config={}
    yaml_file=['config/dataset/common.yaml',
               'config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    # 会输出batch_size*frame_num批次的数据
    dataset=RenderDataset(config,type='train',frame_num=5)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )     
    for data in dataloader:
        for key,value in data.items():
            print('{}形状为{}'.format(key,value.shape))
            # 需要[len,H,W,3]，且为像素取值范围为0-255
            # torchvision.io.write_video('temp.mp4', ((value+1)/2*255).permute(0,2,3,1).cpu(), fps=5)
