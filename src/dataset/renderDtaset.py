import glob
import zlib
import pickle
import torch
import numpy as np
import dlib
import random
from torchvision import transforms


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

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file=self.filenames[idx]
        # 解压pkl文件
        with open(file,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        out={}
        # 下列的self.process方法中，需要根据frame_index拿出合法数据
        video_data,data['frame_index']=self.process_video(data)
        # [frame num,H,W,3]
        src=(video_data[0]/255*2)-1
        # [frame num,H,W,3]
        tgt=(video_data[1]/255*2)-1
        #[frame num,3,H,W]
        out['src']=torch.Tensor(src).float().permute(0,3,1,2)
        # [frame num,3,H,W]
        out['target']=torch.Tensor(tgt).float().permute(0,3,1,2)

        # [frame num,win size,9]
        pose_data=self.process_pose(data)
        # [frame num,9,win size]
        pose_data=torch.tensor(pose_data).float().permute(0,2,1)
        # [frame num,win size,64]
        exp_3DMM=self.process_3DMM(data)
        # [frame num,64,win size]
        exp_3DMM=torch.tensor(exp_3DMM).float().permute(0,2,1)
        driving_src=torch.cat((exp_3DMM,pose_data),dim=1)
        out['driving_src']= driving_src

        return out

    def process_video(self,data):
        r'''返回(src,target),frame_index这样的数据'''
        video_data=data['face_video']

        # 由于要生成序列性的视频，需要取一段连续的序列
        # 从video中随机选择frame_num张帧
        temp_index=random.sample(range(len(video_data)),self.frame_num)
        src=[]
        target=[]
        for i in range(self.frame_num):
            target.append(video_data[temp_index[i]])
            src.append(video_data[temp_index[(i+1)%self.frame_num]])
        frame_index=temp_index
        src=np.array(src)
        target=np.array(target)

        return (src,target),frame_index

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

        frame_index_list=data['frame_index']
        # [frame num,win size,9]
        num_ans=self.get_window(frame_index_list,pose_params)
        
        return np.array(num_ans)

    def process_3DMM(self,data):
        frame_index_list=data['frame_index']
        data_3DMM=data['face_coeff']['coeff']

        # [len,64]
        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range

        frame_index_list=data['frame_index']
        # [frame num,win size,64]
        exp_3DMM=self.get_window(frame_index_list,face3d_exp)

        return np.array(exp_3DMM)

    def collater(self, samples):
        # 对齐数据
        src = [data for s in samples for data in s['src']]
        target = [data for s in samples for data in s['target']]
        driving_src = [data for s in samples for data in s['driving_src']]


        src=torch.stack(src)
        target=torch.stack(target)
        driving_src=torch.stack(driving_src)

        data={}
        data.update({
            'src': src,
            'target': target,
            'driving_src': driving_src,
        })
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
    yaml_file=['config/data_process/common.yaml','config/dataset/common.yaml',
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
            torchvision.io.write_video('temp.mp4', ((value+1)/2*255).permute(0,2,3,1).cpu(), fps=5)
