import glob
import zlib
import pickle
import torch
import numpy as np
import dlib
import cv2
from torch.nn.utils.rnn import pad_sequence
import random
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

class RenderDataset(torch.utils.data.Dataset):
    r'''用于训练render模块的数据集，需要获得输入，以及LOSS函数需要的输入
    由于gpu大小问题，只能将帧长度设置为70，在这里进行更改'''

    def __init__(self,config,type='train',max_len=None):
        format_path=config['format_output_path']
        self.filenames=sorted(glob.glob(f'{format_path}/{type}/*/*.pkl'))
        self.transformed_video_args=config['augmentation_params']
        self.max_len=max_len

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
        video_data,data['frame_index']=self.process_video(data,self.max_len)
        out['img']=torch.tensor(video_data[0]).float()
        out['video_input']=torch.tensor(video_data[1]).float()
        out['raw_video']=torch.tensor(video_data[2]).float()

        audio_data=self.process_audio(data)
        out['audio_input']=torch.tensor(audio_data[0]).float()

        pose_data=self.process_pose(data)
        out['pose']=torch.tensor(pose_data).float()

        return out

    def process_video(self,data,max_len=None):
        r'''返回(img,video_input,raw_video),frame_index这样的数据'''
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
        # 到时候需要这么写
        # raw_video=self.read_video(data['path'])
        raw_video=video_data/255
        img=raw_video[0]
        return (img,input_video,raw_video),frame_index
    
    def read_video(self,path):
        reader = imageio.get_reader(path)
        driving_video = []
        try:
            driving_video=[im for im in reader]
        except RuntimeError:
            pass
        reader.close()
        return driving_video

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

        pose_params = np.concatenate((angles, translations, crop), axis=1)

        # 获取有效数据，3DMM不像音频，不存在越界情况，因此可以直接通过frame_index获得
        frame_index_list=data['frame_index']
        pose=pose_params[frame_index_list]

        return np.array(pose)

    def collater(self, samples):
        # 对齐数据
        audio_input = [s['audio_input'] for s in samples]
        video_input = [s['video_input'] for s in samples]
        pose = [s['pose'] for s in samples]
        img = [s['img'] for s in samples]
        raw_video = [s['raw_video'] for s in samples]

        audio_input=pad_sequence(audio_input,batch_first =True)
        video_input=pad_sequence(video_input,batch_first =True)
        pose=pad_sequence(pose,batch_first =True)
        img=pad_sequence(img,batch_first =True)
        raw_video=pad_sequence(raw_video,batch_first =True)

        data={}
        data.update({
            'audio_input': audio_input,
            'video_input': video_input,
            'pose': pose,
            'img': img,
            'raw_video': raw_video
        })
        return data

# 测试代码
if __name__=='__main__':
    import os,sys
    import yaml

    config={}
    yaml_file=['config/data_process/common.yaml','config/dataset/common.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
       
    dataset=RenderDataset(config,type='train',max_len=50)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3, 
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collater
    )     
    for data in dataloader:
        for key,value in data.items():
            print('{}形状为{}'.format(key,value.shape))
