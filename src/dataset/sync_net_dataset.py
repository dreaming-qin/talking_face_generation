import glob
import zlib
import pickle
import torch
import random
import os



# 测试代码
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.pop(0)
    for _ in range(2):
        path=os.path.dirname(path)
    sys.path.append(path)
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))


from src.util.util_3dmm import get_lm_by_3dmm



class SyncNetDataset(torch.utils.data.Dataset):
    r'''用于训练EXP 3DMM提取模块的数据集，需要获得输入，以及LOSS函数需要的输入'''

    def __init__(self,config,type='train',sample_per_video=1):
        r'''sample per video指的是从每个视频中获得多少样例（正样例和负样例总和）'''
        format_path=config['format_output_path']
        self.filenames=sorted(glob.glob(f'{format_path}/{type}/*/*.pkl'))
        self.type=type
        self.sample_per_video=sample_per_video

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        r'''1.获得id和exp，用于等会合成唇部图片
            2. 获得mfcc
        '''
        out={}
        sample=self.filenames[idx]

        with open(sample,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)

        # 1. 获得3DMM 
        data_3DMM=data['face_coeff']['coeff']
        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
        face3d_exp=torch.tensor(face3d_exp).float()
        face3d_id=data_3DMM[:, 0:80]
        face3d_id=torch.tensor(face3d_id).float()

        #  2.获得mfcc
        mfcc=torch.tensor(data['audio_mfcc']).float()

        # 对齐3dmm数据和mfcc
        if len(face3d_exp)<len(mfcc):
            mfcc=mfcc[:len(face3d_exp)]
        else:
            face3d_exp=face3d_exp[:len(mfcc)]
            face3d_id=face3d_id[:len(mfcc)]
        
        # 放数据
        out['mfcc']=mfcc
        out['exp']=face3d_exp
        out['id']=face3d_id
        return out

    def get_sample_per_video(self,exp_3dmm,id_3dmm,mfcc):
        r'''获得所有样例后，从样例中拿取正负样本
        输入：
            exp_3dmm(B,len[i],64)，tensor
            id_3dmm(B,len[i],80)，tensor
            mfcc(B,len[i],28,12)，tensor
        输出：
            exp_3dmm(B*sample_per_video,64)，tensor
            id_3dmm(B*sample_per_video,80)，tensor
            mfcc(B*sample_per_video,28,12)，tensor
            label：(B*sample_per_video), tensor，正样本是1，负样本是0
        '''

        y_len=[len(c) for c in mfcc]
        exp_lst, id_lst,mfcc_lst, label_lst = [], [], [],[]
        for video_idx in range(len(mfcc)):
            for _ in range(self.sample_per_video):
                if self.type=='train':
                    is_pos_sample = random.choice([True, False])
                else:
                    is_pos_sample = True
                choose_idx = random.randint(a=0, b=y_len[video_idx]-1)
                exp_clip=exp_3dmm[video_idx][choose_idx]
                id_clip=id_3dmm[video_idx][choose_idx]
                if is_pos_sample:
                    mouth_clip = mfcc[video_idx][choose_idx]
                    label_lst.append(1.)
                else:
                    random_cnt=random.random()
                    if  random_cnt< 0.5:
                        # 从别人的视频中选MFCC
                        wrong_video_idx = video_idx
                        while wrong_video_idx == video_idx:
                            wrong_video_idx = random.randint(a=0, b=len(y_len)-1)
                        wrong_choose_idx = random.randint(a=0, b=y_len[wrong_video_idx]-1)
                        mouth_clip = mfcc[wrong_video_idx][wrong_choose_idx]
                    else :
                        # 从自己的视频中选MFCC
                        wrong_choose_idx = choose_idx
                        while abs(wrong_choose_idx-choose_idx)<5:
                            wrong_choose_idx = random.randint(a=0, b=y_len[video_idx]-1)
                        mouth_clip = mfcc[video_idx][wrong_choose_idx]
                    label_lst.append(0.)
                exp_lst.append(exp_clip)
                id_lst.append(id_clip)
                mfcc_lst.append(mouth_clip)


        mfcc_lst = torch.stack(mfcc_lst)
        id_lst = torch.stack(id_lst)
        exp_lst = torch.stack(exp_lst)
        labels = torch.tensor(label_lst).float()
        
        return exp_lst,id_lst,mfcc_lst,labels

    def collater(self, samples):
        r'''需要hugbert和唇部旁边的20个landmark，landmark注意归一化
            返回：
                hugbert: [frame*2,1024]
                landmark: [frame,40]
        '''
        # 对齐数据
        mfcc=[s['mfcc'] for s in samples]
        face3d_exp=[s['exp'] for s in samples]
        face3d_id=[s['id'] for s in samples]
        face3d_exp,face3d_id,mfcc,label=self.get_sample_per_video(face3d_exp,face3d_id,mfcc)
        data={'exp':face3d_exp,'id':face3d_id,'mfcc':mfcc,'label':label}
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
       
    dataset=SyncNetDataset(config,type='train',sample_per_video=20)
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


