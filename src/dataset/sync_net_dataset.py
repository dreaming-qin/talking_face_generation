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
        r'''获得id和exp，用于等会合成landmark
            获得hubert
        '''
        out={}
        sample=self.filenames[idx]

        with open(sample,'rb') as f:
            byte_file=f.read()
        byte_file=zlib.decompress(byte_file)
        data= pickle.loads(byte_file)
        data_3DMM=data['face_coeff']['coeff']

        face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
        face3d_exp=torch.tensor(face3d_exp).float()

        face3d_id=data_3DMM[:, 0:80]
        face3d_id=torch.tensor(face3d_id).float()

        hubert=torch.tensor(data['audio_hugebert']).float()

        # 对齐3dmm数据和hubert
        if len(face3d_exp)*2<len(hubert):
            hubert=hubert[:len(face3d_exp)*2]
        else:
            face3d_exp=face3d_exp[:len(hubert)//2]
            face3d_id=face3d_id[:len(hubert)//2]
        
        out['hubert']=hubert
        mouth_lamdmark=get_lm_by_3dmm(face3d_id,face3d_exp)[:,48:]
        mouth_lamdmark=mouth_lamdmark.detach()
        # 得到mouth landmark后，因为要作为模型输入，需要标准化到（-1,1）中
        # 1.按照原点对齐
        original = torch.sum(mouth_lamdmark,dim=1) / mouth_lamdmark.shape[1]
        mouth_lamdmark =mouth_lamdmark - original.reshape(-1,1,2)
        # 2.规整到（-1,1）中
        mouth_lamdmark=mouth_lamdmark.permute(0,2,1).reshape(-1,40)
        max_lmd=torch.max(torch.abs(mouth_lamdmark),dim=1)[0]
        mouth_lamdmark=(mouth_lamdmark.T/max_lmd).T
        out['mouth_landmark']=mouth_lamdmark
        return out

    def get_sample_per_video(self,mouth_landmark,hubert):
        r'''获得所有样例后，从样例中拿取正负样本
        输入：
            mouth_landmark：(B,len[i],20*2)，tensor
            hubert：(B, len[i],2, 1024)，tensor
        输出：
            landmark：(B*sample_per_video, 5, 20*2)，tensor
            hubert：(B*sample_per_video, 10, 1024)，tensor
            label：(B*sample_per_video), tensor，正样本是1，负样本是0
        '''

        # 随机获得数据
        y_len=[len(c) for c in mouth_landmark]
        mouth_lst, mel_lst, label_lst = [], [], []
        while len(mouth_lst) < len(hubert)*self.sample_per_video:
            for i in range(len(mouth_landmark)):
                if self.type=='train':
                    is_pos_sample = random.choice([True, False])
                else:
                    is_pos_sample = True
                exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                mouth_clip = mouth_landmark[i][exp_idx: exp_idx+5]
                assert mouth_clip.shape[0]==5, f"exp_idx={exp_idx},y_len={y_len[i]}"
                if is_pos_sample:
                    mel_clip = hubert[i][exp_idx*2: exp_idx*2 + 10]
                    label_lst.append(1.)
                else:
                    random_cnt=random.random()
                    if  random_cnt< 0.25:
                        wrong_spk_idx = random.randint(a=0, b=len(y_len)-1)
                        wrong_exp_idx = random.randint(a=0, b=y_len[wrong_spk_idx]-1-5)
                        while wrong_exp_idx == exp_idx:
                            wrong_exp_idx = random.randint(a=0, b=y_len[wrong_spk_idx]-1-5)
                        mel_clip = hubert[wrong_spk_idx][wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10
                    elif random_cnt < 0.5:
                        wrong_exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                        while wrong_exp_idx == exp_idx:
                            wrong_exp_idx = random.randint(a=0, b=y_len[i]-1-5)
                        mel_clip = hubert[i][wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10
                    else:
                        left_offset = max(-5, -exp_idx)
                        right_offset = min(5, (y_len[i]-5-exp_idx))
                        exp_offset = random.randint(a=left_offset, b=right_offset)
                        while abs(exp_offset) <= 1:
                            exp_offset = random.randint(a=left_offset, b=right_offset)
                        wrong_exp_idx = exp_offset + exp_idx
                        mel_clip = hubert[i][wrong_exp_idx*2: wrong_exp_idx*2 + 10]
                        assert mel_clip.shape[0]==10, y_len[i]-wrong_exp_idx
                    label_lst.append(0.)
                mouth_lst.append(mouth_clip)
                mel_lst.append(mel_clip)
        
        mel_clips = torch.stack(mel_lst)
        mouth_clips = torch.stack(mouth_lst).reshape(len(mouth_lst),5,-1)
        labels = torch.tensor(label_lst).float()
        
        return mouth_clips,mel_clips,labels

    def collater(self, samples):
        r'''需要hugbert和唇部旁边的20个landmark，landmark注意归一化
            返回：
                hugbert: [frame*2,1024]
                landmark: [frame,40]
        '''
        # 对齐数据
        landmark=[s['mouth_landmark'] for s in samples]
        hubert=[s['hubert'] for s in samples]
        landmark,hubert,label=self.get_sample_per_video(landmark,hubert)
        data={'hubert':hubert,'mouth_landmark':landmark,'label':label}
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


