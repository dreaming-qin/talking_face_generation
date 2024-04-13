import os
import torch
from tqdm import tqdm
import glob
import zlib
import pickle
import numpy as np
import imageio
import torchvision
import shutil


# test
if __name__=='__main__':
    import os
    import sys
    path=sys.path[0]
    sys.path.append(os.path.join(path,'Deep3DFaceRecon_pytorch'))
    sys.path.append(os.path.join(path,'syncnet_python'))
    sys.path.append(os.path.join(path,'TDDFA_V2_master'))


from src.model.render.render import Render
from src.model.exp3DMM.exp3DMM import Exp3DMM
from src.util.model_util import freeze_params
from src.util.util import get_window
from test_render import save_video

'''对exp 3dmm，需要知道输入来源不同的情况，因此得补全'''

@torch.no_grad()
def generate_video(config):
    '''输入是auido mfcc,style_clip ,video, img。
    输出是视频'''

    # 加载device
    device = torch.device(config['device_id'][0] if torch.cuda.is_available() else "cpu")

    #render model
    render=Render(config['mapping_net'],config['warpping_net'],
                         config['editing_net'],config['common'])
    render=render.to(device)
    render= torch.nn.DataParallel(render, device_ids=config['device_id'])
    # 必须得有预训练文件
    print('render加载预训练模型{}'.format(config['render_pre_train']))
    state_dict=torch.load(config['render_pre_train'],map_location=torch.device('cpu'))
    render.load_state_dict(state_dict)
    freeze_params(render)
    render.eval()
    
    # 3dmm model
    exp_model=Exp3DMM(config)
    exp_model=exp_model.to(device)
    exp_model= torch.nn.DataParallel(exp_model, device_ids=config['device_id'])
    print('exp 3dmm加载预训练模型{}'.format(config['exp_3dmm_pre_train']))
    # 必须得有预训练文件
    state_dict=torch.load(config['exp_3dmm_pre_train'],map_location=torch.device('cpu'))
    exp_model.load_state_dict(state_dict)
    freeze_params(exp_model)
    exp_model.eval()

    # 拿数据
    data=get_data(config)

    print('生成视频中...')
    # 生成exp 3dmm
    exp_result=exp_model(data['video'].permute(0,1,4,2,3),data['audio'])

    driving_src=torch.cat((exp_result,data['pose']),dim=2)
    driving_src=get_window(driving_src,config['render_win_size'])
    # (len,dim,win)
    driving_src=driving_src.squeeze().permute(0,2,1)
    # 送入render
    fake_video=[]
    # 由于GPU限制，10张10张送
    frame_num=config['frame_num']
    i=-frame_num
    for i in range(0,len(data['img'])-frame_num,frame_num):
        output_dict = render(data['img'][i:i+frame_num], driving_src[i:i+frame_num])
        # 得到结果(B,3,H,W)
        fake_video.append(output_dict['fake_image'])
    output_dict = render(data['img'][i+frame_num:], driving_src[i+frame_num:])
    fake_video.append(output_dict['fake_image'])
    fake_video=torch.cat(fake_video)

    # 保存结果
    file='id_{}_pose_{}_audio_{}.mp4'.format(os.path.basename(config['img_file']).replace('.pkl',''),
                                             os.path.basename(config['pose_file']).replace('.pkl',''),
                                             os.path.basename(config['audio_file']).replace('.pkl',''))
    save_file='{}/result/merge/{}'.format(config['result_dir'],file)
    # 从左到右以此是（源图片，姿势视频，音频视频，生成视频）
    # test
    mask_video=data['video'].squeeze(0)[5:-5]

    img_video=data['img'].permute(0,2,3,1)
    pose_real_video=data['pose_real_video']
    audio_real_video=data['audio_real_video']
    fake_video=fake_video.permute(0,2,3,1)
    video=torch.cat((mask_video,img_video,pose_real_video,audio_real_video,fake_video),dim=2)
    save_video(video,data,save_file)

    return

def decompress(file):
    # 解压pkl文件
    with open(file,'rb') as f:
        byte_file=f.read()
    byte_file=zlib.decompress(byte_file)
    data= pickle.loads(byte_file)
    return data

def get_data(config):
    '''
        获得输入，输出，并放到device上
    '''
    device = torch.device(config['device_id'][0] if torch.cuda.is_available() else "cpu")

    # 存最后的data
    ans={}

    audio_file=config['audio_file']
    img_file=config['img_file']
    pose_file=config['pose_file']
    audio_data=decompress(audio_file)
    img_data=decompress(img_file)
    pose_data=decompress(pose_file)

    # 第一步，拿用到的数据
    # 音频路径
    ans['path']=audio_data['path']

    # pose gt视频
    ans['pose_real_video']=torch.tensor((pose_data['face_video']/255*2)-1).float().to(device)
    # audio gt视频
    ans['audio_real_video']=torch.tensor((audio_data['face_video']/255*2)-1).float().to(device)

    # 选第一张为源图片
    # (3,h,w)
    img_video=torch.tensor((img_data['face_video']/255*2)-1).float().to(device)
    ans['img']=img_video[0].expand(len(ans['audio_real_video']),-1,-1,-1).permute(0,3,1,2)

    # 输入视频
    video=pose_data['face_video']
    # mask_list=pose_data['mouth_mask']
    # for img,mask in zip(video,mask_list):
    #     if mask[1]-mask[0]>0 and mask[3]-mask[2]>0:
    #         mask_width=config['mask_width']
    #         mask_height=config['mask_height']
    #         top,bottom,left,right=mask
    #         top_temp=max(0,min(top,top-((mask_height-bottom+top)//2)))
    #         bottom_temp=min(img.shape[0],max(bottom,bottom+((mask_height-bottom+top)//2)))
    #         left_temp=max(0,min(left,left-((mask_width-right+left)//2)))
    #         right_temp=min(img.shape[1],max(right,right+((mask_width-right+left)//2)))
    #         noise=np.random.randint(0,256,(bottom_temp-top_temp,right_temp-left_temp,3))
    #         img[top_temp:bottom_temp,left_temp:right_temp]=noise
    ans['video']=torch.tensor((video/255*2)-1).float().to(device)

    # 输入音频
    ans['audio']=torch.tensor(audio_data['audio_mfcc']).float().to(device)

    # pose信息
    mat_dict = pose_data['face_coeff']
    np_3dmm = mat_dict["coeff"]
    angles = np_3dmm[:, 224:227]
    translations = np_3dmm[:, 254:257]
    np_trans_params = mat_dict["transform_params"]
    crop = np_trans_params[:, -3:]
    # [len,9]
    pose_params = np.concatenate((angles, translations, crop), axis=1)
    ans['pose']=torch.tensor(pose_params).float().to(device)

    # 第二步，对齐数据
    min_len=min(len(ans['audio_real_video']),len(ans['pose_real_video']))
    ans['pose']=ans['pose'][:min_len]
    ans['pose_real_video']=ans['pose_real_video'][:min_len]
    ans['audio_real_video']=ans['audio_real_video'][:min_len]
    ans['img']=ans['img'][:min_len]
    # 输入音频
    ans['audio']=ans['audio'][:min_len]
    # 因为audio窗口问题，需要扩展
    temp_first=ans['audio'][0].expand(config['audio_win_size'],-1,-1)
    temp_last=ans['audio'][-1].expand(config['audio_win_size'],-1,-1)
    ans['audio']=torch.cat((temp_first,ans['audio'],temp_last))
    # 输入视频
    ans['video']=ans['video'][:min_len]
    # 因为video窗口问题，需要扩展
    temp_first=ans['video'][0].expand(config['audio_win_size'],-1,-1,-1)
    temp_last=ans['video'][-1].expand(config['audio_win_size'],-1,-1,-1)
    ans['video']=torch.cat((temp_first,ans['video'],temp_last))

    # 第三步，最后的一些处理
    for key,value in ans.items():
        if 'path' == key:
            continue
        ans[key]=value.unsqueeze(0)
    ans['img']=ans['img'].squeeze(0)
    ans['audio_real_video']=ans['audio_real_video'].squeeze(0)
    ans['pose_real_video']=ans['pose_real_video'].squeeze(0)

    return ans


if __name__ == '__main__':
    import os,sys
    import yaml

    # audio_file='audio_happy007.mp4'
    # img_file='img_sad015.mp4'
    # pose_file='pose_angry011.mp4'

    

    config={}
    yaml_file=['config/dataset/common.yaml','config/model/render.yaml'
               ,'config/model/exp3DMM.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
    
    # 由于GPU限制，得10张10张的往GPU送
    config['frame_num']=10
    # 设置生成的视频数量最大值
    config['video_num']=100


    # id_W016_front_sad_level_2_015_pose_M019_front_happy_level_1_025_audio_M003_front_angry_level_3_011
    # id_M003_front_angry_level_1_030_pose_M019_front_sad_level_3_013_audio_M003_front_angry_level_3_011
    # id_M003_front_angry_level_1_030_pose_M003_front_angry_level_3_011_audio_M003_front_angry_level_3_011
    # 当前的输入文件为了省事，是面向已经处理好的pkl文件
    config['audio_file']='data_mead/format_data/train/0/M019_front_sad_level_3_013.pkl'
    config['img_file']='data_mead/format_data/train/26/M003_front_angry_level_3_011.pkl'
    config['pose_file']='data_mead/format_data/train/0/M019_front_sad_level_3_013.pkl'


    # config['audio_file']='data_mead/format_data/test/0/M003_front_disgusted_level_1_009.pkl'
    # config['img_file']='data_mead/format_data/test/0/M005_front_neutral_level_1_001.pkl'
    # config['pose_file']='data_mead/format_data/test/0/M009_front_angry_level_2_022.pkl'



    generate_video(config)

    # test，获得其它方法的结果
    # method=['make','pc_avs','exp3DMM']
    # for b in method:
    #     config['result_dir']=f'result/{b}'
    #     get_metrices(config)
