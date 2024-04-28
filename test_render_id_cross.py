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
    
    # 拿数据
    data=get_data(config)

    print('生成视频中...')
    # 送入render
    fake_video=[]
    input_img=data['input_img']
    driving_src=data['driving_src']
    # 由于GPU限制，10张10张送
    frame_num=config['frame_num']
    i=-frame_num
    for i in range(0,len(input_img)-frame_num,frame_num):
        output_dict = render(input_img[i:i+frame_num], driving_src[i:i+frame_num])
        # 得到结果(B,3,H,W)
        fake_video.append(output_dict['fake_image'])
    output_dict = render(input_img[i+frame_num:], driving_src[i+frame_num:])
    fake_video.append(output_dict['fake_image'])
    fake_video=torch.cat(fake_video)

    # 从左到右以此是（参考图片，GT视频，生成视频）
    file='src_{}_tgt_{}.mp4'.format(os.path.basename(config['src_file']).replace('.pkl',''),
        os.path.basename(config['target_file']).replace('.pkl',''))
    
    real_video=data['face_video']
    fake_video=fake_video.permute(0,2,3,1)
    save_file='{}/result/merge/{}'.format(config['result_dir'],file)
    video=torch.cat((input_img[0].permute(1,2,0).expand(len(real_video),-1,-1,-1)
                     ,real_video,fake_video),dim=2)
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

    src_data=decompress(config['src_file'])
    tgt_data=decompress(config['target_file'])

    ans={}
    # 把数据放到device中
    ans['face_coeff']={}
    ans['face_coeff']['coeff']=torch.tensor(tgt_data['face_coeff']['coeff']).float().to(device)
    ans['face_coeff']['transform_params']= \
        torch.tensor(tgt_data['face_coeff']['transform_params']).float().to(device)

    #[B,3,H,W]
    face_video=torch.tensor((src_data['face_video']/255*2)-1).float().to(device)
    real_video=face_video.permute(0,3,1,2)
    # 选第一张作为源图片
    input_img=real_video[0].expand(len(ans['face_coeff']['coeff']),-1,-1,-1)
    # 获得驱动源(B,73,win size)
    driving_src=get_drving(ans,config['render_win_size'])


    # 第一步，拿用到的数据
    # 音频路径
    ans['path']=tgt_data['path']
    ans['input_img']=input_img
    ans['driving_src']=driving_src
    ans['face_video']=torch.tensor((tgt_data['face_video']/255*2)-1).float().to(device)

    return ans

def get_drving(data,win_size):
    '''输入是字典数据，value是torch类型
    输出是驱动3dmm，形状是[B,73(exp 3dmm+pose),2*win_size+1]'''
    # 表情3dmm信息
    data_3DMM=data['face_coeff']['coeff']
    # [len,64]
    face3d_exp = data_3DMM[:, 80:144]  # expression 3DMM range
    
    # 姿势信息
    mat_dict = data['face_coeff']
    np_3dmm = mat_dict["coeff"]
    angles = np_3dmm[:, 224:227]
    translations = np_3dmm[:, 254:257]
    np_trans_params = mat_dict["transform_params"]
    crop = np_trans_params[:, -3:]
    # [len,9]
    pose_params = torch.cat((angles, translations, crop), dim=1)

    driving_src=torch.cat((face3d_exp,pose_params),dim=1).unsqueeze(0)

    driving_src=get_window(driving_src,win_size)
    driving_src=driving_src.squeeze().permute(0,2,1)
    return driving_src

if __name__ == '__main__':
    import os,sys
    import yaml

    # audio_file='audio_happy007.mp4'
    # img_file='img_sad015.mp4'
    # pose_file='pose_angry011.mp4'

    

    config={}
    yaml_file=['config/dataset/common.yaml','config/model/render.yaml']
    for a in yaml_file:
        with open(a,'r',encoding='utf8') as f:
            config.update(yaml.safe_load(f))
    
    # 由于GPU限制，得10张10张的往GPU送
    config['frame_num']=10
    # 设置生成的视频数量最大值
    config['video_num']=100

    # 当前的输入文件为了省事，是面向已经处理好的pkl文件
    # config['src_file']='data_mead/format_data/test/0/M007_front_happy_level_2_007.pkl'
    config['src_file']='data_mead/format_data/test/0/M022_front_neutral_level_1_022.pkl'
    config['target_file']='data_mead/format_data/test/0/M005_front_surprised_level_3_003.pkl'
    # config['target_file']='data_mead/format_data/test/0/M003_front_angry_level_3_025.pkl'
    # config['target_file']='data_mead/format_data/test/0/M005_front_angry_level_3_019.pkl'

        
    generate_video(config)

    # test，获得其它方法的结果
    # method=['make','pc_avs','exp3DMM']
    # for b in method:
    #     config['result_dir']=f'result/{b}'
    #     get_metrices(config)
